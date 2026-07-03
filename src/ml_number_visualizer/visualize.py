import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from torch import Tensor, nn, optim

from ml_number_visualizer.dataloader import get_dataset
from ml_number_visualizer.neural_networks import QMNISTClassifier
from ml_number_visualizer.protocols import TeacherModel


# Generative prior (red channel)
class LightningVAE(L.LightningModule):
    def __init__(self, latent_dim: int = 16, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.hparams["learning_rate"] = learning_rate

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z).view(-1, 1, 28, 28), mu, logvar

    def training_step(self, batch: tuple[Tensor, Any], _: int):
        x, _ = batch
        recon_batch, mu, logvar = self(x)
        BCE = F.binary_cross_entropy(recon_batch, x, reduction="sum") / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = BCE + KLD
        self.log("vae_train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])


# Surrogates (blue & green channels)
class BaseSurrogate(L.LightningModule):
    def __init__(self, teacher_model: TeacherModel, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.teacher_model = teacher_model

        # Only freeze the model if it's a PyTorch module
        if isinstance(self.teacher_model, nn.Module):
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def _get_target_probs(self, x: Tensor) -> Tensor:
        if isinstance(self.teacher_model, nn.Module):
            with torch.no_grad():
                logits = self.teacher_model(x)
                return F.softmax(logits, dim=1)
        else:
            x_flat = x.view(x.size(0), -1).detach().cpu().numpy()
            probs = self.teacher_model.predict_proba(x_flat)
            return torch.tensor(probs, dtype=torch.float32, device=self.device)

    def _shared_distillation_step(self, batch: tuple[Tensor, Any], step_name: str):
        x, _ = batch
        target_probs = self._get_target_probs(x)
        surrogate_logits = self(x)
        loss = F.cross_entropy(surrogate_logits, target_probs)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[Tensor, Any], _: int):
        return self._shared_distillation_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Any], _: int):
        return self._shared_distillation_step(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class LightningSurrogateFlat(BaseSurrogate):
    def __init__(self, teacher_model: TeacherModel, learning_rate: float = 1e-3):
        super().__init__(teacher_model, learning_rate)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class LightningSurrogateCNN(BaseSurrogate):
    def __init__(self, teacher_model: TeacherModel, learning_rate: float = 1e-3):
        super().__init__(teacher_model, learning_rate)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# Dream generators
def dream_human_prior(
    oracle_model: nn.Module, vae: LightningVAE, device: torch.device, num_steps: int = 1500
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = 10
    target_classes = torch.arange(batch_size, device=device, dtype=torch.long)

    z = torch.randn((batch_size, vae.latent_dim), device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        images = vae.decoder(z).view(batch_size, 1, 28, 28)
        logits = oracle_model(images)
        loss = F.cross_entropy(logits, target_classes)
        loss.backward()
        optimizer.step()

    final_images = vae.decoder(z).view(batch_size, 1, 28, 28)
    probs = F.softmax(oracle_model(final_images), dim=1)
    confidences = probs[torch.arange(batch_size), target_classes].detach().cpu().numpy()

    return final_images.detach().cpu().squeeze(1).numpy(), confidences


def dream_input_space(
    surrogate: nn.Module, device: torch.device, num_steps: int = 1500, tv_weight: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = 10
    target_classes = torch.arange(batch_size, device=device, dtype=torch.long)

    raw_images = (torch.randn((batch_size, 1, 28, 28), device=device) * 0.1) - 2.0
    raw_images.requires_grad_(True)
    optimizer = optim.Adam([raw_images], lr=0.1)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        synthetic_images = torch.sigmoid(raw_images)
        logits = surrogate(synthetic_images)

        classification_loss = F.cross_entropy(logits, target_classes)
        l2_loss = 0.002 * torch.norm(synthetic_images)

        diff_h = torch.abs(synthetic_images[:, :, :, :-1] - synthetic_images[:, :, :, 1:])
        diff_v = torch.abs(synthetic_images[:, :, :-1, :] - synthetic_images[:, :, 1:, :])
        tv_loss = tv_weight * (torch.sum(diff_h) + torch.sum(diff_v))

        binarization_loss = 0.5 * torch.mean(synthetic_images * (1.0 - synthetic_images))

        total_loss = classification_loss + l2_loss + tv_loss + binarization_loss
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                blurred = TF.gaussian_blur(synthetic_images, kernel_size=3, sigma=0.5)  # type: ignore
                blurred_clamped = torch.clamp(blurred, 1e-4, 1.0 - 1e-4)
                raw_images.copy_(torch.logit(blurred_clamped))

    final_images = torch.sigmoid(raw_images)
    probs = F.softmax(surrogate(final_images), dim=1)
    confidences = probs[torch.arange(batch_size), target_classes].detach().cpu().numpy()

    return final_images.detach().cpu().squeeze(1).numpy(), confidences


# Analysis engine
def analyze_model(
    teacher_model: TeacherModel,
    model_name: str,
    vae: LightningVAE,
    train_loader,
    val_loader,
    device: torch.device,
    num_steps: int = 1500,
    target_digits: Iterable[int] = range(10),
) -> None:
    logger.info(f"\n{'=' * 50}\nAnalyzing Architecture: '{model_name.upper()}'\n{'=' * 50}")

    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=2, mode="min")

    # Train flat surrogate (blue channel)
    logger.info("Training relationally optimal surrogate (Flat)...")
    flat_surrogate = LightningSurrogateFlat(teacher_model=teacher_model)
    flat_trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[early_stop],
        enable_model_summary=False,
        logger=False,
    )
    flat_trainer.fit(flat_surrogate, train_dataloaders=train_loader, val_dataloaders=val_loader)
    flat_surrogate.to(device).eval()
    for p in flat_surrogate.parameters():
        p.requires_grad = False

    # Train CNN surrogate (green channel)
    logger.info("Training spatially optimal surrogate (CNN)...")
    cnn_surrogate = LightningSurrogateCNN(teacher_model=teacher_model)
    cnn_trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[early_stop],
        enable_model_summary=False,
        logger=False,
    )
    cnn_trainer.fit(cnn_surrogate, train_dataloaders=train_loader, val_dataloaders=val_loader)
    cnn_surrogate.to(device).eval()
    for p in cnn_surrogate.parameters():
        p.requires_grad = False

    # Determine differentiable oracle for VAE
    if isinstance(teacher_model, nn.Module):
        oracle_model = teacher_model
    else:
        logger.info(f"'{model_name}' is not differentiable. Using CNN Surrogate as Oracle for VAE.")
        oracle_model = cnn_surrogate

    # Generate the 3 dreams
    vae_images, vae_confs = dream_human_prior(oracle_model, vae, device, num_steps)
    logger.info("Dreaming GREEN Channel (Spatial Truth via CNN)...")
    cnn_images, cnn_confs = dream_input_space(cnn_surrogate, device, num_steps, tv_weight=1e-5)
    logger.info("Dreaming BLUE Channel (Relational Truth via Flat)...")
    flat_images, flat_confs = dream_input_space(flat_surrogate, device, num_steps, tv_weight=1e-4)

    # Composite requested digits
    plot_path = Path(f"plots/{model_name}")
    plot_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Compositing representational maps to {plot_path}/")

    for digit in target_digits:
        rgb_map = np.zeros((28, 28, 3), dtype=np.float32)
        rgb_map[:, :, 0] = vae_images[digit]
        rgb_map[:, :, 1] = cnn_images[digit]
        rgb_map[:, :, 2] = flat_images[digit]
        rgb_map = np.clip(rgb_map, 0.0, 1.0)

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_map)

        mean_conf = (vae_confs[digit] + cnn_confs[digit] + flat_confs[digit]) / 3.0 * 100
        title_text = (
            f"{model_name.title()} Internal Representation: '{digit}'\n"
            + f"Mean Confidence: {mean_conf:.1f}%"
        )

        plt.title(title_text, fontsize=9, pad=10)
        plt.axis("off")
        plt.savefig(plot_path / f"{digit}.png", bbox_inches="tight", dpi=150)
        plt.close()


# Public pipeline API
def generate_all_digits(num_steps: int = 1500) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using execution target: {device}")

    train_loader, val_loader, _ = get_dataset()

    logger.info("Training Global VAE (Human Prior)...")
    vae = LightningVAE()
    vae_trainer = L.Trainer(
        max_epochs=3, accelerator="auto", devices="auto", enable_model_summary=False, logger=False
    )
    vae_trainer.fit(vae, train_dataloaders=train_loader)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    architectures: tuple[Literal["flat", "cnn", "vit"], ...] = ("flat", "cnn", "vit")

    for strategy in architectures:
        model = QMNISTClassifier(model_type=strategy)
        try:
            model.load_state_dict(
                torch.load(f"./models/nn_{strategy}.pth", map_location=device, weights_only=True)
            )
        except FileNotFoundError:
            logger.warning(f"Could not find weights for '{strategy}'. Skipping.")
            continue

        model.to(device)
        analyze_model(model, strategy, vae, train_loader, val_loader, device, num_steps)

    logger.success("All models successfully visualized via RGB Compositing!")


def generate_digit_for_sklearn_model(
    model_name: str, target_digit: int, num_steps: int = 1500
) -> None:
    """Wrapper to run a specific Sklearn or PyTorch model for a single digit."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loader, val_loader, _ = get_dataset()

    vae = LightningVAE()
    vae_trainer = L.Trainer(
        max_epochs=3, accelerator="auto", devices="auto", enable_model_summary=False, logger=False
    )
    vae_trainer.fit(vae, train_dataloaders=train_loader)
    vae.to(device).eval()

    with open(f"./models/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    analyze_model(
        model,
        model_name,
        vae,
        train_loader,
        val_loader,
        device,
        num_steps,
        target_digits=[target_digit],
    )


def generate_legend() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    plot_dir = project_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    legend_path = plot_dir / "legend.png"
    logger.info(f"Exporting standalone color matrix legend to: {legend_path}")

    _, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axis("off")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", edgecolor="none", label="Red Channel: Human Prior"),
        Patch(facecolor="green", edgecolor="none", label="Green Channel: Spatial Machine Logic"),
        Patch(facecolor="blue", edgecolor="none", label="Blue Channel: Relational Logic"),
        Patch(
            facecolor="magenta",
            edgecolor="none",
            label="Magenta (Red + Blue): Human-Machine Feature Overlap",
        ),
        Patch(
            facecolor="yellow",
            edgecolor="none",
            label="Yellow (Red + Green): Structurally Bound Machine Agreement",
        ),
        Patch(
            facecolor="cyan",
            edgecolor="none",
            label="Cyan (Green + Blue): Universal Non-Human Machine Shortcut",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="White (All Combined): Universal Global Agreement",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center",
        frameon=True,
        facecolor="#F5F5F5",
        edgecolor="#D3D3D3",
        fontsize=9,
        title="Representation Interpretations",
        title_fontsize=10,
    )
    plt.savefig(legend_path, bbox_inches="tight", dpi=150)
    plt.close()


generate_legend()
