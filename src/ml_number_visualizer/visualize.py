import io
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 784),
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
        BCE = F.binary_cross_entropy_with_logits(recon_batch, x, reduction="sum") / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = BCE + KLD
        self.log("vae_train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])


# Surrogates (blue & green channels)
class BaseSurrogate(L.LightningModule):
    def __init__(
        self,
        teacher_model: TeacherModel,
        learning_rate: float = 1e-3,
        init_temperature: float = 4.0,
        min_temperature: float = 1.0,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.teacher_model = teacher_model
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.weight_decay = weight_decay

        # Start at the maximum high-entropy temperature
        self.temperature = init_temperature

    def _get_target_probs(self, x: Tensor) -> Tensor:
        return self.teacher_model.get_target_probs(x)

    def _shared_distillation_step(self, batch: tuple[Tensor, Any], step_name: str):
        x, _ = batch

        with torch.no_grad():
            teacher_probs = torch.clamp(self._get_target_probs(x), min=1e-7, max=1.0)
            teacher_soft = torch.pow(teacher_probs, 1.0 / self.temperature)
            teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)

        surrogate_logits = self(x)
        surrogate_log_probs = F.log_softmax(surrogate_logits / self.temperature, dim=1)
        loss = F.kl_div(surrogate_log_probs, teacher_soft, reduction="batchmean") * (
            self.temperature**2
        )
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log("distill_temp", self.temperature, prog_bar=False)
        return loss

    def training_step(self, batch: tuple[Tensor, Any], _: int):
        return self._shared_distillation_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Any], _: int):
        return self._shared_distillation_step(batch, "val")

    @property
    def max_epochs(self) -> int:
        max_epochs = self.trainer.max_epochs if self.trainer else None
        return max_epochs if max_epochs else 10

    def on_train_epoch_end(self):
        max_epochs = self.max_epochs
        epoch = self.current_epoch

        if max_epochs > 1:
            decay_step = (self.init_temperature - self.min_temperature) / (max_epochs - 1)
            self.temperature = max(
                self.min_temperature, self.init_temperature - (epoch * decay_step)
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


class LightningSurrogateFlat(BaseSurrogate):
    def __init__(self, teacher_model: TeacherModel, **kwargs):
        super().__init__(teacher_model, **kwargs)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 2048),
            nn.Mish(),
            nn.Linear(2048, 1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 10),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class LightningSurrogateCNN(BaseSurrogate):
    def __init__(self, teacher_model: TeacherModel, **kwargs):
        super().__init__(teacher_model, **kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# Dream generators
def generate_vae_channel_differentiable(
    oracle_model: nn.Module, vae: LightningVAE, device: torch.device, num_steps: int = 1500
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = 10
    target_classes = torch.arange(batch_size, device=device, dtype=torch.long)

    z = torch.randn((batch_size, vae.latent_dim), device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=0.1)

    for _ in range(num_steps):
        optimizer.zero_grad()
        images = vae.decoder(z).sigmoid().view(batch_size, 1, 28, 28)
        logits = oracle_model(images)
        loss = F.cross_entropy(logits, target_classes)
        loss.backward()
        optimizer.step()

    final_images = vae.decoder(z).sigmoid().view(batch_size, 1, 28, 28)
    probs = F.softmax(oracle_model(final_images), dim=1)
    confidences = probs[torch.arange(batch_size), target_classes].detach().cpu().numpy()

    return final_images.detach().cpu().squeeze(1).numpy(), confidences


def generate_vae_channel_non_differentiable(
    teacher_model: TeacherModel,
    vae: LightningVAE,
    train_loader,
    val_loader,
    device: torch.device,
    num_steps: int = 1500,
) -> tuple[np.ndarray, np.ndarray, LightningSurrogateFlat]:
    logger.info("Model is not differentiable. Training spatial oracle surrogate for VAE...")
    surrogate = LightningSurrogateFlat(teacher_model=teacher_model)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=2, mode="min")

    cnn_trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[early_stop],
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="bf16-mixed",
    )
    cnn_trainer.fit(surrogate, train_dataloaders=train_loader, val_dataloaders=val_loader)

    surrogate.to(device).eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    images, confs = generate_vae_channel_differentiable(surrogate, vae, device, num_steps)
    return images, confs, surrogate


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


# Shared computation core — called by both the static pipeline and the video pipeline.
def _compute_representations(
    teacher_model: TeacherModel,
    vae: LightningVAE,
    train_loader,
    val_loader,
    device: torch.device,
    num_steps: int = 1500,
    max_surrogate_epochs: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train surrogates, dream channels, and return raw arrays.

    Returns:
        (vae_images, cnn_images, flat_images, vae_confs, cnn_confs, flat_confs)
        Each *_images array is shape (10, 28, 28); each *_confs array is shape (10,).
    """
    # Flat surrogate — blue channel (relational logic)
    logger.info("Training relationally optimal surrogate (Flat)...")
    flat_surrogate = LightningSurrogateFlat(teacher_model=teacher_model)
    flat_trainer = L.Trainer(
        max_epochs=max_surrogate_epochs,
        accelerator="auto",
        callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.001, patience=2, mode="min")],
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="bf16-mixed",
    )
    flat_trainer.fit(flat_surrogate, train_dataloaders=train_loader, val_dataloaders=val_loader)
    flat_surrogate.to(device).eval()
    for p in flat_surrogate.parameters():
        p.requires_grad = False

    # VAE red channel — differentiable path uses direct backprop, else trains a CNN oracle
    if teacher_model.is_differentiable:
        oracle_model = teacher_model.get_differentiable_model()
        vae_images, vae_confs = generate_vae_channel_differentiable(
            oracle_model, vae, device, num_steps
        )

        # CNN surrogate — green channel (spatial machine logic)
        logger.info("Training spatially optimal surrogate (CNN)...")
        cnn_surrogate = LightningSurrogateCNN(teacher_model=teacher_model)
        cnn_trainer = L.Trainer(
            max_epochs=max_surrogate_epochs,
            accelerator="auto",
            callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.001, patience=2, mode="min")],
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            precision="bf16-mixed",
        )
        cnn_trainer.fit(
            cnn_surrogate, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        cnn_surrogate.to(device).eval()
        for p in cnn_surrogate.parameters():
            p.requires_grad = False
    else:
        vae_images, vae_confs, cnn_surrogate = generate_vae_channel_non_differentiable(
            teacher_model, vae, train_loader, val_loader, device, num_steps
        )

    logger.info("Dreaming GREEN + BLUE Channels in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_cnn = executor.submit(dream_input_space, cnn_surrogate, device, num_steps, 1e-5)
        fut_flat = executor.submit(dream_input_space, flat_surrogate, device, num_steps, 1e-4)
        cnn_images, cnn_confs = fut_cnn.result()
        flat_images, flat_confs = fut_flat.result()

    return vae_images, cnn_images, flat_images, vae_confs, cnn_confs, flat_confs


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
    logger.info(f"Analyzing Architecture: '{model_name}'")

    vae_images, cnn_images, flat_images, vae_confs, cnn_confs, flat_confs = (
        _compute_representations(teacher_model, vae, train_loader, val_loader, device, num_steps)
    )

    plot_path = Path(f"plots/{model_name}")
    plot_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Compositing representational maps to {plot_path}/")

    def _save_digit_plot(digit: int) -> None:
        rgb_map = np.zeros((28, 28, 3), dtype=np.float32)
        rgb_map[:, :, 0] = vae_images[digit]
        rgb_map[:, :, 1] = cnn_images[digit]
        rgb_map[:, :, 2] = flat_images[digit]
        rgb_map = np.clip(rgb_map, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb_map)

        mean_conf = (vae_confs[digit] + cnn_confs[digit] + flat_confs[digit]) / 3.0 * 100
        title_text = (
            f"{model_name.title()} Internal Representation: '{digit}'\n"
            + f"Mean Confidence: {mean_conf:.1f}%"
        )

        ax.set_title(title_text, fontsize=9, pad=10)
        ax.axis("off")
        fig.savefig(plot_path / f"{digit}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)

    with ThreadPoolExecutor() as executor:
        list(executor.map(_save_digit_plot, target_digits))


# Public pipeline API
def generate_visualizations_for_model(
    model_name: str,
    teacher_model: TeacherModel,
    target_digits: Iterable[int] = range(10),
    num_steps: int = 1500,
) -> None:
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
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="bf16-mixed",
    )
    vae_trainer.fit(vae, train_dataloaders=train_loader)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    analyze_model(
        teacher_model=teacher_model,
        model_name=model_name,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_steps=num_steps,
        target_digits=target_digits,
    )

    logger.success(f"Visualizations completed for '{model_name}'!")


def generate_visualizations_for_models(
    models: Mapping[str, TeacherModel],
    target_digits: Iterable[int] = range(10),
    num_steps: int = 1500,
) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using execution target: {device}. Processing {len(models)} models.")

    train_loader, val_loader, _ = get_dataset()

    logger.info("Training Global VAE (Human Prior)...")
    vae = LightningVAE()
    vae_trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="bf16-mixed",
    )
    vae_trainer.fit(vae, train_dataloaders=train_loader)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Process each model sequentially
    for model_name, teacher_adapter in models.items():
        try:
            analyze_model(
                teacher_model=teacher_adapter,
                model_name=model_name,
                vae=vae,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_steps=num_steps,
                target_digits=target_digits,
            )
            logger.success(f"Successfully composited visualizations for '{model_name}'.")

        except Exception as e:
            logger.error(f"Pipeline failed for model '{model_name}': {e}")
            continue

    logger.success("Batch visualization pipeline complete!")


# ---------------------------------------------------------------------------
# Evolution video pipeline
# ---------------------------------------------------------------------------

def _make_epoch_frame(
    label: str,
    model_name: str,
    vae_images: np.ndarray,
    cnn_images: np.ndarray,
    flat_images: np.ndarray,
    vae_confs: np.ndarray,
    cnn_confs: np.ndarray,
    flat_confs: np.ndarray,
) -> "PIL.Image.Image":  # noqa: F821
    """Render a 2×5 grid of all 10 digit representations into a PIL Image."""
    from PIL import Image

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle(f"{model_name.upper()} — {label}", fontsize=14, fontweight="bold")

    for digit in range(10):
        ax = axes[digit // 5][digit % 5]
        rgb_map = np.zeros((28, 28, 3), dtype=np.float32)
        rgb_map[:, :, 0] = vae_images[digit]
        rgb_map[:, :, 1] = cnn_images[digit]
        rgb_map[:, :, 2] = flat_images[digit]
        rgb_map = np.clip(rgb_map, 0.0, 1.0)
        mean_conf = (vae_confs[digit] + cnn_confs[digit] + flat_confs[digit]) / 3.0 * 100
        ax.imshow(rgb_map)
        ax.set_title(f"'{digit}' — {mean_conf:.0f}%", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() detaches from the BytesIO buffer


def generate_video_for_model(
    model_name: str,
    snapshot_paths: list[Path],
    teacher_factory: Callable[[Path], TeacherModel],
    frame_labels: list[str],
    vae: LightningVAE,
    train_loader,
    val_loader,
    device: torch.device,
    num_steps: int = 300,
    max_surrogate_epochs: int = 3,
) -> None:
    """Generate an animated GIF showing representation evolution across snapshots."""
    from PIL import Image  # noqa: F401 — ensure PIL is imported for type

    logger.info(f"Generating evolution video for '{model_name}' ({len(snapshot_paths)} frames)...")
    frames: list = []

    for snapshot_path, label in zip(snapshot_paths, frame_labels, strict=True):
        logger.info(f"  [{model_name}] {label} ...")
        try:
            teacher = teacher_factory(snapshot_path)
            arrays = _compute_representations(
                teacher, vae, train_loader, val_loader, device, num_steps, max_surrogate_epochs
            )
            vae_imgs, cnn_imgs, flat_imgs, vae_c, cnn_c, flat_c = arrays
            frames.append(
                _make_epoch_frame(label, model_name, vae_imgs, cnn_imgs, flat_imgs, vae_c, cnn_c, flat_c)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  Skipping frame '{label}' for '{model_name}': {exc}")

    if not frames:
        logger.warning(f"No frames generated for '{model_name}' — skipping GIF.")
        return

    plot_path = Path(f"plots/{model_name}")
    plot_path.mkdir(parents=True, exist_ok=True)
    gif_path = plot_path / "evolution.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,  # ms per frame
        loop=0,
    )
    logger.success(f"Evolution GIF saved → {gif_path}")


def generate_videos_for_models(
    nn_snapshots: dict[str, tuple[list[Path], list[str]]],
    sklearn_snapshots: dict[str, tuple[list[Path], list[str]]],
    num_steps: int = 300,
    max_surrogate_epochs: int = 3,
) -> None:
    """Generate training-evolution GIFs for all models with available snapshots."""
    import joblib

    from ml_number_visualizer.neural_networks import QMNISTClassifier
    from ml_number_visualizer.protocols import PyTorchAdapter, SklearnAdapter

    all_snapshots = {**nn_snapshots, **sklearn_snapshots}
    if not all_snapshots:
        logger.info("No snapshots available — skipping video generation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataset()

    # One shared VAE is enough — it represents the global human prior, not the teacher.
    logger.info("Training Global VAE for evolution videos...")
    vae = LightningVAE()
    vae_trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        precision="bf16-mixed",
    )
    vae_trainer.fit(vae, train_dataloaders=train_loader)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Unified loop — dispatch on name prefix to pick the right teacher factory.
    for model_name, (paths, labels) in all_snapshots.items():
        if model_name.startswith("nn_"):
            strategy = model_name.split("_", 1)[1]  # "nn_flat" → "flat"

            # Default-arg capture prevents the classic loop-variable closure bug.
            def _factory(path: Path, _s: str = strategy) -> TeacherModel:
                m = QMNISTClassifier(model_type=_s)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m = m.to(device).eval()
                for param in m.parameters():
                    param.requires_grad = False
                return PyTorchAdapter(m)

        else:
            def _factory(path: Path) -> TeacherModel:  # type: ignore[misc]
                return SklearnAdapter(joblib.load(path))

        try:
            generate_video_for_model(
                model_name=model_name,
                snapshot_paths=paths,
                teacher_factory=_factory,
                frame_labels=labels,
                vae=vae,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_steps=num_steps,
                max_surrogate_epochs=max_surrogate_epochs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Video generation failed for '{model_name}': {exc}")

    logger.success("Evolution video generation complete!")


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
