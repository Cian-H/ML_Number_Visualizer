from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from loguru import logger
from torch import Tensor, optim

from ml_number_visualizer.neural_networks import QMNISTClassifier


def generate_digit(model, target_digit: int, plot_path: Path, num_steps: int = 1500, lr: float = 0.1):
    logger.info(f"Generating '{target_digit}' representation...")
    for param in model.parameters():
        param.requires_grad = False

    # Start with a gray image with slight noise, rather than pure static
    synthetic_image = torch.randn((1, 1, 28, 28)) * 0.1
    synthetic_image.requires_grad = True

    optimizer = optim.Adam([synthetic_image], lr=lr)
    target_class = torch.tensor([target_digit])

    for step in range(1, num_steps+1):
        optimizer.zero_grad()

        # Forward pass
        logits = model(synthetic_image)
        classification_loss = F.cross_entropy(logits, target_class)

        # L2 Regularization (keep pixels from blowing up)
        l2_loss = 0.01 * torch.norm(synthetic_image)

        # Total Variation (TV) Loss (force adjacent pixels to be similar)
        # Calculates the difference between neighboring pixels
        tv_loss = torch.sum(torch.abs(synthetic_image[:, :, :, :-1] - synthetic_image[:, :, :, 1:])) + \
                  torch.sum(torch.abs(synthetic_image[:, :, :-1, :] - synthetic_image[:, :, 1:, :]))
        tv_weight = 1e-5 # Higher = smoother, Lower = noisier

        # Total objective
        total_loss = classification_loss + l2_loss + (tv_weight * tv_loss)

        total_loss.backward()
        optimizer.step()

        # Every 50 steps, slightly blur the image to kill the high-frequency static
        if step % 50 == 0:
            with torch.no_grad():
                # Apply a subtle 3x3 blur
                blurred = TF.gaussian_blur(synthetic_image, kernel_size=3, sigma=0.5) # type: ignore
                synthetic_image.copy_(blurred)

        if step == num_steps:
            prob = F.softmax(logits, dim=1)[0, target_digit].item()
            logger.info(f"Loss: {total_loss.item():.4f}, Conf: {prob*100:.2f}%")

    logger.info("Done! Plotting the result...")
    final_img = synthetic_image.detach().cpu().squeeze().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(final_img, cmap="gray")
    plt.title(f"Model's Dream of a '{target_digit}' (Smoothed)")
    plt.axis("off")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path.with_suffix(".png"))
    plt.close()

def generate_digits(model, plot_path: Path, num_steps: int = 1000, lr: float = 0.1):
    for i in range(10):
        generate_digit(model, i, plot_path / str(i), num_steps=num_steps, lr=lr)

def generate_all_digits(num_steps: int = 1000, lr: float = 0.1):
    for s in ("flat", "cnn", "vit"):
        logger.info(f"Generating number representations for {s}")
        model = QMNISTClassifier(model_type=s)
        model.load_state_dict(torch.load(f"./models/nn_{s}.pth", weights_only=True))
        model.eval()
        plot_path = Path(f"plots/{s}")
        generate_digits(model, plot_path, num_steps=num_steps, lr=lr)

def generate_all_digits_batched(
    num_steps: int = 1500,
    lr: float = 0.1,
    tv_weight: float = 1e-5
) -> None:
    # Automatically select the best hardware accelerator available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU accelerator found. Falling back to CPU.")

    logger.info(f"Using device execution target: {device}")

    architectures: tuple[Literal[flat, cnn, vit], ...] = ("flat", "cnn", "vit")

    for strategy in architectures:
        logger.info(f"Initializing architecture configuration for: '{strategy}'")

        # Load and freeze the classifier weights
        model = QMNISTClassifier(model_type=strategy)
        model.load_state_dict(torch.load(f"./models/nn_{strategy}.pth", map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        # Parallelization Matrix: Shape (10, 1, 28, 28) -> Represents digits 0 through 9
        batch_size = 10
        synthetic_images: Tensor = torch.randn((batch_size, 1, 28, 28), device=device) * 0.1
        synthetic_images.requires_grad_(True)

        optimizer = optim.Adam([synthetic_images], lr=lr)
        target_classes: Tensor = torch.arange(batch_size, device=device, dtype=torch.long)

        logger.info(f"Optimizing 10-digit generation matrix for '{strategy}' simultaneously...")

        for step in range(1, num_steps + 1):
            optimizer.zero_grad()

            # Batched forward pass: shape (10, 10)
            logits: Tensor = model(synthetic_images)

            # Cross-entropy loss averaged across the batch dimension
            classification_loss: Tensor = F.cross_entropy(logits, target_classes)

            # Batched L2 Regularization
            l2_loss: Tensor = 0.01 * torch.norm(synthetic_images)

            # Batched Total Variation (TV) Loss across spatial channels
            diff_h: Tensor = torch.abs(synthetic_images[:, :, :, :-1] - synthetic_images[:, :, :, 1:])
            diff_v: Tensor = torch.abs(synthetic_images[:, :, :-1, :] - synthetic_images[:, :, 1:, :])
            tv_loss: Tensor = torch.sum(diff_h) + torch.sum(diff_v)

            # Total objective balance
            total_loss: Tensor = classification_loss + l2_loss + (tv_weight * tv_loss)

            total_loss.backward()
            optimizer.step()

            # Every 50 steps, apply batched 3x3 spatial blurring to kill high-frequency static
            if step % 50 == 0:
                with torch.no_grad():
                    blurred: Tensor = TF.gaussian_blur(synthetic_images, kernel_size=3, sigma=0.5)
                    synthetic_images.copy_(blurred)

            if step == num_steps:
                probabilities: Tensor = F.softmax(logits, dim=1)
                # Extract diagonal elements corresponding to targeted index metrics
                target_probs = probabilities[torch.arange(batch_size), target_classes]
                avg_conf = target_probs.mean().item() * 100
                logger.info(f"[{strategy.upper()}] Final Step Loss: {total_loss.item():.4f} | Mean Target Confidence: {avg_conf:.2f}%")

        # Plotting & Export Pipeline
        plot_path = Path(f"plots/{strategy}")
        plot_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting generated plots to: {plot_path}/")

        # Move back to host RAM for plotting operations
        final_images = synthetic_images.detach().cpu().squeeze(1).numpy()

        for digit in range(batch_size):
            plt.figure(figsize=(4, 4))
            plt.imshow(final_images[digit], cmap="gray")
            plt.title(f"Model Dream: '{digit}' ({strategy.upper()})")
            plt.axis("off")

            individual_plot = plot_path / f"{digit}.png"
            plt.savefig(individual_plot, bbox_inches="tight")
            plt.close()
