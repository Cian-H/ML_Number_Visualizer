import numpy as np
from loguru import logger
from torch.utils.data import DataLoader


def extract_numpy_data(
    dataloader: DataLoader,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    max_samples = max_samples if max_samples else len(dataloader)

    logger.info(f"Extracting up to {max_samples} samples from DataLoader into NumPy arrays...")

    x_list = []
    y_list = []

    samples_collected = 0

    for images, labels in dataloader:
        flat_images = images.view(images.size(0), -1).numpy()

        x_list.append(flat_images)
        y_list.append(labels.numpy())

        samples_collected += images.size(0)
        if samples_collected >= max_samples:
            break

    X = np.vstack(x_list)[:max_samples]
    y = np.concatenate(y_list)[:max_samples]

    logger.info(f"Extraction complete. X shape: {X.shape}, y shape: {y.shape}")

    return X, y
