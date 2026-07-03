import typer


def main():
    from pathlib import Path

    import torch
    from loguru import logger

    from .dataloader import get_dataset
    from .neural_networks import train_neural_networks
    from .protocols import LazyModelAdapter
    from .sklearn_models import train_sklearn
    from .visualize import generate_visualizations_for_models

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataset()

    train_neural_networks(train_loader, val_loader, test_loader)
    train_sklearn(train_loader, val_loader, test_loader)

    logger.info("Generating digit representations for models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {p.stem: LazyModelAdapter(p, device) for p in Path("./models").glob("[!.]*")}
    generate_visualizations_for_models(models)
    logger.success("DONE!")


if __name__ == "__main__":
    typer.run(main)
