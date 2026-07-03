import typer


def main():
    from loguru import logger

    from .dataloader import get_dataset
    from .neural_networks import train_neural_networks

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataset()

    train_neural_networks(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    typer.run(main)
