import typer


def main():
    from loguru import logger

    from .dataloader import get_dataset
    from .neural_networks import train_neural_networks
    from .visualize import generate_all_digits_batched

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataset()

    train_neural_networks(train_loader, val_loader, test_loader)

    logger.info("Generating digit representations for models...")
    generate_all_digits_batched()
    logger.success("DONE!")


if __name__ == "__main__":
    typer.run(main)
