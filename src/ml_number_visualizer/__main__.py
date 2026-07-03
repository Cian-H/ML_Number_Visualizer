import typer


def main():
    from loguru import logger

    from .dataloader import get_dataset
    from .neural_networks import train_neural_networks
    from .sklearn_models import train_sklearn
    from .visualize import generate_all_digits, generate_digit_for_sklearn_model

    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataset()

    train_neural_networks(train_loader, val_loader, test_loader)
    train_sklearn(train_loader, val_loader, test_loader)

    logger.info("Generating digit representations for models...")
    generate_all_digits()
    generate_digit_for_sklearn_model(
        model_name="sklearn_rf",
        target_digit=8,
        num_steps=1500,
    )
    logger.success("DONE!")


if __name__ == "__main__":
    typer.run(main)
