import lightning as L
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim


class FlatStrategy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNNStrategy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Mish(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Mish(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 128), nn.Mish(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ViTStrategy(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, embed_dim=64, heads=4, depth=3):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size**2

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.contiguous().view(B, -1, self.patch_size * self.patch_size)

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed

        x = self.transformer(x)
        return self.mlp_head(x[:, 0])


STRATEGY_REGISTRY = {"flat": FlatStrategy, "cnn": CNNStrategy, "vit": ViTStrategy}


class QMNISTClassifier(L.LightningModule):
    def __init__(self, model_type: str = "flat", learning_rate: float = 1e-3, **kwargs):
        super().__init__()

        model_type = model_type.lower().strip()
        if model_type not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{model_type}'. "
                f"Available options are: {list(STRATEGY_REGISTRY.keys())}"
            )

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.backbone = STRATEGY_REGISTRY[model_type](**kwargs)

    def forward(self, x):
        return self.backbone(x)

    def _shared_step(self, batch, _, step_name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(f"{step_name}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_neural_networks(train_loader, val_loader, test_loader):
    logger.info("Running NN Models...")

    for s in STRATEGY_REGISTRY:
        logger.info(f"Initializing {s}...")
        model = QMNISTClassifier(model_type=s, learning_rate=1e-3)
        trainer = L.Trainer(
            max_epochs=5,
            accelerator="auto",
            devices="auto",
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader)
        torch.save(model.state_dict(), f"./models/nn_{s}.pth")
