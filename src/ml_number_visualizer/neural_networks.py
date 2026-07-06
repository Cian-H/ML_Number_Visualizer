from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn, optim

torch.set_float32_matmul_precision("high")


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


class FlashAttentionBlock(nn.Module):
    """Multi-head self-attention backed by F.scaled_dot_product_attention.

    On CUDA with BF16/FP16, PyTorch dispatches SDPA to the Flash Attention
    kernel automatically — no extra dependencies required.
    """

    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim) each
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class FlashTransformerBlock(nn.Module):
    """Pre-norm transformer encoder block wrapping FlashAttentionBlock."""

    def __init__(self, embed_dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashAttentionBlock(embed_dim, heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTStrategy(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, embed_dim=64, heads=4, depth=3):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size**2

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer = nn.Sequential(
            *[FlashTransformerBlock(embed_dim, heads) for _ in range(depth)]
        )
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


class EpochSnapshotCallback(L.Callback):
    """Saves a bare state_dict after every training epoch for video generation."""

    def __init__(self, model_name: str, snapshot_dir: Path = Path("./checkpoints/snapshots")):
        super().__init__()
        self.save_dir = Path(snapshot_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        path = self.save_dir / f"epoch_{trainer.current_epoch:03d}.pth"
        torch.save(pl_module.state_dict(), path)
        self.paths.append(path)


def train_neural_networks(
    train_loader, val_loader, test_loader
) -> dict[str, tuple[list[Path], list[str]]]:
    logger.info("Running NN Models...")
    snapshots: dict[str, tuple[list[Path], list[str]]] = {}

    for s in STRATEGY_REGISTRY:
        logger.info(f"Initializing {s}...")
        model_name = f"nn_{s}"
        snapshot_cb = EpochSnapshotCallback(model_name=model_name)
        model = QMNISTClassifier(model_type=s, learning_rate=1e-3)
        trainer = L.Trainer(
            max_epochs=10,
            accelerator="auto",
            devices="auto",
            precision="bf16-mixed",
            enable_checkpointing=False,
            callbacks=[snapshot_cb],
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader)
        torch.save(model.state_dict(), f"./models/nn_{s}.pth")
        paths = snapshot_cb.paths
        labels = [f"Epoch {i}" for i in range(len(paths))]
        snapshots[model_name] = (paths, labels)

    return snapshots
