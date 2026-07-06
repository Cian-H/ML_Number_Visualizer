import joblib
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from ml_number_visualizer.neural_networks import QMNISTClassifier


@runtime_checkable
class TeacherModel(Protocol):
    @property
    def is_differentiable(self) -> bool:
        """Returns True if the model natively supports gradient flow."""
        ...

    def get_target_probs(self, x: Tensor) -> Tensor:
        """Returns output probabilities for a given batch of images."""
        ...

    def get_differentiable_model(self) -> nn.Module:
        """Returns the underlying PyTorch model if differentiable, otherwise raises an exception."""
        ...


class PyTorchAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def is_differentiable(self) -> bool:
        return True

    def get_target_probs(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.model(x)
            return F.softmax(logits, dim=1)

    def get_differentiable_model(self) -> nn.Module:
        return self.model


class SklearnAdapter:
    def __init__(self, model):
        self.model = model

    @property
    def is_differentiable(self) -> bool:
        return False

    def get_target_probs(self, x: Tensor) -> Tensor:
        x_flat = x.view(x.size(0), -1).detach().cpu().numpy()
        probs = self.model.predict_proba(x_flat)
        return torch.tensor(probs, dtype=torch.float32, device=x.device)

    def get_differentiable_model(self) -> nn.Module:
        raise ValueError("Scikit-Learn models are not natively differentiable.")


class LazyModelAdapter:
    def __init__(
        self, path: Path | str, device: torch.device, torch_module: nn.Module | None = None
    ):
        self.path = Path(path)
        self.device = device
        try:
            torch_module = (
                torch_module if torch_module else QMNISTClassifier(self.path.stem.split("_")[1])
            )
        except ValueError:
            torch_module = None
        self.torch_module = torch_module
        self._inner_adapter: TeacherModel | None = None

    def _load_if_needed(self) -> None:
        if self._inner_adapter is not None:
            return

        logger.info(f"Lazily loading model from {self.path}...")

        # Scikit-Learn Model
        if self.path.suffix in [".pkl", ".joblib"]:
            model = joblib.load(self.path)
            self._inner_adapter = SklearnAdapter(model)

        # PyTorch Model
        elif self.path.suffix in [".pth", ".pt"]:
            self.torch_module.load_state_dict(
                torch.load(self.path, map_location=self.device, weights_only=True)
            )
            model = self.torch_module.to(self.device)
            self._inner_adapter = PyTorchAdapter(model)

        else:
            raise ValueError(f"Unsupported file format for lazy loading: {self.path.suffix}")

    @property
    def is_differentiable(self) -> bool:
        self._load_if_needed()
        assert self._inner_adapter is not None
        return self._inner_adapter.is_differentiable

    def get_target_probs(self, x: Tensor) -> Tensor:
        self._load_if_needed()
        assert self._inner_adapter is not None
        return self._inner_adapter.get_target_probs(x)

    def get_differentiable_model(self) -> nn.Module:
        self._load_if_needed()
        assert self._inner_adapter is not None
        return self._inner_adapter.get_differentiable_model()
