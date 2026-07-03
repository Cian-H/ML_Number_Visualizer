from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np
from torch import Tensor, nn


@runtime_checkable
class TorchTeacher(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def eval(self) -> Any: ...

@runtime_checkable
class SklearnTeacher(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

TeacherModel = TorchTeacher | SklearnTeacher
