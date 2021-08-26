import numpy as np
from numpy.typing import ArrayLike, NDArray

from typing import Protocol

from .kernel_specs import KernelSpec, TopLevelKernelSpec
from .dataset_adapters import Dataset, ModelPredictions
from .run_settings import RunSettings


class AutoGpModel(Protocol):
    data: Dataset

    def __init__(
        self,
        kernel_spec: KernelSpec,
        data: Dataset,
        run_settings: RunSettings,
    ) -> None:
        ...

    def fit(self, data: Dataset) -> None:
        ...

    def predict(self, x: ArrayLike) -> ModelPredictions:
        ...

    def residuals(self) -> np.ndarray:
        ...

    def to_spec(self) -> TopLevelKernelSpec:
        ...

    def print_fitted_kernel(self) -> None:
        ...

    def bic(self) -> float:
        ...

    def log_likelihood(self) -> float:
        ...
