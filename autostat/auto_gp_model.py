import numpy as np
from numpy.typing import ArrayLike, NDArray

from typing import Protocol

from .kernel_specs import KernelSpec, AdditiveKernelSpec
from .dataset_adapters import Dataset, ModelPredictions
from .constraints import KernelConstraints


class AutoGpModel(Protocol):
    data: Dataset
    constraints: KernelConstraints

    def __init__(self, kernel_spec: KernelSpec, data: Dataset) -> None:
        ...

    def fit(self, data: Dataset) -> None:
        ...

    def predict(self, x: ArrayLike) -> ModelPredictions:
        ...

    def residuals(self) -> np.ndarray:
        ...

    def to_spec(self) -> AdditiveKernelSpec:
        ...

    def print_fitted_kernel(self) -> None:
        ...

    def bic(self) -> float:
        ...

    def log_likelihood(self) -> float:
        ...

    # @staticmethod
    # def get_kernel_constraints_from_data(data) -> KernelConstraints:
    #     ...
