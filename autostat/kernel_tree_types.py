from collections import namedtuple
import torch

from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from numpy.typing import ArrayLike

from typing import Generic, NamedTuple, Union, Protocol, TypeVar
from dataclasses import asdict, dataclass, astuple, field


# A kernel spec is composed of a top level AdditiveKernelSpec
# Each summand of an additive kernel is a product
# each product has a scalar and one or more operands
# product operands may be either base kernels or additive kernels

KernelSpec = Union["ArithmeticKernelSpec", "BaseKernelSpec"]

ArithmeticKernelSpec = Union["AdditiveKernelSpec", "ProductKernelSpec"]


BaseKernelSpecTypes = Union[
    type["RBFKernelSpec"],
    type["PeriodicKernelSpec"],
    type["LinearKernelSpec"],
    type["RQKernelSpec"],
]


ProductOperandSpec = Union[
    "RBFKernelSpec",
    "PeriodicKernelSpec",
    "LinearKernelSpec",
    "RQKernelSpec",
    "AdditiveKernelSpec",
]


class KernelSpecFns:
    def spec_str(self, verbose=True, pretty=True) -> str:
        ...

    def num_params(self) -> int:
        ...

    def __str__(self) -> str:
        return self.spec_str(True, True)

    def __repr__(self) -> str:
        return self.spec_str(True, False)

    def schema(self) -> str:
        return self.spec_str(False, False)

    def __iter__(self):
        yield from astuple(self)

    def _asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class AdditiveKernelSpec(KernelSpecFns):
    operands: list["ProductKernelSpec"] = field(default_factory=list)

    def num_params(self) -> int:
        return sum(op.num_params() for op in self.operands)

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            return "(" + " + ".join(operandStrings) + ")"
        else:
            return f"ADD({', '.join(operandStrings)})"


@dataclass(frozen=True)
class ProductKernelSpec(KernelSpecFns):
    operands: list[Union["BaseKernelSpec", AdditiveKernelSpec]] = field(
        default_factory=list
    )
    scalar: float = 1

    def num_params(self) -> int:
        # 1 for scalar, plus child params
        return 1 + sum(op.num_params() for op in self.operands)

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            scalar_str = f"{self.scalar:.2f} * " if verbose else ""
            return scalar_str + " * ".join(operandStrings)
        else:
            scalar_str = f"{self.scalar:.2f}, " if verbose else ""
            return f"PROD({scalar_str}{', '.join(operandStrings)})"


@dataclass(frozen=True)
class RBFKernelSpec(KernelSpecFns):
    length_scale: float = 1

    def num_params(self) -> int:
        return 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"RBF(l={self.length_scale:.2f})"
        else:
            return "RBF"


@dataclass(frozen=True)
class PeriodicKernelSpec(KernelSpecFns):
    length_scale: float = 1
    period: float = 1

    def num_params(self) -> int:
        return 2

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"PER(per={self.period:.2f},l={self.length_scale:.2f})"
        else:
            return "PER"


@dataclass(frozen=True)
class RQKernelSpec(KernelSpecFns):
    length_scale: float = 1
    alpha: float = 1

    def num_params(self) -> int:
        return 2

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"RQ(α={self.alpha:.2f},l={self.length_scale:.2f})"
        else:
            return "RQ"


@dataclass(frozen=True)
class LinearKernelSpec(KernelSpecFns):
    variance: float = 1

    def num_params(self) -> int:
        return 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"LIN(v={self.variance:.2f})"
        else:
            return "LIN"


BaseKernelSpec = Union[
    RBFKernelSpec, PeriodicKernelSpec, LinearKernelSpec, RQKernelSpec
]

# def clone_spec(spec: KernelSpec, kwargs: dict = {}) -> KernelSpec:
#     return spec.__class__({**spec._asdict(), **kwargs})


NdGen = TypeVar("NdGen", torch.Tensor, np.ndarray)


@dataclass
class DatasetGeneric(Generic[NdGen]):
    __slots__ = ("train_x", "train_y", "test_x", "test_y")

    train_x: NdGen
    train_y: NdGen
    test_x: NdGen
    test_y: NdGen

    def __iter__(self):
        yield from astuple(self)


class NpDataSet(DatasetGeneric[np.ndarray]):
    ...


class TorchDataSet(DatasetGeneric[torch.Tensor]):
    ...


Dataset = Union[TorchDataSet, NpDataSet]

Dataseries = Union[torch.Tensor, ArrayLike]


class ModelPredictions(NamedTuple):
    y: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class AutoGpModel(Protocol):
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


# class GpytorchModel(NamedTuple):
#     model: ExactGPModel
#     likelihood: gpytorch.likelihoods.GaussianLikelihood
#     kernel_tree: KernelTreeNode


# class SklearnModel(NamedTuple):
#     model: GaussianProcessRegressor
#     kernel_tree: KernelTreeNode


# GPModel = Union[GpytorchModel, SklearnModel]


class GPPrediction(NamedTuple):
    x: ArrayLike
    mean: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
