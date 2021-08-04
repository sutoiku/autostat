from collections import namedtuple
import torch

# from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from numpy.typing import ArrayLike, NDArray

from typing import Any, Generic, NamedTuple, Union, Protocol, TypeVar, Type, Any
from dataclasses import asdict, dataclass, astuple, field, replace


# A kernel spec is composed of a top level AdditiveKernelSpec
# Each summand of an additive kernel is a product
# each product has a scalar and one or more operands
# product operands may be either base kernels or additive kernels

# KernelSpec = Union["ArithmeticKernelSpec", "BaseKernelSpec"]

ArithmeticKernelSpec = Union["AdditiveKernelSpec", "ProductKernelSpec"]


ProductOperandSpec = Union[
    "RBFKernelSpec",
    "PeriodicKernelSpec",
    "LinearKernelSpec",
    "RQKernelSpec",
    "AdditiveKernelSpec",
]

T = TypeVar("T", bound="KernelSpec")


@dataclass(frozen=True)
class KernelSpec:
    def __init__(self, kwargs: dict[str, Any] = {}) -> None:
        ...

    def spec_str(self, verbose=True, pretty=True) -> str:
        ...

    def num_params(self) -> int:
        ...

    def fit_count(self) -> int:
        ...

    def schema(self) -> str:
        return self.spec_str(False, False)

    def clone_update(self: T, kwargs: dict[str, Any] = {}) -> T:
        return replace(self, **kwargs)

    def __iter__(self):
        yield from astuple(self)

    def _asdict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.spec_str(True, True)

    def __repr__(self) -> str:
        return self.spec_str(True, False)


@dataclass(frozen=True)
class AdditiveKernelSpec(KernelSpec):
    operands: list["ProductKernelSpec"] = field(default_factory=list)

    def num_params(self) -> int:
        return sum(op.num_params() for op in self.operands)

    def fit_count(self) -> int:
        return sum(op.fit_count() for op in self.operands)

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            return "(" + " + ".join(operandStrings) + ")"
        else:
            return f"ADD({', '.join(operandStrings)})"


@dataclass(frozen=True)
class ProductKernelSpec(KernelSpec):
    operands: list[Union["BaseKernelSpec", AdditiveKernelSpec]] = field(
        default_factory=list
    )
    scalar: float = 1

    def num_params(self) -> int:
        # 1 for scalar, plus child params
        return 1 + sum(op.num_params() for op in self.operands)

    def fit_count(self) -> int:
        this_fit = 0 if self.scalar == 1 else 1
        return sum(op.fit_count() for op in self.operands) + this_fit

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            scalar_str = f"{self.scalar:.2f} * " if verbose else ""
            return scalar_str + " * ".join(operandStrings)
        else:
            scalar_str = f"{self.scalar:.2f}, " if verbose else ""
            return f"PROD({scalar_str}{', '.join(operandStrings)})"

    def clone_update(self, kwargs: dict[str, Any] = {}) -> "ProductKernelSpec":
        cloned_operands = [op.clone_update() for op in self.operands]
        return replace(
            self, **{"operands": cloned_operands, "scalar": self.scalar, **kwargs}
        )


@dataclass(frozen=True)
class RBFKernelSpec(KernelSpec):
    length_scale: float = 1

    def num_params(self) -> int:
        return 1

    def fit_count(self) -> int:
        return 0 if self.length_scale == 1 else 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"RBF(l={self.length_scale:.2f})"
        else:
            return "RBF"


@dataclass(frozen=True)
class PeriodicKernelSpec(KernelSpec):
    length_scale: float = 1
    period: float = 1

    def num_params(self) -> int:
        return 2

    def fit_count(self) -> int:
        return 0 if (self.length_scale == 1 and self.period == 1) else 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"PER(per={self.period:.3f},l={self.length_scale:.2f})"
        else:
            return "PER"


@dataclass(frozen=True)
class RQKernelSpec(KernelSpec):
    length_scale: float = 1
    alpha: float = 1

    def num_params(self) -> int:
        return 2

    def fit_count(self) -> int:
        return 0 if (self.length_scale == 1 and self.alpha == 1) else 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"RQ(α={self.alpha:.2f},l={self.length_scale:.2f})"
        else:
            return "RQ"


@dataclass(frozen=True)
class LinearKernelSpec(KernelSpec):
    variance: float = 1

    def num_params(self) -> int:
        return 1

    def fit_count(self) -> int:
        return 0 if (self.variance == 1) else 1

    def spec_str(self, verbose=True, pretty=True) -> str:
        if verbose:
            return f"LIN(v={self.variance:.2f})"
        else:
            return "LIN"


BaseKernelSpec = Union[
    RBFKernelSpec, PeriodicKernelSpec, LinearKernelSpec, RQKernelSpec
]


NdGen = TypeVar("NdGen", torch.Tensor, NDArray[np.float_])


@dataclass
class DatasetGeneric(Generic[NdGen]):
    __slots__ = ("train_x", "train_y", "test_x", "test_y")

    train_x: NdGen
    train_y: NdGen
    test_x: NdGen
    test_y: NdGen

    def __iter__(self):
        yield from astuple(self)


class NpDataSet(DatasetGeneric[NDArray[np.float_]]):
    ...


class TorchDataSet(DatasetGeneric[torch.Tensor]):
    ...


Dataset = Union[TorchDataSet, NpDataSet]


class ModelPredictions(NamedTuple):
    y: NDArray[np.float_]
    lower: NDArray[np.float_]
    upper: NDArray[np.float_]


# class GpytorchModel(NamedTuple):
#     model: ExactGPModel
#     likelihood: gpytorch.likelihoods.GaussianLikelihood
#     kernel_tree: KernelTreeNode


# class SklearnModel(NamedTuple):
#     model: GaussianProcessRegressor
#     kernel_tree: KernelTreeNode


# GPModel = Union[GpytorchModel, SklearnModel]
