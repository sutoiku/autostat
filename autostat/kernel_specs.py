from collections import namedtuple
import torch


import numpy as np
from numpy.typing import NDArray
import typing as ty

# from typing import ty.Any, Generic, NamedTuple, Union, TypeVar
from dataclasses import asdict, dataclass, astuple, field, replace, InitVar


# A kernel spec is composed of a top level AdditiveKernelSpec
# Each summand of an additive kernel is a product
# each product has a scalar and one or more operands
# product operands may be either base kernels or additive kernels


GenericKernelSpec = ty.TypeVar("GenericKernelSpec", bound="KernelSpec")


@dataclass(frozen=True)
class KernelSpec:
    def __init__(self, kwargs: dict[str, ty.Any] = {}) -> None:
        ...

    def spec_str(self, verbose: bool = True, pretty: bool = True) -> str:
        ...

    def num_params(self) -> int:
        ...

    def fit_count(self) -> int:
        ...

    def schema(self) -> str:
        return self.spec_str(False, False)

    def clone_update(
        self: GenericKernelSpec, kwargs: dict[str, ty.Any] = {}
    ) -> GenericKernelSpec:
        return replace(self, **kwargs)

    def __iter__(self):
        yield from astuple(self)

    def asdict(self) -> dict[str, ty.Any]:
        return asdict(self)

    def astuple(self) -> tuple[ty.Any, ...]:
        return astuple(self)

    def __str__(self) -> str:
        return self.spec_str(True, True)

    def __repr__(self) -> str:
        return self.spec_str(True, False)


class BaseKernelSpec(KernelSpec):
    kernel_name: InitVar[str]
    pp_replacements: InitVar[dict[str, str]]

    def spec_str(self, verbose: bool, pretty: bool) -> str:
        name = str(self.kernel_name)
        if verbose:
            param_str = ",".join([f"{k}={v:.4f}" for k, v in self.asdict().items()])
            for str1, str2 in self.pp_replacements.items():
                param_str = param_str.replace(str1, str2)
            return f"{name}({param_str})"
        else:
            return name

    def num_params(self) -> int:
        return len(self.astuple())

    def fit_count(self) -> int:
        return sum(v != 1 for v in self.astuple())


@dataclass(frozen=True)
class RBFKernelSpec(BaseKernelSpec):
    length_scale: float = 1

    kernel_name: InitVar[str] = "RBF"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l"}


@dataclass(frozen=True)
class PeriodicKernelSpec(BaseKernelSpec):
    length_scale: float = 1
    period: float = 1

    kernel_name: InitVar[str] = "PER"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l", "period": "p"}


@dataclass(frozen=True)
class RQKernelSpec(BaseKernelSpec):
    length_scale: float = 1
    alpha: float = 1

    kernel_name: InitVar[str] = "RQ"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l", "alpha": "α"}


@dataclass(frozen=True)
class LinearKernelSpec(BaseKernelSpec):
    variance: float = 1

    kernel_name: InitVar[str] = "LIN"
    pp_replacements: InitVar[dict[str, str]] = {"variance": "var"}


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


ProductOperandSpec = ty.Union[
    BaseKernelSpec,
    AdditiveKernelSpec,
]


@dataclass(frozen=True)
class ProductKernelSpec(KernelSpec):
    operands: list[ProductOperandSpec] = field(default_factory=list)
    scalar: float = 1

    kernel_name: InitVar[str] = "PROD"

    def num_params(self) -> int:
        # 1 for scalar, plus child params
        return 1 + sum(op.num_params() for op in self.operands)

    def fit_count(self) -> int:
        this_fit = 0 if self.scalar == 1 else 1
        return sum(op.fit_count() for op in self.operands) + this_fit

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            scalar_str = f"{self.scalar:.4f} * " if verbose else ""
            return scalar_str + " * ".join(operandStrings)
        else:
            scalar_str = f"{self.scalar:.4f}, " if verbose else ""
            return f"PROD({scalar_str}{', '.join(operandStrings)})"

    def clone_update(self, kwargs: dict[str, ty.Any] = {}) -> "ProductKernelSpec":
        cloned_operands = [op.clone_update() for op in self.operands]
        return replace(
            self, **{"operands": cloned_operands, "scalar": self.scalar, **kwargs}
        )


NdGen = ty.TypeVar("NdGen", torch.Tensor, NDArray[np.float_])


@dataclass
class DatasetGeneric(ty.Generic[NdGen]):
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


Dataset = ty.Union[TorchDataSet, NpDataSet]


class ModelPredictions(ty.NamedTuple):
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
