import torch
import typing as ty
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple


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
