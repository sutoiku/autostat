import typing as ty
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class Dataset:
    train_x: NDArray[np.float_]
    train_y: NDArray[np.float_]
    test_x: ty.Union[NDArray[np.float_], None]
    test_y: ty.Union[NDArray[np.float_], None]


class ModelPredictions(ty.NamedTuple):
    y: NDArray[np.float_]
    std: NDArray[np.float_]
    cov: NDArray[np.float_]
