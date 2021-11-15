import typing as ty
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class Dataset:
    train_x: NDArray[np.float_]
    train_y: NDArray[np.float_]
    test_x: NDArray[np.float_]
    test_y: NDArray[np.float_]


class ModelPredictions(ty.NamedTuple):
    y: NDArray[np.float_]
    std: NDArray[np.float_]
    # upper: NDArray[np.float_]
