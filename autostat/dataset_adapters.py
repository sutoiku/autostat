import typing as ty
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class Dataset:
    __slots__ = ("train_x", "train_y", "test_x", "test_y")

    train_x: NDArray[np.float_]
    train_y: NDArray[np.float_]
    test_x: NDArray[np.float_]
    test_y: NDArray[np.float_]

    def __iter__(self):
        yield from astuple(self)


class ModelPredictions(ty.NamedTuple):
    y: NDArray[np.float_]
    lower: NDArray[np.float_]
    upper: NDArray[np.float_]
