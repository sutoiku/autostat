from dataclasses import dataclass
from typing import NamedTuple, Union
from .kernel_tree_types import Dataset


default_constraints = {
    "PER": {"min_periods": 2, "min_data_points_per_period": 4},
    "RBF": {},
    "LIN": {},
}


class ConstraintBounds(NamedTuple):
    lower: float
    upper: float


CB = ConstraintBounds


class PeriodicKernelConstraints(NamedTuple):
    length_scale: Union[ConstraintBounds, None] = None
    period: Union[ConstraintBounds, None] = None


@dataclass
class KernelConstraints:
    PER: PeriodicKernelConstraints


def constraints_from_data(d: Dataset, constraints=None) -> KernelConstraints:
    constraints = constraints or default_constraints

    min_x_diff = d.train_x.diff().min()
    periodicity_min = constraints["PER"]["min_data_points_per_period"] * min_x_diff

    x_range = d.train_x.max() - d.train_x.min()
    periodicity_max = x_range / constraints["PER"]["min_periods"]

    return KernelConstraints(
        PeriodicKernelConstraints(period=CB(periodicity_min, periodicity_max))
    )
