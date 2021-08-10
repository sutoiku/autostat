from typing import NamedTuple
import numpy as np
from .kernel_specs import Dataset


default_constraint_heuristics = {
    "PER": {
        "min_periods": 5,
        "min_data_points_per_period": 10,
        "max_length_scale_as_mult_of_max_period": 5,
        "min_length_scale_as_mult_of_min_period": 0.5,
    },
    "RBF": {},
    "LIN": {},
}


class ConstraintBounds(NamedTuple):
    lower: float
    upper: float


CB = ConstraintBounds


def cb_default():
    return ConstraintBounds(1e-5, 1e5)


class PeriodicKernelConstraints(NamedTuple):
    length_scale: ConstraintBounds = cb_default()
    period: ConstraintBounds = cb_default()


class KernelConstraints(NamedTuple):
    PER: PeriodicKernelConstraints


def constraints_from_data(d: Dataset, heuristics=None) -> KernelConstraints:
    heuristics = heuristics or default_constraint_heuristics

    min_x_diff = np.diff(d.train_x.flatten()).min()
    periodicity_min = heuristics["PER"]["min_data_points_per_period"] * min_x_diff

    x_range = d.train_x.max() - d.train_x.min()
    periodicity_max = x_range / heuristics["PER"]["min_periods"]

    PER_length_scale_min = (
        periodicity_min * heuristics["PER"]["min_length_scale_as_mult_of_min_period"]
    )
    PER_length_scale_max = (
        periodicity_max * heuristics["PER"]["max_length_scale_as_mult_of_max_period"]
    )

    return KernelConstraints(
        PeriodicKernelConstraints(
            period=CB(periodicity_min, periodicity_max),
            length_scale=CB(PER_length_scale_min, PER_length_scale_max),
        )
    )


def default_constraints() -> KernelConstraints:
    return KernelConstraints(
        PeriodicKernelConstraints(period=cb_default(), length_scale=cb_default())
    )
