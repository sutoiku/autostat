import typing as ty
import numpy as np
from .dataset_adapters import Dataset

from .kernel_specs import (
    AdditiveKernelSpec,
    KernelSpec,
    BaseKernelSpec,
    GenericKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
    ConstraintBounds as CB,
    ProductKernelSpec,
    TopLevelKernelSpec,
)


default_constraint_heuristics = {
    "PER": {
        "min_periods": 5,
        "min_data_points_per_period": 5,
        "max_length_scale_as_mult_of_max_period": 5,
        "min_length_scale_as_mult_of_min_period": 0.5,
    },
    "RBF": {},
    "LIN": {},
}


def kernel_proto_constrained_with_data(
    kernel: GenericKernelSpec, d: Dataset, heuristics=None
) -> GenericKernelSpec:
    heuristics = heuristics or default_constraint_heuristics

    if isinstance(kernel, PeriodicNoConstKernelSpec) or isinstance(
        kernel, PeriodicKernelSpec
    ):

        min_x_diff = np.diff(d.train_x.flatten()).min()
        periodicity_min = heuristics["PER"]["min_data_points_per_period"] * min_x_diff

        x_range = d.train_x.max() - d.train_x.min()
        periodicity_max = x_range / heuristics["PER"]["min_periods"]

        length_scale_min = (
            periodicity_min
            * heuristics["PER"]["min_length_scale_as_mult_of_min_period"]
        )
        length_scale_max = (
            periodicity_max
            * heuristics["PER"]["max_length_scale_as_mult_of_max_period"]
        )

        period_bounds = CB(periodicity_min, periodicity_max)
        length_scale_bounds = CB(length_scale_min, length_scale_max)

        return ty.cast(GenericKernelSpec, kernel).clone_update(
            {
                "period_bounds": period_bounds,
                "length_scale_bounds": length_scale_bounds,
                "period": period_bounds.clamp(kernel.period),
                "length_scale": length_scale_bounds.clamp(kernel.length_scale),
            }
        )
    else:
        return kernel


def update_kernel_protos_constrained_with_data(
    kernels: list[GenericKernelSpec], d: Dataset, heuristics=None
) -> list[GenericKernelSpec]:
    return [kernel_proto_constrained_with_data(k, d, heuristics) for k in kernels]


T = ty.TypeVar("T")


def set_constraints_on_spec(
    spec: T, constrained_base_kernels: list[BaseKernelSpec]
) -> T:
    CBK = constrained_base_kernels
    if (
        isinstance(spec, TopLevelKernelSpec)
        or isinstance(spec, AdditiveKernelSpec)
        or isinstance(spec, ProductKernelSpec)
    ):
        operands = [set_constraints_on_spec(subspec, CBK) for subspec in spec.operands]
        return spec.clone_update({"operands": operands})
    else:
        base_kernel = list(filter(lambda x: type(x) == type(spec), CBK))[0]

        if isinstance(spec, PeriodicNoConstKernelSpec) or isinstance(
            spec, PeriodicKernelSpec
        ):
            # FIXME: type handling here sucks
            base_kernel = ty.cast(
                PeriodicKernelSpec,
                base_kernel,
            )
            return spec.clone_update(
                {
                    "period_bounds": base_kernel.period_bounds,
                    "length_scale_bounds": base_kernel.length_scale_bounds,
                }
            )
        else:
            return spec
