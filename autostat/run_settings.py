from dataclasses import dataclass, field
import typing as ty

from .constraints import KernelConstraints, default_constraints

import inspect
import autostat.kernel_specs as ks

from .kernel_specs import (
    TopLevelKernelSpec,
    BaseKernelSpec,
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
)

from .kernel_swaps import top_level_spec_from_base_kernel


default_base_kernel_classes: list[type[BaseKernelSpec]] = [
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
]


kernel_prototypes_from_classes = lambda classes: list(c() for c in classes)

base_kernel_prototypes = kernel_prototypes_from_classes(default_base_kernel_classes)


def starting_kernel_specs(kernel_classes) -> list[TopLevelKernelSpec]:
    return [top_level_spec_from_base_kernel(k()) for k in kernel_classes]


default_initial_kernels = lambda: starting_kernel_specs(default_base_kernel_classes)

default_kernels_prototypes = lambda: base_kernel_prototypes


@dataclass(frozen=True)
class RunSettings:
    kernel_constraints: KernelConstraints = field(default_factory=default_constraints)

    initial_kernels: list[TopLevelKernelSpec] = field(
        default_factory=default_initial_kernels
    )

    base_kernel_prototypes: list[BaseKernelSpec] = field(
        default_factory=default_kernels_prototypes
    )

    max_search_depth: int = 5

    kernel_priors: None = None
    log_level: None = None


def get_kernel_class_short_name_mapping() -> dict[str, BaseKernelSpec]:
    shortname_to_kernel_class = {}
    for name, obj in inspect.getmembers(ks):
        if inspect.isclass(obj) and issubclass(obj, ks.BaseKernelSpec):
            for item in inspect.getmembers(obj):
                if item[0] == "kernel_name":
                    shortname_to_kernel_class[item[1]] = obj
    return shortname_to_kernel_class


def init_run_settings_from_shorthand_args(
    base_kernel_shortnames: list[str], max_search_depth: int = 5
) -> RunSettings:

    kernel_class_short_name_mapping = get_kernel_class_short_name_mapping()

    base_kernel_classes: list[BaseKernelSpec] = []
    for bksn in base_kernel_shortnames:
        # print(bksn, kernel_class_short_name_mapping[bksn])
        try:
            base_kernel_classes.append(kernel_class_short_name_mapping[bksn])
        except:
            valid_names = ", ".join(kernel_class_short_name_mapping.keys())
            raise ValueError(
                f'Invalid base kernel name "{bksn}"; must be one of {valid_names}'
            )

    return RunSettings(
        initial_kernels=starting_kernel_specs(base_kernel_classes),
        base_kernel_prototypes=kernel_prototypes_from_classes(base_kernel_classes),
        max_search_depth=max_search_depth,
    )
