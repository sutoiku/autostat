import typing as ty

from .constraints import KernelConstraints, default_constraints
from .kernel_specs import (
    AdditiveKernelSpec,
    BaseKernelSpec,
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
)

from .kernel_swaps import addititive_base_term_with_scalar


base_kernel_classes: list[type[BaseKernelSpec]] = [
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
]

base_kernel_prototypes = list(c() for c in base_kernel_classes)


def starting_kernel_specs(kernel_classes) -> list[AdditiveKernelSpec]:
    return [addititive_base_term_with_scalar(k()) for k in kernel_classes]


class RunSettings(ty.NamedTuple):
    kernel_constraints: KernelConstraints = default_constraints()
    initial_kernels: list[AdditiveKernelSpec] = starting_kernel_specs(
        base_kernel_classes
    )
    base_kernel_prototypes: list[BaseKernelSpec] = base_kernel_prototypes
    max_search_depth: int = 5

    kernel_priors: None = None
    log_level: None = None
