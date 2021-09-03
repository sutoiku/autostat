from .dataset_adapters import Dataset
from dataclasses import dataclass, field, replace
import typing as ty

from .constraints import update_kernel_protos_constrained_with_data


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


def get_kernel_class_short_name_mapping() -> dict[str, BaseKernelSpec]:
    shortname_to_kernel_class = {}
    for name, obj in inspect.getmembers(ks):
        if inspect.isclass(obj) and issubclass(obj, ks.BaseKernelSpec):
            for item in inspect.getmembers(obj):
                if item[0] == "kernel_name":
                    shortname_to_kernel_class[item[1]] = obj
    return shortname_to_kernel_class


def kernel_protos_from_names(base_kernel_shortnames: list[str]):
    kernel_class_short_name_mapping = get_kernel_class_short_name_mapping()

    base_kernel_classes: list[BaseKernelSpec] = []
    for bksn in base_kernel_shortnames:

        try:
            base_kernel_classes.append(kernel_class_short_name_mapping[bksn])
        except:
            valid_names = ", ".join(kernel_class_short_name_mapping.keys())
            raise ValueError(
                f'Invalid base kernel name "{bksn}"; must be one of {valid_names}'
            )

    return kernel_prototypes_from_classes(base_kernel_classes)


####


@dataclass(frozen=True)
class RunSettings:
    initial_kernels: list[TopLevelKernelSpec] = field(
        default_factory=default_initial_kernels
    )
    base_kernel_prototypes: list[BaseKernelSpec] = field(
        default_factory=default_kernels_prototypes
    )
    kernel_priors: None = None

    expand_kernel_specs_as_sums: bool = False

    log_level: None = None
    max_search_depth: int = 5

    def replace_base_kernels_by_names(self, names: list[str]) -> "RunSettings":
        return replace(self, base_kernel_prototypes=kernel_protos_from_names(names))

    def replace_init_kernel_proto_constraints_using_dataset(
        self, dataset: Dataset
    ) -> "RunSettings":
        initial_kernels = update_kernel_protos_constrained_with_data(
            self.base_kernel_prototypes, dataset
        )
        initial_kernels = [
            TopLevelKernelSpec.from_base_kernel(k) for k in initial_kernels
        ]

        return replace(self, initial_kernels=initial_kernels)
