from .kernel_specs import GenericKernelSpec


def sort_specs_by_type(
    kernels: list[GenericKernelSpec],
) -> list[GenericKernelSpec]:
    return sorted(kernels, key=lambda node: node.spec_str(True, True))
