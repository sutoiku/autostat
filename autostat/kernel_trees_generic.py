from .kernel_tree_types import (
    Dataset,
    KernelSpec,
    ArithmeticKernelSpec,
    BaseKernelSpec,
    BaseKernelSpecTypes,
    AdditiveKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    RQKernelSpec,
    AutoGpModel,
)


def kernel_type(k: KernelSpec) -> str:
    if isinstance(k, RBFKernelSpec):
        return f"RBF"

    elif isinstance(k, LinearKernelSpec):
        return f"LIN"

    elif isinstance(k, PeriodicKernelSpec):
        return f"PER"

    elif isinstance(k, RQKernelSpec):
        return f"RQ"

    elif isinstance(k, AdditiveKernelSpec):
        return f"ADD"

    elif isinstance(k, ProductKernelSpec):
        return f"PROD"

    else:
        raise TypeError("Invalid kernel type")


# def kernel_str_op_node(k: ArithmeticKernelSpec, details: bool = False) -> str:
#     children = f", ".join(sorted([k.spec_str(False, False) for k in k.operands]))
#     return f"{children}"


# def kernel_str(k: KernelSpec, details: bool = False) -> str:
#     if details:
#         if isinstance(k, RBFKernelSpec):
#             details_str = f"(l={k.length_scale:.3f})"

#         elif isinstance(k, LinearKernelSpec):
#             details_str = f"(var={k.variance:.3f})"

#         elif isinstance(k, PeriodicKernelSpec):
#             details_str = f"(per={k.period:.3f},l={k.length_scale:.3f})"

#         elif isinstance(k, RQKernelSpec):
#             details_str = f"(α={k.alpha:.3f},l={k.length_scale:.3f})"

#         else:
#             raise TypeError("Invalid kernel type")

#     elif isinstance(k, AdditiveKernelSpec) or isinstance(k, ProductKernelSpec):
#         details_str = f"({kernel_str_op_node(k,details)})"
#     else:
#         details_str = ""

#     scale = f"{k.scalar} * " if isinstance(k, ProductKernelSpec) else ""

#     return kernel_type(k) + details_str


# def num_kernel_params(k: KernelSpec) -> int:
#     if isinstance(k, RBFKernelSpec) or isinstance(k, LinearKernelSpec):
#         return 2  # scale and one parameter

#     elif isinstance(k, PeriodicKernelSpec) or isinstance(k, RQKernelSpec):
#         return 3  # scale and 2 params

#     elif isinstance(k, AdditiveKernelSpec) or isinstance(k, ProductKernelSpec):
#         return 1 + sum(
#             num_kernel_params(child) for child in k.operands
#         )  # scale + child params

#     else:
#         raise TypeError("Invalid kernel type")


# base_kernels = [
#     KernelTreeLeafNode(RBFKernel),
#     KernelTreeLeafNode(RQKernel),
#     KernelTreeLeafNode(LinearKernel),
#     KernelTreeLeafNode(PeriodicKernel),
# ]
