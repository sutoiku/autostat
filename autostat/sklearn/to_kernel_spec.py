from autostat.constraints import constraints_from_data
import typing as ty

from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Kernel,
    Product,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)

from ..kernel_tree_types import (
    AdditiveKernelSpec,
    KernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    RQKernelSpec,
)
from .custom_periodic_kernel import PeriodicKernelNoConstant


def remove_nones(L: list) -> list:
    return [x for x in L if x is not None]


def remove_white_kernels(L: list[Kernel]) -> list[Kernel]:
    return [k for k in L if not isinstance(k, WhiteKernel)]


def to_kernel_spec_additive(kernel: Sum) -> AdditiveKernelSpec:
    params = kernel.get_params()
    operands: list[ProductKernelSpec] = []
    kernels_to_check = [params["k1"], params["k2"]]

    while kernels_to_check:
        this_k = kernels_to_check.pop()
        params = this_k.get_params()
        if isinstance(this_k, Sum):
            kernels_to_check.append(params["k1"])
            kernels_to_check.append(params["k2"])
        elif isinstance(this_k, Product):
            operands.append(to_kernel_spec_product(this_k))
        elif isinstance(this_k, WhiteKernel):
            pass
        else:
            print("invalid kernel type -- type(kernel):", type(kernel))
            print(kernel)
            raise TypeError(
                "Invalid sklearn kernel type for to_kernel_spec_additive; must be additive or product"
            )

    return AdditiveKernelSpec(operands=operands)


def to_kernel_spec_product(kernel: Product) -> ProductKernelSpec:
    params = kernel.get_params()
    operands: list[ProductOperandSpec] = []
    scalar = 1
    kernels_to_check = [params["k1"], params["k2"]]

    while kernels_to_check:
        this_k = kernels_to_check.pop()
        params = this_k.get_params()
        if isinstance(this_k, Product):
            kernels_to_check.append(params["k1"])
            kernels_to_check.append(params["k2"])
        elif isinstance(this_k, Sum):
            operands.append(to_kernel_spec_additive(this_k))
        elif isinstance(this_k, ConstantKernel):
            scalar = params["constant_value"]
        elif (
            isinstance(this_k, RBF)
            or isinstance(this_k, RationalQuadratic)
            or isinstance(this_k, PeriodicKernelNoConstant)
            or isinstance(this_k, DotProduct)
        ):
            spec = ty.cast(ProductOperandSpec, to_kernel_spec_inner(this_k))
            operands.append(spec)
        else:
            print("invalid kernel type for to_kernel_spec_product:", type(this_k))
            print(this_k)
            raise TypeError("Invalid sklearn kernel type for to_kernel_spec_product")

    return ProductKernelSpec(operands=operands, scalar=scalar)


def to_kernel_spec_inner(kernel: Kernel) -> KernelSpec:
    params = kernel.get_params()
    if isinstance(kernel, RBF):
        inner: KernelSpec = RBFKernelSpec(length_scale=params["length_scale"])

    elif isinstance(kernel, DotProduct):
        # inner = (sigma_0=kernel.variance)
        inner = LinearKernelSpec(variance=params["sigma_0"])

    elif isinstance(kernel, PeriodicKernelNoConstant):
        inner = PeriodicKernelSpec(
            length_scale=params["length_scale"], period=params["periodicity"]
        )

    elif isinstance(kernel, RationalQuadratic):
        inner = RQKernelSpec(length_scale=params["length_scale"], alpha=params["alpha"])

    elif isinstance(kernel, Sum):
        inner = to_kernel_spec_additive(kernel)

    elif isinstance(kernel, Product):
        inner = to_kernel_spec_product(kernel)

    else:
        print("invalid kernel type -- type(kernel):", type(kernel))
        print(kernel)
        raise TypeError("Invalid sklearn kernel type")

    return inner


def to_kernel_spec(kernel: ty.Union[Sum, Product]) -> AdditiveKernelSpec:
    # NOTE: from sklearn, top level product specs (scalar times 1 or more base kernels)
    # will NOT be wrapped in an additive kernel
    inner_spec = to_kernel_spec_inner(kernel)
    if isinstance(inner_spec, AdditiveKernelSpec):
        return inner_spec
    elif isinstance(inner_spec, ProductKernelSpec):
        return AdditiveKernelSpec(operands=[inner_spec])
    else:
        print(
            "invalid inner kernel type for to_kernel_spec_top_level:", type(inner_spec)
        )
        print(inner_spec)
        raise TypeError("invalid inner kernel type for to_kernel_spec_top_level")
