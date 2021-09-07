import typing as ty


from gpytorch.kernels import (
    Kernel,
    PeriodicKernel,
    LinearKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    ProductKernel,
    AdditiveKernel,
)
from sklearn.gaussian_process.kernels import RationalQuadratic

from ..kernel_specs import (
    AdditiveKernelSpec,
    KernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    RQKernelSpec,
    TopLevelKernelSpec,
)
from .custom_periodic_kernel import PeriodicKernelNoConstant


ProductOperandKernel = ty.Union[
    AdditiveKernel, RBFKernel, PeriodicKernel, LinearKernel, RQKernel
]


@ty.overload
def to_kernel_spec_inner(kernel: ScaleKernel) -> ProductKernelSpec:  # type: ignore
    ...


@ty.overload
def to_kernel_spec_inner(kernel: ProductOperandKernel) -> ProductOperandSpec:  # type: ignore
    ...


def to_kernel_spec_inner(kernel: Kernel) -> KernelSpec:
    lengthscale = float(kernel.lengthscale.item()) if kernel.lengthscale else 1.0

    if isinstance(kernel, RBFKernel):
        inner: KernelSpec = RBFKernelSpec(length_scale=lengthscale)

    elif isinstance(kernel, LinearKernel):
        # inner = (sigma_0=kernel.variance)
        inner = LinearKernelSpec(variance=kernel.variance.item())

    elif isinstance(kernel, PeriodicKernel):
        inner = PeriodicKernelSpec(
            length_scale=lengthscale, period=kernel.period_length.item()
        )

    # elif isinstance(kernel, PeriodicKernelNoConstant):
    #     inner = PeriodicNoConstKernelSpec(
    #         length_scale=lengthscale, period=kernel.period_length
    #     )

    elif isinstance(kernel, RQKernel):
        inner = RQKernelSpec(length_scale=lengthscale, alpha=kernel.alpha.item())

    elif isinstance(kernel, AdditiveKernel):
        # inner = to_kernel_spec_additive(kernel)
        subkernels = ty.cast(list[ScaleKernel], kernel.kernels)
        operands = [to_kernel_spec_inner(sub_spec) for sub_spec in subkernels]
        inner = AdditiveKernelSpec(operands)

    elif isinstance(kernel, ScaleKernel):
        # NOTE: ScaleKernel must always contain a product kernel
        scalar = kernel.outputscale.item()
        prod_kernel = kernel.base_kernel
        subkernels = ty.cast(list[ProductOperandKernel], prod_kernel.kernels)
        operands = [to_kernel_spec_inner(sub_spec) for sub_spec in subkernels]
        inner = ProductKernelSpec(operands, scalar)  # type: ignore

    else:
        print("invalid kernel type -- type(kernel):", type(kernel))
        print(kernel)
        raise TypeError("Invalid kernel type for translation to spec")

    return inner


def to_kernel_spec(kernel: AdditiveKernel, noise: float) -> TopLevelKernelSpec:
    if not isinstance(kernel, AdditiveKernel):
        raise TypeError(
            "to_kernel_spec expects a Gpytorch AdditiveKernel (which must contain one or more ScaleKernels wrapping ProductKernels"
        )

    subkernels = ty.cast(list[ScaleKernel], kernel.kernels)
    for k in subkernels:
        if not isinstance(k, ScaleKernel):
            raise TypeError(
                "Immediate children of Gpytorch AdditiveKernels input to to_kernel_spec must be ScaleKernels wrapping ProductKernels"
            )
    operands = [to_kernel_spec_inner(inner) for inner in subkernels]

    # ty.cast(list[ProductKernelSpec],

    return TopLevelKernelSpec(
        operands=operands,
        noise=noise,
    )
