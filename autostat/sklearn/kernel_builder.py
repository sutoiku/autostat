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
    ExpSineSquared,
)

from .custom_periodic_kernel import PeriodicKernelNoConstant

from ..kernel_specs import (
    AdditiveKernelSpec,
    KernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
    ProductKernelSpec,
    RBFKernelSpec,
    RQKernelSpec,
    TopLevelKernelSpec,
)


def build_kernel_additive(kernel_spec: AdditiveKernelSpec) -> ty.Union[Sum, Product]:

    inner = build_kernel(kernel_spec.operands[0])
    if len(kernel_spec.operands) == 1:
        return ty.cast(Product, inner)

    for product in kernel_spec.operands[1:-1]:
        inner += build_kernel(product)
    inner += build_kernel(kernel_spec.operands[-1])
    return inner


def build_kernel(kernel_spec: KernelSpec) -> Kernel:

    if isinstance(kernel_spec, RBFKernelSpec):
        inner = RBF(length_scale=kernel_spec.length_scale)

    elif isinstance(kernel_spec, LinearKernelSpec):
        inner = DotProduct(sigma_0=kernel_spec.variance)

    elif isinstance(kernel_spec, PeriodicNoConstKernelSpec):
        kwargs = {
            "periodicity_bounds": kernel_spec.period_bounds,
            "length_scale_bounds": kernel_spec.length_scale_bounds,
        }
        inner = PeriodicKernelNoConstant(
            length_scale=kernel_spec.length_scale,
            periodicity=kernel_spec.period,
            **kwargs
        )

    elif isinstance(kernel_spec, PeriodicKernelSpec):
        kwargs = {
            "periodicity_bounds": kernel_spec.period_bounds,
            "length_scale_bounds": kernel_spec.length_scale_bounds,
        }
        inner = ExpSineSquared(
            length_scale=kernel_spec.length_scale,
            periodicity=kernel_spec.period,
            **kwargs
        )

    elif isinstance(kernel_spec, RQKernelSpec):
        inner = RationalQuadratic(
            length_scale=kernel_spec.length_scale, alpha=kernel_spec.alpha
        )

    elif isinstance(kernel_spec, AdditiveKernelSpec):
        inner = build_kernel_additive(kernel_spec)

    elif isinstance(kernel_spec, ProductKernelSpec):
        inner = ConstantKernel(constant_value=kernel_spec.scalar)
        for operand in kernel_spec.operands:
            inner *= build_kernel(operand)

    else:
        print("invalid kernel_spec type -- type(kernel_spec):", type(kernel_spec))
        print(kernel_spec)
        raise TypeError("Invalid kernel spec type")

    if isinstance(kernel_spec, TopLevelKernelSpec):
        inner = build_kernel_additive(kernel_spec)
        inner = inner + WhiteKernel(noise_level=kernel_spec.noise)

    return inner
