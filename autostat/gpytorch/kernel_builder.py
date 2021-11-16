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

from gpytorch.constraints import Interval

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
    ConstraintBounds,
)


"""
In gpytorch, any compositional kernel will be represented as a top level AdditiveKernel containing
one or more ScaleKernels, which will always contain a product kernel.
"""


def bounds_to_interval(cb: ConstraintBounds) -> Interval:
    return Interval(lower_bound=cb.lower, upper_bound=cb.upper)


@ty.overload
def build_kernel(kernel_spec: TopLevelKernelSpec) -> AdditiveKernel:
    ...


@ty.overload
def build_kernel(kernel_spec: KernelSpec) -> Kernel:
    ...


def build_kernel(kernel_spec: KernelSpec) -> Kernel:

    if isinstance(kernel_spec, RBFKernelSpec):
        inner = RBFKernel(
            lengthscale_constraint=bounds_to_interval(kernel_spec.length_scale_bounds)
        )
        inner.lengthscale = kernel_spec.length_scale

    elif isinstance(kernel_spec, LinearKernelSpec):
        inner = LinearKernel(
            variance_constraint=bounds_to_interval(kernel_spec.variance_bounds)
        )
        inner.variance = kernel_spec.variance

    elif isinstance(kernel_spec, PeriodicNoConstKernelSpec):

        inner = PeriodicKernelNoConstant(
            lengthscale_constraint=bounds_to_interval(kernel_spec.length_scale_bounds),
            period_length_constraint=bounds_to_interval(kernel_spec.period_bounds),
        )
        inner.period_length = kernel_spec.period
        inner.lengthscale = kernel_spec.length_scale

    elif isinstance(kernel_spec, PeriodicKernelSpec):

        inner = PeriodicKernel(
            lengthscale_constraint=bounds_to_interval(kernel_spec.length_scale_bounds),
            period_length_constraint=bounds_to_interval(kernel_spec.period_bounds),
        )
        inner.period_length = kernel_spec.period
        inner.lengthscale = kernel_spec.length_scale

    elif isinstance(kernel_spec, RQKernelSpec):
        inner = RQKernel(
            lengthscale_constraint=bounds_to_interval(kernel_spec.length_scale_bounds),
            alpha_constraint=bounds_to_interval(kernel_spec.alpha_bounds),
        )
        inner.lengthscale = kernel_spec.length_scale
        inner.alpha = kernel_spec.alpha

    elif isinstance(kernel_spec, AdditiveKernelSpec) or isinstance(
        kernel_spec, TopLevelKernelSpec
    ):
        operands = [build_kernel(product_spec) for product_spec in kernel_spec.operands]
        inner = AdditiveKernel(*operands)

    elif isinstance(kernel_spec, ProductKernelSpec):
        operands = [build_kernel(sub_spec) for sub_spec in kernel_spec.operands]
        inner = ScaleKernel(ProductKernel(*operands))
        inner.outputscale = kernel_spec.scalar

    else:
        print("invalid kernel_spec type -- type(kernel_spec):", type(kernel_spec))
        print(kernel_spec)
        raise TypeError("Invalid kernel spec type")

    return inner
