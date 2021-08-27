from ..constraints import KernelConstraints, default_constraints
import typing as ty


# from sklearn.gaussian_process.kernels import (
#     RBF,
#     ConstantKernel,
#     DotProduct,
#     Kernel,
#     Product,
#     RationalQuadratic,
#     Sum,
#     WhiteKernel,
#     ExpSineSquared,
# )

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
)


"""
In gpytorch, any compositional kernel will be represented as a top level AdditiveKernel containing
one or more ScaleKernels, which will always contain a product kernel.
"""


@ty.overload
def build_kernel(
    kernel_spec: TopLevelKernelSpec, constraints: KernelConstraints
) -> AdditiveKernel:
    ...


@ty.overload
def build_kernel(kernel_spec: KernelSpec, constraints: KernelConstraints) -> Kernel:
    ...


def build_kernel(kernel_spec: KernelSpec, constraints: KernelConstraints) -> Kernel:

    constraints = constraints or default_constraints()

    if isinstance(kernel_spec, RBFKernelSpec):
        inner = RBFKernel()
        inner.lengthscale = kernel_spec.length_scale

    elif isinstance(kernel_spec, LinearKernelSpec):
        inner = LinearKernel()
        inner.variance = kernel_spec.variance

    elif isinstance(kernel_spec, PeriodicNoConstKernelSpec):
        # kwargs = {
        #     "periodicity_bounds": constraints.PER.period,
        #     "length_scale_bounds": constraints.PER.length_scale,
        # }
        # inner = PeriodicKernelNoConstant(
        #     length_scale=kernel_spec.length_scale,
        #     periodicity=kernel_spec.period,
        #     **kwargs
        # )
        raise NotImplementedError()

    elif isinstance(kernel_spec, PeriodicKernelSpec):

        periodicity_bounds = constraints.PER.period
        length_scale_bounds = constraints.PER.length_scale

        inner = PeriodicKernel(
            lengthscale_constraint=Interval(
                lower_bound=length_scale_bounds.lower,
                upper_bound=length_scale_bounds.upper,
            ),
            period_length_constraint=Interval(
                lower_bound=periodicity_bounds.lower,
                upper_bound=periodicity_bounds.upper,
            ),
        )
        inner.period_length = kernel_spec.period
        inner.lengthscale = kernel_spec.length_scale

    elif isinstance(kernel_spec, RQKernelSpec):
        inner = RQKernel()
        inner.lengthscale = kernel_spec.length_scale
        inner.alpha = kernel_spec.alpha

    elif isinstance(kernel_spec, AdditiveKernelSpec) or isinstance(
        kernel_spec, TopLevelKernelSpec
    ):
        operands = [
            build_kernel(product_spec, constraints)
            for product_spec in kernel_spec.operands
        ]
        inner = AdditiveKernel(*operands)

    elif isinstance(kernel_spec, ProductKernelSpec):
        operands = [
            build_kernel(sub_spec, constraints) for sub_spec in kernel_spec.operands
        ]
        inner = ScaleKernel(ProductKernel(*operands))
        inner.outputscale = kernel_spec.scalar

    else:
        print("invalid kernel_spec type -- type(kernel_spec):", type(kernel_spec))
        print(kernel_spec)
        raise TypeError("Invalid kernel spec type")

    # if isinstance(kernel_spec, TopLevelKernelSpec):
    #     inner = build_kernel_additive(kernel_spec, constraints)
    #     inner = inner + WhiteKernel(noise_level=kernel_spec.noise)

    return inner
