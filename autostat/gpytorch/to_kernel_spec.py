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


# def to_kernel_spec_additive(kernel: Sum) -> AdditiveKernelSpec:
#     params = kernel.get_params()
#     operands: list[ProductKernelSpec] = []
#     kernels_to_check = [params["k1"], params["k2"]]

#     while kernels_to_check:
#         this_k = kernels_to_check.pop()
#         params = this_k.get_params()
#         if isinstance(this_k, Sum):
#             kernels_to_check.append(params["k1"])
#             kernels_to_check.append(params["k2"])
#         elif isinstance(this_k, Product):
#             operands.append(to_kernel_spec_product(this_k))
#         elif isinstance(this_k, WhiteKernel):
#             pass
#         else:
#             print("invalid kernel type -- type(kernel):", type(kernel))
#             print(kernel)
#             raise TypeError(
#                 "Invalid sklearn kernel type for to_kernel_spec_additive; must be additive or product"
#             )

#     return AdditiveKernelSpec(operands=operands)


# def to_kernel_spec_product(kernel: Product) -> ProductKernelSpec:
#     params = kernel.get_params()
#     operands: list[ProductOperandSpec] = []
#     scalar = 1
#     kernels_to_check = [params["k1"], params["k2"]]

#     while kernels_to_check:
#         this_k = kernels_to_check.pop()
#         params = this_k.get_params()
#         if isinstance(this_k, Product):
#             kernels_to_check.append(params["k1"])
#             kernels_to_check.append(params["k2"])
#         elif isinstance(this_k, Sum):
#             operands.append(to_kernel_spec_additive(this_k))
#         elif isinstance(this_k, ConstantKernel):
#             scalar = params["constant_value"]
#         elif (
#             isinstance(this_k, RBF)
#             or isinstance(this_k, RationalQuadratic)
#             or isinstance(this_k, PeriodicKernelNoConstant)
#             or isinstance(this_k, DotProduct)
#             or isinstance(this_k, ExpSineSquared)
#         ):
#             spec = ty.cast(ProductOperandSpec, to_kernel_spec_inner(this_k))
#             operands.append(spec)
#         else:
#             print("invalid kernel type for to_kernel_spec_product:", type(this_k))
#             print(this_k)
#             raise TypeError("Invalid sklearn kernel type for to_kernel_spec_product")

#     return ProductKernelSpec(operands=operands, scalar=scalar)


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
        operands = [to_kernel_spec_inner(sub_spec) for sub_spec in kernel.kernels]
        inner = AdditiveKernelSpec(operands)

    elif isinstance(kernel, ScaleKernel):
        # NOTE: ScaleKernel must always contain a product kernel
        scalar = kernel.outputscale.item()
        prod_kernel = kernel.base_kernel
        # inner = to_kernel_spec_product(kernel)
        operands = [to_kernel_spec_inner(sub_spec) for sub_spec in prod_kernel.kernels]
        inner = ProductKernelSpec(operands, scalar)

    else:
        print("invalid kernel type -- type(kernel):", type(kernel))
        print(kernel)
        raise TypeError("Invalid kernel type for translation to spec")

    return inner


def to_kernel_spec(kernel: AdditiveKernel, noise: float) -> TopLevelKernelSpec:
    # inner_spec =
    subkernels = kernel.kernels
    return TopLevelKernelSpec(
        operands=[to_kernel_spec_inner(inner) for inner in subkernels],
        noise=noise,
    )

    # inner_spec = to_kernel_spec_inner(kernel)
    # if isinstance(inner_spec, AdditiveKernelSpec):
    #     # NOTE: sklearn Sum kernels including a WhiteKernel must map to a TopLevelKernelSpec
    #     if isinstance(kernel.k1, WhiteKernel) or isinstance(kernel.k2, WhiteKernel):
    #         wk = kernel.k1 if isinstance(kernel.k1, WhiteKernel) else kernel.k2
    #         return TopLevelKernelSpec(
    #             operands=inner_spec.operands, noise=wk.get_params()["noise_level"]
    #         )

    #     return TopLevelKernelSpec(operands=inner_spec.operands)
    # elif isinstance(inner_spec, ProductKernelSpec):
    #     return TopLevelKernelSpec(operands=[inner_spec])
    # else:
    #     print(
    #         "invalid inner kernel type for to_kernel_spec_top_level:", type(inner_spec)
    #     )
    #     print(inner_spec)
    #     raise TypeError("invalid inner kernel type for to_kernel_spec_top_level")
