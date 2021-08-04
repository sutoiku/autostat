from autostat.constraints import constraints_from_data
import typing as ty

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Product,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)

from ..kernel_tree_types import (
    AdditiveKernelSpec,
    ArithmeticKernelSpec,
    Dataset,
    KernelSpec,
    LinearKernelSpec,
    ModelPredictions,
    NpDataSet,
    PeriodicKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    RQKernelSpec,
)


def build_kernel_additive(kernel_spec: AdditiveKernelSpec) -> ty.Union[Sum, Product]:

    inner = build_kernel(kernel_spec.operands[0])
    if len(kernel_spec.operands) == 1:
        return ty.cast(Product, inner)

    for product in kernel_spec.operands[1:-1]:
        inner += build_kernel(product)
    inner += build_kernel(kernel_spec.operands[-1])
    return inner


def build_kernel(kernel_spec: KernelSpec, top_level=False) -> Kernel:
    if isinstance(kernel_spec, RBFKernelSpec):
        inner = RBF(length_scale=kernel_spec.length_scale)

    elif isinstance(kernel_spec, LinearKernelSpec):
        inner = DotProduct(sigma_0=kernel_spec.variance)

    elif isinstance(kernel_spec, PeriodicKernelSpec):
        inner = ExpSineSquared(
            length_scale=kernel_spec.length_scale,
            periodicity=kernel_spec.period,
        )

    elif isinstance(kernel_spec, RQKernelSpec):
        inner = RationalQuadratic(
            length_scale=kernel_spec.length_scale, alpha=kernel_spec.alpha
        )

    elif isinstance(kernel_spec, AdditiveKernelSpec):
        inner = build_kernel_additive(kernel_spec)

    elif isinstance(kernel_spec, ProductKernelSpec):
        # print("ProductKernelSpec: ", kernel_spec)
        # print("ProductKernelSpec operands:", kernel_spec.operands)
        inner = ConstantKernel(constant_value=kernel_spec.scalar)
        # for i in range(0, len(kernel_spec.operands)):
        for operand in kernel_spec.operands:
            # print("ProductKernelSpec operand", operand)
            inner *= build_kernel(operand, False)

    else:
        print("invalid kernel_spec type -- type(kernel_spec):", type(kernel_spec))
        print(kernel_spec)
        raise TypeError("Invalid kernel spec type")

    inner = inner + WhiteKernel(noise_level=0.001) if top_level else inner

    return inner
