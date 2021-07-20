import typing as T

import numpy as np
from numpy.typing import ArrayLike
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

from .kernel_tree_types import (
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
from .math import calc_bic


def build_kernel_additive(kernel_spec: AdditiveKernelSpec) -> T.Union[Sum, Product]:

    inner = build_kernel(kernel_spec.operands[0])
    if len(kernel_spec.operands) == 1:
        return T.cast(Product, inner)

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
            or isinstance(this_k, ExpSineSquared)
            or isinstance(this_k, DotProduct)
        ):
            operands.append(to_kernel_spec_inner(this_k))
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

    elif isinstance(kernel, ExpSineSquared):
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


def to_kernel_spec(kernel: T.Union[Sum, Product]) -> AdditiveKernelSpec:
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


class SklearnGPModel:
    def __init__(self, kernel_spec: KernelSpec, data: Dataset, alpha=1e-7) -> None:
        self.kernel_spec = kernel_spec
        kernel = build_kernel(kernel_spec, top_level=True)
        self.data = T.cast(NpDataSet, data)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False
        )

    def fit(self, data: Dataset) -> None:
        self.gp.fit(data.train_x, data.train_y)
        # print("\n---post-fit ---")
        # print(self.gp)
        # print("\n")

    def log_likelihood(self) -> float:
        k = T.cast(Kernel, self.gp.kernel_)
        ll: float = T.cast(float, self.gp.log_marginal_likelihood(k.theta))
        return ll

    def bic(self) -> float:
        return calc_bic(
            self.kernel_spec.num_params(),
            self.data.train_x.shape[0],
            self.log_likelihood(),
        )

    def predict(self, x: ArrayLike) -> ModelPredictions:
        y_pred, y_std = T.cast(
            tuple[np.ndarray, np.ndarray], self.gp.predict(x, return_std=True)
        )
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()
        return ModelPredictions(y_pred, y_pred - 2 * y_std, y_pred + 2 * y_std)

    def residuals(self) -> np.ndarray:
        yHat, _, _ = self.predict(self.data.train_x)
        return self.data.train_y - yHat

    def print_fitted_kernel(self):
        print(self.gp.kernel_)

    def to_spec(self) -> AdditiveKernelSpec:
        return to_kernel_spec(T.cast(Sum, self.gp.kernel_))
