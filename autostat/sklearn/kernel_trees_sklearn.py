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
from .kernel_builder import build_kernel
from ..math import calc_bic


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


class SklearnGPModel:
    def __init__(self, kernel_spec: KernelSpec, data: Dataset, alpha=1e-7) -> None:
        self.kernel_spec = kernel_spec
        self.data = ty.cast(NpDataSet, data)
        self.constraints = constraints_from_data(self.data)
        kernel = build_kernel(kernel_spec, constraints=self.constraints, top_level=True)

        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False
        )

    def fit(self, data: Dataset) -> None:
        self.gp.fit(data.train_x, data.train_y)
        # print("\n---post-fit ---")
        # print(self.gp)
        # print("\n")

    def log_likelihood(self) -> float:
        k = ty.cast(Kernel, self.gp.kernel_)
        ll: float = ty.cast(float, self.gp.log_marginal_likelihood(k.theta))
        return ll

    def bic(self) -> float:
        return calc_bic(
            self.kernel_spec.num_params(),
            self.data.train_x.shape[0],
            self.log_likelihood(),
        )

    def predict(self, x: ArrayLike) -> ModelPredictions:
        y_pred, y_std = ty.cast(
            tuple[NDArray[np.float_], NDArray[np.float_]],
            self.gp.predict(x, return_std=True),
        )

        # y_pred, y_std = self.gp.predict(x, return_std=True)

        y_pred = y_pred.flatten()
        y_std = y_std.flatten()
        return ModelPredictions(y_pred, y_pred - 2 * y_std, y_pred + 2 * y_std)

    def residuals(self) -> NDArray[np.float_]:
        yHat, _, _ = self.predict(self.data.train_x)
        yHat = yHat.flatten()
        train_y = self.data.train_y.flatten()
        residuals = train_y - yHat
        return residuals

    def print_fitted_kernel(self):
        print(self.gp.kernel_)

    def to_spec(self) -> AdditiveKernelSpec:
        return to_kernel_spec(ty.cast(Sum, self.gp.kernel_))