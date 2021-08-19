# from autostat.constraints import constraints_from_data
import typing as ty

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    Sum,
)

from .to_kernel_spec import to_kernel_spec

from ..kernel_specs import (
    AdditiveKernelSpec,
    KernelSpec,
)
from ..run_settings import RunSettings
from ..dataset_adapters import Dataset, NpDataSet, ModelPredictions
from .kernel_builder import build_kernel
from ..math import calc_bic


class SklearnGPModel:
    def __init__(
        self,
        kernel_spec: KernelSpec,
        data: Dataset,
        run_settings: RunSettings,
        alpha=1e-7,
    ) -> None:
        self.kernel_spec = kernel_spec
        self.data = ty.cast(NpDataSet, data)
        self.run_settings = run_settings

        kernel = build_kernel(
            kernel_spec,
            constraints=self.run_settings.kernel_constraints,
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=3
        )

    def fit(self, data: Dataset) -> None:
        self.gp.fit(data.train_x, data.train_y)

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
