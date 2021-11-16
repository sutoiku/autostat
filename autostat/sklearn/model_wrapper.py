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
    TopLevelKernelSpec,
    KernelSpec,
)
from ..run_settings import RunSettings
from ..dataset_adapters import Dataset, ModelPredictions
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
        self.data = data
        self.run_settings = run_settings

        kernel = build_kernel(kernel_spec)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=run_settings.sklean_n_restarts_optimizer,
        )
        self.training_predictions: ty.Union[ModelPredictions, None] = None
        self.test_predictions: ty.Union[ModelPredictions, None] = None

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

    def predict_train(self):
        return self._predict_cached(train=True)

    def predict_test(self):
        return self._predict_cached(train=False)

    def _predict_cached(self, train=True) -> ModelPredictions:
        if train:
            if self.training_predictions is None:
                self.training_predictions = self._predict(self.data.train_x)
            return self.training_predictions
        else:
            if self.test_predictions is None:
                self.test_predictions = self._predict(self.data.test_x)
            return self.test_predictions

    def _predict(self, x: ArrayLike) -> ModelPredictions:
        y_pred, y_std = ty.cast(
            tuple[NDArray[np.float_], NDArray[np.float_]],
            self.gp.predict(x, return_std=True),
        )
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()
        return ModelPredictions(y_pred, y_std)

    def residuals(self) -> NDArray[np.float_]:
        yHat, _ = self.predict_train()
        yHat = yHat.flatten()
        train_y = self.data.train_y.flatten()
        residuals = train_y - yHat
        return residuals

    def print_fitted_kernel(self):
        print(self.gp.kernel_)

    def to_spec(self) -> TopLevelKernelSpec:
        return to_kernel_spec(ty.cast(Sum, self.gp.kernel_))

    def prediction_log_prob_score(self) -> float:
        # NOTE: we don't need the covariance for this score b/c we're only
        # concerned about the epsilon between the predicted latent function y_pred
        # and the observation y_test. Under the assumption that we have noisy observations--
        # y_test = y_pred + ε , with ε ~ N(0, σ^2 * I)
        #  -- then the prob of seeing a collection of epsilons
        # ε = y_test - y_pred
        # does not depend on the covariance structure of the kernel matrix
        y_pred, y_std = self.predict_test()
        y_test = self.data.test_y
        z_score_sqr = ((y_pred - y_test) / y_std) ** 2
        N = len(y_pred)

        log_prob_score = -0.5 * N * np.log(2 * np.pi) - np.sum(
            0.5 * z_score_sqr + np.log(y_std)
        )
        return log_prob_score
