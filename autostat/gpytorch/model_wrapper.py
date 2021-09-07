import numpy as np
from numpy.typing import ArrayLike, NDArray
import typing as ty


from gpytorch.kernels import Kernel, AdditiveKernel
import gpytorch as gp
import torch
from botorch import fit_gpytorch_model


from .to_kernel_spec import to_kernel_spec

from ..kernel_specs import (
    TopLevelKernelSpec,
)
from ..run_settings import RunSettings
from ..dataset_adapters import Dataset, ModelPredictions
from .kernel_builder import build_kernel
from ..math import calc_bic


class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


class GpytorchGPModel:
    def __init__(
        self,
        kernel_spec: TopLevelKernelSpec,
        data: Dataset,
        run_settings: RunSettings,
        use_cuda: bool = True,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

        self.data = data
        self.run_settings = run_settings

        self.kernel_spec = kernel_spec
        self.built_kernel = build_kernel(kernel_spec)

        self.likelihood = gp.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.initialize(noise=kernel_spec.noise)

        self.model = ExactGPModel(
            self._np_to_dev(data.train_x).flatten(),
            self._np_to_dev(data.train_y).flatten(),
            self.likelihood,
            self.built_kernel,
        ).to(self.device)

        self.mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def _np_to_dev(self, arr):
        return torch.from_numpy(arr).to(self.device)

    def fit(self, data: Dataset) -> None:
        self.model.train()
        self.likelihood.train()
        fit_gpytorch_model(self.mll)

    def log_likelihood(self) -> float:
        with torch.no_grad():
            x = self._np_to_dev(self.data.train_x)
            Y = self.data.train_y

            covar_module = ty.cast(torch.Tensor, self.model.covar_module(x))
            L = covar_module.detach().cholesky().clone().cpu().numpy()

            # L = self.model.covar_module(x).detach().cholesky().clone().cpu().numpy()
            N = Y.shape[0]
            sigma2 = ty.cast(
                gp.likelihoods.GaussianLikelihood, self.model.likelihood
            ).noise.item()

            K = L @ L.T + sigma2 * np.eye(N)
            K_inv = np.linalg.inv(K)
            _, log_det_K = np.linalg.slogdet(K)

        return (-0.5 * (Y.T @ K_inv @ Y + log_det_K + N * np.log(2 * np.pi))).item()

    def bic(self) -> float:
        return calc_bic(
            self.kernel_spec.num_params(),
            self.data.train_x.shape[0],
            self.log_likelihood(),
        )

    def predict(self, x: ArrayLike) -> ModelPredictions:
        x_torch = self._np_to_dev(x).flatten()
        self.model.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = ty.cast(
                gp.distributions.MultivariateNormal,
                self.likelihood(self.model(x_torch)),
            )

            lower, upper = observed_pred.confidence_region()

            return ModelPredictions(
                observed_pred.mean.cpu().numpy(),
                lower.cpu().numpy(),
                upper.cpu().numpy(),
            )

    def residuals(self) -> NDArray[np.float_]:
        yHat, _, _ = self.predict(self.data.train_x)
        yHat = yHat.flatten()
        train_y = self.data.train_y.flatten()
        residuals = train_y - yHat
        return residuals

    def to_spec(self) -> TopLevelKernelSpec:
        return to_kernel_spec(self.built_kernel, float(self.likelihood.noise.item()))
