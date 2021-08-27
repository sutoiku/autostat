import typing as ty

import numpy as np
from numpy.typing import ArrayLike, NDArray

# from sklearn.gaussian_process import GaussianProcessRegressor

# from sklearn.gaussian_process.kernels import (
#     Kernel,
#     Sum,
# )

from gpytorch.kernels import Kernel, AdditiveKernel
import gpytorch as gp
import torch
from botorch import fit_gpytorch_model


from .to_kernel_spec import to_kernel_spec

from ..kernel_specs import (
    TopLevelKernelSpec,
    KernelSpec,
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
        self.kernel_spec = kernel_spec
        self.data = data
        self.run_settings = run_settings

        self.likelihood = gp.likelihoods.GaussianLikelihood()
        self.likelihood.initialize(noise=kernel_spec.noise)

        self.built_kernel = build_kernel(
            kernel_spec,
            constraints=self.run_settings.kernel_constraints,
        )

        self.model = ExactGPModel(x, y, self.likelihood, self.built_kernel)

        self.mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def fit(self, data: Dataset) -> None:
        fit_gpytorch_model(self.mll)

    def log_likelihood(self) -> float:
        ...
        # k = ty.cast(Kernel, self.gp.kernel_)
        # ll: float = ty.cast(float, self.gp.log_marginal_likelihood(k.theta))
        # return ll

    def bic(self) -> float:
        return calc_bic(
            self.kernel_spec.num_params(),
            self.data.train_x.shape[0],
            self.log_likelihood(),
        )

    def predict(self, x: ArrayLike) -> ModelPredictions:
        # y_pred, y_std = ty.cast(
        #     tuple[NDArray[np.float_], NDArray[np.float_]],
        #     self.gp.predict(x, return_std=True),
        # )
        self.model.eval()
        self.likelihood.eval()
        x = torch.tensor(x, dtype=torch.float64, device=self.device).flatten()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
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
