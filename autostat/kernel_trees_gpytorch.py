from gpytorch import kernels
from gpytorch.likelihoods import likelihood
import torch
import gpytorch
from gpytorch.kernels import (
    Kernel,
    ScaleKernel,
    AdditiveKernel,
    ProductKernel,
    PeriodicKernel,
    LinearKernel,
    RBFKernel,
    RQKernel,
)
from torch import tensor
from torch.functional import Tensor


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel: Kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit(model, likelihood, train_x, train_y, training_iters=50):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.debug(False):
        for i in range(training_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()
    return mll


def num_params_by_kernel_type(k: type[Kernel]):
    if k is RBFKernel:
        return 2
    elif k is LinearKernel:
        return 1
    elif k is PeriodicKernel:
        return 2
    elif k is RQKernel:
        return 2
    else:
        raise TypeError("Invalid kernel type")


def kernel_type_str(k: type[Kernel]):
    if k is RBFKernel:
        return "RBF"
    elif k is LinearKernel:
        return "LIN"
    elif k is PeriodicKernel:
        return "PER"
    elif k is RQKernel:
        return "RQ"
    elif k is AdditiveKernel:
        return "ADD"
    elif k is ProductKernel:
        return "PROD"
    else:
        raise TypeError("Invalid kernel type")


def initialize_model_from_kernel_spec(kernel_tree: KernelTreeNode) -> GPModel:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # if likelihood_noise is not None:
    likelihood.initialize(noise=0.01)
    likelihood.cuda()
    likelihood.double()

    built_kernel = build(kernel_tree)
    print("before training:", built_kernel_str(built_kernel))
    model = ExactGPModel(train_x, train_y, likelihood, built_kernel)
    model.cuda()
    model.double()

    return GpytorchModel(model, likelihood, kernel_tree)
