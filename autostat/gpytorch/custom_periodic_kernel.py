import math
import torch

from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel

from scipy.special import i0e, i1e

import warnings


class i0eTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        device = input.device
        if torch.any(input < 0):
            raise ValueError("i0eTorchFunction only accepts positive inputs")
        input_np = input.detach().cpu().numpy()
        result_np = i0e(input_np)
        result = torch.from_numpy(result_np).to(device)
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, result = ctx.saved_tensors
        device = input.device
        g = torch.from_numpy(i1e(input.detach().cpu().numpy())).to(device) - result
        return g * grad_output


i0e_torch = i0eTorchFunction.apply


class PeriodicKernelNoConstant(Kernel):
    r"""Computes a covariance matrix based on the periodic kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            k_{\text{Periodic}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left(
            -2 \sum_i
            \frac{\sin ^2 \left( \frac{\pi}{p} (\mathbf{x_{1,i}} - \mathbf{x_{2,i}} ) \right)}{\lambda}
            \right)
        \end{equation*}

    where

    * :math:`p` is the period length parameter.
    * :math:`\lambda` is a lengthscale parameter.

    Equation is based on [David Mackay's Introduction to Gaussian Processes equation 47]
    (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.1927&rep=rep1&type=pdf)
    albeit without feature-specific lengthscales and period lengths. The exponential
    coefficient was changed and lengthscale is not squared to maintain backwards compatibility

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        This kernel does not have an ARD lengthscale or period length option.

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`period_length_prior` (Prior, optional):
            Set this if you want to apply a prior to the period length parameter.  Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the lengthscale. Default: `Positive`.
        :attr:`period_length_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the period length. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale/period length can take
            (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size = `*batch_shape x 1 x 1`.
        :attr:`period_length` (Tensor):
            The period length parameter. Size = `*batch_shape x 1 x 1`.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernelNoConstant())
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernelNoConstant())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernelNoConstant(batch_size=2))
        >>> covar = covar_module(x)  # Output: LazyVariable of size (2 x 10 x 10)
    """

    has_lengthscale = True

    def __init__(
        self, period_length_prior=None, period_length_constraint=None, **kwargs
    ):
        super(PeriodicKernelNoConstant, self).__init__(**kwargs)

        if period_length_constraint is None:
            period_length_constraint = Positive()

        self.register_parameter(
            name="raw_period_length",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if period_length_prior is not None:
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        if value < 0.01:
            warnings.warn(
                "PeriodicKernelNoConstant suffers from numerical instability for small values of period_length",
            )
        self._set_period_length(value)

    @property
    def lengthscale(self):
        return super().lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        if value < 0.01:
            warnings.warn(
                "PeriodicKernelNoConstant suffers from numerical instability for small values of lengthscale"
            )
        self._set_lengthscale(value)
        # return super().lengthscale

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(
            raw_period_length=self.raw_period_length_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.period_length).mul(math.pi)
        x2_ = x2.div(self.period_length).mul(math.pi)
        diff_pi_over_period = x1_.unsqueeze(-2) - x2_.unsqueeze(-3)

        expBess0 = i0e_torch(1 / self.lengthscale ** 2)

        exp = (
            diff_pi_over_period.squeeze(-1)
            .sin()
            .div(self.lengthscale)
            .pow(2)
            .mul(-2.0)
            .exp_()
        )

        res = (expBess0 - exp) / (expBess0 - 1)

        if diag:
            res = res.squeeze(0)
        return res
