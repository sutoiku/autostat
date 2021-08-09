from sklearn.gaussian_process.kernels import (
    Kernel,
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Hyperparameter,
)

from scipy.special import iv

import numpy as np

# from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform


class PeriodicKernelNoConstant(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    r"""Periodic kernel "without constant"
    Described in appendix A of https://arxiv.org/pdf/1402.4304.pdf

    https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/matlab/custom-cov/covPeriodicNoDC.m

    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.
    periodicity : float > 0, default=1.0
        The periodicity of the kernel.
    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.
    periodicity_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'periodicity'.
        If set to "fixed", 'periodicity' cannot be changed during
        hyperparameter tuning.

    """

    def __init__(
        self,
        length_scale=1.0,
        periodicity=1.0,
        length_scale_bounds=(1e-5, 1e5),
        periodicity_bounds=(1e-5, 1e5),
    ):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.length_scale_bounds = length_scale_bounds
        self.periodicity_bounds = periodicity_bounds

    @property
    def hyperparameter_length_scale(self):
        """Returns the length scale"""
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_periodicity(self):
        return Hyperparameter("periodicity", "numeric", self.periodicity_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(pdist(X, metric="euclidean"))
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="euclidean")

        l = self.length_scale
        p = self.periodicity

        one_over_lsqr = 1 / (l ** 2)
        exp_one_over_lsqr = np.exp(one_over_lsqr)

        bess_0 = iv(0, one_over_lsqr)

        period_dist = (dists * np.pi) / p
        two_period_dist = 2 * period_dist
        cos_two_period_dist = np.cos(two_period_dist)
        exp_scaled_dists = np.exp(one_over_lsqr * cos_two_period_dist)

        # arg = np.pi * dists / self.periodicity
        # sin_of_arg = np.sin(arg)
        # K = np.exp(-2 * (sin_of_arg / self.length_scale) ** 2)

        K = (exp_scaled_dists - bess_0) / (exp_one_over_lsqr - bess_0)

        if eval_gradient:
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                bess_1 = iv(1, one_over_lsqr)
                length_scale_gradient = (
                    -2
                    * exp_one_over_lsqr
                    * (
                        bess_0
                        - bess_1
                        - exp_scaled_dists
                        + exp_scaled_dists * np.cos(period_dist)
                    )
                    - 2 * exp_scaled_dists * (bess_1 - bess_0 * cos_two_period_dist)
                ) / ((bess_0 - exp_one_over_lsqr) ** 2 * l ** 3)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:
                periodicity_gradient = (
                    exp_scaled_dists
                    * one_over_lsqr
                    * two_period_dist
                    * np.sin(two_period_dist)
                ) / (p * (exp_one_over_lsqr - bess_0))
                periodicity_gradient = periodicity_gradient[:, :, np.newaxis]
            else:  # p is kept fixed
                periodicity_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((length_scale_gradient, periodicity_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale={1:.3g}, periodicity={2:.3g})".format(
            self.__class__.__name__, self.length_scale, self.periodicity
        )
