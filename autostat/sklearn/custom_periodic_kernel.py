from sklearn.gaussian_process.kernels import (
    Kernel,
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Hyperparameter,
)

from scipy.special import iv, ive

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
        self.length_scale: float = length_scale
        self.periodicity: float = periodicity
        self.length_scale_bounds: tuple[float, float] = length_scale_bounds
        self.periodicity_bounds: tuple[float, float] = periodicity_bounds

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

        e_bess_0 = ive(0, 1 / l ** 2)

        period_dist = (dists * np.pi) / p

        S = -2 * np.sin(period_dist) ** 2
        exp_S_over_lsqr = np.exp(S / l ** 2)

        K = (e_bess_0 - exp_S_over_lsqr) / (e_bess_0 - 1)

        if eval_gradient:
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                # NOTE: we're deviating from the matlab source:
                #  https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/gpml/cov/covPeriodicNoDC.m
                # because comparison with finite differences suggested that
                # having a specialized version for (1/l**2) < 3.75 performed
                # worse than using this single implementation in all cases

                e_bess_1 = ive(1, 1 / l ** 2)

                numerator = 2 * (
                    -e_bess_0
                    + e_bess_1
                    + exp_S_over_lsqr * (1 - e_bess_1 + (e_bess_0 - 1) * (1 + S))
                )

                denominator = (e_bess_0 - 1) ** 2 * l ** 3

                length_scale_gradient = numerator / denominator
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]

            else:  # length_scale is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))

            # gradient with respect to p
            if not self.hyperparameter_periodicity.fixed:

                numerator = (
                    2 * dists * exp_S_over_lsqr * np.pi * np.sin(2 * period_dist)
                )

                denominator = (e_bess_0 - 1) * (l * p) ** 2

                periodicity_gradient = -numerator / denominator
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
