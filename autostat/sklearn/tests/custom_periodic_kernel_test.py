import numpy as np
import pytest

from ..custom_periodic_kernel import PeriodicKernelNoConstant


class TestGradients:
    @pytest.mark.parametrize("param_init_val", [0.001, 0.04, 0.3, 0.8, 1.5, 10.5, 15])
    def test_grad_length_scale(self, param_init_val):
        param = "length_scale"
        mat_slice = 0

        N = 50
        max_entry_diff = 0.01
        max_mse = max_entry_diff ** 2
        X = np.linspace([-1], [1], N)

        delta = 1e-7

        param_init = param_init_val

        kwargs1 = {param: param_init}
        k1 = PeriodicKernelNoConstant(**kwargs1)
        kMat1, grads = k1(X, eval_gradient=True)

        kwargs2 = {param: param_init + delta}
        k2 = PeriodicKernelNoConstant(**kwargs2)
        kMat2 = k2(X, eval_gradient=False)

        grad = grads[:, :, mat_slice]
        finite_diff_grad = (kMat2 - kMat1) / delta

        assert np.mean((grad - finite_diff_grad) ** 2) < max_mse

    @pytest.mark.parametrize(
        "param_init_val", [0.012, 0.04, 0.115, 0.15, 0.3, 0.8, 1.5, 10.5, 15]
    )
    def test_grad_periodicity(self, param_init_val):
        param = "periodicity"
        mat_slice = 1

        N = 50
        max_entry_diff = 0.01
        max_mse = max_entry_diff ** 2
        X = np.linspace([-1], [1], N)

        # NOTE: we're at the limit of what finite differences can do...
        # going from 1e-11 to 1e-12 makes the test _fail_ on th first case
        delta = 1e-11

        param_init = param_init_val

        kwargs1 = {param: param_init}
        k1 = PeriodicKernelNoConstant(**kwargs1)
        kMat1, grads = k1(X, eval_gradient=True)

        kwargs2 = {param: param_init + delta}
        k2 = PeriodicKernelNoConstant(**kwargs2)
        kMat2 = k2(X, eval_gradient=False)

        grad = grads[:, :, mat_slice]
        finite_diff_grad = (kMat2 - kMat1) / delta

        assert np.mean((grad - finite_diff_grad) ** 2) < max_mse
