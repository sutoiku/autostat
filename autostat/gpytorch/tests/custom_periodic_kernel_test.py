import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from ..custom_periodic_kernel import PeriodicKernelNoConstant, i0e_torch
from ...tests.PERnc_testValuesAndGrads import PERnc_testValuesAndGrads
from scipy.special import i0e
from ...kernel_specs import PeriodicNoConstKernelSpec
from ..kernel_builder import build_kernel

def tensor(a:float)->torch.Tensor:
    return torch.Tensor()

class Test_i0e:
    def test_i0e_torch_call(self):
        input = torch.randn(20, 20, dtype=torch.float64, requires_grad=False) ** 2
        # assert np.all(
        #     np.abs(i0e_torch(input).detach().numpy() - input.detach().numpy()) < 1e-4
        # )
        out_torch = i0e_torch(input).detach().cpu().numpy()
        out_np = i0e(input.detach().cpu().numpy())

        assert np.all(np.abs(out_torch - out_np) < 1e-4)
        # assert out_torch == out_np

    def test_grad_check(self):
        input = (
            1e6
            + torch.randn(20, 20, dtype=torch.float64, requires_grad=True, device="cpu")
            ** 2,
        )
        test = gradcheck(i0e_torch, input, eps=1e-9, atol=1e-4)
        assert test

    def test_raise_on_negative(self):
        input = -torch.randn(20, 20, dtype=torch.float64, requires_grad=True) ** 2
        with pytest.raises(ValueError):
            i0e_torch(input)

    
    @pytest.mark.parametrize("PERnc_case", PERnc_testValuesAndGrads[:10])
    def test_grad_length_scale(self, PERnc_case):
        (dist,l,p),value,dKdl,dKdp = PERnc_case
        spec = PeriodicNoConstKernelSpec(length_scale=l,period=p)
        kernel = build_kernel(spec)
        



    # @pytest.mark.parametrize("param_init_val", [0.001, 0.04, 0.3, 0.8, 1.5, 10.5, 15])
    # def test_grad_length_scale(self, param_init_val):
    #     param = "length_scale"
    #     mat_slice = 0

    #     N = 50
    #     max_entry_diff = 0.01
    #     max_mse = max_entry_diff ** 2
    #     X = np.linspace([-1], [1], N)

    #     delta = 1e-7

    #     param_init = param_init_val

    #     kwargs1 = {param: param_init}
    #     k1 = PeriodicKernelNoConstant(**kwargs1)
    #     kMat1, grads = k1(X, eval_gradient=True)

    #     kwargs2 = {param: param_init + delta}
    #     k2 = PeriodicKernelNoConstant(**kwargs2)
    #     kMat2 = k2(X, eval_gradient=False)

    #     grad = grads[:, :, mat_slice]
    #     finite_diff_grad = (kMat2 - kMat1) / delta

    #     assert np.mean((grad - finite_diff_grad) ** 2) < max_mse

    # @pytest.mark.parametrize(
    #     "param_init_val", [0.012, 0.04, 0.115, 0.15, 0.3, 0.8, 1.5, 10.5, 15]
    # )
    # def test_grad_periodicity(self, param_init_val):
    #     param = "periodicity"
    #     mat_slice = 1

    #     N = 50
    #     max_entry_diff = 0.01
    #     max_mse = max_entry_diff ** 2
    #     X = np.linspace([-1], [1], N)

    #     # NOTE: we're at the limit of what finite differences can do...
    #     # going from 1e-11 to 1e-12 makes the test _fail_ on th first case
    #     delta = 1e-11

    #     param_init = param_init_val

    #     kwargs1 = {param: param_init}
    #     k1 = PeriodicKernelNoConstant(**kwargs1)
    #     kMat1, grads = k1(X, eval_gradient=True)

    #     kwargs2 = {param: param_init + delta}
    #     k2 = PeriodicKernelNoConstant(**kwargs2)
    #     kMat2 = k2(X, eval_gradient=False)

    #     grad = grads[:, :, mat_slice]
    #     finite_diff_grad = (kMat2 - kMat1) / delta

    #     assert np.mean((grad - finite_diff_grad) ** 2) < max_mse
