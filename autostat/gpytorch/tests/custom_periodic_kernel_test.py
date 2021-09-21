import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from ..custom_periodic_kernel import PeriodicKernelNoConstant, i0e_torch
from ...tests.PERnc_testValuesAndGrads import PERnc_testValuesAndGrads
from scipy.special import i0e
from ...kernel_specs import PeriodicNoConstKernelSpec
from ..kernel_builder import build_kernel


def tensor(a: float) -> torch.Tensor:
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


class Test_PeriodicKernelNoConstant:
    @pytest.mark.parametrize("PERnc_case", PERnc_testValuesAndGrads[:])
    def test_values_close(self, PERnc_case):
        (value, dKdl, dKdp), dist, l, p = PERnc_case
        kernel = PeriodicKernelNoConstant()
        kernel.lengthscale = l
        kernel.period_length = p
        X = torch.tensor([[0], [dist]])
        Gram = kernel(X, X).evaluate()
        if l < 0.02 or p < 0.2:
            target = 2e-2
        else:
            target = 2e-4
        assert np.abs(Gram[0, 1].item() - value) < target

    @pytest.mark.parametrize("PERnc_case", PERnc_testValuesAndGrads[:])
    def test_grad_length_scale_close(self, PERnc_case):
        (value, dKdl, dKdp), dist, l, p = PERnc_case
        kernel = PeriodicKernelNoConstant()
        kernel.lengthscale = l
        kernel.period_length = p
        X = torch.tensor([[0], [dist]])
        Gram = kernel(X, X).evaluate()
        k = Gram[0, 1]
        k.backward()
        dkdl_custom = kernel.get_parameter("raw_lengthscale").grad.item()

        if l < 0.02 or p < 0.2:
            target = 2e-2
        else:
            target = 2e-4
        assert np.abs(dkdl_custom - dKdl) < target

    @pytest.mark.parametrize("PERnc_case", PERnc_testValuesAndGrads[:])
    def test_grad_periodicity_close(self, PERnc_case):
        (value, dKdl, dKdp), dist, l, p = PERnc_case
        kernel = PeriodicKernelNoConstant()
        kernel.lengthscale = l
        kernel.period_length = p
        X = torch.tensor([[0], [dist]])
        Gram = kernel(X, X).evaluate()
        k = Gram[0, 1]
        k.backward()
        dkdp_custom = kernel.get_parameter("raw_period_length").grad.item()

        if l < 0.02 or p < 0.2:
            target = 2e-2
        else:
            target = 2e-4

        if abs(dKdp) > 4:
            ratio = dkdp_custom / dKdp
            assert 1 - target < ratio and ratio < 1 + target
        else:
            assert np.abs(dkdp_custom - dKdp) < target
