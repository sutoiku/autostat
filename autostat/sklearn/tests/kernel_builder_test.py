from sklearn.gaussian_process.kernels import (
    Kernel,
    RBF,
    WhiteKernel,
    RationalQuadratic,
    ConstantKernel,
    DotProduct,
    Product,
    Sum,
    ExpSineSquared,
)

import typing as ty

from ...kernel_specs import (
    RBFKernelSpec as RBF_spec,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    PeriodicNoConstKernelSpec as PERnc,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    ConstraintBounds as CB,
)

from ...run_settings import starting_kernel_specs, default_base_kernel_classes

from ..kernel_builder import build_kernel
from ..custom_periodic_kernel import PeriodicKernelNoConstant


class TestBuildKernel:
    def test_starting_kernel_specs(self):
        [build_kernel(k) for k in starting_kernel_specs(default_base_kernel_classes)]


class TestBuildKernelWithConstraints:
    def test_build_PER_default_constraints(self):
        k = ty.cast(ExpSineSquared, build_kernel(PER()))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == CB()
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == CB()

    def test_build_PERnc_default_constraints(self):
        k = ty.cast(PeriodicKernelNoConstant, build_kernel(PERnc()))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == CB()
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == CB()

    def test_build_PER_constrained(self):
        spec = PER().clone_update(
            {"length_scale_bounds": CB(10, 20), "period_bounds": CB(0.5, 1.5)}
        )

        k = ty.cast(ExpSineSquared, build_kernel(spec))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == (10, 20)
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == (0.5, 1.5)

    def test_build_PERnc_constrained(self):
        spec = PERnc().clone_update(
            {"length_scale_bounds": CB(10, 20), "period_bounds": CB(0.5, 1.5)}
        )

        k = ty.cast(PeriodicKernelNoConstant, build_kernel(spec))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == (10, 20)
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == (0.5, 1.5)

    def test_build_periodic_constrained_composite(self):

        per_spec = PER().clone_update(
            {"length_scale_bounds": CB(10, 20), "period_bounds": CB(0.5, 1.5)}
        )

        spec = ADD([PROD([LIN()]), PROD([per_spec])])

        k = build_kernel(spec)
        p_bounds = k.get_params()["k2__k2__periodicity_bounds"]
        l_bounds = k.get_params()["k2__k2__length_scale_bounds"]

        assert tuple(p_bounds) == (0.5, 1.5)
        assert tuple(l_bounds) == (10, 20)
