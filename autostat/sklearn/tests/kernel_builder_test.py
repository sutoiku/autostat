from sklearn.gaussian_process.kernels import (
    Kernel,
    RBF,
    WhiteKernel,
    RationalQuadratic,
    ConstantKernel,
    DotProduct,
    Product,
    Sum,
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
)

from ...run_settings import starting_kernel_specs, default_base_kernel_classes
from ...constraints import (
    KernelConstraints,
    ConstraintBounds as CB,
    PeriodicKernelConstraints,
    cb_default,
    default_constraints,
)

from ..kernel_builder import build_kernel
from ..custom_periodic_kernel import PeriodicKernelNoConstant


class TestBuildKernel:
    def test_starting_kernel_specs(self):
        [
            build_kernel(k, default_constraints())
            for k in starting_kernel_specs(default_base_kernel_classes)
        ]


class TestBuildKernelWithConstraints:
    def test_build_periodic_default_constraints(self):
        k = ty.cast(
            PeriodicKernelNoConstant, build_kernel(PER(), default_constraints())
        )
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == cb_default()
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == cb_default()

    def test_build_periodic_default_constraints_PERnc(self):
        k = ty.cast(
            PeriodicKernelNoConstant, build_kernel(PERnc(), default_constraints())
        )
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == cb_default()
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == cb_default()

    def test_build_periodic_constrained(self):
        constraints = KernelConstraints(
            PeriodicKernelConstraints(length_scale=CB(10, 20), period=CB(0.5, 1.5))
        )
        k = ty.cast(PeriodicKernelNoConstant, build_kernel(PER(), constraints))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == (10, 20)
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == (0.5, 1.5)

    def test_build_periodic_constrained_composite(self):
        constraints = KernelConstraints(
            PeriodicKernelConstraints(length_scale=CB(10, 20), period=CB(0.5, 1.5))
        )

        spec = ADD([PROD([LIN()]), PROD([PER()])])

        k = build_kernel(spec, constraints)
        p_bounds = k.get_params()["k2__k2__periodicity_bounds"]
        l_bounds = k.get_params()["k2__k2__length_scale_bounds"]

        assert tuple(p_bounds) == (0.5, 1.5)
        assert tuple(l_bounds) == (10, 20)
