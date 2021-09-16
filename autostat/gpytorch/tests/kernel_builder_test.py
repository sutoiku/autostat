from gpytorch.kernels import (
    Kernel,
    PeriodicKernel,
    LinearKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    ProductKernel,
    AdditiveKernel,
)

from pytest import approx

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

from ..custom_periodic_kernel import PeriodicKernelNoConstant

from ...run_settings import starting_kernel_specs, default_base_kernel_classes

from ..kernel_builder import build_kernel


class TestBuildKernel:
    def test_starting_kernel_specs(self):
        [build_kernel(k) for k in starting_kernel_specs(default_base_kernel_classes)]


class TestBuildKernelWithConstraints:
    def test_build_periodic_default_constraints(self):
        k = ty.cast(PeriodicKernel, build_kernel(PER()))

        assert approx(k.raw_lengthscale_constraint.lower_bound.item, CB().lower)  # type: ignore

        assert approx(k.raw_lengthscale_constraint.upper_bound.item, CB().upper)  # type: ignore
        assert approx(
            k.raw_period_length_constraint.lower_bound.item, CB().lower  # type: ignore
        )
        assert approx(
            k.raw_period_length_constraint.upper_bound.item, CB().upper  # type: ignore
        )

    def test_build_periodic_no_const_default_constraints(self):
        k = ty.cast(PeriodicKernelNoConstant, build_kernel(PERnc()))

        assert approx(k.raw_lengthscale_constraint.lower_bound.item, CB().lower)  # type: ignore

        assert approx(k.raw_lengthscale_constraint.upper_bound.item, CB().upper)  # type: ignore
        assert approx(
            k.raw_period_length_constraint.lower_bound.item, CB().lower  # type: ignore
        )
        assert approx(
            k.raw_period_length_constraint.upper_bound.item, CB().upper  # type: ignore
        )

    # def test_build_periodic_default_constraints_PERnc(self):
    #     k = ty.cast(PeriodicKernel, build_kernel(PERnc()))
    #     assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == CB()
    #     assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == CB()

    def test_build_periodic_constrained(self):

        k = ty.cast(
            PeriodicKernel,
            build_kernel(
                PER(period_bounds=CB(0.5, 1.5), length_scale_bounds=CB(1, 20))
            ),
        )

        assert approx(k.raw_lengthscale_constraint.lower_bound.item, 1)  # type: ignore
        assert approx(k.raw_lengthscale_constraint.upper_bound.item, 20)  # type: ignore
        assert approx(k.raw_period_length_constraint.lower_bound.item, 0.5)  # type: ignore
        assert approx(k.raw_period_length_constraint.upper_bound.item, 1.5)  # type: ignore

    def test_build_periodic_no_const_constrained(self):

        k = ty.cast(
            PeriodicKernelNoConstant,
            build_kernel(
                PERnc(period_bounds=CB(0.5, 1.5), length_scale_bounds=CB(1, 20))
            ),
        )

        assert approx(k.raw_lengthscale_constraint.lower_bound.item, 1)  # type: ignore
        assert approx(k.raw_lengthscale_constraint.upper_bound.item, 20)  # type: ignore
        assert approx(k.raw_period_length_constraint.lower_bound.item, 0.5)  # type: ignore
        assert approx(k.raw_period_length_constraint.upper_bound.item, 1.5)  # type: ignore
