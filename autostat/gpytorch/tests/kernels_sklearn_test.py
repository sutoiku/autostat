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

from gpytorch.constraints import Interval


from ...kernel_specs import (
    RBFKernelSpec as RBF_spec,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    PeriodicNoConstKernelSpec as PERnc,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    TopLevelKernelSpec as TOP,
    ConstraintBounds as CB,
)

import typing as ty

from ...run_settings import starting_kernel_specs, default_base_kernel_classes

from ..to_kernel_spec import to_kernel_spec, to_kernel_spec_inner
from ..kernel_builder import build_kernel, bounds_to_interval
from ..custom_periodic_kernel import PeriodicKernelNoConstant

param_bounds_interval = lambda: bounds_to_interval(CB())


def gpy_init_PER(lengthscale=1, periodicity=1):
    k = PeriodicKernel(
        period_length_constraint=param_bounds_interval(),
        lengthscale_constraint=param_bounds_interval(),
    )
    k.lengthscale = lengthscale
    k.period_length = periodicity
    return k


def gpy_init_RBF(lengthscale=1):
    k = RBFKernel(lengthscale_constraint=param_bounds_interval())
    k.lengthscale = lengthscale
    return k


def gpy_init_LIN(variance=1):
    k = LinearKernel(variance_constraint=param_bounds_interval())
    k.variance = variance
    return k


def gpy_init_RQ(lengthscale=1, alpha=1):
    k = RQKernel(
        lengthscale_constraint=param_bounds_interval(),
        alpha_constraint=param_bounds_interval(),
    )
    k.lengthscale = lengthscale
    k.alpha = alpha
    return k


def gpy_wrap_scale_prod(kernels, scalar=1):
    k = ScaleKernel(ProductKernel(*kernels))
    k.outputscale = scalar
    return k


def gpy_wrap_add(kernels, scalar=1):
    return AdditiveKernel(*kernels)


class TestToKernelSpec:
    def test_one_term_product_kernel_PER(self):
        noise = 0.4312
        scale = 5343
        k_gpy = gpy_wrap_add([gpy_wrap_scale_prod([gpy_init_PER()], scale)])
        k_autostat = TOP([PROD([PER()], scale)], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_one_term_product_kernel_RBF(self):
        noise = 0.4312
        scale = 5343
        k_gpy = gpy_wrap_add([gpy_wrap_scale_prod([gpy_init_RBF()], scale)])
        k_autostat = TOP([PROD([RBF_spec()], scale)], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_one_term_product_kernel_LIN(self):
        noise = 0.4312
        scale = 5343
        k_gpy = gpy_wrap_add([gpy_wrap_scale_prod([gpy_init_LIN()], scale)])
        k_autostat = TOP([PROD([LIN()], scale)], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_one_term_product_kernel_RQ(self):
        noise = 0.4312
        scale = 5343
        k_gpy = gpy_wrap_add([gpy_wrap_scale_prod([gpy_init_RQ()], scale)])
        k_autostat = TOP([PROD([RQ()], scale)], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_simple_composite_kernel(self):
        noise = 0.56424
        k_gpy = gpy_wrap_add(
            [
                gpy_wrap_scale_prod([gpy_init_RQ()], 22),
                gpy_wrap_scale_prod([gpy_init_PER()], 787),
            ]
        )
        k_autostat = TOP([PROD([RQ()], 22), PROD([PER()], 787)], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_more_complex_composite_kernel(self):
        noise = 675437.345234
        k_gpy = gpy_wrap_add(
            [
                gpy_wrap_scale_prod(
                    [
                        gpy_init_PER(),
                        gpy_wrap_add(
                            [
                                gpy_wrap_scale_prod([gpy_init_LIN()], 25),
                                gpy_wrap_scale_prod([gpy_init_RBF()], 16),
                            ]
                        ),
                    ],
                    64,
                ),
            ]
        )

        k_autostat = TOP([64 * PER() * (25 * LIN() + 16 * RBF_spec())], noise)

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

    def test_very_complex_parameterized_composite_kernel(self):
        noise = 675437.345234
        k_gpy = gpy_wrap_add(
            [
                gpy_wrap_scale_prod(
                    [gpy_init_RQ(alpha=3, lengthscale=5), gpy_init_LIN(variance=7)],
                    16,
                ),
                gpy_wrap_scale_prod(
                    [
                        gpy_init_PER(lengthscale=9, periodicity=3),
                        gpy_wrap_add(
                            [
                                gpy_wrap_scale_prod([gpy_init_LIN(variance=77)], 81),
                                gpy_wrap_scale_prod([gpy_init_RBF(lengthscale=13)], 25),
                            ]
                        ),
                    ],
                    64,
                ),
            ]
        )

        k_autostat = TOP(
            [
                PROD([RQ(alpha=3, length_scale=5), LIN(variance=7)], 16),
                PROD(
                    [
                        PER(length_scale=9, period=3),
                        ADD(
                            [
                                PROD([RBF_spec(length_scale=13)], 25),
                                PROD([LIN(variance=77)], 81),
                            ]
                        ),
                    ],
                    64,
                ),
            ],
            noise=noise,
        )

        assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))


class TestSklearnToSpecAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in [gpy_init_RBF, gpy_init_RQ, gpy_init_LIN, gpy_init_PER]:
            assert str(k()) == str(build_kernel(to_kernel_spec_inner(k())))

    def test_parameterized_base_kernels(self):
        for k in [
            gpy_init_RBF(lengthscale=0.5),
            gpy_init_RQ(lengthscale=0.2, alpha=3.3),
            gpy_init_LIN(variance=1.7),
            gpy_init_PER(lengthscale=8.9, periodicity=0.13),
        ]:
            assert str(k) == str(build_kernel(to_kernel_spec_inner(k)))

    def test_simple_composite_kernel_round_trip(self):
        noise = 0.56424
        k = gpy_wrap_add(
            [
                gpy_wrap_scale_prod([gpy_init_RQ()], 22),
                gpy_wrap_scale_prod([gpy_init_PER()], 787),
            ]
        )
        # k_autostat = TOP([PROD([RQ()], 22), PROD([PER()], 787)], noise)

        # assert str(k_autostat) == str(to_kernel_spec(k_gpy, noise))

        spec = to_kernel_spec(k, noise)

        assert str(k) == str(build_kernel(spec))

    def test_very_complex_parameterized_composite_kernel_round_trip(self):
        # k = (1 * RBF()) + (4 * PeriodicKernelNoConstant()) + WhiteKernel(0.454)
        noise = 675437.345234
        k = gpy_wrap_add(
            [
                gpy_wrap_scale_prod(
                    [gpy_init_LIN(variance=7), gpy_init_RQ(alpha=3, lengthscale=5)],
                    16,
                ),
                gpy_wrap_scale_prod(
                    [
                        gpy_wrap_add(
                            [
                                gpy_wrap_scale_prod([gpy_init_RBF(lengthscale=13)], 25),
                                gpy_wrap_scale_prod([gpy_init_LIN(variance=77)], 81),
                            ]
                        ),
                        gpy_init_PER(lengthscale=9, periodicity=3),
                    ],
                    64,
                ),
            ]
        )
        spec = to_kernel_spec(k, noise)
        # assert spec == k

        assert str(k) == str(build_kernel(spec))


class TestSpecToSklearnAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in default_base_kernel_classes:
            assert str(k()) == str(
                to_kernel_spec_inner(build_kernel(k()))  # type: ignore
            )

    def test_parameterized_base_kernels(self):
        for k in [
            RBF_spec(length_scale=0.5),
            RQ(length_scale=0.2, alpha=3.3),
            LIN(variance=1.7),
            PER(length_scale=8.9, period=0.13),
        ]:
            assert str(k) == str(to_kernel_spec_inner(build_kernel(k)))


class TestSpecToSklearnAndBackRoundTrips_CompleteSpecs:
    def test_starting_kernel_specs(self):
        noise = TOP().noise

        for k in starting_kernel_specs(default_base_kernel_classes):
            built_kernel = build_kernel(k)
            unbuilt_kernel = to_kernel_spec(built_kernel, noise=noise)
            assert str(k) == str(unbuilt_kernel)
