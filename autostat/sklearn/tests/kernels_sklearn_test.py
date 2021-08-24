from sklearn.gaussian_process.kernels import (
    ExpSineSquared,
    Kernel,
    RBF,
    WhiteKernel,
    RationalQuadratic,
    ConstantKernel,
    DotProduct,
    Product,
    Sum,
)


from ...kernel_specs import (
    RBFKernelSpec as RBF_spec,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    PeriodicNoConstKernelSpec as PERnc,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    TopLevelKernelSpec as TOP,
)

import typing as ty

from ...run_settings import starting_kernel_specs, default_base_kernel_classes
from ...constraints import default_constraints

from ..to_kernel_spec import to_kernel_spec, to_kernel_spec_inner
from ..kernel_builder import build_kernel
from ..custom_periodic_kernel import PeriodicKernelNoConstant


class TestToKernelSpec:
    def test_one_term_product_kernel(self):
        k_sklearn = 22 * PeriodicKernelNoConstant()
        k_autostat = TOP([PROD([PERnc()], 22)])

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_one_term_product_kernel_2(self):
        k_sklearn = 22 * ExpSineSquared()
        k_autostat = TOP([PROD([PER()], 22)])

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_simple_composite_kernel(self):
        k_sklearn = (22 * RationalQuadratic()) + (44 * PeriodicKernelNoConstant())
        k_autostat = TOP([PROD([RQ()], 22), PROD([PERnc()], 44)])
        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_more_complex_composite_kernel(self):
        k_sklearn = 64 * PeriodicKernelNoConstant() * (25 * RBF() + 81 * DotProduct())

        k_autostat = TOP(
            [
                PROD([PERnc(), ADD([PROD([RBF_spec()], 25), PROD([LIN()], 81)])], 64),
            ]
        )

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_very_complex_composite_kernel(self):
        k_sklearn = (16 * RationalQuadratic() * DotProduct()) + (
            64 * PeriodicKernelNoConstant() * (25 * RBF() + 81 * DotProduct())
        )
        k_autostat = TOP(
            [
                PROD([RQ(), LIN()], 16),
                PROD([PERnc(), ADD([PROD([RBF_spec()], 25), PROD([LIN()], 81)])], 64),
            ]
        )

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_very_complex_parameterized_composite_kernel(self):
        k_sklearn = (
            16 * RationalQuadratic(alpha=3, length_scale=5) * DotProduct(sigma_0=7)
        ) + (
            64
            * PeriodicKernelNoConstant(length_scale=9, periodicity=3)
            * (25 * RBF(length_scale=13) + 81 * DotProduct(sigma_0=77))
        )
        k_autostat = TOP(
            [
                PROD([RQ(alpha=3, length_scale=5), LIN(variance=7)], 16),
                PROD(
                    [
                        PERnc(length_scale=9, period=3),
                        ADD(
                            [
                                PROD([RBF_spec(length_scale=13)], 25),
                                PROD([LIN(variance=77)], 81),
                            ]
                        ),
                    ],
                    64,
                ),
            ]
        )

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))


class TestSklearnToSpecAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in [RBF, RationalQuadratic, PeriodicKernelNoConstant, DotProduct]:
            assert str(k()) == str(
                build_kernel(to_kernel_spec_inner(k()), default_constraints())
            )

    def test_parameterized_base_kernels(self):
        for k in [
            RBF(length_scale=0.5),
            RationalQuadratic(length_scale=0.2, alpha=3.3),
            DotProduct(sigma_0=1.7),
            PeriodicKernelNoConstant(length_scale=8.9, periodicity=0.13),
        ]:
            assert str(k) == str(
                build_kernel(to_kernel_spec_inner(k), default_constraints())
            )

    def test_simple_sum_kernel(self):
        k = (1 * RBF()) + (4 * PeriodicKernelNoConstant()) + WhiteKernel(0.454)
        spec = to_kernel_spec(k)
        # assert spec == k

        assert k == build_kernel(spec, default_constraints())


class TestSpecToSklearnAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in default_base_kernel_classes:
            assert str(k()) == str(
                to_kernel_spec_inner(build_kernel(k(), default_constraints()))
            )

    def test_parameterized_base_kernels(self):
        for k in [
            RBF_spec(length_scale=0.5),
            RQ(length_scale=0.2, alpha=3.3),
            LIN(variance=1.7),
            PER(length_scale=8.9, period=0.13),
        ]:
            assert str(k) == str(
                to_kernel_spec_inner(build_kernel(k, default_constraints()))
            )


class TestSpecToSklearnAndBackRoundTrips_CompleteSpecs:
    def test_starting_kernel_specs(self):
        # [build_kernel(k) for k in starting_kernel_specs()]

        for k in starting_kernel_specs(default_base_kernel_classes):
            built_kernel = ty.cast(Sum, build_kernel(k, default_constraints()))
            unbuilt_kernel = to_kernel_spec(built_kernel)
            assert str(k) == str(unbuilt_kernel)
