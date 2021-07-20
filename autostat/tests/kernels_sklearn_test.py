from sklearn.gaussian_process.kernels import (
    Kernel,
    RBF,
    WhiteKernel,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel,
    DotProduct,
    Product,
    Sum,
)


from ..kernel_tree_types import (
    KernelSpec,
    Dataset,
    KernelSpec,
    ArithmeticKernelSpec,
    BaseKernelSpec,
    RBFKernelSpec as RBF_spec,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
)

from ..kernel_swaps import base_kernel_classes
from ..kernel_search import starting_kernel_specs

from ..kernel_trees_sklearn import (
    build_kernel,
    # to_kernel_spec,
    to_kernel_spec,
    to_kernel_spec_inner,
)


class TestBuildKernel:
    def test_starting_kernel_specs(self):
        [build_kernel(k) for k in starting_kernel_specs()]


class TestToKernelSpec:
    def test_one_term_product_kernel(self):
        k_sklearn = 22 * ExpSineSquared()
        k_autostat = ADD([PROD([PER()], 22)])

        # print(str(k_autostat))
        # print(str(to_kernel_spec(k_sklearn)))
        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_simple_composite_kernel(self):
        k_sklearn = (22 * RationalQuadratic()) + (44 * ExpSineSquared())
        k_autostat = ADD([PROD([RQ()], 22), PROD([PER()], 44)])
        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_more_complex_composite_kernel(self):
        k_sklearn = 64 * ExpSineSquared() * (25 * RBF() + 81 * DotProduct())

        k_autostat = ADD(
            [
                PROD([PER(), ADD([PROD([RBF_spec()], 25), PROD([LIN()], 81)])], 64),
            ]
        )

        # print(k_sklearn)
        # print(str(k_autostat))
        # print(str(to_kernel_spec(k_sklearn)))

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_very_complex_composite_kernel(self):
        k_sklearn = (16 * RationalQuadratic() * DotProduct()) + (
            64 * ExpSineSquared() * (25 * RBF() + 81 * DotProduct())
        )
        k_autostat = ADD(
            [
                PROD([RQ(), LIN()], 16),
                PROD([PER(), ADD([PROD([RBF_spec()], 25), PROD([LIN()], 81)])], 64),
            ]
        )

        # print(k_sklearn)
        # print(str(k_autostat))
        # print(str(to_kernel_spec(k_sklearn)))

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))

    def test_very_complex_parameterized_composite_kernel(self):
        k_sklearn = (
            16 * RationalQuadratic(alpha=3, length_scale=5) * DotProduct(sigma_0=7)
        ) + (
            64
            * ExpSineSquared(length_scale=9, periodicity=3)
            * (25 * RBF(length_scale=13) + 81 * DotProduct(sigma_0=77))
        )
        k_autostat = ADD(
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
            ]
        )

        # print(k_sklearn)
        # print(str(k_autostat))
        # print(str(to_kernel_spec(k_sklearn)))

        assert str(k_autostat) == str(to_kernel_spec(k_sklearn))


class TestSklearnToSpecAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in [RBF, RationalQuadratic, ExpSineSquared, DotProduct]:
            assert str(k()) == str(build_kernel(to_kernel_spec_inner(k())))

    def test_parameterized_base_kernels(self):
        for k in [
            RBF(length_scale=0.5),
            RationalQuadratic(length_scale=0.2, alpha=3.3),
            DotProduct(sigma_0=1.7),
            ExpSineSquared(length_scale=8.9, periodicity=0.13),
        ]:
            assert str(k) == str(build_kernel(to_kernel_spec_inner(k)))

    def test_simple_sum_kernel(self):
        k = (1 * RBF()) + (4 * ExpSineSquared())

        assert k == build_kernel(to_kernel_spec(k))

    # def test_simple_product(self):
    #     k = 4 * RBF() * ExpSineSquared()

    #     print(k)
    #     print(build_kernel(to_kernel_spec(k)))
    #     assert k == build_kernel(to_kernel_spec(k))

    # def test_simple_composite_kernel(self):
    #     k = 9 * RBF() * ExpSineSquared() + 16 * DotProduct()
    #     assert k == build_kernel(to_kernel_spec(k))


class TestSpecToSklearnAndBackRoundTrips_InnerSpecs:
    def test_base_kernels(self):
        for k in base_kernel_classes:
            assert str(k()) == str(to_kernel_spec_inner(build_kernel(k())))

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
        # [build_kernel(k) for k in starting_kernel_specs()]

        for k in starting_kernel_specs():
            assert str(k) == str(to_kernel_spec(build_kernel(k)))


# def test_other_base_kernels():
#     spec = AdditiveKernelSpec(operands=[RBFKernelSpec(),LinearKernelSpec()])
#     assert
