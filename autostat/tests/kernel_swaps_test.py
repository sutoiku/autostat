from ..kernel_swaps import (
    dedupe_kernels,
    other_base_kernels,
    product_operand_swaps,
    product_subtree_swaps,
    sort_list_of_operand_lists,
    sort_specs_by_type,
    base_subtree_swaps,
    simplify_additive_kernel_spec,
)
from ..kernel_specs import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
)

from ..run_settings import base_kernel_prototypes

# base_kernel_prototypes = [c() for c in base_kernel_classes]


def test_other_base_kernels():
    assert other_base_kernels(RBF(), base_kernel_prototypes) == [
        LIN(),
        PER(),
    ]


class TestSimplifyAdditiveKernel:
    def test_simple_case(self):
        k = ADD([PROD([ADD([PROD([LIN()]), PROD([RBF()])])])])
        k_simple = ADD([PROD([LIN()]), PROD([RBF()])])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_simple_case_with_scalars(self):
        k = ADD([PROD([ADD([PROD([LIN()], 5), PROD([RBF()], 7)])], 10)])
        k_simple = ADD([PROD([LIN()], 50), PROD([RBF()], 70)])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_simple_case_with_scalars_and_params(self):
        k = ADD([PROD([ADD([PROD([LIN(345)], 5), PROD([RBF(876)], 7)])], 10)])
        k_simple = ADD([PROD([LIN(345)], 50), PROD([RBF(876)], 70)])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_non_simplifiable_case(self):
        k = ADD(
            [
                PROD(
                    [
                        ADD([PROD([LIN()]), PROD([RBF()])]),
                        ADD([PROD([LIN()]), PROD([RBF()])]),
                    ]
                )
            ]
        )
        # k_simple = ADD([PROD([LIN()]), PROD([RBF()])])
        assert simplify_additive_kernel_spec(k) == k

    def test_non_simplifiable_case_with_params(self):
        k = ADD(
            [
                PROD(
                    [
                        ADD([PROD([LIN(12)]), PROD([RBF(34)])]),
                        ADD([PROD([LIN(56)]), PROD([RBF(78)])]),
                    ],
                    123,
                )
            ]
        )
        # k_simple = ADD([PROD([LIN()]), PROD([RBF()])])
        assert simplify_additive_kernel_spec(k) == k

    def test_multilayer_case(self):
        k = ADD([PROD([ADD([PROD([LIN()]), PROD([RBF()])])]), PROD([RQ()])])
        k_simple = ADD([PROD([LIN()]), PROD([RBF()]), PROD([RQ()])])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_multilayer_case_with_scalars(self):
        k = ADD(
            [PROD([ADD([PROD([LIN()], 5), PROD([RBF()], 7)])], 10), PROD([RQ()], 9)]
        )
        k_simple = ADD([PROD([LIN()], 50), PROD([RBF()], 70), PROD([RQ()], 9)])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_multilayer_case_with_scalars_and_params(self):
        k = ADD(
            [
                PROD([ADD([PROD([LIN(345)], 5), PROD([RBF(876)], 7)])], 10),
                PROD([RQ(75, 99)], 9),
            ]
        )
        k_simple = ADD(
            [PROD([LIN(345)], 50), PROD([RBF(876)], 70), PROD([RQ(75, 99)], 9)]
        )
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_multilayer_case_2(self):
        # k = ADD([PROD([ADD(PROD([LIN()]), PROD([PER()]))]), PROD(PER()), PROD(RBF())])
        k = ADD(
            [PROD([ADD([PROD([LIN()]), PROD([PER()])])]), PROD([PER()]), PROD([RBF()])]
        )
        k_simple = ADD([PROD([LIN()]), PROD([PER()]), PROD([PER()]), PROD([RBF()])])
        assert simplify_additive_kernel_spec(k) == k_simple

    def test_multilayer_case_2_scalars_and_param(self):
        # k = ADD([PROD([ADD(PROD([LIN()]), PROD([PER()]))]), PROD(PER()), PROD(RBF())])
        k = ADD(
            [
                PROD([ADD([PROD([LIN(2)], 3), PROD([PER(4)], 5)])], 11),
                PROD([PER(6)], 7),
                PROD([RBF(8)], 9),
            ]
        )
        k_simple = ADD(
            [
                PROD([LIN(2)], 33),
                PROD([PER(4)], 55),
                PROD([PER(6)], 7),
                PROD([RBF(8)], 9),
            ]
        )
        assert simplify_additive_kernel_spec(k) == k_simple


class TestBaseSubtreeSwaps:
    def test_simplest_case(self):
        assert base_subtree_swaps(RBF(), base_kernel_prototypes) == sort_specs_by_type(
            [
                LIN(),
                PER(),
                ADD([PROD([RBF()]), PROD([RBF()])]),
                ADD([PROD([RBF()]), PROD([LIN()])]),
                ADD([PROD([RBF()]), PROD([PER()])]),
                PROD([RBF(), RBF()]),
                PROD([RBF(), LIN()]),
                PROD([RBF(), PER()]),
            ]
        )

    def test_with_params(self):
        assert sort_specs_by_type(
            base_subtree_swaps(RBF(2.2), base_kernel_prototypes)
        ) == sort_specs_by_type(
            [
                LIN(),
                PER(),
                ADD([PROD([RBF(2.2)]), PROD([RBF()])]),
                ADD([PROD([RBF(2.2)]), PROD([LIN()])]),
                ADD([PROD([RBF(2.2)]), PROD([PER()])]),
                PROD([RBF(2.2), RBF()]),
                PROD([RBF(2.2), LIN()]),
                PROD([RBF(2.2), PER()]),
            ]
        )


class TestProductOperandSwaps:
    def test_simplest_case(self):
        swapped_operands = sort_list_of_operand_lists(
            product_operand_swaps([RBF(), LIN(), PER()], 0, base_kernel_prototypes)
        )
        target_operands = sort_list_of_operand_lists(
            [
                # other base kernels swapped in for index kernels
                [LIN(), LIN(), PER()],
                [PER(), LIN(), PER()],
                # all base kernels included with index kernel in sum
                [ADD([PROD([RBF()]), PROD([RBF()])]), LIN(), PER()],
                [ADD([PROD([RBF()]), PROD([LIN()])]), LIN(), PER()],
                [ADD([PROD([RBF()]), PROD([PER()])]), LIN(), PER()],
            ]
        )
        assert swapped_operands == target_operands


class TestProductSubtreeSwaps:
    def test_simplest_case(self):
        swapped_kernels = product_subtree_swaps(PROD([RBF()]), base_kernel_prototypes)
        target_kernels = sort_specs_by_type(
            [
                # existing product with all base kernels appended
                PROD([RBF(), RBF()]),
                PROD([RBF(), LIN()]),
                PROD([RBF(), PER()]),
                # existing product with OTHER base kernels swapped in
                PROD([LIN()]),
                PROD([PER()]),
                PROD([ADD([PROD([RBF()]), PROD([RBF()])])]),
                PROD([ADD([PROD([RBF()]), PROD([LIN()])])]),
                PROD([ADD([PROD([RBF()]), PROD([PER()])])]),
            ]
        )

        assert len(swapped_kernels) == len(target_kernels)
        for i in range(len(swapped_kernels)):
            assert swapped_kernels[i] == target_kernels[i]


class TestDedupeKernels:
    def test_simplest_case(self):
        with_dupes = sort_specs_by_type(
            [RBF(), PER(), RBF(), PER(), LIN(), RBF(), RQ()]
        )
        without_dupes = sort_specs_by_type([RBF(), PER(), LIN(), RQ()])
        assert dedupe_kernels(with_dupes) == without_dupes

    def test_simplest_case_with_param_precedence(self):
        with_dupes = sort_specs_by_type(
            [RBF(), PER(3, 5), RBF(), PER(), LIN(), RBF(4), RQ()]
        )
        without_dupes = sort_specs_by_type([RBF(4), PER(3, 5), LIN(), RQ()])
        assert dedupe_kernels(with_dupes) == without_dupes

    def test_medium_case_with_param_precedence(self):
        with_dupes = sort_specs_by_type(
            [
                RBF(),
                PROD([RBF(4), RQ(5, 6)]),
                RBF(),
                PROD([RBF(7), RQ()], 9),
                RBF(),
                PROD([RBF(10), RQ(11, 12)], 13),
                RBF(),
                PROD([ADD([PROD([RBF(3)]), PROD([RBF()])])]),
                RBF(),
                PROD([ADD([PROD([RBF()]), PROD([RBF(5)], 5)])]),
                RBF(),
                PROD([ADD([PROD([RBF(40)]), PROD([RBF()], 30)])], 20),
            ]
        )
        without_dupes = sort_specs_by_type(
            [
                RBF(),
                PROD([RBF(10), RQ(11, 12)], 13),
                PROD([ADD([PROD([RBF(40)]), PROD([RBF()], 30)])], 20),
            ]
        )
        assert dedupe_kernels(with_dupes) == without_dupes
