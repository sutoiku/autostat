from ..kernel_swaps import (
    other_base_kernels,
    product_operand_swaps,
    product_subtree_swaps,
    sort_list_of_operand_lists,
    sort_specs_by_type,
    base_subtree_swaps,
)
from ..kernel_tree_types import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
)


def test_other_base_kernels():
    assert other_base_kernels(RBF()) == [
        RQ(),
        LIN(),
        PER(),
    ]


class TestBaseSubtreeSwaps:
    def test_simplest_case(self):
        assert base_subtree_swaps(RBF()) == sort_specs_by_type(
            [
                RQ(),
                LIN(),
                PER(),
                ADD([PROD([RBF()]), PROD([RBF()])]),
                ADD([PROD([RBF()]), PROD([RQ()])]),
                ADD([PROD([RBF()]), PROD([LIN()])]),
                ADD([PROD([RBF()]), PROD([PER()])]),
                PROD([RBF(), RBF()]),
                PROD([RBF(), RQ()]),
                PROD([RBF(), LIN()]),
                PROD([RBF(), PER()]),
            ]
        )


class TestProductOperandSwaps:
    def test_simplest_case(self):
        swapped_operands = sort_list_of_operand_lists(
            product_operand_swaps([RBF(), LIN(), RQ()], 0)
        )
        target_operands = sort_list_of_operand_lists(
            [
                # other base kernels swapped in for index kernels
                [RQ(), LIN(), RQ()],
                [LIN(), LIN(), RQ()],
                [PER(), LIN(), RQ()],
                # all base kernels included with index kernel in sum
                [ADD([PROD([RBF()]), PROD([RBF()])]), LIN(), RQ()],
                [ADD([PROD([RBF()]), PROD([LIN()])]), LIN(), RQ()],
                [ADD([PROD([RBF()]), PROD([PER()])]), LIN(), RQ()],
                [ADD([PROD([RBF()]), PROD([RQ()])]), LIN(), RQ()],
            ]
        )
        assert swapped_operands == target_operands


class TestProductSubtreeSwaps:
    def test_simplest_case(self):
        swapped_kernels = product_subtree_swaps(PROD([RBF()]))
        target_kernels = sort_specs_by_type(
            [
                # existing product with all base kernels appended
                PROD([RBF(), RBF()]),
                PROD([RBF(), RQ()]),
                PROD([RBF(), LIN()]),
                PROD([RBF(), PER()]),
                # existing product with OTHER base kernels swapped in
                PROD([RQ()]),
                PROD([LIN()]),
                PROD([PER()]),
                PROD([ADD([PROD([RBF()]), PROD([RBF()])])]),
                PROD([ADD([PROD([RBF()]), PROD([RQ()])])]),
                PROD([ADD([PROD([RBF()]), PROD([LIN()])])]),
                PROD([ADD([PROD([RBF()]), PROD([PER()])])]),
            ]
        )

        assert len(swapped_kernels) == len(target_kernels)
        for i in range(len(swapped_kernels)):
            assert swapped_kernels[i] == target_kernels[i]
