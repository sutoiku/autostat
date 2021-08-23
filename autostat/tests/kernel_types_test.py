import pytest
from itertools import permutations

from ..kernel_specs import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    TopLevelKernelSpec,
)


class TestCompositeKernelInitOperandSorting:
    def test_additive(self):
        a = [
            PROD([RBF(13.353)]),
            PROD([PER()]),
            PROD([LIN(12)]),
            PROD([RBF(14)]),
            PROD([RBF(20)]),
        ]

        b = [
            PROD([LIN(12)]),
            PROD([PER()]),
            PROD([RBF(14)]),
            PROD([RBF(13.353)]),
            PROD([RBF(20)]),
        ]

        assert ADD(a) == ADD(b)
        assert TopLevelKernelSpec(a) == TopLevelKernelSpec(b)

    def test_prod(self):
        a = [
            RBF(13.353),
            PER(),
            LIN(12),
            RBF(14),
            RBF(20),
        ]

        b = [
            LIN(12),
            PER(),
            RBF(14),
            RBF(13.353),
            RBF(20),
        ]

        assert PROD(a) == PROD(b)

    def test_composite(self):
        a = [
            PROD([RBF(13.353), RQ(4), PER(12)]),
            PROD([LIN(12)]),
            PROD([RBF(14)]),
            PROD([RBF(20)]),
            PROD([PER(), ADD([PROD([LIN()]), PROD([RQ(343)])])]),
        ]

        b = [
            PROD([LIN(12)]),
            PROD([ADD([PROD([LIN()]), PROD([RQ(343)])]), PER()]),
            PROD([RBF(14)]),
            PROD([RQ(4), PER(12), RBF(13.353)]),
            PROD([RBF(20)]),
        ]

        assert ADD(a) == ADD(b)
        assert TopLevelKernelSpec(a) == TopLevelKernelSpec(b)


class TestCloneUpdate:
    def test_simple_case(self):
        k = RBF(13.353)
        l = 353.645
        assert k.clone_update({"length_scale": l}) == RBF(l)

    def test_composite_case(self):
        k = PROD([RBF(13.353), PER(0.36543, 12.1)], 35.132)
        s = 353.645
        assert k.clone_update({"scalar": s}) == PROD(
            [RBF(13.353), PER(0.36543, 12.1)], s
        )

    def test_nested_composite_case(self):
        k_fn = lambda s: PROD(
            [
                RBF(13.353),
                PER(0.36543, 12.1),
                ADD([PROD([LIN(34)], 876), PROD([RQ(12.23)], 645.121)]),
            ],
            s,
        )
        s0 = 353.645
        s1 = 4567.234
        assert k_fn(s0).clone_update({"scalar": s1}) == k_fn(s1)


class TestMathOpMult:
    def test_base_kernel_mult(self):

        assert RBF(13.353) * PER() == PROD([RBF(13.353), PER()], 1)

    def test_base_kernel_mult_by_numeric(self):
        assert RBF(13.353) * 5 == PROD([RBF(13.353)], 5)
        assert RBF(13.353) * 5.34 == PROD([RBF(13.353)], 5.34)

        assert 5 * RBF(13.353) == PROD([RBF(13.353)], 5)
        assert 5.34 * RBF(13.353) == PROD([RBF(13.353)], 5.34)

    def test_product_kernel_mult(self):
        assert RBF(13.353) * 5 * PER() == PROD([RBF(13.353), PER()], 5)
        assert RBF(13.353) * PER() * 5.34 == PROD([RBF(13.353), PER()], 5.34)

        assert PER() * 5 * RBF(13.353) == PROD([PER(), RBF(13.353)], 5)
        assert PER() * 5.34 * RBF(13.353) == PROD([PER(), RBF(13.353)], 5.34)

    @pytest.mark.parametrize(
        "expression",
        [
            "a * b * c * d * e",
            "(a * b) * c * d * e",
            "(a * b * c) * d * e",
            "a * (b * c * d) * e",
            "a * b * c * (d * e)",
            "a * b * (c * d) * e",
            "a * (b * c * d * e)",
            "a * e * c * d * b",
            "a * d * c * b * e",
            "c * b * a * d * e",
            "d * b * c * a * e",
            "e * c * b * d * a",
            "e * c * (b * d) * a",
            "(e * c * b) * d * a",
            "(e * c) * b * (d * a)",
            "e * (c * b * d) * a",
        ],
    )
    def test_product_kernel_mult_grouped_and_permuted(self, expression):
        a = PER(5, 9)
        b = 12.434
        c = RBF(4.67)
        d = LIN(762)
        e = 34.66

        final = PROD(
            [
                PER(5, 9),
                RBF(4.67),
                LIN(762),
            ],
            12.434 * 34.66,
        )

        assert final == eval(expression)


class TestMathOpAdd:
    def test_base_kernel_add(self):
        assert RBF(13.353) + PER() == ADD([PROD([RBF(13.353)]), PROD([PER()])])

    def test_base_kernel_left_add_several(self):
        a = LIN(12)
        b = RBF(13.353) + PER()
        assert a + b == ADD([PROD([RBF(13.353)]), PROD([PER()]), PROD([LIN(12)])])

    def test_base_kernel_right_add_several(self):
        a = RBF(13.353) + PER()
        b = LIN(12)
        assert a + b == ADD([PROD([RBF(13.353)]), PROD([PER()]), PROD([LIN(12)])])

    def test_base_kernel_add_several(self):
        a = RBF(13.353) + PER()
        b = LIN(12) + RQ(52)
        assert a + b == ADD(
            [PROD([RBF(13.353)]), PROD([PER()]), PROD([LIN(12)]), PROD([RQ(52)])]
        )


class TestMathOpAddAndMult:
    def test_nested_composite_case(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = ADD([PROD([LIN(34)], 876), PROD([RQ(12.23)], 645.121)])

        A = PROD([a, b, c], 12.34)
        B = a * b * c * 12.34

        assert A == B

    @pytest.mark.parametrize(
        "expression",
        [" * ".join(order) for order in permutations(["a", "b", "c", "d", "e"])],
    )
    def test_nested_composite_case_reorder(self, expression):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = ADD([PROD([LIN(34)], 876), PROD([RQ(12.23)], 645.121)])
        d = 12.3435
        e = ADD([PROD([PER()], 43), PROD([RQ(7)], 0.65)])

        assert a * b * c * d * e == eval(expression)

    def test_add_times_add(self):
        ...
