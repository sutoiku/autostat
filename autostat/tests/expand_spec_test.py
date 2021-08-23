import pytest

from ..kernel_specs import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
    TopLevelKernelSpec as TOP,
)

from ..expand_spec import (
    expand_product_spec,
    expand_additive_spec,
    NotExpandableProductSpecException,
    expand_spec,
)


class TestExpandAddSpec:
    def test_expand_no_op(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        assert expand_additive_spec(a + b + c + c) == a + b + c + c


class TestExpandProductSpec:
    def test_expand_no_op(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        with pytest.raises(NotExpandableProductSpecException):
            expand_product_spec(a * b * c)

    def test_expand_simple(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)

        A = expand_product_spec(a * (b + c))
        B = a * b + a * c

        assert A == B

    def test_expand_simple_2(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        assert expand_product_spec(a * (b + c) * d) == a * b * d + a * c * d
        assert expand_product_spec((b + c) * d) == b * d + c * d
        assert expand_product_spec((b + c) * a * d) == a * b * d + a * c * d

    def test_expand_composite(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        assert expand_product_spec((a + b) * (c + d)) == a * c + a * d + b * c + b * d

    def test_expand_composite_2(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        e = RBF(7645)
        assert (
            expand_product_spec((a + b) * (c + d) * e)
            == a * c * e + a * d * e + b * c * e + b * d * e
        )

    def test_expand_composite_3(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        e = RBF(7645)
        assert expand_product_spec((a + b + c) * e) == a * e + b * e + c * e

    def test_expand_composite_4(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        e = RBF(7645)
        f = LIN(453531)
        assert (
            expand_product_spec((a + b) * (c + d) * (e + f))
            == a * c * e
            + a * d * e
            + b * c * e
            + b * d * e
            + a * c * f
            + a * d * f
            + b * c * f
            + b * d * f
        )

    def test_expand_composite_scalars(self):
        a = RBF(13.353)
        b = 232.545

        c = LIN(34)
        d = 87.2321

        e = RBF(7645)
        f = 7656.12121

        S = b * f
        T = d * f
        assert expand_product_spec((a * b + c * d) * (e * f)) == (a * e * S + c * e * T)


class TestExpandTopLevelSpec:
    def test_expand_top_level_composite(self):
        a = RBF(13.353)
        b = PER(0.36543, 12.1)
        c = LIN(34)
        d = RQ(12.23)
        e = RBF(7645)
        f = LIN(453531)
        g = PER(97)

        A = (a + b) * (c + d) * (e + f) + g

        B = (
            a * c * e
            + a * d * e
            + b * c * e
            + b * d * e
            + a * c * f
            + a * d * f
            + b * c * f
            + b * d * f
            + g
        )
        T1 = TOP.from_additive(A)
        T2 = TOP.from_additive(B)
        assert expand_spec(T1) == expand_spec(T2)
