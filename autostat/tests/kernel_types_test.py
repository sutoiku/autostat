from ..kernel_specs import (
    RBFKernelSpec as RBF,
    RQKernelSpec as RQ,
    LinearKernelSpec as LIN,
    PeriodicKernelSpec as PER,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
)


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
