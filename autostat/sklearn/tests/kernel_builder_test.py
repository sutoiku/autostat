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

import typing as ty

from ...kernel_tree_types import (
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

from ...kernel_swaps import base_kernel_classes
from ...kernel_search import starting_kernel_specs
from ...constraints import (
    KernelConstraints,
    ConstraintBounds as CB,
    PeriodicKernelConstraints,
    cb_default,
    default_constraints,
)

from ..kernel_trees_sklearn import to_kernel_spec, to_kernel_spec_inner
from ..kernel_builder import build_kernel, build_kernel_additive


class TestBuildKernel:
    def test_starting_kernel_specs(self):
        [build_kernel(k) for k in starting_kernel_specs()]


class TestBuildKernelWithConstraints:
    def test_build_periodic_default_constraints(self):
        k = ty.cast(ExpSineSquared, build_kernel(PER()))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == cb_default()
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == cb_default()

    def test_build_periodic_constrained(self):
        constraints = KernelConstraints(
            PeriodicKernelConstraints(length_scale=CB(10, 20), period=CB(0.5, 1.5))
        )
        k = ty.cast(ExpSineSquared, build_kernel(PER(), constraints))
        assert tuple(k.hyperparameter_length_scale.bounds.flatten()) == (10, 20)
        assert tuple(k.hyperparameter_periodicity.bounds.flatten()) == (0.5, 1.5)

    def test_build_periodic_constrained_composite(self):
        constraints = KernelConstraints(
            PeriodicKernelConstraints(length_scale=CB(10, 20), period=CB(0.5, 1.5))
        )

        spec = ADD([PROD([PER()], scalar=6), PROD([LIN()], scalar=13)])

        k = build_kernel(spec, constraints)

        k_PER = ty.cast(ExpSineSquared, k.k1.k2)
        assert tuple(k_PER.hyperparameter_length_scale.bounds.flatten()) == (10, 20)
        assert tuple(k_PER.hyperparameter_periodicity.bounds.flatten()) == (0.5, 1.5)