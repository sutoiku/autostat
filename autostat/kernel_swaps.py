#  from multipledispatch import dispatch

from typing import NamedTuple, TypeVar, Union, cast

from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

from .kernel_tree_types import (
    Dataset,
    KernelSpec,
    ArithmeticKernelSpec,
    BaseKernelSpec,
    BaseKernelSpecTypes,
    AdditiveKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    RQKernelSpec,
    AutoGpModel,
)

from .kernel_trees_generic import kernel_type

base_kernel_classes: list[type[BaseKernelSpec]] = [
    RBFKernelSpec,
    RQKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
]


# FIX: need to simplify kernels like:
#  'ADD(PROD(ADD(PROD(LIN), PROD(RBF))))',
#  'ADD(PROD(ADD(PROD(PER), PROD(RBF))))',
#  'ADD(PROD(ADD(PROD(RBF), PROD(RBF))))',
#  'ADD(PROD(ADD(PROD(RBF), PROD(RQ))))',
#  -- ADD(PROD_with_one_add_term(ADD(...))) -> ADD(...)


class KernelInitialValues(NamedTuple):
    PER_period: float = 1
    PER_length_scale: float = 1


def initialize_spec(
    k: BaseKernelSpec, v: KernelInitialValues = KernelInitialValues()
) -> BaseKernelSpec:

    if isinstance(k, PeriodicKernelSpec):
        return PeriodicKernelSpec(
            **{
                **k._asdict(),
                **{"length_scale": v.PER_length_scale, "period": v.PER_period},
            }
        )

    else:
        return k.__class__()


def other_base_kernels(
    kernel: BaseKernelSpec, init_vals: KernelInitialValues = KernelInitialValues()
) -> list[BaseKernelSpec]:
    return [
        initialize_spec(k(), init_vals)
        for k in base_kernel_classes
        if k.__name__ != kernel.__class__.__name__
    ]


GenericKernelSpecClasses = TypeVar(
    "GenericKernelSpecClasses",
    KernelSpec,
    BaseKernelSpec,
    "AdditiveKernelSpec",
    "ProductKernelSpec",
    ProductOperandSpec,
)


def sort_specs_by_type(
    kernels: list[GenericKernelSpecClasses],
) -> list[GenericKernelSpecClasses]:
    return sorted(kernels, key=lambda node: node.spec_str(False, False))


def sort_operand_list(
    operands: list[GenericKernelSpecClasses],
) -> list[GenericKernelSpecClasses]:
    return sorted(operands, key=lambda operand: operand.spec_str(False, False))


def sort_list_of_operand_lists(
    operand_lists: list[list[GenericKernelSpecClasses]],
) -> list[list[GenericKernelSpecClasses]]:
    return sorted(
        operand_lists, key=lambda operand_list: str(sort_operand_list(operand_list))
    )


def dedupe_kernels(
    kernel: list[GenericKernelSpecClasses],
) -> list[GenericKernelSpecClasses]:
    subtree_dict = {k.spec_str(False, False): k for k in kernel}
    return sort_specs_by_type(list(subtree_dict.values()))


def product_wrapped_base_kernel(
    kernel: BaseKernelSpec,
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> ProductKernelSpec:
    return ProductKernelSpec(operands=[initialize_spec(kernel, initial_vals)], scalar=1)


def addititive_base_term_with_scalar(kernel: BaseKernelSpec) -> AdditiveKernelSpec:
    return AdditiveKernelSpec(operands=[product_wrapped_base_kernel(kernel)])


def base_subtree_swaps(
    node: BaseKernelSpec, initial_vals: KernelInitialValues = KernelInitialValues()
) -> list[Union[BaseKernelSpec, AdditiveKernelSpec, ProductKernelSpec]]:
    # other base kernels
    nodes_out = cast(list[KernelSpec], other_base_kernels(node, initial_vals))
    # this base kernel with sums and products of all base kernels
    for bk in base_kernel_classes:
        nodes_out.append(
            AdditiveKernelSpec(
                [
                    product_wrapped_base_kernel(node),
                    product_wrapped_base_kernel(bk(), initial_vals),
                ]
            )
        )
        nodes_out.append(ProductKernelSpec([node, initialize_spec(bk(), initial_vals)]))
    return dedupe_kernels(nodes_out)


def product_operand_swaps(
    operands: list[ProductOperandSpec],
    index: int,
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> list[list[ProductOperandSpec]]:
    # take a list of product operands and an index to a new
    # list of lists of operands with all valid swaps at that index

    operand_lists: list[list[ProductOperandSpec]] = []

    before = operands[:index]
    current = operands[index]
    after = operands[index + 1 :]
    # in case of a Additive subkernel, recurse
    if isinstance(current, AdditiveKernelSpec):
        new_subtrees = additive_subtree_swaps(current, initial_vals)
        for new_subtree in new_subtrees:
            operand_lists.append([*before, new_subtree, *after])
    else:
        # in this case, we have a base kernel
        # valid swaps of this base kernel for others
        for bk in other_base_kernels(current, initial_vals):
            operand_lists.append([*before, bk, *after])
        # valid sums of this base with other base kernels
        for bk_type in base_kernel_classes:
            new_sum_kernel = AdditiveKernelSpec(
                operands=[
                    product_wrapped_base_kernel(current, initial_vals),
                    product_wrapped_base_kernel(bk_type(), initial_vals),
                ]
            )
            operand_lists.append([*before, new_sum_kernel, *after])

    return operand_lists


def product_subtree_swaps(
    node: ProductKernelSpec,
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> list[ProductKernelSpec]:

    nodes_out: list[ProductKernelSpec] = []
    scalar = node.scalar

    for i in range(len(node.operands)):
        for operands in product_operand_swaps(node.operands, i, initial_vals):
            nodes_out.append(ProductKernelSpec(operands=operands, scalar=scalar))

    # # this product with all other base kernels included
    for bk_type in base_kernel_classes:
        nodes_out.append(
            ProductKernelSpec(
                operands=[*node.operands, initialize_spec(bk_type(), initial_vals)],
                scalar=scalar,
            )
        )
    return dedupe_kernels(nodes_out)


def additive_subtree_swaps(
    node: AdditiveKernelSpec,
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> list[AdditiveKernelSpec]:

    nodes_out: list[AdditiveKernelSpec] = []

    # this sum of kernels with all other base kernels added,
    for bk in base_kernel_classes:
        nodes_out.append(
            AdditiveKernelSpec(
                operands=node.operands
                + [product_wrapped_base_kernel(bk(), initial_vals)]
            )
        )

    # if this sum of kernels has 2 or more summands, then
    # this sum of kernels as a unit multiplied by all base kernels
    if len(node.operands) >= 2:
        for bk in base_kernel_classes:
            nodes_out.append(
                AdditiveKernelSpec(
                    [ProductKernelSpec([node, initialize_spec(bk(), initial_vals)])]
                )
            )

    # for each summand, all valid replacement subkernels
    for i in range(len(node.operands)):
        before = node.operands[:i]
        current = node.operands[i]
        after = node.operands[i + 1 :]
        for new_subtree in product_subtree_swaps(
            current, initial_vals=KernelInitialValues()
        ):
            nodes_out.append(
                AdditiveKernelSpec(operands=before + [new_subtree] + after)
            )

    return dedupe_kernels(nodes_out)
