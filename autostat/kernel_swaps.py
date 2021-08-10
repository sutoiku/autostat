from typing import NamedTuple, TypeVar, Union, cast

from .kernel_specs import (
    KernelSpec,
    BaseKernelSpec,
    AdditiveKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    PeriodicKernelSpec,
    GenericKernelSpec,
)


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
        return k.clone_update()


def other_base_kernels(
    kernel: BaseKernelSpec,
    base_kernel_classes: list[type[BaseKernelSpec]],
    init_vals: KernelInitialValues = KernelInitialValues(),
) -> list[BaseKernelSpec]:
    return [
        initialize_spec(k(), init_vals)
        for k in base_kernel_classes
        if k.__name__ != kernel.__class__.__name__
    ]


def sort_specs_by_type(
    kernels: list[GenericKernelSpec],
) -> list[GenericKernelSpec]:
    return sorted(kernels, key=lambda node: node.spec_str(True, True))


def sort_operand_list(
    operands: list[GenericKernelSpec],
) -> list[GenericKernelSpec]:
    return sorted(operands, key=lambda operand: operand.spec_str(False, False))


def sort_list_of_operand_lists(
    operand_lists: list[list[GenericKernelSpec]],
) -> list[list[GenericKernelSpec]]:
    return sorted(
        operand_lists, key=lambda operand_list: str(sort_operand_list(operand_list))
    )


def dedupe_kernels(
    kernels: list[GenericKernelSpec],
) -> list[GenericKernelSpec]:
    subtree_dict: dict[str, GenericKernelSpec] = {}
    # for each kernel spec, keep the kernel matching that spec
    # that has the greatest number of fitted params
    for k in kernels:
        key = k.schema()
        if (
            key in subtree_dict and k.fit_count() > subtree_dict[key].fit_count()
        ) or key not in subtree_dict:
            subtree_dict[key] = k
    return sort_specs_by_type(list(subtree_dict.values()))


def simplify_additive_kernel_spec(kernel: AdditiveKernelSpec) -> AdditiveKernelSpec:
    # This function simplifies kernels like:
    # (1)
    # `ADD([PROD(ADD([PROD(LIN), PROD(RBF)]))])`
    # down to e.g. `ADD([PROD(LIN), PROD(RBF)])`.
    # (2)
    # `ADD([PROD(ADD([PROD(LIN), PROD(RBF)])), PROD(RQ), PROD(PER)])`
    # down to e.g. `ADD([PROD(LIN), PROD(RBF), PROD(RQ), PROD(PER)])`.

    # given an ADD operator, for each operand (which must be PROD),
    # check whether the PROD has only one ADD, and if it does,
    # then push these inner ADDs back up to the parent add, adjusting
    # scalars as needed

    # print(f"### simplify in : {str(kernel)}")

    final_summands: list[ProductKernelSpec] = []

    for prod_op in kernel.operands:

        if len(prod_op.operands) > 1:
            # if this PROD kernel has multiple operands, append them as is
            # this means that we DO NOT simplify ADD*ADD*... , ADD*X*... , X*X*...
            # for any base kernel X (in these cases, simplification would require expanding products)
            final_summands.append(prod_op.clone_update())
        else:
            lone_multiplicand = prod_op.operands[0]
            if isinstance(lone_multiplicand, AdditiveKernelSpec):
                # in this case:
                # 1) this product kernel has only one multiplicand
                # 2) this lone multiplicand is an additive kernel
                # this being the case, it's valid to
                # remove the middle product layer, and propogate its scalar
                # down to the product terms below the second sum
                outer_scalar = prod_op.scalar
                inner_add = lone_multiplicand
                final_summands += [
                    summand.clone_update({"scalar": summand.scalar * outer_scalar})
                    for summand in inner_add.operands
                ]
            else:
                # if the inner kernel within the product is not a SUM, then it is
                # the product of the base kernel with a scalar, so append as is
                final_summands.append(prod_op.clone_update())
    kernel_out = AdditiveKernelSpec(final_summands)

    # print(f"### simplify out: {str(kernel_out)}")

    return kernel_out


def product_wrapped_base_kernel(
    kernel: BaseKernelSpec,
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> ProductKernelSpec:
    return ProductKernelSpec(operands=[initialize_spec(kernel, initial_vals)], scalar=1)


def addititive_base_term_with_scalar(kernel: BaseKernelSpec) -> AdditiveKernelSpec:
    return AdditiveKernelSpec(operands=[product_wrapped_base_kernel(kernel)])


def base_subtree_swaps(
    node: BaseKernelSpec,
    base_kernel_classes: list[type[BaseKernelSpec]],
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> list[Union[BaseKernelSpec, AdditiveKernelSpec, ProductKernelSpec]]:
    # other base kernels
    nodes_out = cast(
        list[Union[BaseKernelSpec, AdditiveKernelSpec, ProductKernelSpec]],
        other_base_kernels(node, base_kernel_classes, initial_vals),
    )
    # this base kernel with sums and products of all base kernels
    for bk in base_kernel_classes:
        # all sums
        nodes_out.append(
            AdditiveKernelSpec(
                [
                    product_wrapped_base_kernel(node.clone_update()),
                    product_wrapped_base_kernel(bk(), initial_vals),
                ]
            )
        )
        # all products
        nodes_out.append(ProductKernelSpec([node, initialize_spec(bk(), initial_vals)]))
    return dedupe_kernels(nodes_out)


def product_operand_swaps(
    operands: list[ProductOperandSpec],
    index: int,
    base_kernel_classes: list[type[BaseKernelSpec]],
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
        new_subtrees = additive_subtree_swaps(
            current, base_kernel_classes, initial_vals
        )
        for new_subtree in new_subtrees:
            operand_lists.append([*before, new_subtree, *after])
    else:
        # in this case, we have a base kernel
        # valid swaps of this base kernel for others
        for bk in other_base_kernels(current, base_kernel_classes, initial_vals):
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
    base_kernel_classes: list[type[BaseKernelSpec]],
    initial_vals: KernelInitialValues = KernelInitialValues(),
) -> list[ProductKernelSpec]:

    nodes_out: list[ProductKernelSpec] = []
    scalar = node.scalar

    for i in range(len(node.operands)):
        for operands in product_operand_swaps(
            node.operands, i, base_kernel_classes, initial_vals
        ):
            nodes_out.append(ProductKernelSpec(operands=operands, scalar=scalar))

    # this product with all other base kernels included
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
    base_kernel_classes: list[type[BaseKernelSpec]],
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
            current, base_kernel_classes, initial_vals=KernelInitialValues()
        ):
            nodes_out.append(
                AdditiveKernelSpec(operands=before + [new_subtree] + after)
            )

    return dedupe_kernels([simplify_additive_kernel_spec(spec) for spec in nodes_out])
    # return dedupe_kernels(nodes_out)
