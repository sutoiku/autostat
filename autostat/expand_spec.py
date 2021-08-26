from .kernel_specs import (
    ProductOperandSpec,
    TopLevelKernelSpec as TOP,
    AdditiveKernelSpec as ADD,
    ProductKernelSpec as PROD,
)


def is_expandable_product(spec: PROD) -> bool:
    return any(isinstance(op, ADD) for op in spec.operands)


class NotExpandableProductSpecException(Exception):
    def __init__(
        self,
        msg="Given Kernel spec is not expandable; check is_expandable_product before expanding",
        *args,
        **kwargs
    ):
        super().__init__(msg, *args, **kwargs)


def expand_additive_spec(spec: ADD) -> ADD:
    ops = []

    for op in spec.operands:
        if is_expandable_product(op):
            ops += expand_product_spec(op).operands
        else:
            ops.append(op)

    return ADD(ops)


def expand_product_spec(spec: PROD) -> ADD:
    if not is_expandable_product(spec):
        raise NotExpandableProductSpecException()

    first_additive_found = False
    first_additive_operand = None
    other_operands: list[ProductOperandSpec] = []
    for op in spec.operands:
        if isinstance(op, ADD) and not first_additive_found:
            first_additive_operand = op
            first_additive_found = True
        else:
            other_operands.append(op)

    if first_additive_operand is None:
        raise ValueError(
            "Given KernelSpec is not expandable; check is_expandable_product before expanding"
        )

    expanded_spec = ADD(
        operands=[
            PROD(
                [*product_spec_inner.operands, *other_operands],
                scalar=spec.scalar * product_spec_inner.scalar,
            )
            for product_spec_inner in first_additive_operand.operands
        ]
    )

    return expand_additive_spec(expanded_spec)


def expand_spec(spec: TOP) -> TOP:
    operands = expand_additive_spec(spec).operands
    return spec.clone_update({"operands": operands})
