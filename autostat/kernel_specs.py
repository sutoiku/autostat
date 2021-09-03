import typing as ty


from dataclasses import asdict, dataclass, astuple, field, replace, InitVar


# A kernel spec is composed of a top level AdditiveKernelSpec
# Each summand of an additive kernel is a product
# each product has a scalar and one or more operands
# product operands may be either base kernels or additive kernels


GenericKernelSpec = ty.TypeVar("GenericKernelSpec", bound="KernelSpec")


class ConstraintBounds(ty.NamedTuple):
    lower: float = 0.00001
    upper: float = 1000

    def interval_proportion(self, a: float) -> float:
        if a < 0.0 or a > 1.0:
            raise ValueError(f"interval_proportion argument must be in [0,1], got {a}")
        return self.lower + a * (self.upper - self.lower)

    def clamp(self, x: float) -> float:
        if x < self.lower:
            return self.interval_proportion(0.01)
        elif x > self.upper:
            return self.interval_proportion(0.99)
        else:
            return x


def sort_specs_by_type(
    kernels: list[GenericKernelSpec],
) -> list[GenericKernelSpec]:
    return sorted(kernels, key=lambda node: node.spec_str(True, True))


@dataclass(frozen=True)
class KernelSpec:
    def __init__(self, kwargs: dict[str, ty.Any] = {}) -> None:
        ...

    def spec_str(self, verbose: bool = True, pretty: bool = True) -> str:
        ...

    def num_params(self) -> int:
        ...

    def fit_count(self) -> int:
        ...

    def schema(self) -> str:
        return self.spec_str(False, False)

    def clone_update(
        self: GenericKernelSpec, kwargs: dict[str, ty.Any] = {}
    ) -> GenericKernelSpec:
        for k, v in kwargs.items():
            bounds_key = k + "_bounds"
            if hasattr(self, bounds_key):
                bounds = ty.cast(ConstraintBounds, getattr(self, bounds_key))
                if v < bounds.lower or v > bounds.upper:
                    raise ValueError(
                        f"Attempted to clone_update '{k}' with value {v} outside of allowed bounds {bounds}"
                    )
        return replace(self, **kwargs)

    def __iter__(self):
        yield from self.params()

    def param_dict(self) -> dict[str, ty.Any]:
        return {
            k: v for k, v in asdict(self).items() if not isinstance(v, ConstraintBounds)
        }

    def params(self) -> tuple[ty.Any, ...]:
        return tuple(
            item for item in astuple(self) if not isinstance(item, ConstraintBounds)
        )

    def __str__(self) -> str:
        return self.spec_str(True, True)

    def __repr__(self) -> str:
        return self.spec_str(True, False)

    def __add__(self, other: "KernelSpec") -> "AdditiveKernelSpec":
        ...

    def __radd__(self, other: "KernelSpec") -> "AdditiveKernelSpec":
        return self.__add__(other)

    def __mul__(self, other: ty.Union["KernelSpec", float]) -> "ProductKernelSpec":
        ...

    def __rmul__(self, other: ty.Union["KernelSpec", float]) -> "ProductKernelSpec":
        return self.__mul__(other)


class BaseKernelSpec(KernelSpec):
    kernel_name: InitVar[str]
    pp_replacements: InitVar[dict[str, str]]

    def spec_str(self, verbose: bool = True, pretty: bool = True) -> str:
        name = str(self.kernel_name)
        if verbose:
            param_str = ",".join([f"{k}={v:.4f}" for k, v in self.param_dict().items()])
            for str1, str2 in ty.cast(dict[str, str], self.pp_replacements).items():
                param_str = param_str.replace(str1, str2)
            return f"{name}({param_str})"
        else:
            return name

    def num_params(self) -> int:
        return len(self.params())

    def fit_count(self) -> int:
        return sum(v != 1 for v in self.params())

    def __add__(self, other: KernelSpec) -> "AdditiveKernelSpec":
        if isinstance(other, BaseKernelSpec):
            return AdditiveKernelSpec(operands=[1 * self, 1 * other])

        elif isinstance(other, AdditiveKernelSpec):
            return AdditiveKernelSpec(operands=[1 * self, *other.operands])

        elif isinstance(other, ProductKernelSpec):
            return AdditiveKernelSpec(operands=[1 * self, other])

        else:
            raise TypeError(f"__add__ not defined for BaseKernelSpec and {type(other)}")

    def __mul__(self, other: ty.Union["KernelSpec", float]) -> "ProductKernelSpec":
        if isinstance(other, float) or isinstance(other, int):
            return ProductKernelSpec([self], scalar=float(other))

        elif isinstance(other, BaseKernelSpec):
            return ProductKernelSpec([self, other])

        elif isinstance(other, AdditiveKernelSpec):
            return ProductKernelSpec([self, other])

        else:
            return NotImplemented


@dataclass(frozen=True)
class RBFKernelSpec(BaseKernelSpec):
    length_scale: float = 1.0

    length_scale_bounds: ConstraintBounds = ConstraintBounds()

    kernel_name: InitVar[str] = "RBF"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l"}


@dataclass(frozen=True)
class PeriodicKernelSpec(BaseKernelSpec):
    length_scale: float = 1.0
    period: float = 1.0

    length_scale_bounds: ConstraintBounds = ConstraintBounds()
    period_bounds: ConstraintBounds = ConstraintBounds()

    kernel_name: InitVar[str] = "PER"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l", "period": "p"}


@dataclass(frozen=True)
class PeriodicNoConstKernelSpec(BaseKernelSpec):
    length_scale: float = 1.0
    period: float = 1.0

    length_scale_bounds: ConstraintBounds = ConstraintBounds()
    period_bounds: ConstraintBounds = ConstraintBounds()

    kernel_name: InitVar[str] = "PERnc"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l", "period": "p"}


@dataclass(frozen=True)
class RQKernelSpec(BaseKernelSpec):
    length_scale: float = 1.0
    alpha: float = 1.0

    length_scale_bounds: ConstraintBounds = ConstraintBounds()
    alpha_bounds: ConstraintBounds = ConstraintBounds()

    kernel_name: InitVar[str] = "RQ"
    pp_replacements: InitVar[dict[str, str]] = {"length_scale": "l", "alpha": "α"}


@dataclass(frozen=True)
class LinearKernelSpec(BaseKernelSpec):
    variance: float = 1.0

    variance_bounds: ConstraintBounds = ConstraintBounds()

    kernel_name: InitVar[str] = "LIN"
    pp_replacements: InitVar[dict[str, str]] = {"variance": "var"}


##############


# # @sort_operands
@dataclass(frozen=True)
class AdditiveKernelSpec(KernelSpec):
    operands: list["ProductKernelSpec"] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "operands", sort_specs_by_type(self.operands))

    def num_params(self) -> int:
        return sum(op.num_params() for op in self.operands)

    def fit_count(self) -> int:
        return sum(op.fit_count() for op in self.operands)

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            return "(" + " + ".join(operandStrings) + ")"
        else:
            return f"ADD({', '.join(operandStrings)})"

    def __add__(self, other: "KernelSpec") -> "AdditiveKernelSpec":
        if isinstance(other, BaseKernelSpec):
            ops = [*self.operands, 1 * other]

        elif isinstance(other, AdditiveKernelSpec):
            ops = [*self.operands, *other.operands]

        elif isinstance(other, ProductKernelSpec):
            ops = [*self.operands, other]

        else:
            return NotImplemented
        return AdditiveKernelSpec(ops)

    def __mul__(self, other: ty.Union["KernelSpec", float]) -> "ProductKernelSpec":
        if isinstance(other, float) or isinstance(other, int):
            return ProductKernelSpec([self], scalar=float(other))

        elif isinstance(other, BaseKernelSpec) or isinstance(other, AdditiveKernelSpec):
            return ProductKernelSpec([self, other], scalar=1.0)

        elif isinstance(other, ProductKernelSpec):
            return ProductKernelSpec(other.operands + [self], other.scalar)

        else:
            return NotImplemented


@dataclass(frozen=True)
class TopLevelKernelSpec(AdditiveKernelSpec):
    operands: list["ProductKernelSpec"] = field(default_factory=list)
    noise: float = 0.01

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            v = f" + {self.noise:.4f}*ε" if verbose else ""
            return " + ".join(operandStrings) + v
        else:
            v = f",σ={self.noise:.4f}" if verbose else ""
            return f"TOP({', '.join(operandStrings)}{v})"

    @staticmethod
    def from_additive(spec: AdditiveKernelSpec, noise: float = None):
        ops = spec.operands
        return TopLevelKernelSpec(ops, noise) if noise else TopLevelKernelSpec(ops)

    @staticmethod
    def from_base_kernel(
        spec: BaseKernelSpec, scalar: float = None, noise: float = None
    ):
        prod = (
            ProductKernelSpec([spec], scalar) if scalar else ProductKernelSpec([spec])
        )
        return (
            TopLevelKernelSpec([prod], noise) if noise else TopLevelKernelSpec([prod])
        )


ProductOperandSpec = ty.Union[
    BaseKernelSpec,
    AdditiveKernelSpec,
]


@dataclass(frozen=True)
class ProductKernelSpec(KernelSpec):
    operands: list[ProductOperandSpec] = field(default_factory=list)
    scalar: float = 1.0

    kernel_name: InitVar[str] = "PROD"

    def __post_init__(self, _):
        object.__setattr__(self, "operands", sort_specs_by_type(self.operands))

    def num_params(self) -> int:
        # 1 for scalar, plus child params
        return 1 + sum(op.num_params() for op in self.operands)

    def fit_count(self) -> int:
        this_fit = 0 if self.scalar == 1 else 1
        return sum(op.fit_count() for op in self.operands) + this_fit

    def spec_str(self, verbose=True, pretty=True) -> str:
        operandStrings = sorted(op.spec_str(verbose, pretty) for op in self.operands)
        if pretty:
            scalar_str = f"{self.scalar:.4f} * " if verbose else ""
            return scalar_str + " * ".join(operandStrings)
        else:
            scalar_str = f"{self.scalar:.4f}, " if verbose else ""
            return f"PROD({scalar_str}{', '.join(operandStrings)})"

    def clone_update(self, kwargs: dict[str, ty.Any] = {}) -> "ProductKernelSpec":
        cloned_operands = [op.clone_update() for op in self.operands]
        return replace(
            self, **{"operands": cloned_operands, "scalar": self.scalar, **kwargs}
        )

    def __mul__(self, other: ty.Union["KernelSpec", float]) -> "ProductKernelSpec":
        if isinstance(other, float) or isinstance(other, int):
            return self.clone_update({"scalar": other * self.scalar})

        elif isinstance(other, BaseKernelSpec):
            return self.clone_update({"operands": self.operands + [other]})

        elif isinstance(other, AdditiveKernelSpec):
            return self.clone_update({"operands": self.operands + [other]})

        elif isinstance(other, ProductKernelSpec):
            return self.clone_update(
                {
                    "operands": self.operands + other.operands,
                    "scalar": self.scalar * other.scalar,
                }
            )
        else:
            return NotImplemented

    def __add__(self, other: "KernelSpec") -> AdditiveKernelSpec:
        if isinstance(other, ProductKernelSpec):
            return AdditiveKernelSpec([self, other])

        else:
            return NotImplemented
