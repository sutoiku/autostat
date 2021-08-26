from dataclasses import dataclass
import typing as ty

import numpy as np
from numpy.lib.twodim_base import diag
from numpy.typing import NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from .kernel_specs import ProductKernelSpec, TopLevelKernelSpec, KernelSpec

from .sklearn.kernel_builder import build_kernel
from .constraints import default_constraints


def gram_matrix_from_spec(
    spec: KernelSpec, x: NDArray[np.float_]
) -> NDArray[np.float_]:
    gp = GaussianProcessRegressor(kernel=build_kernel(spec, default_constraints()))
    kernel = ty.cast(Kernel, gp.kernel)
    K = kernel(x)
    return ty.cast(NDArray[np.float_], K)


@dataclass
class DecompositionData:
    x: NDArray[np.float_]
    components: list[tuple[ProductKernelSpec, NDArray[np.float_], NDArray[np.float_]]]


def decompose_spec(
    spec: TopLevelKernelSpec, x: NDArray[np.float_], y: NDArray[np.float_]
) -> DecompositionData:
    K_full = gram_matrix_from_spec(spec, x)
    L_inv = np.linalg.inv(np.linalg.cholesky(K_full))
    K_inv = L_inv.T @ L_inv

    components = []

    for sub_spec in spec.operands:
        # spec_str = sub_spec.spec_str(True, False)
        K_i = gram_matrix_from_spec(sub_spec, x)
        components.append(
            (
                sub_spec,
                ty.cast(NDArray[np.float_], K_i @ K_inv @ y),
                ty.cast(NDArray[np.float_], np.sqrt(np.diag(K_i - K_i @ K_inv @ K_i))),
            )
        )

    return DecompositionData(x, components)
