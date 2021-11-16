import numpy as np
from numpy.fft import fft, fftfreq

from .kernel_specs import (
    BaseKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
)


def init_period_from_residuals(residuals: np.ndarray) -> float:
    residuals = residuals.flatten()
    N = len(residuals)
    yf = fft(residuals)[: N // 2]
    T = 2 / N
    xf = fftfreq(N, T)[: N // 2]
    return 1 / xf[np.abs(yf) == max(np.abs(yf))][0]


def intialize_base_kernel_prototypes_from_residuals(
    residuals, base_kernel_prototypes: list[BaseKernelSpec]
) -> list[BaseKernelSpec]:
    protos: list[BaseKernelSpec] = []
    for bk in base_kernel_prototypes:
        if isinstance(bk, PeriodicKernelSpec) or isinstance(
            bk, PeriodicNoConstKernelSpec
        ):
            period = init_period_from_residuals(residuals)
            period = np.clip(period, *bk.period_bounds)

            length_scale = np.clip(period / 2, *bk.length_scale_bounds)

            protos.append(
                bk.clone_update({"period": period, "length_scale": length_scale})
            )
        else:
            protos.append(bk.clone_update())
    return protos
