from autostat.decomposition import decompose_spec
import time
from typing import NamedTuple, cast

import numpy as np
from numpy.fft import fft, fftfreq

from .auto_gp_model import AutoGpModel
from .kernel_specs import (
    BaseKernelSpec,
    TopLevelKernelSpec,
    PeriodicKernelSpec,
    PeriodicNoConstKernelSpec,
)


from .dataset_adapters import Dataset

from .utils.logger import BasicLogger, Logger

from .kernel_swaps import top_level_spec_swaps

from .run_settings import RunSettings

from .plots import plot_decomposition, plot_model

from .expand_spec import expand_spec


class ScoredKernelInfo(NamedTuple):
    name: str
    spec_pre_fit: TopLevelKernelSpec
    spec_fitted: TopLevelKernelSpec
    model: AutoGpModel
    bic: float
    log_likelihood: float


KernelScores = dict[str, "ScoredKernelInfo"]


def score_kernel_spec(
    kernel_spec: TopLevelKernelSpec,
    data: Dataset,
    model_class: type[AutoGpModel],
    run_settings: RunSettings,
    logger: Logger = None,
) -> ScoredKernelInfo:
    logger = logger or BasicLogger()
    tic = time.perf_counter()

    model = model_class(kernel_spec, data, run_settings=run_settings)

    model.fit(data)

    log_likelihood = model.log_likelihood()
    num_params = kernel_spec.num_params()
    bic = model.bic()

    fig, ax = plot_model(model, data)

    fitted_spec = model.to_spec()

    spec_str = f"""{fitted_spec.spec_str(False,True)}
{fitted_spec.spec_str(False,False)} -- bic: {bic:.2f}, log likelihood: {log_likelihood:.3f}, M: {num_params}
{fitted_spec.spec_str(True,True)}"""

    ax.set_title(spec_str)

    toc = time.perf_counter()
    logger.print(f"**{fitted_spec.spec_str(False,True)}** -- fit in: {toc-tic:.3f} s")
    logger.show(fig)

    return ScoredKernelInfo(
        kernel_spec.spec_str(False, False),
        kernel_spec,
        fitted_spec,
        model,
        bic,
        log_likelihood,
    )


def score_kernel_specs(
    specs: list[TopLevelKernelSpec],
    data: Dataset,
    model_class: type[AutoGpModel],
    kernel_scores: KernelScores,
    run_settings: RunSettings,
    logger: Logger = None,
) -> KernelScores:
    logger = logger or BasicLogger()

    for spec in specs:
        spec_str = spec.spec_str(False, False)
        if spec_str in kernel_scores:
            continue
        kernel_scores[spec_str] = score_kernel_spec(
            spec, data, model_class, run_settings, logger
        )
    return kernel_scores


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
            protos.append(
                bk.clone_update({"period": period, "length_scale": period / 2})
            )
        else:
            protos.append(bk.clone_update())
    return protos


def get_best_kernel_name_and_info(
    kernel_scores: KernelScores,
) -> tuple[str, ScoredKernelInfo]:
    return min(
        kernel_scores.items(), key=lambda name_score_info: name_score_info[1].bic
    )


def get_best_kernel_info(
    kernel_scores: KernelScores,
) -> ScoredKernelInfo:
    return min(kernel_scores.values(), key=lambda name_score_info: name_score_info.bic)


def kernel_search(
    data: Dataset,
    model_class: type[AutoGpModel],
    run_settings: RunSettings,
    kernel_scores: KernelScores = None,
    logger: Logger = None,
) -> KernelScores:

    kernel_scores = kernel_scores or {}
    logger = logger or BasicLogger()
    best_model = None

    for i in range(run_settings.max_search_depth):
        tic = time.perf_counter()
        logger.print(f"# DEPTH {i}")

        # set up kernels for this depth
        if i == 0:
            specs = run_settings.initial_kernels
        else:
            best_kernel_info = get_best_kernel_info(kernel_scores)
            residuals = best_kernel_info.model.residuals()
            base_kernel_prototypes = intialize_base_kernel_prototypes_from_residuals(
                residuals, run_settings.base_kernel_prototypes
            )

            proto_str = "\n".join(str(k) for k in base_kernel_prototypes)
            logger.print(f"### prototype kernels from residuals:\n {proto_str}")
            specs = top_level_spec_swaps(
                best_kernel_info.spec_fitted,
                base_kernel_prototypes,
                run_settings.expand_kernel_specs_as_sums,
            )

        logger.print(f"### specs to check at depth {i}")
        logger.print("\n".join(["* " + str(sp) for sp in specs]))

        # score the kernels for this depth
        kernel_scores = score_kernel_specs(
            specs, data, model_class, kernel_scores, run_settings, logger
        )

        best_kernel_info = get_best_kernel_info(kernel_scores)

        best_model = best_kernel_info.model
        best_fitted_spec = best_kernel_info.spec_fitted

        best_kernel_str = f"""Best at depth {i}:   {best_fitted_spec.spec_str(False,True)}  -- bic: {best_kernel_info.bic:.2f}, log likelihood: {best_kernel_info.log_likelihood:.3f}, M: {best_fitted_spec.num_params()}
{best_fitted_spec.spec_str(False,False)} 
{best_fitted_spec.spec_str(True,True)}"""

        logger.print("## " + best_kernel_str)

        fig, ax = plot_model(best_model, data)
        ax.set_title(best_kernel_str)
        logger.show(fig)

        expanded_spec = expand_spec(best_fitted_spec)

        logger.print(f"best spec expanded:\n{expanded_spec.spec_str(True,True)}")

        decomp = decompose_spec(expanded_spec, data.train_x, data.train_y)
        fig = plot_decomposition(decomp)
        logger.show(fig)

        toc = time.perf_counter()
        logger.print(f"depth {i} complete in: {toc-tic:.3f} s")

    return kernel_scores


def find_best_kernel_and_predict(
    data: Dataset,
    model_class: type[AutoGpModel],
    run_settings: RunSettings,
    kernel_scores: KernelScores = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_scores = kernel_search(
        data, model_class, run_settings=run_settings, kernel_scores=kernel_scores
    )
    best_kernel_info = get_best_kernel_info(kernel_scores)

    best_model = cast(AutoGpModel, best_kernel_info.model)
    return best_model.predict(data.test_x)
