import time
from typing import NamedTuple, Union, cast

from matplotlib import pyplot as plt
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


class ScoredKernelInfo(NamedTuple):
    name: str
    spec_pre_fit: TopLevelKernelSpec
    spec_fitted: TopLevelKernelSpec
    model: Union["AutoGpModel", None]
    bic: float
    log_likelihood: float


KernelScores = dict[str, "ScoredKernelInfo"]


def plot_observations(X, Y, ax):
    ax.plot(X.flatten(), Y.flatten(), "k.", markersize=1)


def plot_predictions(pred_x, pred_mean_y, lower_y, upper_y, ax):
    # Plot predictive means as blue line
    ax.plot(pred_x.flatten(), pred_mean_y.flatten(), "r")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        pred_x.flatten(),
        lower_y.flatten(),
        upper_y.flatten(),
        alpha=0.5,
    )


def plot_model(model: AutoGpModel, data: Dataset):
    train_x, train_y, test_x, test_y = data

    _, ax = plt.subplots(1, 1, figsize=(14, 3))
    plot_observations(train_x, train_y, ax)
    plot_observations(test_x, test_y, ax)

    y, l, u = model.predict(data.train_x)

    plot_predictions(data.train_x, y, l, u, ax)

    y, l, u = model.predict(data.test_x)
    plot_predictions(data.test_x, y, l, u, ax)

    return ax


def score_kernel_spec(
    kernel_spec: TopLevelKernelSpec,
    data: Dataset,
    model_class: type[AutoGpModel],
    logger: Logger = None,
) -> ScoredKernelInfo:
    logger = logger or BasicLogger()
    tic = time.perf_counter()

    model = model_class(kernel_spec, data)

    model.fit(data)

    log_likelihood = model.log_likelihood()
    num_params = kernel_spec.num_params()
    bic = model.bic()

    ax = plot_model(model, data)

    fitted_spec = model.to_spec()

    spec_str = f"""{fitted_spec.spec_str(False,True)}
{fitted_spec.spec_str(False,False)} -- bic: {bic:.2f}, log likelihood: {log_likelihood:.3f}, M: {num_params}
{fitted_spec.spec_str(True,True)}"""

    ax.set_title(spec_str)

    toc = time.perf_counter()
    logger.print(f"**{fitted_spec.spec_str(False,True)}** -- fit in: {toc-tic:.3f} s")
    logger.show(plt.gcf())

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
    logger: Logger = None,
) -> KernelScores:
    logger = logger or BasicLogger()

    for spec in specs:
        spec_str = spec.spec_str(False, False)
        if spec_str in kernel_scores:
            continue
        kernel_scores[spec_str] = score_kernel_spec(spec, data, model_class, logger)
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
    kernel_scores: KernelScores = None,
    run_settings: RunSettings = RunSettings(),
    logger: Logger = None,
) -> KernelScores:

    kernel_scores = kernel_scores or {}
    logger = logger or BasicLogger()
    best_model = None

    for i in range(run_settings.max_search_depth):
        tic = time.perf_counter()
        logger.print(f"# DEPTH {i}")
        if i == 0:
            specs = run_settings.initial_kernels
        else:
            best_kernel_info = get_best_kernel_info(kernel_scores)
            residuals = cast(AutoGpModel, best_kernel_info.model).residuals()
            base_kernel_prototypes = intialize_base_kernel_prototypes_from_residuals(
                residuals, run_settings.base_kernel_prototypes
            )
            proto_str = "\n".join(str(k) for k in base_kernel_prototypes)
            logger.print(f"### prototype kernels from residuals:\n {proto_str}")
            specs = top_level_spec_swaps(
                best_kernel_info.spec_fitted,
                base_kernel_prototypes,
            )

        logger.print(f"### specs to check at depth {i}")
        logger.print("\n".join(["* " + str(sp) for sp in specs]))

        kernel_scores = score_kernel_specs(
            specs, data, model_class, kernel_scores, logger
        )

        best_kernel_info = get_best_kernel_info(kernel_scores)

        best_model = cast(AutoGpModel, best_kernel_info.model)
        best_fitted_spec = best_kernel_info.spec_fitted

        best_kernel_str = f"""Best at depth {i}:   {best_fitted_spec.spec_str(False,True)}  -- bic: {best_kernel_info.bic:.2f}, log likelihood: {best_kernel_info.log_likelihood:.3f}, M: {best_fitted_spec.num_params()}
{best_fitted_spec.spec_str(False,False)} 
{best_fitted_spec.spec_str(True,True)}"""

        logger.print("## " + best_kernel_str)

        ax = plot_model(best_model, data)
        ax.set_title(best_kernel_str)
        logger.show(plt.gcf())
        toc = time.perf_counter()
        logger.print(f"depth {i} complete in: {toc-tic:.3f} s")

    return kernel_scores


def find_best_kernel_and_predict(
    data: Dataset,
    model_class: type[AutoGpModel],
    kernel_scores: KernelScores = None,
    run_settings: RunSettings = RunSettings(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_scores = kernel_search(
        data,
        model_class,
        kernel_scores,
        run_settings,
    )
    best_kernel_info = get_best_kernel_info(kernel_scores)

    best_model = cast(AutoGpModel, best_kernel_info.model)
    return best_model.predict(data.test_x)
