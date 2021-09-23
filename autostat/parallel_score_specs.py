import ray

from autostat.decomposition import decompose_spec
import time
import typing as ty
import numpy as np

from .auto_gp_model import AutoGpModel
from .kernel_specs import TopLevelKernelSpec
from .dataset_adapters import Dataset
from .utils.logger import JupyterLogger, Logger, QueingLogger
from .run_settings import RunSettings
from .plots import plot_decomposition, plot_model

# from .kernel_search import ScoredKernelInfo, KernelScores


class ScoreKernelSpecArgs(ty.NamedTuple):
    kernel_spec: TopLevelKernelSpec
    data: Dataset
    model_class: type[AutoGpModel]
    run_settings: RunSettings
    logger: Logger


class ScoredKernelInfo(ty.NamedTuple):
    name: str
    spec_pre_fit: TopLevelKernelSpec
    spec_fitted: TopLevelKernelSpec
    model: AutoGpModel
    bic: float
    log_likelihood: float


KernelScores = dict[str, "ScoredKernelInfo"]


@ray.remote
def score_kernel_spec(args: ScoreKernelSpecArgs) -> tuple[ScoredKernelInfo, Logger]:
    (kernel_spec, data, model_class, run_settings, logger) = args

    logger = logger or JupyterLogger()
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

    return (
        ScoredKernelInfo(
            kernel_spec.spec_str(False, False),
            kernel_spec,
            fitted_spec,
            model,
            bic,
            log_likelihood,
        ),
        logger,
    )


# g = ray.remote(f)

# t0 = time.time()
# results = ray.get([g.remote(s) for s in specs])
# for result in results:
#     print(result)


def parallel_score_kernel_specs(
    specs: list[TopLevelKernelSpec],
    data: Dataset,
    model_class: type[AutoGpModel],
    kernel_scores: KernelScores,
    run_settings: RunSettings,
    logger: Logger = None,
) -> KernelScores:
    logger = logger or JupyterLogger()

    specs = [spec for spec in specs if spec.spec_str(False, False) not in kernel_scores]

    ray_data = ray.put(data)
    ray_model_class = ray.put(model_class)
    ray_run_settings = ray.put(run_settings)

    # score_args = [
    #     (spec, ray_data, ray_model_class, ray_run_settings, QueingLogger())
    #     for spec in specs
    # ]
    score_args = [
        (spec, data, model_class, run_settings, QueingLogger()) for spec in specs
    ]

    kernel_scores_and_logs: list[tuple[ScoredKernelInfo, QueingLogger]] = ray.get(
        [score_kernel_spec.remote(args) for args in score_args]
    )
    for _, logs in kernel_scores_and_logs:
        logs.flush_queue_to_logger(logger)

    kernel_scores = {score.name: score for score, _ in kernel_scores_and_logs}

    return kernel_scores
