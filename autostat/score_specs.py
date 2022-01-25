import ray
from ray.util.queue import Queue
from enum import Enum

from autostat.decomposition import decompose_spec
import time
import typing as ty
import numpy as np
import random
import os

from .auto_gp_model import CompositionalGPModel
from .kernel_specs import TopLevelKernelSpec
from .dataset_adapters import Dataset
from .utils.logger import JupyterLogger, Logger, SerializedLogQueue
from .run_settings import KernelSearchSettings
from .plots import plot_decomposition, plot_model

# from .kernel_search import ScoredKernelInfo, KernelScores


class ScoreKernelSpecArgs(ty.NamedTuple):
    kernel_spec: TopLevelKernelSpec
    data: Dataset
    model_class: type[CompositionalGPModel]
    run_settings: KernelSearchSettings
    logger: SerializedLogQueue


class ScoredKernelInfo(ty.NamedTuple):
    name: str
    spec_pre_fit: TopLevelKernelSpec
    spec_fitted: TopLevelKernelSpec
    model: CompositionalGPModel
    bic: float
    log_likelihood: float
    log_likelihood_test: float

    # def clear_model


class TaskState(Enum):
    RUNNING = 1
    FULL = 0


KernelScores = dict[str, "ScoredKernelInfo"]


def score_kernel_spec(
    args: ScoreKernelSpecArgs,
) -> tuple[ty.Union[ScoredKernelInfo, None], SerializedLogQueue]:
    (kernel_spec, data, model_class, run_settings, logger) = args
    random.seed(run_settings.seed)
    np.random.seed(run_settings.seed)

    # try:
    tic = time.perf_counter()

    model = model_class(kernel_spec, data, run_settings=run_settings)

    model.fit(data)

    log_likelihood = model.log_likelihood()
    num_params = kernel_spec.num_params()
    bic = model.bic()

    fig, ax = plot_model(model, data)

    prediction_log_likelihood = model.log_likelihood_test()

    fitted_spec = model.to_spec()

    spec_str = f"""{fitted_spec.spec_str(False,True)}   --  {fitted_spec.spec_str(False,False)}
bic: {bic:.2f}, M: {num_params}, log likelihood: {log_likelihood:.3f}, pred. score: {prediction_log_likelihood:.3f}
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
            prediction_log_likelihood,
        ),
        logger,
    )


#     except Exception as e:
#         logger.print(
#             f"""
# ## Failed to fit {kernel_spec.schema()}
# Error:
# {str(e)}
# """
#         )
#         return (None, logger)


def score_kernel_specs(
    specs: list[TopLevelKernelSpec],
    data: Dataset,
    model_class: type[CompositionalGPModel],
    kernel_scores: KernelScores,
    run_settings: KernelSearchSettings,
    logger: Logger = None,
) -> KernelScores:
    logger = logger or JupyterLogger()

    score_args = [
        ScoreKernelSpecArgs(spec, data, model_class, run_settings, SerializedLogQueue())
        for spec in specs
    ]

    if run_settings.use_parallel:
        gpu_needed = run_settings.gpu_memory_share_needed if run_settings.use_gpu else 0

        parallel_score_kernel_spec = ray.remote(
            num_cpus=1,
            num_gpus=gpu_needed,
        )(score_kernel_spec)

        kernel_scores_and_logs: list[
            tuple[ty.Union[ScoredKernelInfo, None], SerializedLogQueue]
        ] = ray.get([parallel_score_kernel_spec.remote(args) for args in score_args])
    else:
        kernel_scores_and_logs = [score_kernel_spec(args) for args in score_args]

    kernel_scores = {}
    for score_info, logs in kernel_scores_and_logs:
        logs.flush_queue_to_logger(logger)
        if score_info is not None:
            kernel_scores[score_info.name] = score_info

    return kernel_scores
