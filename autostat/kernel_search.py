from autostat.decomposition import decompose_spec
import time
from typing import NamedTuple, cast
import ray
import numpy as np

from .auto_gp_model import AutoGpModel
from .kernel_specs import TopLevelKernelSpec
from .dataset_adapters import Dataset
from .utils.logger import JupyterLogger, Logger, SerializedLogQueue
from .kernel_swaps import top_level_spec_swaps
from .run_settings import RunSettings, Backend
from .plots import plot_decomposition, plot_model
from .expand_spec import expand_spec
from .kernel_spec_initialization import intialize_base_kernel_prototypes_from_residuals
from .score_specs import (
    score_kernel_specs,
    ScoredKernelInfo,
    KernelScores,
)
from .constraints import set_constraints_on_spec


from autostat.sklearn.model_wrapper import SklearnGPModel
from autostat.gpytorch.model_wrapper import GpytorchGPModel


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
    run_settings: RunSettings,
    kernel_scores: KernelScores = None,
    logger: Logger = None,
) -> KernelScores:

    kernel_scores = kernel_scores or {}
    logger = logger or JupyterLogger()
    best_model = None

    # FIXME move to general init? or perhaps this IS the actual init...
    if run_settings.use_parallel:
        ray.init(num_cpus=run_settings.num_cpus, ignore_reinit_error=True)
    if run_settings.backend == Backend.GPYTORCH:
        model_class = GpytorchGPModel
    else:
        model_class = SklearnGPModel

    logger.print(str(run_settings.initial_kernels))

    for i in range(run_settings.max_search_depth):
        tic = time.perf_counter()
        logger.print(f"# DEPTH {i}")

        # set up kernels for this depth
        if i == 0:
            new_specs = run_settings.initial_kernels
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
            new_specs = [
                set_constraints_on_spec(spec, run_settings.base_kernel_prototypes)
                for spec in specs
                if spec.spec_str(False, False) not in kernel_scores
            ]

        logger.print(f"### specs to check at depth {i}")
        logger.print("\n".join(["* " + str(sp) for sp in new_specs]))

        # score the kernels for this depth
        kernel_scores = score_kernel_specs(
            new_specs, data, model_class, kernel_scores, run_settings, logger
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
    run_settings: RunSettings,
    kernel_scores: KernelScores = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_scores = kernel_search(
        data, run_settings=run_settings, kernel_scores=kernel_scores
    )
    best_kernel_info = get_best_kernel_info(kernel_scores)

    best_model = cast(AutoGpModel, best_kernel_info.model)
    return best_model.predict(data.test_x)
