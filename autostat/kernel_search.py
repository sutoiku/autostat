#  from multipledispatch import dispatch

from typing import NamedTuple, Union, cast

from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

from .kernel_tree_types import (
    Dataset,
    AdditiveKernelSpec,
    AutoGpModel,
)


from .kernel_swaps import (
    KernelInitialValues,
    additive_subtree_swaps,
    base_kernel_classes,
    addititive_base_term_with_scalar,
)


class ScoredKernelInfo(NamedTuple):
    name: str
    spec_pre_fit: AdditiveKernelSpec
    spec_fitted: AdditiveKernelSpec
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
    kernel_spec: AdditiveKernelSpec, data: Dataset, model_class: type[AutoGpModel]
) -> ScoredKernelInfo:

    model = model_class(kernel_spec, data)

    model.fit(data)

    log_likelihood = model.log_likelihood()
    num_params = kernel_spec.num_params()
    bic = model.bic()

    ax = plot_model(model, data)

    fitted_spec = model.to_spec()

    ax.set_title(
        f"""{fitted_spec.spec_str(False,True)}
{fitted_spec.spec_str(False,False)} -- bic: {bic:.2f}, log likelihood: {log_likelihood:.3f}, M: {num_params}
{fitted_spec.spec_str(True,True)}"""
    )

    return ScoredKernelInfo(
        kernel_spec.spec_str(False, False),
        kernel_spec,
        fitted_spec,
        model,
        bic,
        log_likelihood,
    )


def score_kernel_specs(
    specs: list[AdditiveKernelSpec],
    data: Dataset,
    model_class: type[AutoGpModel],
    kernel_scores: KernelScores,
):
    for spec in specs:
        spec_str = spec.spec_str(False, False)
        if spec_str in kernel_scores:
            continue
        kernel_scores[spec_str] = score_kernel_spec(spec, data, model_class)
    return kernel_scores


def init_period_from_residuals(residuals: np.ndarray) -> float:
    residuals = residuals.flatten()
    N = len(residuals)
    yf = fft(residuals)[: N // 2]
    T = 2 / N
    xf = fftfreq(N, T)[: N // 2]
    return 1 / xf[np.abs(yf) == max(np.abs(yf))][0]


def starting_kernel_specs() -> list[AdditiveKernelSpec]:
    return [addititive_base_term_with_scalar(k()) for k in base_kernel_classes]


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
    initial_kernels: list[AdditiveKernelSpec] = starting_kernel_specs(),
    kernel_scores: KernelScores = None,
    search_iterations: int = 3,
):
    kernel_scores = {} if kernel_scores is None else kernel_scores

    specs = initial_kernels
    for i in range(search_iterations):
        kernel_scores = score_kernel_specs(specs, data, model_class, kernel_scores)

        best_kernel_info = get_best_kernel_info(kernel_scores)

        best_model = cast(AutoGpModel, best_kernel_info.model)
        best_fitted_spec = best_kernel_info.spec_fitted

        residuals = best_model.residuals()
        period = init_period_from_residuals(residuals)
        initial_values = KernelInitialValues(period, np.sqrt(period / 2))

        best_kernel_str = f"""BEST ITER {i}:   {best_fitted_spec.spec_str(False,True)}  -- bic: {best_kernel_info.bic:.2f}, log likelihood: {best_kernel_info.log_likelihood:.3f}, M: {best_fitted_spec.num_params()}
{best_fitted_spec.spec_str(False,False)} 
{best_fitted_spec.spec_str(True,True)}"""

        print(best_kernel_str)

        ax = plot_model(best_model, data)
        ax.set_title(best_kernel_str)

        specs = additive_subtree_swaps(best_kernel_info.spec_fitted, initial_values)
        print("---specs next---")
        print("\n".join([str(sp) for sp in specs]))

    return kernel_scores


def find_best_kernel_and_predict(
    data: Dataset,
    model_class: type[AutoGpModel],
    initial_kernels: list[AdditiveKernelSpec] = starting_kernel_specs(),
    kernel_scores: KernelScores = None,
    search_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_scores = kernel_search(
        data,
        model_class,
        initial_kernels,
        kernel_scores,
        search_iterations,
    )
    best_kernel_info = get_best_kernel_info(kernel_scores)

    best_model = cast(AutoGpModel, best_kernel_info.model)
    # best_fitted_spec = best_kernel_info.spec_fitted
    return best_model.predict(data.test_x)
