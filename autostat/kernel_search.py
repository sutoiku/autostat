#  from multipledispatch import dispatch

from typing import NamedTuple, TypeVar, Union, cast

from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

from .kernel_tree_types import (
    Dataset,
    KernelSpec,
    ArithmeticKernelSpec,
    BaseKernelSpec,
    BaseKernelSpecTypes,
    AdditiveKernelSpec,
    ProductKernelSpec,
    ProductOperandSpec,
    RBFKernelSpec,
    LinearKernelSpec,
    PeriodicKernelSpec,
    RQKernelSpec,
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


def score_kernel_spec(
    kernel_spec: AdditiveKernelSpec, data: Dataset, model_class: type[AutoGpModel]
) -> ScoredKernelInfo:

    model = model_class(kernel_spec, data)
    train_x, train_y, test_x, test_y = data
    print("---pre-fit---")
    print(model.gp.kernel)

    model.fit(data)

    log_likelihood = model.log_likelihood()
    num_params = kernel_spec.num_params()
    bic = model.bic()

    print("---post-fit---")
    print(model.gp.kernel_)
    print("\n")

    _, ax = plt.subplots(1, 1, figsize=(14, 3))
    plot_observations(train_x, train_y, ax)
    plot_observations(test_x, test_y, ax)

    y, l, u = model.predict(data.train_x)

    plot_predictions(data.train_x, y, l, u, ax)
    # model.print_fitted_kernel()

    y, l, u = model.predict(data.test_x)
    plot_predictions(data.test_x, y, l, u, ax)

    ax.set_title(
        f"""{kernel_spec.spec_str(False,True)}
{kernel_spec.spec_str(False,False)} -- bic: {bic:.2f}, log likelihood: {log_likelihood:.3f}, M: {num_params}
{kernel_spec.spec_str(True,True)}"""
    )

    return ScoredKernelInfo(
        kernel_spec.spec_str(False, False), kernel_spec, model.to_spec(), model, bic
    )


def score_kernel_specs(
    specs: list[AdditiveKernelSpec],
    data: Dataset,
    model_class: type[AutoGpModel],
    kernel_scores: KernelScores = KernelScores(),
):
    for spec in specs:
        spec_str = repr(spec)
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


def kernel_search(
    data: Dataset,
    model_class: type[AutoGpModel],
    initial_kernels: list[AdditiveKernelSpec] = starting_kernel_specs(),
    search_iterations: int = 3,
    kernel_scores: KernelScores = KernelScores(),
):
    # kernel_scores = KernelScores() if kernel_scores is None else kernel_scores
    specs = initial_kernels
    for i in range(search_iterations):
        kernel_scores = score_kernel_specs(specs, data, model_class, kernel_scores)
        best_kernel_name, best_kernel_info = min(
            kernel_scores.items(), key=lambda name_score_info: name_score_info[1].bic
        )

        best_kernel = kernel_scores[best_kernel_name]

        residuals = cast(AutoGpModel, best_kernel.model).residuals()
        period = init_period_from_residuals(residuals)
        initial_values = KernelInitialValues(period, np.sqrt(period / 2))

        print(
            f"""
---best kernel iter {i};  (bic: {best_kernel_info.bic}) ---
{str(best_kernel_info.spec_fitted)}
"""
        )

        specs = additive_subtree_swaps(best_kernel.spec_fitted, initial_values)
        print("---specs next---")
        print("\n".join([str(sp) for sp in specs]))

    return kernel_scores

    # pred_y, bic, log_likelihood, num_params = overall_best_model_info
    # f, overall_winner_ax = plt.subplots(1, 1, figsize=(14, 3))
    # plot_observations(train_x, train_y, overall_winner_ax)
    # plot_predictions(train_x, pred_y, overall_winner_ax)
    # overall_winner_ax.set_title(
    #     f"{winner_kernel} -- bic: {bic:.2f}, mll: {log_likelihood:.4f}, M: {num_par
