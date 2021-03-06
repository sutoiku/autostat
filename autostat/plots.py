from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import typing as ty

from .compositional_gp_model import CompositionalGPModel
from .dataset_adapters import Dataset
from .decomposition import DecompositionData


def plot_observations(X, Y, ax):
    ax.plot(X.flatten(), Y.flatten(), "k.", markersize=1)


def plot_predictions(pred_x, pred_mean_y, y_std, ax):
    # Plot predictive means as blue line
    ax.plot(pred_x.flatten(), pred_mean_y.flatten(), "r")
    # Shade between the lower and upper confidence bounds
    ax.fill_between(
        pred_x.flatten(),
        pred_mean_y - 2 * y_std,
        pred_mean_y + 2 * y_std,
        alpha=0.5,
    )


def plot_model(model: CompositionalGPModel, data: Dataset):
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    plot_observations(data.train_x, data.train_y, ax)
    plot_observations(data.test_x, data.test_y, ax)

    y, y_std, _ = model.predict_train()

    plot_predictions(data.train_x, y, y_std, ax)

    y, y_std, _ = model.predict_test()
    plot_predictions(data.test_x, y, y_std, ax)

    return fig, ax


def plot_decomposition(d: DecompositionData):
    num_components = len(d.components)

    fig, axes = plt.subplots(
        nrows=num_components, sharex=True, figsize=(14, 3 * num_components)
    )
    if isinstance(axes, plt.Axes):
        axes = [axes]

    axes = ty.cast(list[plt.Axes], axes)

    for (spec, y_comp, y_std), ax in zip(d.components, axes):
        ax.plot(d.x, y_comp)
        # print("y_comp", y_comp.shape, y_comp)
        # print("y_std", y_std.shape, y_std)
        ax.fill_between(
            d.x.flatten(),
            y_comp.flatten() - 2 * y_std.flatten(),
            y_comp.flatten() + 2 * y_std.flatten(),
            alpha=0.5,
        )
        ax.set_title(spec.spec_str(True, True))

    return fig
