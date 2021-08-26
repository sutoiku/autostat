from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import typing as ty

from .auto_gp_model import AutoGpModel
from .dataset_adapters import Dataset
from .decomposition import DecompositionData


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

    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    plot_observations(train_x, train_y, ax)
    plot_observations(test_x, test_y, ax)

    y, l, u = model.predict(data.train_x)

    plot_predictions(data.train_x, y, l, u, ax)

    y, l, u = model.predict(data.test_x)
    plot_predictions(data.test_x, y, l, u, ax)

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
