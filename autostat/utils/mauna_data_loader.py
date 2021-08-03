import typing
from numpy.core.fromnumeric import resize
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import torch


class OpenMlData(typing.NamedTuple):
    data: np.ndarray
    target: np.ndarray


def load_mauna_loa_atmospheric_co2():
    ml_data = typing.cast(
        OpenMlData,
        fetch_openml(data_id=41187, as_frame=False),
    )
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


def scale_split_cuda(X, Y, split=0.2):
    def scale_cuda(V):
        V = MinMaxScaler((-1, 1)).fit_transform(V)
        return torch.tensor(V).flatten().double().cuda()

    N = round(max(X.shape) * (1 - split))
    X = scale_cuda(X.reshape(-1, 1))
    Y = scale_cuda(Y.reshape(-1, 1))

    x_train, x_test = X[:N], X[N:]
    y_train, y_test = Y[:N], Y[N:]

    return x_train, x_test, y_train, y_test


def load_mauna_torch():
    X_raw, Y_raw = load_mauna_loa_atmospheric_co2()
    train_x, test_x, train_y, test_y = scale_split_cuda(X_raw, Y_raw)

    f, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy())
    ax.plot(test_x.cpu().numpy(), test_y.cpu().numpy())

    return train_x, test_x, train_y, test_y


def end_block_split(X, split):
    N = round(max(X.shape) * (1 - split))
    return X[:N], X[N:]


def banded_split(X, end_prop=0.1, inner_prop=0.0, num_inner_test_bands=3):
    # this function splits that data into test and train sets with a banded structure,
    # having two test bands on each end (with a total proportion `end_prop` of data points)
    # and `num_inner_test_bands` number of test intervals in the interior of the dataset

    N = max(X.shape)
    N_end_band = end_prop * N * 0.5
    N_inner = N - 2 * N_end_band
    N_inner_test = N_inner * inner_prop
    N_inner_train = N_inner * (1 - inner_prop)
    N_inner_test_band = N_inner_test / num_inner_test_bands
    N_inner_train_band = N_inner_train / (num_inner_test_bands + 1)

    split_indices = np.cumsum(
        [
            N_end_band,
            *[N_inner_train_band, N_inner_test_band] * num_inner_test_bands,
            N_inner_train_band,
        ]
    ).astype(int)

    data_bands = np.split(X, split_indices)
    train = np.concatenate(data_bands[1::2])
    test = np.concatenate(data_bands[::2])

    return train, test


def scale_split(X, Y, split=0.2, rescale=True, banded=False):
    def scale(V):
        return MinMaxScaler((-1, 1)).fit_transform(V)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    if rescale:
        X = scale(X)
        Y = scale(Y)

    splitter = banded_split if banded else end_block_split

    x_train, x_test = splitter(X, split)
    y_train, y_test = splitter(Y, split)

    return x_train, x_test, y_train, y_test


def load_mauna_numpy(plot=True, rescale=True, banded=False):
    X_raw, Y_raw = load_mauna_loa_atmospheric_co2()
    x_train, x_test, y_train, y_test = scale_split(
        X_raw, Y_raw, rescale=rescale, banded=banded
    )
    if plot:
        f, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(x_train, y_train)
        ax.plot(x_test, y_test)
    return x_train, x_test, y_train, y_test
