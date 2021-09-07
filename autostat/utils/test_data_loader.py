import scipy.io as io
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from ..dataset_adapters import Dataset


files_sorted_by_num_data_points = [
    "01-airline.mat",
    "07-call-centre.mat",
    "08-radio.mat",
    "04-wheat.mat",
    "02-solar.mat",
    "11-unemployment.mat",
    "10-sulphuric.mat",
    "09-gas-production.mat",
    "03-mauna.mat",
    "13-wages.mat",
    "06-internet.mat",
    "05-temperature.mat",
    "12-births.mat",
]

file_names = [
    "01-airline.mat",
    "02-solar.mat",
    "03-mauna.mat",
    "04-wheat.mat",
    "05-temperature.mat",
    "06-internet.mat",
    "07-call-centre.mat",
    "08-radio.mat",
    "09-gas-production.mat",
    "10-sulphuric.mat",
    "11-unemployment.mat",
    "12-births.mat",
    "13-wages.mat",
]


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


def load_test_dataset(
    path: str, file_num: int, split: float = 0.2, **kwargs
) -> Dataset:
    path += file_names[file_num - 1]
    data = io.loadmat(path)

    train_x, test_x, train_y, test_y = scale_split(
        np.array(data["X"]), np.array(data["y"]), split=split, **kwargs
    )

    return Dataset(train_x, train_y, test_x, test_y)
