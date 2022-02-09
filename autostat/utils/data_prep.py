from sklearn.preprocessing import MinMaxScaler
import numpy as np


def end_block_split(X, split: float):
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


def scale_split(X, Y, split: float = 0.2, rescale=True, banded=False):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    splitter = banded_split if banded else end_block_split

    train_x, x_test = splitter(X, split)
    train_y, y_test = splitter(Y, split)

    def scale(x1, x2):
        scaler = MinMaxScaler((-1, 1))
        scaler.fit(x1)
        return scaler.transform(x1), scaler.transform(x2)

    if rescale:
        train_x, x_test = scale(train_x, x_test)
        train_y, y_test = scale(train_y, y_test)

    return train_x, train_y, x_test, y_test
