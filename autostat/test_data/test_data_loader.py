import scipy.io as io
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


def load_matlab_test_data_by_file_num(file_num: int = 3):
    # NOTE: default `file_num: int = 3` is Mauna Loa data
    from importlib.resources import files
    from ..test_data import matlab

    file_path = files(matlab).joinpath(file_names[file_num - 1])
    data = io.loadmat(file_path)

    return np.array(data["X"]), np.array(data["y"])
