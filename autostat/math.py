import numpy as np


def calc_bic(num_params: float, data_size: float, log_likelihood: float) -> float:
    return -2 * log_likelihood + num_params * np.log(data_size)
