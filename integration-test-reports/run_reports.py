from autostat.run_settings import RunSettings
from autostat.utils.logger import Logger
from html_reports import Report
import matplotlib.pyplot as plt
import scipy.io as io


from autostat.utils.mauna_data_loader import load_mauna_numpy, scale_split


from autostat.kernel_search import kernel_search

from autostat.kernel_tree_types import (
    NpDataSet as NpDataSet,
)

from autostat.sklearn.model_wrapper import SklearnGPModel

from datetime import datetime

import csv
import numpy as np

import os
import time

print(os.getcwd())

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def timestamp():
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S")


class HtmlLogger:
    def __init__(self, report) -> None:
        self.report = report

    def print(self, s: str) -> None:
        self.report.add_markdown(s.replace("\n", "\n\n"))

    def show(self, fig) -> None:
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.figure(fig.number)
        self.report.add_figure(options="width=100%")
        plt.close(plt.gcf())


report = Report()
logger = HtmlLogger(report)


def title_separator(title):
    logger.print("-----------")
    logger.print("-----------")
    logger.print("-----------")
    logger.print(f"# ***{title}***")


def run_report_fn(dataset_name: str, report_fn, run_settings):

    title_separator(f"Dataset: {dataset_name}")
    tic = time.perf_counter()
    report_fn(run_settings)
    toc = time.perf_counter()
    logger.print(f"Total time for {dataset_name}: {toc-tic:.3f} s")


def matlab_data_report_fn(file_path):

    data = io.loadmat(file_path)

    def runner(run_settings):
        train_x, test_x, train_y, test_y = scale_split(
            np.array(data["X"]), np.array(data["y"]), split=0.01
        )

        d = NpDataSet(train_x, train_y, test_x, test_y)
        kernel_scores = kernel_search(
            d, SklearnGPModel, run_settings=run_settings, logger=logger
        )

    return runner


matlab_data_path = "data/"

files_sorted_by_num_data_points = [
    "01-airline.mat",
    # "07-call-centre.mat",
    # "08-radio.mat",
    # "04-wheat.mat",
    # "02-solar.mat",
    # "11-unemployment.mat",
    # "10-sulphuric.mat",
    # "09-gas-production.mat",
    # "03-mauna.mat",
    # "13-wages.mat",
    # "06-internet.mat",
    # "05-temperature.mat",
    # "12-births.mat",
]

if __name__ == "__main__":
    print("starting report")
    # run_report_fn("Mauna Loa", run_mauna_loa)
    # run_report_fn("Air passengers", run_air_passengers)
    run_settings = RunSettings()
    for file in files_sorted_by_num_data_points:
        run_report_fn(
            file, matlab_data_report_fn(matlab_data_path + file), run_settings
        )

    report.write_report(filename=f"reports/report_{timestamp()}.html")
    print("report done")
