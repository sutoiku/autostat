import traceback
from autostat.run_settings import KernelSearchSettings, Backend

from autostat.kernel_search import kernel_search, get_best_kernel_info
from autostat.dataset_adapters import Dataset
from autostat.utils.data_prep import scale_split
from autostat.test_data.test_data_loader import load_matlab_test_data_by_file_num

from html_reports import Report
from markdown import markdown
import matplotlib.pyplot as plt

from datetime import datetime

import os
import time

import random
import numpy as np

print(os.getcwd())

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def timestamp():
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S")


class HtmlLogger:
    def __init__(self, report: Report) -> None:
        self.report = report

    def print(self, s: str) -> None:
        self.report.add_markdown(
            s.replace("\n", "\n\n")
            # .replace("<", "&lt;").replace(">", "&gt;")
        )

    def prepend(self, s: str) -> None:
        md = markdown(s, extensions=["fenced_code", "codehilite"])
        self.report.body = [md] + self.report.body

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


matlab_data_path = "data/"

files_sorted_by_num_data_points = [
    "01-airline.mat",
    # "07-call-centre.mat",
    "08-radio.mat",
    # "04-wheat.mat",
    "02-solar.mat",
    # "11-unemployment.mat",
    # "10-sulphuric.mat",
    # # "09-gas-production.mat",
    "03-mauna.mat",
    # # "13-wages.mat",
    "06-internet.mat",
    # "05-temperature.mat",
    "12-births.mat",
]

if __name__ == "__main__":

    random.seed(1234)
    np.random.seed(1234)
    print("starting report")

    run_settings = KernelSearchSettings(
        max_search_depth=4,
        expand_kernel_specs_as_sums=True,
        num_cpus=12,
        use_gpu=False,
        use_parallel=True,
        gpu_memory_share_needed=0.45,
        backend=Backend.SKLEARN,
        cv_split=0.15,
    ).replace_base_kernels_by_names(["PER", "LIN", "RBF"])

    logger.print(str(run_settings))

    logger.print("\n" + str(run_settings.asdict()))

    prediction_scores = []

    for file_name in files_sorted_by_num_data_points:
        file_num = int(file_name[:2])

        x, y = load_matlab_test_data_by_file_num(file_num)
        dataset = Dataset(*scale_split(x, y, split=run_settings.cv_split))

        run_settings = run_settings.replace_kernel_proto_constraints_using_dataset(
            dataset
        )

        title_separator(f"Dataset: {file_name}")
        try:
            tic = time.perf_counter()
            kernel_scores = kernel_search(
                dataset, run_settings=run_settings, logger=logger
            )
            toc = time.perf_counter()
            best_kernel_info = get_best_kernel_info(kernel_scores)
            prediction_scores.append(best_kernel_info.log_likelihood_test)

            logger.print(f"best_kernel_info {str(best_kernel_info)}")

            logger.print(f"Total time for {file_name}: {toc-tic:.3f} s")
        except Exception as e:
            logger.print("##ERROR")
            logger.print(repr(e))
            logger.print(traceback.format_exc())

    logger.prepend(f"prediction_scores: {str(prediction_scores)}")
    logger.prepend(f"sum prediction_scores: {str(sum(prediction_scores))}")

    report.write_report(filename=f"reports/report_{timestamp()}.html")
    print("report done")
