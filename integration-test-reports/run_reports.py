from autostat.run_settings import RunSettings, Backend


from autostat.kernel_search import kernel_search

from autostat.dataset_adapters import Dataset
from autostat.utils.test_data_loader import load_test_dataset

from html_reports import Report
import matplotlib.pyplot as plt

from datetime import datetime

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
        self.report.add_markdown(
            s.replace("\n", "\n\n")
            # .replace("<", "&lt;").replace(">", "&gt;")
        )

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
    # "08-radio.mat",
    "04-wheat.mat",
    # "02-solar.mat",
    "11-unemployment.mat",
    # "10-sulphuric.mat",
    # "09-gas-production.mat",
    "03-mauna.mat",
    # "13-wages.mat",
    # "06-internet.mat",
    "05-temperature.mat",
    "12-births.mat",
]

if __name__ == "__main__":
    print("starting report")

    run_settings = RunSettings(
        max_search_depth=3,
        expand_kernel_specs_as_sums=False,
        num_cpus=12,
        use_gpu=False,
        use_parallel=True,
        gpu_memory_share_needed=0.45,
        backend=Backend.SKLEARN,
    ).replace_base_kernels_by_names(["PER", "LIN", "RBF"])

    logger.print(str(run_settings))

    logger.print("\n" + str(run_settings.asdict()))

    for file_name in files_sorted_by_num_data_points:
        file_num = int(file_name[:2])

        dataset = load_test_dataset(matlab_data_path, file_num, split=0.1)

        run_settings = run_settings.replace_kernel_proto_constraints_using_dataset(
            dataset
        )

        title_separator(f"Dataset: {file_name}")
        tic = time.perf_counter()
        kernel_scores = kernel_search(dataset, run_settings=run_settings, logger=logger)
        toc = time.perf_counter()
        logger.print(f"Total time for {file_name}: {toc-tic:.3f} s")

    report.write_report(filename=f"reports/report_{timestamp()}.html")
    print("report done")
