import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import typing as ty


class Logger(ty.Protocol):
    def print(self, s: str) -> None:
        ...

    def show(self, fig: Figure) -> None:
        ...


class BasicLogger:
    def print(self, s: str) -> None:
        print(s)

    def show(self, fig: Figure) -> None:
        plt.figure(fig.number)
        plt.show()
        plt.close(plt.gcf())


class QueingLogger:
    def __init__(self, non_q_logger: Logger) -> None:
        self.log_queue: list[ty.Union[str, Figure]] = []
        self.non_q_logger = non_q_logger

    def print(self, s: str) -> None:
        self.log_queue.append(s)

    def show(self, fig: Figure) -> None:
        self.log_queue.append(fig)

    def flush_queue(self) -> None:
        while self.log_queue:
            item = self.log_queue.pop(0)
            if isinstance(item, str):
                self.non_q_logger.print(item)
            elif isinstance(item, Figure):
                self.non_q_logger.show(item)
            else:
                raise ValueError("QueingLogger only supports strings and Figures")
