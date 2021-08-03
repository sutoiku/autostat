import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Any, Generic, NamedTuple, Union, Protocol, TypeVar, Type


class Logger(Protocol):
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
