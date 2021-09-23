from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.core.display import HTML, display

import typing as ty

import matplotlib.pyplot as plt
import base64
from io import BytesIO


class HTMLStr(str):
    def __new__(cls, str):
        return super().__new__(cls, str)


class Base64PngString(str):
    def __new__(cls, str):
        return super().__new__(cls, str)

    @classmethod
    def from_fig(cls, fig: Figure) -> Base64PngString:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format="png", bbox_inches="tight", facecolor="white")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        return cls(encoded)

    def to_html_img_str(self) -> HTMLStr:
        return HTMLStr(f'<img src="data:image/png;base64,{self}" width="100%">')


def fig_to_html(fig: Figure) -> str:
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png", bbox_inches="tight")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" width="100%">'


class Logger(ty.Protocol):
    def print(self, s: str) -> None:
        ...

    def show(self, fig: ty.Union[Figure, HTMLStr]) -> None:
        ...


class JupyterLogger:
    def print(self, s: str) -> None:
        print(s)

    def show(self, fig: ty.Union[Figure, HTMLStr]) -> None:
        if isinstance(fig, Figure):
            plt.figure(fig.number)
            plt.show()
            plt.close(fig)
        elif isinstance(fig, HTMLStr):
            display(HTML(fig))
        else:
            raise ValueError(
                f"JupyterLogger can only `show` Figures and HTMLStrs, got {type(fig)}"
            )


class QueingLogger:
    def __init__(
        self,
    ) -> None:
        self.log_queue: list[ty.Union[str, HTMLStr]] = []

    def print(self, s: str) -> None:
        self.log_queue.append(s)

    def show(self, fig: Figure) -> None:
        # self.log_queue.append(fig)
        self.log_queue.append(Base64PngString.from_fig(fig).to_html_img_str())
        plt.close(fig)

    def flush_queue_to_logger(self, logger: Logger) -> None:
        while self.log_queue:
            item = self.log_queue.pop(0)
            if isinstance(item, HTMLStr):
                logger.show(item)
            elif isinstance(item, str):
                logger.print(item)
            else:
                raise ValueError("QueingLogger only supports strings and Figures")
