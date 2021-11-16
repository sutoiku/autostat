import matplotlib.pyplot as plt
from ..logger import Base64PngString

# from .. import logger


class TestBase64PngString:
    def test_is_str(self):
        assert isinstance(Base64PngString("foo"), str)

    def test_from_fig(self):
        plt.plot(range(5))
        fig = plt.gcf()
        assert isinstance(Base64PngString.from_fig(fig), str)

    def test_from_fig_to_html(self):
        plt.plot(range(5))
        fig = plt.gcf()
        b64 = Base64PngString.from_fig(fig).to_html_img_str()
        assert b64[0] == "<" and b64[-1] == ">"
