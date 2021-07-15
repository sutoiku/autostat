from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="autostat",
    version="0.0.1",
    description="An implementation of the Automatic Statistician kernel selection algorithm",
    long_description_content_type="text/markdown",
    long_description=README,
    license="MIT",
    packages=find_packages(),
    author="Brendan Colloran, Sutoiku Inc.",
    author_email="brendan@stoic.com",
    keywords=["Automatic Statistician", "Gaussian Process"],
    url="https://github.com/sutoiku/autostat",
    # download_url='https://pypi.org/.../.../'
)

install_requires = ["scikit-learn", "matplotlib", "numpy", "torch"]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)
