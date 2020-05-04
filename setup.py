# pylint: disable=missing-docstring
from setuptools import setup, find_packages


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


setup(
    name="gym-industrial",
    version="0.0.1",
    author="Ângelo G. Lovatto",
    author_email="angelolovatto@gmail.com",
    description="Industrial Benchmark for OpenAI Gym",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/angelolovatto/gym-industrial",
    packages=find_packages(),
    install_requires=["gym"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
