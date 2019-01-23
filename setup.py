import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cor_control_benchmarks",
    version="0.3.1",
    author="Tim de Bruin",
    author_email="t.d.debruin@tudelft.nl",
    description="This repository contains python (3.6+) implementations of several control benchmarks. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timdebruin/CoR-control-benchmarks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)