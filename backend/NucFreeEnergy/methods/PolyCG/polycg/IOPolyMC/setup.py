from setuptools import setup

setup(
    name="iopolymc",
    version="0.0.3",
    description="A module to read and write PolyMC input and output files",
    url="https://github.com/esasen/IOPolyMC",
    author="Enrico Skoruppa",
    author_email="enrico dot skoruppa at gmail dot com",
    license="MIT",
    packages=["iopolymc"],
    package_dir={
        "iopolymc": "iopolymc",
    },
    include_package_data=True,
    package_data={"": ["database/*"]},
    install_requires=[
        "numpy>=1.20",
    ],
    zip_safe=False,
)
