from setuptools import setup, find_packages

setup(
    name="ssi_scanpy_tools",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scanpy",
        "anndata",
        "numpy",
        "pandas",
    ],
    author="Tu Hu",
    description="SSI tools for Scanpy and AnnData processing.",
    license="MIT",
)
