# setup.py 

from setuptools import setup, find_packages

setup(
    name="surrogateopt_openmdao",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "openmdao", 
        "SMT", 
        "pySOT @ git+https://github.com/christianhauschel/pySOT",
        "proplot",
        "matplotlib<3.5"
    ],
    author="Christian Hauschel",
    description="Surrogate optimization Driver for OpenMDAO",
)