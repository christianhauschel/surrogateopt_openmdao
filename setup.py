# setup.py 

from setuptools import setup, find_packages

setup(
    name="surrogateopt_openmdao",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        "openmdao", 
        "SMT", 
        "pySOT @ git+https://github.com/christianhauschel/pySOT",
        "proplot>=0.9.7",
        "matplotlib<3.5",
        "dill>=0.3.8",
    ],
    author="Christian Hauschel",
    description="Surrogate optimization Driver for OpenMDAO",
)