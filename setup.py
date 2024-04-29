# setup.py 

from setuptools import setup, find_packages

setup(
    name="surrogateopt_openmdao",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openmdao", 
        "SMT", 
        "pySOT"
    ],
    author="Christian Hauschel",
    description="Surrogate optimization Driver for OpenMDAO",
)