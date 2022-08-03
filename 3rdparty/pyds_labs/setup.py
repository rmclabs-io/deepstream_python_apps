from setuptools import find_packages, setup

setup(
    name="pyds_labs",
    version="0.1.0",
    author="Pablo Woolvett",
    author_email="pwoolvett@rmc.cl",
    description="Deepstream extensions",
    packages=find_packages(),
    install_requires=["pyds==1.1.3"],
)
