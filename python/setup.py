from setuptools import setup, find_packages
import os

setup(
    name="domino",
    packages=find_packages(),
)

os.chdir("../build")
setup(
    name="dominoc",
    packages=[""],
    package_dir={"": "."},
    package_data={"": ["dominoc*.so"]},
)

os.chdir("../testing/mculib/python")
setup(
    name="mculib",
    packages=find_packages(),
)
