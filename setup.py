from setuptools import setup, find_packages

setup(
    name="bettyng",
    version="0.1.0",
    author="Roman Alberto Velez Jimenez",
    author_email="roman.velez.jimenez@gmail.com",
    description="Python package to generate betting portfolios",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
