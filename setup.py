from setuptools import setup, find_packages

setup(
    name="adaptive-elicitation",  # name for pip, doesn't affect imports
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)