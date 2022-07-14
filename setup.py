import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py12box_invert",
    version="0.0.1",
    author="Matt Rigby",
    author_email="matt.rigby@bristol.ac.uk",
    description="AGAGE 12-box model inversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)