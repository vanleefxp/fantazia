from setuptools import setup, find_packages

setup (
    name = "fantazia",
    version = "0.0.1",
    author = "F. X. P.",
    author_email = "litran39@hotmail.com",
    description = "A package for math-based music theory computation.",
    long_description = open ( 'README.md' ).read ( ),
    long_description_content_type = "text/markdown",
    packages = find_packages ( ),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.12",
)