#! /usr/bin/env python
"""Toolbox for ensemble learning on class-imbalanced dataset."""

# import codecs

import io
import os

from setuptools import find_packages, setup, Command

# get __version__ from _version.py
ver_file = os.path.join("imbens", "_version.py")
with open(ver_file) as f:
    exec(f.read())

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

DISTNAME = "imbalanced-ensemble"
DESCRIPTION = "Toolbox for ensemble learning on class-imbalanced dataset."

# with codecs.open("README.rst", encoding="utf-8-sig") as f:
#     LONG_DESCRIPTION = f.read()

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

AUTHOR = "Zhining Liu"
AUTHOR_EMAIL = "zhining.liu@outlook.com"
MAINTAINER = "Zhining Liu"
MAINTAINER_EMAIL = "zhining.liu@outlook.com"
URL = "https://github.com/ZhiningLiu1998/imbalanced-ensemble"
PROJECT_URLS = {
    "Documentation": "https://imbalanced-ensemble.readthedocs.io/",
    "Source": "https://github.com/ZhiningLiu1998/imbalanced-ensemble",
    "Tracker": "https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues",
    "Changelog": "https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html",
    "Download": "https://pypi.org/project/imbalanced-ensemble/#files",
}
LICENSE = "MIT"
VERSION = __version__
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
]
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "flake8",
    ],
    "test": [
        "pytest",
        "pytest-cov",
    ],
    "doc": [
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "pydata-sphinx-theme",
        "numpydoc",
        "sphinxcontrib-bibtex",
        "torch",
        "pytest",
    ],
}

setup(
    name=DISTNAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    project_urls=PROJECT_URLS,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
