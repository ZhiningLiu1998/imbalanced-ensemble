#! /usr/bin/env python
"""Toolbox for ensemble learning on class-imbalanced dataset."""

# import codecs

import io
import os

from setuptools import find_packages, setup, Command

# get __version__ from _version.py
ver_file = os.path.join("imbalanced_ensemble", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "imbalanced-ensemble"
DESCRIPTION = "Toolbox for ensemble learning on class-imbalanced dataset."

# with codecs.open("README.rst", encoding="utf-8-sig") as f:
#     LONG_DESCRIPTION = f.read()

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

MAINTAINER = "Zhining Liu"
MAINTAINER_EMAIL = "zhining.liu@outlook.com"
URL = "https://github.com/ZhiningLiu1998/imbalanced-ensemble"
LICENSE = "MIT"
DOWNLOAD_URL = "https://github.com/ZhiningLiu1998/imbalanced-ensemble"
VERSION = __version__
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
INSTALL_REQUIRES = [
    "numpy>=1.16.0",
    "scipy>=0.19.1",
    "pandas>=1.1.3"
    "joblib>=0.11",
    "scikit-learn>=0.24",
    "matplotlib>=3.3.2",
    "seaborn>=0.11.0",
    "tqdm>=4.50.2",
]
EXTRAS_REQUIRE = {
    # "dev": [
    #     "black",
    #     "flake8",
    # ],
    "tests": [
        "pytest",
        "pytest-cov",
    ],
    "docs": [
        "sphinx",
        "sphinx-gallery",
        "numpydoc",
        "pydata-sphinx-theme",
        "sphinxcontrib-bibtex",
    ],
}


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
