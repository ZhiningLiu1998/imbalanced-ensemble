.. imbalanced-ensemble documentation master file, created by
   sphinx-quickstart on Mon May 17 16:20:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imbalanced-ensemble documentation!
===============================================

.. image:: https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbalanced_ensemble_header.png

.. raw:: html

   <p>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">
         <img src="https://img.shields.io/badge/Imbalanced-Ensemble-orange">
      </a>
      <a href='https://imbalanced-ensemble.readthedocs.io/en/latest/?badge=latest'>
         <img src='https://readthedocs.org/projects/imbalanced-ensemble/badge/?version=latest' alt='Documentation Status' />
      </a>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/stargazers">
         <img src="https://img.shields.io/github/stars/ZhiningLiu1998/imbalanced-ensemble">
      </a>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/network/members">
         <img src="https://img.shields.io/github/forks/ZhiningLiu1998/imbalanced-ensemble">
      </a>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues">
         <img src="https://img.shields.io/github/issues/ZhiningLiu1998/imbalanced-ensemble">
      </a>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/master/LICENSE">
         <img src="https://img.shields.io/github/license/ZhiningLiu1998/imbalanced-ensemble">
      </a>
      <a href="https://pypi.org/project/imbalanced-ensemble/">
         <img src="https://badge.fury.io/py/imbalanced-ensemble.svg">
      </a>
      <br>
      <a href="https://www.python.org/">
         <img src="https://img.shields.io/pypi/pyversions/imbalanced-ensemble.svg">
      </a>
      <a href="https://pepy.tech/project/imbalanced-ensemble">
         <img src="https://pepy.tech/badge/imbalanced-ensemble">
      </a>
      <a href="https://pepy.tech/project/imbalanced-ensemble">
         <img src="https://pepy.tech/badge/imbalanced-ensemble/month">
      </a>
   </p>

.. raw:: html

   <h3>
      [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">Github</a>]
      [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">Gallery</a>]
      [<a href="https://pypi.org/project/imbalanced-ensemble/">PyPI</a>]
      [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">Changelog</a>]
      [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/tree/main/imbalanced_ensemble">Source</a>]
      [<a href="https://pypi.org/project/imbalanced-ensemble/#files">Download</a>]
   </h3>

**Date**: |today| **Version**: |version|

**imbalanced-ensemble** (IMBENS, imported as ``imbalanced_ensemble``) is a Python toolbox 
for quick implementing and deploying ensemble learning algorithms on class-imbalanced data.
It was built on the basis of `scikit-learn <https://scikit-learn.org/stable/index.html>`__
and `imbalanced-learn <https://imbalanced-learn.org/stable/>`__.
IMBENS includes more than 15 ensemble imbalanced learning (EIL) algorithms, from the 
classical SMOTEBoost (2003) and RUSBoost (2010) to recent SPE (2020), from resampling-based 
methods to cost-sensitive ensemble learning.

**IMBENS is featured for:**

- Unified, easy-to-use APIs, detailed documentation and examples.
- Capable for multi-class imbalanced learning out-of-box.
- Optimized performance with parallelization when possible using 
  `joblib <https://github.com/joblib/joblib>`__.
- Powerful, customizable, interactive training logging and visualizer.
- Full compatibility with other popular packages like 
  `scikit-learn <https://scikit-learn.org/stable/>`__ and 
  `imbalanced-learn <https://imbalanced-learn.org/stable/>`__.

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   get_start
   install

.. toctree::
   :maxdepth: 2
   :caption: API

   api/ensemble/api
   api/sampler/api
   api/visualizer/api
   api/pipeline/api
   api/datasets/api
   api/metrics/api
   api/utils/api

.. toctree::
   :maxdepth: 3
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 3
   :caption: History

   release_history
