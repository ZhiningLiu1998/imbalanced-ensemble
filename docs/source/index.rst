.. imbalanced-ensemble documentation master file, created by
   sphinx-quickstart on Mon May 17 16:20:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imbalanced-ensemble documentation!
===============================================

.. image:: https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbens-logo.png

.. raw:: html

   <p>
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">
         <img src="https://img.shields.io/badge/Imbalanced-Ensemble-orange">
      </a>
      <a href='https://dl.circleci.com/status-badge/redirect/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main'>
         <img src='https://dl.circleci.com/status-badge/img/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main.svg?style=shield' alt='CircleCI Status' />
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
      [<a href="https://zhuanlan.zhihu.com/p/376572330">知乎/Zhihu</a>]
      [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md">中文README</a>]
      [<a href="https://arxiv.org/abs/2111.12776">arXiv</a>]
   </h3>

**Date**: |today| **Version**: |version|

**Paper**: `IMBENS: Ensemble Class-imbalanced Learning in Python <https://arxiv.org/abs/2111.12776>`_

**Citing US**

If you find IMBENS helpful in your work or research, we would greatly appreciate citations to the following paper [`PDF <(https://arxiv.org/pdf/2111.12776.pdf>`_]::

   @article{liu2021imbens,
      title={IMBENS: Ensemble Class-imbalanced Learning in Python},
      author={Liu, Zhining and Wei, Zhepei and Yu, Erxin and Huang, Qiang and Guo, Kai and Yu, Boyang and Cai, Zhaonian and Ye, Hangting and Cao, Wei and Bian, Jiang and Wei, Pengfei and Jiang, Jing and Chang, Yi},
      journal={arXiv preprint arXiv:2111.12776},
      year={2021}
   }


**imbalanced-ensemble** (IMBENS, imported as ``imbalanced_ensemble``) is a Python toolbox 
for quick implementation, modification, evaluation, and visualization of ensemble learning 
algorithms for class-imbalanced data.
It was built on the basis of `scikit-learn <https://scikit-learn.org/stable/index.html>`__
and `imbalanced-learn <https://imbalanced-learn.org/stable/>`__.
IMBENS includes more than 15 ensemble imbalanced learning (EIL) algorithms, from the 
classical SMOTEBoost (2003) and RUSBoost (2010) to recent SPE (2020), from resampling-based 
methods to cost-sensitive ensemble learning.

**IMBENS is featured for:**

- Unified, easy-to-use APIs, detailed `documentation <https://imbalanced-ensemble.readthedocs.io/>`_ and `examples <https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#>`_.
- Capable for out-of-the-box *multi-class* imbalanced (long-tailed) learning.
- Optimized performance with parallelization when possible using 
  `joblib <https://github.com/joblib/joblib>`__.
- Powerful, customizable, interactive training logging and visualizer.
- Full compatibility with other popular packages like 
  `scikit-learn <https://scikit-learn.org/stable/>`__ and 
  `imbalanced-learn <https://imbalanced-learn.org/stable/>`__.

**API Demo:**

.. code-block:: python

  >>> from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
  >>> from imbalanced_ensemble.datasets import generate_imbalance_data
  >>> from imbalanced_ensemble.utils import evaluate_print
  >>> from imbalanced_ensemble.visualizer import ImbalancedEnsembleVisualizer
  >>> 
  >>> X_train, X_test, y_train, y_test = generate_imbalance_data(
  ...     n_samples=200, weights=[.9,.1], test_size=.5)
  >>> 
  >>> clf = SelfPacedEnsembleClassifier()                    # initialize ensemble
  >>> clf.fit(X_train, y_train)                              
  >>> 
  >>> y_test_pred = clf.predict(X_test)                      # predict labels
  >>> y_test_proba = clf.predict_proba(X_test)               # predict probabilities
  >>> 
  >>> evaluate_print(y_test, y_test_pred, "SPE")             # performance evaluation
  SPE balanced Acc: 0.972 | macro Fscore: 0.886 | macro Gmean: 0.972
  >>> 
  >>> visualizer = ImbalancedEnsembleVisualizer()            # initialize visualizer
  >>> visualizer.fit({'SPE': clf})
  >>> 
  >>> visualizer.performance_lineplot()                      # performance visualization 
  >>> visualizer.confusion_matrix_heatmap()                  # prediction visualization

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
