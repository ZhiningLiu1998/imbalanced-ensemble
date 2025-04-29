.. imbalanced-ensemble documentation master file, created by
   sphinx-quickstart on Mon May 17 16:20:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imbalanced-ensemble documentation!
===============================================

.. image:: https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbens-logo.png

**Date**: |today| **Version**: |version| **Github**: `imbalanced-ensemble <https://github.com/ZhiningLiu1998/imbalanced-ensemble>`_

.. raw:: html
   

   <h3 align="center">
      <!-- [<a href="https://arxiv.org/pdf/2111.12776.pdf">Paper</a>] -->
      <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">📁GitHub</a> | 
      <!-- [<a href="https://imbalanced-ensemble.readthedocs.io">📕Documentation</a>] -->
      <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">🖼️Gallery</a> | 
      <a href="#installation">🔧Installation</a> | 
      <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">📜Changelog</a>
      <!-- [<a href="https://zhuanlan.zhihu.com/p/376572330">Zhihu/知乎</a>] -->
   </h3>

   <h3 align="center">
   ⏳Quick Start with our <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble#5-min-quick-start-with-imbens">5-minute Guide</a> & <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">Detailed Examples</a>
   </h3>

   <table>
      <tr>
         <td>Status</td>
         <td>
               <a href="https://codecov.io/gh/ZhiningLiu1998/imbalanced-ensemble">
                  <img src="https://codecov.io/gh/ZhiningLiu1998/imbalanced-ensemble/branch/main/graph/badge.svg?token=46Y73QPA68"></a>
               <a href='https://dl.circleci.com/status-badge/redirect/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main'>
                  <img src='https://dl.circleci.com/status-badge/img/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main.svg?style=shield' alt='CircleCI Status'></a>
               <a href='https://imbalanced-ensemble.readthedocs.io/en/latest/?badge=latest'>
                  <img alt="Read the Docs" src="https://img.shields.io/readthedocs/imbalanced-ensemble"></a>
                  <!-- <img src='https://readthedocs.org/projects/imbalanced-ensemble/badge/?version=latest'></a> -->
               <a href="https://github.com/psf/black">
                  <img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/master/LICENSE">
                  <img src="https://img.shields.io/github/license/ZhiningLiu1998/imbalanced-ensemble"></a>
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues">
                  <img src="https://img.shields.io/github/issues/ZhiningLiu1998/imbalanced-ensemble?logo=github"></a>
         </td>
      </tr>
      <tr>
         <td>PyPI</td>
         <td>
               <a href="https://pypi.org/project/imbalanced-ensemble/">
                  <img src="https://img.shields.io/badge/PyPi-imbalanced--ensemble-3775A9?logo=pypi&labelColor=white"></a>
               <a href="https://pypi.org/project/imbalanced-ensemble/">
                  <img src="https://img.shields.io/pypi/v/imbalanced-ensemble?logo=pypi&label=version&labelColor=white&color=3775A9"></a>
               <a href="https://www.python.org/">
                  <img src="https://img.shields.io/pypi/pyversions/imbalanced-ensemble.svg?logo=python&labelColor=white"></a>
         </td>
      </tr>
      <tr>
         <td>Traffic</td>
         <td>
               <a href="https://pepy.tech/project/imbalanced-ensemble">
                  <img src="https://img.shields.io/github/stars/ZhiningLiu1998/imbalanced-ensemble"></a>
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/network/members">
                  <img src="https://img.shields.io/github/forks/ZhiningLiu1998/imbalanced-ensemble"></a>
               <a href="https://pepy.tech/project/imbalanced-ensemble">
                  <img src="https://pepy.tech/badge/imbalanced-ensemble"></a>
               <a href="https://pepy.tech/project/imbalanced-ensemble">
                  <img src="https://pepy.tech/badge/imbalanced-ensemble/month"></a>
               <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble#contributors-"><img src="https://img.shields.io/badge/all_contributors-5-orange.svg"></a>
               <!-- ALL-CONTRIBUTORS-BADGE:END -->
         </td>
      </tr>
      <tr>
         <td>Documentation</td>
         <td>
               <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/">
                  <img src="https://img.shields.io/badge/ReadTheDoc-Latest-green?logo=readthedocs&labelColor=376681"></a>
               <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">
                  <img src="https://img.shields.io/badge/Doc-Changelog-blue?logo=readthedocs"></a>
               <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">
                  <img src="https://img.shields.io/badge/Doc-Examples & Gallery-blue?logo=readthedocs"></a>
               <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/api.html">
                  <img src="https://img.shields.io/badge/Doc-API Reference-blue?logo=readthedocs"></a>
         </td>
      </tr>
      <tr>
         <td>Paper & Citation</td>
         <td>
               <a href="https://arxiv.org/abs/2111.12776">
                  <img src="https://img.shields.io/badge/arXiv-2111.12776-B31B1B?logo=arXiv"></a>
               <a href="https://arxiv.org/pdf/2111.12776">
                  <img src="https://img.shields.io/badge/arXiv-PDF-B31B1B?logo=arXiv"></a>
               <a href="https://zhuanlan.zhihu.com/p/376572330">
                  <img src="https://img.shields.io/badge/Blog-知乎/Zhihu-0084ff?logo=Zhihu&labelColor=white"></a>
               <a href="https://scholar.google.com/scholar?q=IMBENS%3A+Ensemble+class-imbalanced+learning+in+Python">
                  <img src="https://img.shields.io/badge/Citation-Bibtex-4285F4?logo=googlescholar&labelColor=white"></a>
         </td>
      </tr>
      <tr>
         <td>Language</td>
         <td>
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">
                  <img src="https://img.shields.io/badge/README-English-blue?logo=github&labelColor=black"></a>
               <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md">
                  <img src="https://img.shields.io/badge/README-中文-blue?logo=github&labelColor=black"></a>
         </td>
      </tr>
   </table>
   <br>

.. .. raw:: html

..    <h3>
..       Links: 
..       [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">Github</a>]
..       [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">Gallery</a>]
..       [<a href="https://pypi.org/project/imbalanced-ensemble/">PyPI</a>]
..       [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">Changelog</a>]
..       [<a href="https://zhuanlan.zhihu.com/p/376572330">知乎/Zhihu</a>]
..       [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md">中文README</a>]
..       [<a href="https://arxiv.org/abs/2111.12776">arXiv</a>]
..    </h3>

**IMBENS** (imported as ``imbens``) is an extensible Python library for quick implementation, evaluation, and comparison for `general class-imbalanced learning solutions <https://github.com/ZhiningLiu1998/awesome-imbalanced-learning>`_.
Currently, IMBENS includes **30+** algorithms for class-imbalanced classification, including under-sampling (selection or generation-based), over-sampling (e.g., SMOTE and its variants), cost-sensitive learning (e.g., AdaCost), and ensemble methods that integrate these techniques (SMOTEBagging, RUSBoost, SelfPacedEnsemble, etc).

IMBENS is built on top of `scikit-learn <https://scikit-learn.org/stable/>`_ design principles and was initially built based on `imbalanced-learn <https://imbalanced-learn.org/stable/>`_, but has since evolved independently and no longer depends on it. Users can take advantage of various utilities from the sklearn community for data processing/cross-validation/hyper-parameter tuning, etc.

🌈 **IMBENS Highlights**

- 🧑‍💻 **Ease-of-use:** Unified user-friendly scikit-learn-style `APIs <https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/api.html>`_.
- 📒 **Documentation:** Detailed `documentation <https://imbalanced-ensemble.readthedocs.io/>`_ and `examples <https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#>`_.
- 🚀 **Efficiency:** Optimized efficiency with parallelization using `joblib <https://github.com/joblib/joblib>`_.
- 📊 **Benchmarking:** Running & comparing multiple models with our `visualizer <#visualize-ensemble-classifiers>`_.
- 📺 **Monitoring:** Powerful, customizable, interactive training `logging <#customizing-training-log>`_.
- 🍻 **Compatibility:** Work seamlessly with `scikit-learn <https://scikit-learn.org/stable/>`_ and other tools for the sklearn community.
- 📈 **Functionality:** Extended existing binary techniques to multi-class setting.
- 👯 **Extensibility:** Implement new methods via well-designed inheritance and polymorphism.
- 🧪 **Testing:** Quality control with extensive unit tests.

✂️ **Use IMBENS for class-imbalanced classification with <5 lines of code:**

.. code-block:: python

   # Train an SPE classifier
   from imbens.ensemble import SelfPacedEnsembleClassifier
   clf = SelfPacedEnsembleClassifier(random_state=42)
   clf.fit(X_train, y_train)

   # Predict with an SPE classifier
   y_pred = clf.predict(X_test)

🤗 **Citing IMBENS**

🍻 We appreciate your citation if you find our work helpful! The BibTeX entry:

.. code-block:: bibtex

   @misc{liu2022imbens,
      author       = {Zhining Liu},
      title        = {IMBENS: Python Toolbox for Class-Imbalanced Ensemble Learning},
      howpublished = {\url{https://github.com/ZhiningLiu1998/imbalanced-ensemble}},
      year         = {2025},
   }

**API Demo:**

.. code-block:: python

  >>> from imbens.ensemble import SelfPacedEnsembleClassifier
  >>> from imbens.datasets import generate_imbalance_data
  >>> from imbens.utils import evaluate_print
  >>> from imbens.visualizer import ImbalancedEnsembleVisualizer
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
   :hidden:
   :maxdepth: 0
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 3
   :caption: History

   release_history
