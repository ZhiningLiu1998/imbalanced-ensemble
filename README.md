![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbens-logo.png)
<!-- ![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/example_gallery_snapshot_horizontal.png) -->

<h1 align="center">
    IMBENS: Class-Imbalance Benchmark and Solutions <br>
</h1>

<h3 align="center">
    <!-- [<a href="https://arxiv.org/pdf/2111.12776.pdf">Paper</a>] -->
    <!-- [<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble">🗂GitHub</a>] -->
    [<a href="https://imbalanced-ensemble.readthedocs.io">📕Documentation</a>]
    [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">🖼️Gallery</a>]
    [<a href="#installation">🛠Installation</a>]
    [<a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">📜Changelog</a>]
    <!-- [<a href="https://zhuanlan.zhihu.com/p/376572330">Zhihu/知乎</a>] -->
</h3>

<h3 align="center">
⏳Quick Start with our <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble#5-min-quick-start-with-imbens">5-minute Guide</a> & <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">Detailed Examples</a>
</h3>

<table align="center">
    <tr>
        <td>Status</td>
        <td>
            <a href="https://codecov.io/gh/ZhiningLiu1998/imbalanced-ensemble">
                <img src="https://codecov.io/gh/ZhiningLiu1998/imbalanced-ensemble/branch/main/graph/badge.svg?token=46Y73QPA68"></a>
            <a href='https://dl.circleci.com/status-badge/redirect/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main'>
                <img src='https://dl.circleci.com/status-badge/img/gh/ZhiningLiu1998/imbalanced-ensemble/tree/main.svg?style=shield' alt='CircleCI Status'></a>
            <a href='https://imbalanced-ensemble.readthedocs.io/en/latest/?badge=latest'>
                <img alt="Read the Docs" src="https://img.shields.io/readthedocs/imbalanced-ensemble"></a>
            <a href="https://github.com/psf/black">
                <img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
            <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/master/LICENSE">
                <img src="https://img.shields.io/github/license/ZhiningLiu1998/imbalanced-ensemble"></a>
            <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues">
                <img src="https://img.shields.io/github/issues/ZhiningLiu1998/imbalanced-ensemble?logo=github"></a>
        </td>
    </tr>
    <tr>
        <td>Releases</td>
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
        <td>Articles</td>
        <td>
            <!-- <a href="https://arxiv.org/abs/2111.12776">
                <img src="https://img.shields.io/badge/arXiv-2111.12776-B31B1B?logo=arXiv"></a> -->
            <a href="https://arxiv.org/pdf/2111.12776">
                <img src="https://img.shields.io/badge/arXiv-Package-B31B1B?logo=arXiv"></a>
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


***IMBENS* (imported as `imbens`) is an extensible Python library for quick implementation, evaluation, and comparison for [general class-imbalanced learning solutions](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning)**. 
Currently, IMBENS includes **30+** algorithms for class-imbalanced classification, including under-sampling (selection or generation-based), over-sampling (e.g., SMOTE and its variants), cost-sensitive learning (e.g., AdaCost), and ensemble methods that integrate these techniques (SMOTEBagging, RUSBoost, SelfPacedEnsemble, etc).

IMBENS is built on top of [scikit-learn](https://scikit-learn.org/stable/) design principles and was initially built based on [imbalanced-learn](https://imbalanced-learn.org/stable/), but has since evolved independently and no longer depends on it. Users can take advantage of various utilities from the sklearn community for data processing/cross-validation/hyper-parameter tuning, etc.

<h2 align="left">🌈 IMBENS Highlights</h2>

- 🧑‍💻 **Ease-of-use:** Unified user-friendly scikit-learn-style [APIs](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/api.html).
- 📒 **Documentation:** Detailed [documentation](https://imbalanced-ensemble.readthedocs.io/) and [examples](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#).
- 🚀 **Efficiency:** Optimized efficiency with parallelization using [joblib](https://github.com/joblib/joblib).
- 📊 **Benchmarking:** Running & comparing multiple models with our [visualizer](#visualize-ensemble-classifiers).
- 📺 **Monitoring:** Powerful, customizable, interactive training [logging]((#customizing-training-log)).
- 🍻 **Compatibility:** Work seamlessly with [scikit-learn](https://scikit-learn.org/stable/) and other tools for sklearn community.
- 📈 **Functionality:** Extended existing binary techniques to multi-class setting.
- 👯 **Extensibility:** Implement new methods via well-designed inheritance and polymorphism.

### ✂️ **Use IMBENS for class-imbalanced classification with <5 lines of code:**

```python
# Train an SPE classifier
from imbens.ensemble import SelfPacedEnsembleClassifier
clf = SelfPacedEnsembleClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict with an SPE classifier
y_pred = clf.predict(X_test)
```

### 🤗 Citing IMBENS Package

We appreciate your citation if you find our work helpful! The BibTeX entry:

```bib
@misc{liu2022imbens,
  author       = {Zhining Liu},
  title        = {IMBENS: Python Toolbox for Class-Imbalanced Ensemble Learning},
  howpublished = {\url{https://github.com/ZhiningLiu1998/imbalanced-ensemble}},
  year         = {2025},
}
```

<!-- ```bib
@article{liu2023imbens,
  title={IMBENS: Ensemble Class-imbalanced Learning in Python},
  author={Liu, Zhining and Kang, Jian and Tong, Hanghang and Chang, Yi},
  journal={arXiv preprint arXiv:2111.12776},
  year={2023}
}
``` -->

### 👯‍♂️ Contribute to IMBENS

Join us and become a contributor!
Please refer to the [contributing guidelines](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/CONTRIBUTING.md).

<h2 align="left">📚 Table of Contents</h2>

- [Installation](#installation)
- [5-min Quick Start with IMBENS](#5-min-quick-start-with-imbens)
  - [A minimal working example](#a-minimal-working-example)
  - [Visualize ensemble classifiers](#visualize-ensemble-classifiers)
  - [Customizing training log](#customizing-training-log)
- [About imbalanced learning](#about-imbalanced-learning)
- [Acknowledgements](#acknowledgements)
- [List of Methods and References](#list-of-methods-and-references)
- [Related Projects](#related-projects)
- [Contributors ✨](#contributors-)


## Installation

It is recommended to use **pip** for installation.  
Please make sure the **latest version** is installed to avoid potential problems:
```shell
$ pip install imbalanced-ensemble            # normal install
$ pip install --upgrade imbalanced-ensemble  # update if needed
```

Or you can install imbalanced-ensemble by clone this repository:
```shell
$ git clone https://github.com/ZhiningLiu1998/imbalanced-ensemble.git
$ cd imbalanced-ensemble
$ pip install .
```

imbalanced-ensemble requires following dependencies:

- [Python](https://www.python.org/) (>=3.6)
- [numpy](https://numpy.org/) (>=1.16.0)
- [pandas](https://pandas.pydata.org/) (>=1.1.3)
- [scipy](https://www.scipy.org/) (>=1.9.1)
- [joblib](https://pypi.org/project/joblib/) (>=0.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=1.2.0)
- [matplotlib](https://matplotlib.org/) (>=3.3.2)
- [seaborn](https://seaborn.pydata.org/) (>=0.11.0)
- [tqdm](https://tqdm.github.io/) (>=4.50.2)
- [openml](https://www.openml.org/) (>=0.14.0)

## 5-min Quick Start with IMBENS

**Here, we provide some quick guides to help you get started with IMBENS.**  
**We strongly encourage users to check out the [**example gallery**](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#) for more comprehensive usage examples, which demonstrate many advanced features of IMBENS.**

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/example_gallery_snapshot.png)

### A minimal working example

Taking self-paced ensemble [1] as an example, it only requires less than 10 lines of code to deploy it:

```python
>>> from imbens.ensemble import SelfPacedEnsembleClassifier
>>> from sklearn.datasets import make_classification
>>> from sklearn.model_selection import train_test_split
>>> 
>>> X, y = make_classification(n_samples=1000, n_classes=3,
...                            n_informative=4, weights=[0.2, 0.3, 0.5],
...                            random_state=0)
>>> X_train, X_test, y_train, y_test = train_test_split(
...                            X, y, test_size=0.2, random_state=42)
>>> clf = SelfPacedEnsembleClassifier(random_state=0)
>>> clf.fit(X_train, y_train)
SelfPacedEnsembleClassifier(...)
>>> clf.predict(X_test)  
array([...])
```

### Visualize ensemble classifiers

The [`imbens.visualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/api.html) sub-module provide an [`ImbalancedEnsembleVisualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbens.visualizer.ImbalancedEnsembleVisualizer.html).
It can be used to visualize the ensemble estimator(s) for further information or comparison.
Please refer to [**visualizer documentation**](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbens.visualizer.ImbalancedEnsembleVisualizer.html) and [**examples**](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html) for more details.

**Fit an ImbalancedEnsembleVisualizer**
```python
from imbens.ensemble import SelfPacedEnsembleClassifier
from imbens.ensemble import RUSBoostClassifier
from imbens.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

# Fit ensemble classifiers
init_kwargs = {'estimator': DecisionTreeClassifier()}
ensembles = {
    'spe': SelfPacedEnsembleClassifier(**init_kwargs).fit(X_train, y_train),
    'rusboost': RUSBoostClassifier(**init_kwargs).fit(X_train, y_train),
    'easyens': EasyEnsembleClassifier(**init_kwargs).fit(X_train, y_train),
}

# Fit visualizer
from imbens.visualizer import ImbalancedEnsembleVisualizer
visualizer = ImbalancedEnsembleVisualizer().fit(ensembles=ensembles)
```
**Plot performance curves**
```python
fig, axes = visualizer.performance_lineplot()
```
![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/examples/visualize_performance_example.png)

**Plot confusion matrices**
```python
fig, axes = visualizer.confusion_matrix_heatmap()
```
![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/examples/visualize_confusion_matrix_example.png)

### Customizing training log

All ensemble classifiers in IMBENS support customizable training logging.
The training log is controlled by 3 parameters `eval_datasets`, `eval_metrics`, and `training_verbose` of the `fit()` method.
Read more details in the [**fit documentation**](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.SelfPacedEnsembleClassifier.html#imbens.ensemble.SelfPacedEnsembleClassifier.fit).

**Enable auto training log**
```python
clf.fit(..., train_verbose=True)
```
```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             ┃                          ┃            Data: train             ┃
┃ #Estimators ┃    Class Distribution    ┃               Metric               ┃
┃             ┃                          ┃  acc    balanced_acc   weighted_f1 ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃      1      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.838      0.877          0.839    ┃
┃      5      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.924      0.949          0.924    ┃
┃     10      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.954      0.970          0.954    ┃
┃     15      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.979      0.986          0.979    ┃
┃     20      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.990      0.993          0.990    ┃
┃     25      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.994      0.996          0.994    ┃
┃     30      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.988      0.992          0.988    ┃
┃     35      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.999      0.999          0.999    ┃
┃     40      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.995      0.997          0.995    ┃
┃     45      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.995      0.997          0.995    ┃
┃     50      ┃ {0: 150, 1: 150, 2: 150} ┃ 0.993      0.995          0.993    ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃    final    ┃ {0: 150, 1: 150, 2: 150} ┃ 0.993      0.995          0.993    ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```


**Customize granularity and content of the training log**
```python
clf.fit(..., 
        train_verbose={
            'granularity': 10,
            'print_distribution': False,
            'print_metrics': True,
        })
```

<details><summary> Click to view example output </summary>

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             ┃            Data: train             ┃
┃ #Estimators ┃               Metric               ┃
┃             ┃  acc    balanced_acc   weighted_f1 ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃      1      ┃ 0.964      0.970          0.964    ┃
┃     10      ┃ 1.000      1.000          1.000    ┃
┃     20      ┃ 1.000      1.000          1.000    ┃
┃     30      ┃ 1.000      1.000          1.000    ┃
┃     40      ┃ 1.000      1.000          1.000    ┃
┃     50      ┃ 1.000      1.000          1.000    ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃    final    ┃ 1.000      1.000          1.000    ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

</details>

**Add evaluation dataset(s)**
```python
  clf.fit(..., 
          eval_datasets={
              'valid': (X_valid, y_valid)
          })
```

<details><summary> Click to view example output </summary>

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃             ┃            Data: train             ┃            Data: valid             ┃
┃ #Estimators ┃               Metric               ┃               Metric               ┃
┃             ┃  acc    balanced_acc   weighted_f1 ┃  acc    balanced_acc   weighted_f1 ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃      1      ┃ 0.939      0.961          0.940    ┃ 0.935      0.933          0.936    ┃
┃     10      ┃ 1.000      1.000          1.000    ┃ 0.971      0.974          0.971    ┃
┃     20      ┃ 1.000      1.000          1.000    ┃ 0.982      0.981          0.982    ┃
┃     30      ┃ 1.000      1.000          1.000    ┃ 0.983      0.983          0.983    ┃
┃     40      ┃ 1.000      1.000          1.000    ┃ 0.983      0.982          0.983    ┃
┃     50      ┃ 1.000      1.000          1.000    ┃ 0.983      0.982          0.983    ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃    final    ┃ 1.000      1.000          1.000    ┃ 0.983      0.982          0.983    ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

</details>

**Customize evaluation metric(s)**
```python
from sklearn.metrics import accuracy_score, f1_score
clf.fit(..., 
        eval_metrics={
            'acc': (accuracy_score, {}),
            'weighted_f1': (f1_score, {'average':'weighted'}),
        })
```

<details><summary> Click to view example output </summary>

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃             ┃     Data: train      ┃     Data: valid      ┃
┃ #Estimators ┃        Metric        ┃        Metric        ┃
┃             ┃  acc    weighted_f1  ┃  acc    weighted_f1  ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━┫
┃      1      ┃ 0.942      0.961     ┃ 0.919      0.936     ┃
┃     10      ┃ 1.000      1.000     ┃ 0.976      0.976     ┃
┃     20      ┃ 1.000      1.000     ┃ 0.977      0.977     ┃
┃     30      ┃ 1.000      1.000     ┃ 0.981      0.980     ┃
┃     40      ┃ 1.000      1.000     ┃ 0.980      0.979     ┃
┃     50      ┃ 1.000      1.000     ┃ 0.981      0.980     ┃
┣━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━┫
┃    final    ┃ 1.000      1.000     ┃ 0.981      0.980     ┃
┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━┛
```

</details>

## About imbalanced learning

**Class-imbalance** (also known as the **long-tail problem**) is the fact that the classes are not represented equally in a classification problem, which is quite common in practice. For instance, fraud detection, prediction of rare adverse drug reactions and prediction gene families. Failure to account for the class imbalance often causes inaccurate and decreased predictive performance of many classification algorithms. **Imbalanced learning** aims to tackle the class imbalance problem to learn an unbiased model from imbalanced data.

For more resources on imbalanced learning, please refer to [**awesome-imbalanced-learning**](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning).

## Acknowledgements

IMBENS was initially developed on top of the awesome [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn), but has undergone heavy developments to implement many important techniques and features.
The infrastructure also underwent significant refactoring to support advanced ensemble learning features that are essential to practical usability (fine-grained training control, parallel computing, multi-class support, training logs, visualization, etc).

## List of Methods and References

***(Click to jump to the API reference page)**

- **Under-sampling**
  - *Selection-based*
    1. **[`RandomUnderSampler`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.RandomUnderSampler.html)**
    2. **[`NearMiss`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.NearMiss.html)** Mani, I., & Zhang, I. (2003, August). kNN approach to unbalanced data distributions: a case study involving information extraction. In Proceedings of workshop on learning from imbalanced datasets (Vol. 126, No. 1, pp. 1-7). United States: ICML.
    3. **[`InstanceHardnessThreshold`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.InstanceHardnessThreshold.html)** Smith, M. R., Martinez, T., & Giraud-Carrier, C. (2014). An instance level analysis of data complexity. Machine learning, 95, 225-256.
  - *Generation-based*
    1. **[`ClusterCentroids`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.ClusterCentroids.html)** Lin, W. C., Tsai, C. F., Hu, Y. H., & Jhang, J. S. (2017). Clustering-based undersampling in class-imbalanced data. Information Sciences, 409, 17-26.
- **Cleaning**
  - *Distance-based*
    1. **[`TomekLinks`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.TomekLinks.html)** Tomek, I. (1976). Two modifications of CNN.
    2. **[`EditedNearestNeighbours`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.CondensedNearestNeighbour.html)** Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, and Cybernetics, (3), 408-421.
    3. **[`RepeatedEditedNearestNeighbours`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.RepeatedEditedNearestNeighbours.html)** Tomek, I. (1976). An experiment with the edited nearest-nieghbor rule
    4. **[`AllKNN`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.AllKNN.html)** Tomek, I. (1976). An experiment with the edited nearest-nieghbor rule.
    5. **[`OneSidedSelection`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.OneSidedSelection.html)** Kubat, M., & Matwin, S. (1997, July). Addressing the curse of imbalanced training sets: one-sided selection. In Icml (Vol. 97, No. 1, p. 179).
    6. **[`NeighbourhoodCleaningRule`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.NeighbourhoodCleaningRule.html)** Laurikkala, J. (2001). Improving identification of difficult small classes by balancing class distribution. In Artificial Intelligence in Medicine: 8th Conference on Artificial Intelligence in Medicine in Europe, AIME 2001 Cascais, Portugal, July 1–4, 2001, Proceedings 8 (pp. 63-66). Springer Berlin Heidelberg.
- **Oversamping**
  - *Generation-based*
    1. **[`RandomOverSampler`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.RandomOverSampler.html)**
    2. **[`SMOTE`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.SMOTE.html)** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.
    3. **[`BorderlineSMOTE`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.BorderlineSMOTE.html)** Han, H., Wang, W. Y., & Mao, B. H. (2005, August). Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning. In International conference on intelligent computing (pp. 878-887). Berlin, Heidelberg: Springer Berlin Heidelberg.
    4. **[`SVMSMOTE`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.SVMSMOTE.html)** Nguyen, H. M., Cooper, E. W., & Kamei, K. (2011). Borderline over-sampling for imbalanced data classification. International Journal of Knowledge Engineering and Soft Data Paradigms, 3(1), 4-21.
    5. **[`ADASYN`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/sampler/_autosummary/imbens.sampler.ADASYN.html)** He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. In 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence) (pp. 1322-1328). Ieee.
- **Ensemble Modeling**
  - *Under-sampling + Ensemble*
    1. **[`SelfPacedEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.SelfPacedEnsembleClassifier.html)** Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., & Liu, T. Y. (2020, April). Self-paced ensemble for highly imbalanced massive data classification. In 2020 IEEE 36th international conference on data engineering (ICDE) (pp. 841-852). IEEE.
    2. **[`BalanceCascadeClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.BalanceCascadeClassifier.html)** Liu, X. Y., Wu, J., & Zhou, Z. H. (2008). Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39(2), 539-550.
    3. **[`BalancedRandomForestClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.BalancedRandomForestClassifier.html)** Khoshgoftaar, T. M., Golawala, M., & Van Hulse, J. (2007, October). An empirical study of learning from imbalanced data using random forest. In 19th IEEE international conference on tools with artificial intelligence (ICTAI 2007) (Vol. 2, pp. 310-317). IEEE.
    4. **[`EasyEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.EasyEnsembleClassifier.html)** Liu, X. Y., Wu, J., & Zhou, Z. H. (2008). Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39(2), 539-550.
    5. **[`RUSBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.RUSBoostClassifier.html)** Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J., & Napolitano, A. (2009). RUSBoost: A hybrid approach to alleviating class imbalance. IEEE transactions on systems, man, and cybernetics-part A: systems and humans, 40(1), 185-197.
    6. **[`UnderBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.UnderBaggingClassifier.html)** Barandela, R., Valdovinos, R. M., & Sánchez, J. S. (2003). New applications of ensembles of classifiers. Pattern Analysis & Applications, 6, 245-256.
  - *Over-sampling + Ensemble*
    1. **[`OverBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.OverBoostClassifier.html)** Chawla, N. V., Lazarevic, A., Hall, L. O., & Bowyer, K. W. (2003, September). SMOTEBoost: Improving prediction of the minority class in boosting. In European conference on principles of data mining and knowledge discovery (pp. 107-119). Berlin, Heidelberg: Springer Berlin Heidelberg.
    2. **[`SMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.SMOTEBoostClassifier.html)** Chawla, N. V., Lazarevic, A., Hall, L. O., & Bowyer, K. W. (2003, September). SMOTEBoost: Improving prediction of the minority class in boosting. In European conference on principles of data mining and knowledge discovery (pp. 107-119). Berlin, Heidelberg: Springer Berlin Heidelberg.
    3. **[`OverBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.OverBaggingClassifier.html)** Wang, S., & Yao, X. (2009, March). Diversity analysis on imbalanced data sets by using ensemble models. In 2009 IEEE symposium on computational intelligence and data mining (pp. 324-331). IEEE.
    4. **[`SMOTEBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.SMOTEBaggingClassifier.html)** Wang, S., & Yao, X. (2009, March). Diversity analysis on imbalanced data sets by using ensemble models. In 2009 IEEE symposium on computational intelligence and data mining (pp. 324-331). IEEE.
- **Reweighting-based**
  - *Cost-sensitive Learning*
    1. **[`AdaCostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.AdaCostClassifier.html)** Fan, W., Stolfo, S. J., Zhang, J., & Chan, P. K. (1999, June). AdaCost: misclassification cost-sensitive boosting. In Icml (Vol. 99, pp. 97-105).
    2. **[`AdaUBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.AdaUBoostClassifier.html)** Karakoulas, G., & Shawe-Taylor, J. (1998). Optimizing classifers for imbalanced training sets. Advances in neural information processing systems, 11.
    3. **[`AsymBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.AsymBoostClassifier.html)** Viola, P., & Jones, M. (2001). Fast and robust classification using asymmetric adaboost and a detector cascade. Advances in neural information processing systems, 14.
- **Compatible**
  - **[`CompatibleAdaBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.CompatibleAdaBoostClassifier.html)** Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
  - **[`CompatibleBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbens.ensemble.CompatibleBaggingClassifier.html)** Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17):1–5, 2017.

## Related Projects

**Check out [Zhining](https://zhiningliu.com/)'s other open-source projects!**  
<table style="font-size:15px;">
  <tr>
    <td align="center"><a href="https://github.com/ZhiningLiu1998/awesome-imbalanced-learning"><img src="https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/thumbnails/awesomeil-thumb.png" height="80px" alt=""/><br /><sub><b>Imbalanced Learning [Awesome]</b></sub></a><br />
      <a href="https://github.com/ZhiningLiu1998/awesome-imbalanced-learning/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/ZhiningLiu1998/awesome-imbalanced-learning?style=social">
      </a>
    </td>
    <td align="center"><a href="https://github.com/ZhiningLiu1998/awesome-awesome-machine-learning"><img src="https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/thumbnails/awesomeml-thumb.png" height="80px" alt=""/><br /><sub><b>Machine Learning [Awesome]</b></sub></a><br />
      <a href="https://github.com/ZhiningLiu1998/awesome-awesome-machine-learning/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/ZhiningLiu1998/awesome-awesome-machine-learning?style=social">
      </a>
    </td>
    <td align="center"><a href="https://github.com/ZhiningLiu1998/self-paced-ensemble"><img src="https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/thumbnails/spe-thumb-1.png" height="80px" alt=""/><br /><sub><b>Self-paced Ensemble [ICDE]</b></sub></a><br />
      <a href="https://github.com/ZhiningLiu1998/self-paced-ensemble/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/ZhiningLiu1998/self-paced-ensemble?style=social">
      </a>
    </td>
    <td align="center"><a href="https://github.com/ZhiningLiu1998/mesa"><img src="https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/thumbnails/mesa-thumb.png" height="80px" alt=""/><br /><sub><b>Meta-Sampler [NeurIPS]</b></sub></a><br />
      <a href="https://github.com/ZhiningLiu1998/mesa/stargazers">
      <img alt="GitHub stars" src="https://img.shields.io/github/stars/ZhiningLiu1998/mesa?style=social">
      </a>
    </td>
  </tr>
</table>


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://zhiningliu.com"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt="Zhining Liu"/><br /><sub><b>Zhining Liu</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/commits?author=ZhiningLiu1998" title="Code">💻</a> <a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-ZhiningLiu1998" title="Maintenance">🚧</a> <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3AZhiningLiu1998" title="Bug reports">🐛</a> <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/commits?author=ZhiningLiu1998" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leaphan"><img src="https://avatars.githubusercontent.com/u/35593707?v=4?s=100" width="100px;" alt="leaphan"/><br /><sub><b>leaphan</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Aleaphan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hannanhtang"><img src="https://avatars.githubusercontent.com/u/23587399?v=4?s=100" width="100px;" alt="hannanhtang"/><br /><sub><b>hannanhtang</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Ahannanhtang" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/huajuanren"><img src="https://avatars.githubusercontent.com/u/37321841?v=4?s=100" width="100px;" alt="H.J.Ren"/><br /><sub><b>H.J.Ren</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Ahuajuanren" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://datamodelsanalytics.com"><img src="https://avatars.githubusercontent.com/u/42288570?v=4?s=100" width="100px;" alt="Marc Skov Madsen"/><br /><sub><b>Marc Skov Madsen</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3AMarcSkovMadsen" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
