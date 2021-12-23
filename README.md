![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbens-logo.png)

<!-- ![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbalanced_ensemble_header.png) -->

<!-- ![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/example_gallery_snapshot_horizontal.png) -->

<h1 align="center">
  IMBENS: Class-imbalanced Ensemble Learning in Python
</h1>

<p align="center">
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
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/pypi/pyversions/imbalanced-ensemble.svg">
  </a>
  <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=ZhiningLiu1998.imbalanced-ensemble">
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble#contributors-"><img src="https://img.shields.io/badge/all_contributors-4-orange.svg"></a>
<!-- ALL-CONTRIBUTORS-BADGE:END -->
  <a href="https://pepy.tech/project/imbalanced-ensemble">
    <img src="https://pepy.tech/badge/imbalanced-ensemble">
  </a>
  <a href="https://pepy.tech/project/imbalanced-ensemble">
    <img src="https://pepy.tech/badge/imbalanced-ensemble/month">
  </a>
</p>

**Links: 
  <a href="https://imbalanced-ensemble.readthedocs.io/">Documentation</a> |
  <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#">Gallery</a> |
  <a href="https://pypi.org/project/imbalanced-ensemble/">PyPI</a> |
  <a href="https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html">Changelog</a> |
  <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/tree/main/imbalanced_ensemble">Source</a> |
  <a href="https://pypi.org/project/imbalanced-ensemble/#files">Download</a> |
  <a href="https://zhuanlan.zhihu.com/p/376572330">知乎/Zhihu</a> |
  <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md">中文README</a> |
  <a href="https://arxiv.org/abs/2111.12776">arXiv</a>**

**Paper: [IMBENS: Ensemble Class-imbalanced Learning in Python](https://arxiv.org/abs/2111.12776)**

***imbalanced-ensemble* (IMBENS, imported as `imbalanced_ensemble`)** is a Python toolbox for quick implementation, modification, evaluation, and visualization of ensemble learning algorithms for class-imbalanced data.
The problem of learning from imbalanced data is known as imbalanced learning or long-tail learning (under multi-class scenario). See related papers/libraries/resources [here](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning).

Currently (v0.1), IMBENS includes more than 15 ensemble imbalanced learning algorithms, from the classical *SMOTEBoost* (2003), *RUSBoost* (2010) to recent [*Self-paced Ensemble*](https://github.com/ZhiningLiu1998/self-paced-ensemble) (2020), from *resampling* to *cost-sensitive learning*. More algorithms will be included in the future. We also provide detailed documentation and examples across various algorithms. See full list of implemented methods [here](#list-of-implemented-methods).

<!-- **Read more at: [[知乎/Zhihu](https://zhuanlan.zhihu.com/p/376572330)] [[中文README](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/docs/README_CN.md)].** -->


<!-- ## Citing us -->

**If you find IMBENS helpful in your work or research, we would greatly appreciate citations to the following [paper](https://arxiv.org/pdf/2111.12776.pdf):**

```bib
@article{liu2021imbens,
  title={IMBENS: Ensemble Class-imbalanced Learning in Python},
  author={Liu, Zhining and Wei, Zhepei and Yu, Erxin and Huang, Qiang and Guo, Kai and Yu, Boyang and Cai, Zhaonian and Ye, Hangting and Cao, Wei and Bian, Jiang and Wei, Pengfei and Jiang, Jing and Chang, Yi},
  journal={arXiv preprint arXiv:2111.12776},
  year={2021}
}
```

**IMBENS is featured for:**
- &#x1F34E; **Unified, easy-to-use APIs, detailed [documentation](https://imbalanced-ensemble.readthedocs.io/) and [examples](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#).**
- &#x1F34E; **Capable for out-of-the-box *multi-class* imbalanced (long-tailed) learning.**
- &#x1F34E; **Optimized performance with parallelization when possible using [joblib](https://github.com/joblib/joblib).**
- &#x1F34E; **Powerful, customizable, interactive training logging and visualizer.**
- &#x1F34E; **Full compatibility with other popular packages like [scikit-learn](https://scikit-learn.org/stable/) and [imbalanced-learn](https://imbalanced-learn.org/stable/).**

**API Demo:**
```python
# Train an SPE classifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
clf = SelfPacedEnsembleClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict with an SPE classifier
y_pred = clf.predict(X_test)
```


### Table of Contents

- [Installation](#installation)
- [Highlights](#highlights)
- [List of implemented methods](#list-of-implemented-methods)
- [5-min Quick Start with IMBENS](#5-min-quick-start-with-imbens)
  - [A minimal working example](#a-minimal-working-example)
  - [Visualize ensemble classifiers](#visualize-ensemble-classifiers)
  - [Customizing training log](#customizing-training-log)
- [About imbalanced learning](#about-imbalanced-learning)
- [Acknowledgements](#acknowledgements)
- [References](#references)
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
- [scipy](https://www.scipy.org/) (>=0.19.1)
- [joblib](https://pypi.org/project/joblib/) (>=0.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=1.0.0)
- [matplotlib](https://matplotlib.org/) (>=3.3.2)
- [seaborn](https://seaborn.pydata.org/) (>=0.11.0)
- [tqdm](https://tqdm.github.io/) (>=4.50.2)


## Highlights

- &#x1F34E; ***Unified, easy-to-use API design.***  
All ensemble learning methods implemented in IMBENS share a unified API design. 
Similar to sklearn, all methods have functions (e.g., `fit()`, `predict()`, `predict_proba()`) that allow users to deploy them with only a few lines of code.
- &#x1F34E; ***Extended functionalities, wider application scenarios.***  
*All methods in IMBENS are ready for **multi-class imbalanced classification**.* We extend binary ensemble imbalanced learning methods to get them to work under the multi-class scenario. Additionally, for supported methods, we provide more training options like class-wise resampling control, balancing scheduler during the ensemble training process, etc.
- &#x1F34E; ***Detailed training log, quick intuitive visualization.***   
We provide additional parameters (e.g., `eval_datasets`, `eval_metrics`, `training_verbose`) in `fit()` for users to control the information they want to monitor during the ensemble training. We also implement an [`EnsembleVisualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html) to quickly visualize the ensemble estimator(s) for providing further information/conducting comparison. See an example [here](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/basic/plot_basic_example.html#sphx-glr-auto-examples-basic-plot-basic-example-py).
- &#x1F34E; ***Wide compatiblilty.***   
IMBENS is designed to be compatible with [scikit-learn](https://scikit-learn.org/stable/) (sklearn) and also other compatible projects like [imbalanced-learn](https://imbalanced-learn.org/stable/). Therefore, users can take advantage of various utilities from the sklearn community for data processing/cross-validation/hyper-parameter tuning, etc.

<!-- ## Background

Class-imbalance (also known as the long-tail problem in multi-class) is the fact that the classes are not represented equally in a classification problem, which is quite common in practice. For instance, fraud detection, prediction of rare adverse drug reactions and prediction gene families. Failure to account for the class imbalance often causes inaccurate and decreased predictive performance of many classification algorithms.

Imbalanced learning (IL) aims to tackle the class imbalance problem to learn an unbiased model from imbalanced data. This is usually achieved by changing the training data distribution by resampling or reweighting. However, naive resampling or reweighting may introduce bias/variance to the training data, especially when the data has class-overlapping or contains noise.

Ensemble imbalanced learning (EIL) is known to effectively improve typical IL solutions by combining the outputs of multiple classifiers, thereby reducing the variance introduce by resampling/reweighting. -->

## List of implemented methods

**Currently (v0.1.3, 2021/06), *16* ensemble imbalanced learning methods were implemented:  
(Click to jump to the document page)**

- **Resampling-based**
  - *Under-sampling + Ensemble*
    1. **[`SelfPacedEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html) [1] ([in Github](https://github.com/ZhiningLiu1998/self-paced-ensemble))**
    2. **[`BalanceCascadeClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalanceCascadeClassifier.html) [2]**
    3. **[`BalancedRandomForestClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalancedRandomForestClassifier.html) [3] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html))**
    4. **[`EasyEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.EasyEnsembleClassifier.html) [2] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html))**
    5. **[`RUSBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.RUSBoostClassifier.html) [4] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html))**
    6. **[`UnderBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.UnderBaggingClassifier.html) [5] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
  - *Over-sampling + Ensemble*
    1. **[`OverBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBoostClassifier.html)**
    2. **[`SMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBoostClassifier.html) [6]**
    3. **[`KmeansSMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.KmeansSMOTEBoostClassifier.html)**
    4. **[`OverBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBaggingClassifier.html) [5] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
    5. **[`SMOTEBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBaggingClassifier.html) [7] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
- **Reweighting-based**
  - *Cost-sensitive Learning*
    1. **[`AdaCostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaCostClassifier.html) [8]**
    2. **[`AdaUBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaUBoostClassifier.html) [9]**
    3. **[`AsymBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AsymBoostClassifier.html) [10]**
- **Compatible**
  - **[`CompatibleAdaBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleAdaBoostClassifier.html) [11]**
  - **[`CompatibleBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleBaggingClassifier.html) [12]**

> **Note: `imbalanced-ensemble` is still under development, please see [API reference](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/api.html) for the latest list.**

## 5-min Quick Start with IMBENS

**Here, we provide some quick guides to help you get started with IMBENS.**  
**We strongly encourage users to check out the [**example gallery**](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#) for more comprehensive usage examples, which demonstrate many advanced features of IMBENS.**

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/example_gallery_snapshot.png)

### A minimal working example

Taking self-paced ensemble [1] as an example, it only requires less than 10 lines of code to deploy it:

```python
>>> from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
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

The [`imbalanced_ensemble.visualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/api.html) sub-module provide an [`ImbalancedEnsembleVisualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html).
It can be used to visualize the ensemble estimator(s) for further information or comparison.
Please refer to [**visualizer documentation**](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html) and [**examples**](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html) for more details.

**Fit an ImbalancedEnsembleVisualizer**
```python
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
from imbalanced_ensemble.ensemble import RUSBoostClassifier
from imbalanced_ensemble.ensemble import EasyEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier

# Fit ensemble classifiers
init_kwargs = {'base_estimator': DecisionTreeClassifier()}
ensembles = {
    'spe': SelfPacedEnsembleClassifier(**init_kwargs).fit(X_train, y_train),
    'rusboost': RUSBoostClassifier(**init_kwargs).fit(X_train, y_train),
    'easyens': EasyEnsembleClassifier(**init_kwargs).fit(X_train, y_train),
}

# Fit visualizer
from imbalanced_ensemble.visualizer import ImbalancedEnsembleVisualizer
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
Read more details in the [**fit documentation**](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html#imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.fit).

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

***Many samplers and utilities are adapted from* [imbalanced-learn](https://imbalanced-learn.org/), *which is an amazing project!***


## References

| #   | Reference |
|-----|-------|
| [1] | Zhining Liu, Wei Cao, Zhifeng Gao, Jiang Bian, Hechang Chen, Yi Chang, and Tie-Yan Liu. 2019. Self-paced Ensemble for Highly Imbalanced Massive Data Classification. 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020, pp. 841-852. |
| [2] | X.-Y. Liu, J. Wu, and Z.-H. Zhou, Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539–550, 2009. |
| [3] | Chen, Chao, Andy Liaw, and Leo Breiman. “Using random forest to learn imbalanced data.” University of California, Berkeley 110 (2004): 1-12. |
| [4] | C. Seiffert, T. M. Khoshgoftaar, J. Van Hulse, and A. Napolitano, Rusboost: A hybrid approach to alleviating class imbalance. IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, vol. 40, no. 1, pp. 185–197, 2010. |
| [5] | Maclin, R., & Opitz, D. (1997). An empirical evaluation of bagging and boosting. AAAI/IAAI, 1997, 546-551. |
| [6] | N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer, Smoteboost: Improving prediction of the minority class in boosting. in European conference on principles of data mining and knowledge discovery. Springer, 2003, pp. 107–119|
| [7] | S. Wang and X. Yao, Diversity analysis on imbalanced data sets by using ensemble models. in 2009 IEEE Symposium on Computational Intelligence and Data Mining. IEEE, 2009, pp. 324–331.|
| [8] | Fan, W., Stolfo, S. J., Zhang, J., & Chan, P. K. (1999, June). AdaCost: misclassification cost-sensitive boosting. In Icml (Vol. 99, pp. 97-105). |
| [9] | Shawe-Taylor, G. K. J., & Karakoulas, G. (1999). Optimizing classifiers for imbalanced training sets. Advances in neural information processing systems, 11(11), 253. |
| [10] | Viola, P., & Jones, M. (2001). Fast and robust classification using asymmetric adaboost and a detector cascade. Advances in Neural Information Processing System, 14. |
| [11] | Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139. |
| [12] | Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140. |
| [13] | Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17):1–5, 2017. |
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://zhiningliu.com"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zhining Liu</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/commits?author=ZhiningLiu1998" title="Code">💻</a> <a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-ZhiningLiu1998" title="Maintenance">🚧</a> <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3AZhiningLiu1998" title="Bug reports">🐛</a> <a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/commits?author=ZhiningLiu1998" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/leaphan"><img src="https://avatars.githubusercontent.com/u/35593707?v=4?s=100" width="100px;" alt=""/><br /><sub><b>leaphan</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Aleaphan" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/hannanhtang"><img src="https://avatars.githubusercontent.com/u/23587399?v=4?s=100" width="100px;" alt=""/><br /><sub><b>hannanhtang</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Ahannanhtang" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/huajuanren"><img src="https://avatars.githubusercontent.com/u/37321841?v=4?s=100" width="100px;" alt=""/><br /><sub><b>H.J.Ren</b></sub></a><br /><a href="https://github.com/ZhiningLiu1998/imbalanced-ensemble/issues?q=author%3Ahuajuanren" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
