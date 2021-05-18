<h1 align="center">
Imbalanced Ensemble (IMBENS): <i>ensemble learning for class-imbalanced data in Python.</i>
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


***imbalanced-ensemble* (imported as `imbalanced_ensemble`) is a Python toolbox for quick implementing and deploying ensemble imbalanced learning algorithms.**
This package aims to provide users with easy-to-use ensemble imbalanced learning (EIL) methods and related utilities, so that everyone can quickly deploy EIL algorithms to their tasks. The EIL methods implemented in this package have unified APIs and are compatible with other popular Python machine-learning packages such as [scikit-learn](https://scikit-learn.org/stable/) and [imbalanced-learn](https://imbalanced-learn.org/stable/).

**Installation documentation, API documentation, and examples can be found on the [documentation](https://imbalanced-ensemble.readthedocs.io/).**

### Installation

imbalanced-ensemble requires following dependencies:

- [numpy](https://numpy.org/) (>=1.16.0)
- [pandas](https://pandas.pydata.org/) (>=1.1.3)
- [scipy](https://www.scipy.org/) (>=0.19.1)
- [joblib](https://pypi.org/project/joblib/) (>=0.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.24)
- [matplotlib](https://matplotlib.org/) (>=3.3.2)
- [seaborn](https://seaborn.pydata.org/) (>=0.11.0)
- [tqdm](https://tqdm.github.io/) (>=4.50.2)

**You can install imbalanced-ensemble from [PyPI](https://pypi.org/project/imbalanced-ensemble/) by running:**
```shell
$ pip install imbalanced-ensemble
```

**Or you can install imbalanced-ensemble by clone this repository:**
```shell
$ git clone https://github.com/ZhiningLiu1998/imbalanced-ensemble.git
$ cd imbalanced-ensemble
$ python setup.py install
```


### Table of Contents

- [Highlights](#highlights)
- [Background](#background)
- [List of implemented methods](#list-of-implemented-methods)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Highlights

- &#x1F34E; ***Unified, easy-to-use API design.***  
All ensemble learning methods implemented in IMBENS share a unified API design. 
Similar to sklearn, all methods have functions (e.g., `fit()`, `predict()`, `predict_proba()`) that allow users to deploy them with only a few lines of code.
- &#x1F34E; ***Extended functionalities, wider application scenarios.***  
*All methods in IMBENS are ready for **multi-class imbalanced classification**.* We extend binary ensemble imbalanced learning methods to get them to work under the multi-class scenario. Additionally, for supported methods, we provide more training options like class-wise resampling control, balancing scheduler during the ensemble training process, etc.
- &#x1F34E; ***Detailed training log, quick intuitive visualization.***   
We provide additional parameters (e.g., `eval_datasets`, `eval_metrics`, `training_verbose`) in `fit()` for users to control the information they want to monitor during the ensemble training. We also implement an [`EnsembleVisualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html) to quickly visualize the ensemble estimator(s) for providing further information/conducting comparison. See an example [here](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/basic/plot_basic_example.html#sphx-glr-auto-examples-basic-plot-basic-example-py).
- &#x1F34E; ***Wide compatiblilty.***   
Imbalanced-ensemble (IMBENS) is designed to be compatible with [scikit-learn](https://scikit-learn.org/stable/) (sklearn) and also other compatible projects like [imbalanced-learn](https://imbalanced-learn.org/stable/). Therefore, users can take advantage of various utilities from the sklearn community for data processing/cross-validation/hyper-parameter tuning, etc.

## Background

Class-imbalance (also known as the long-tail problem in multi-class) is the fact that the classes are not represented equally in a classification problem, which is quite common in practice. For instance, fraud detection, prediction of rare adverse drug reactions and prediction gene families. Failure to account for the class imbalance often causes inaccurate and decreased predictive performance of many classification algorithms.

Imbalanced learning (IL) aims to tackle the class imbalance problem to learn an unbiased model from imbalanced data. This is usually achieved by changing the training data distribution by resampling or reweighting. However, naive resampling or reweighting may introduce bias/variance to the training data, especially when the data has class-overlapping or contains noise.

Ensemble imbalanced learning (EIL) is known to effectively improve typical IL solutions by combining the outputs of multiple classifiers, thereby reducing the variance introduce by resampling/reweighting.

## List of implemented methods

**Currently, *16* ensemble imbalanced learning methods were implemented:**

- Resampling-based
  - Under-sampling + Ensemble
    1. [`SelfPacedEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html) [1] ([in Github](https://github.com/ZhiningLiu1998/self-paced-ensemble))
    2. [`BalanceCascadeClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalanceCascadeClassifier.html) [2] 
    3. [`BalancedRandomForestClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalancedRandomForestClassifier.html) [3] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html))
    4. [`EasyEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.EasyEnsembleClassifier.html) [2] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html))
    5. [`RUSBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.RUSBoostClassifier.html) [4] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html))
    6. [`UnderBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.UnderBaggingClassifier.html) [5] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))
  - Over-sampling + Ensemble
    1. [`OverBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBoostClassifier.html)
    2. [`SMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBoostClassifier.html) [6]
    3. [`KmeansSMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.KmeansSMOTEBoostClassifier.html)
    4. [`OverBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBaggingClassifier.html) [5] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))
    5. [`SMOTEBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBaggingClassifier.html) [7] ([imblearn version](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))
- Reweighting-based
  - Cost-sensitive Learning
    1. [`AdaCostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaCostClassifier.html) [8]
    2. [`AdaUBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaUBoostClassifier.html) [9]
    3. [`AsymBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AsymBoostClassifier.html) [10]
- Compatible
  - [`CompatibleAdaBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleAdaBoostClassifier.html) [11]
  - [`CompatibleBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleBaggingClassifier.html) [12]


> **Note: `imbalanced-ensemble` is still under development.**

## Usage

**Taking self-paced ensemble [1] as an example, it only requires less than 10 lines of code to deploy it:**

```python
>>> from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
>>> from sklearn.datasets import make_classification
>>> 
>>> X, y = make_classification(n_samples=1000, n_classes=3,
...                            n_informative=4, weights=[0.2, 0.3, 0.5],
...                            random_state=0)
>>> clf = SelfPacedEnsembleClassifier(random_state=0)
>>> clf.fit(X, y)  
SelfPacedEnsembleClassifier(...)
>>> clf.predict(X)  
array([...])
```

**For more examples, please refer to the [documentation](https://imbalanced-ensemble.readthedocs.io/).**


## Acknowledgements

***many samplers and utilities are adapted from* [imbalanced-learn](https://imbalanced-learn.org/), *which is an amazing project!***


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