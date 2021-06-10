![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/imbalanced_ensemble_header.png)

<h2 align="center">
  Imbalanced Ensemble: 在多类别不平衡(长尾)数据上部署集成学习算法
</h2>

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

#### 链接：[[Github](https://github.com/ZhiningLiu1998/imbalanced-ensemble)] [[API/文档/使用手册](https://imbalanced-ensemble.readthedocs.io/)] [[PyPI](https://pypi.org/project/imbalanced-ensemble/)] [[版本历史](https://imbalanced-ensemble.readthedocs.io/en/latest/release_history.html)] [[源代码](https://github.com/ZhiningLiu1998/imbalanced-ensemble/tree/main/imbalanced_ensemble)] [[下载](https://pypi.org/project/imbalanced-ensemble/#files)]

**imbalanced-ensemble（IMBENS）是一个 Python 库/软件包。它主要用于在类别不平衡数据上快速实现和部署集成学习算法。截至目前（v0.1.3, 2021/06/08），IMBENS已实现了14种不同的不平衡集成学习算法，从经典的SMOTEBoost (2003) 到最近的 SPE (2020)，从欠采样、过采样到代价敏感学习，全部包括在内。IMBENS实现的大部分方法都具有详细的 [文档和使用手册](https://imbalanced-ensemble.readthedocs.io/)，并将在未来继续更新加入其他方法。**

**欢迎 star / issue / PR !**

**IMBENS的主要特性有：**

- &#x1F34E; **统一易用的API设计，便于使用和二次开发**
- &#x1F34E; **所有实现的方法均原生支持多分类不平衡问题**
- &#x1F34E; **在可能的情况下，使用 [joblib](https://github.com/joblib/joblib) 实现并行训练/预测以优化性能**
- &#x1F34E; **强大的、可定制的、交互式的模型训练日志记录和可视化工具**
- &#x1F34E; **完全兼容其他的流行软件包，如 [scikit-learn](https://scikit-learn.org/stable/) 和 [imbalanced-learn](https://imbalanced-learn.org/stable/)**

**API 使用示例：**
```python
# Train an SPE classifier
from imbalanced_ensemble.ensemble import SelfPacedEnsembleClassifier
clf = SelfPacedEnsembleClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict with an SPE classifier
y_pred = clf.predict(X_test)
```

## 目录

- [目录](#目录)
- [安装IMBENS](#安装imbens)
- [已实现的方法](#已实现的方法)
- [5分钟快速上手IMBENS](#5分钟快速上手imbens)
    - [基础示例](#基础示例)
    - [对集成分类器进行可视化](#对集成分类器进行可视化)
    - [自定义训练日志](#自定义训练日志)
- [有关类别不平衡学习](#有关类别不平衡学习)
- [参考文献](#参考文献)

## 安装IMBENS

推荐使用pip进行安装：
```shell
$ pip install imbalanced-ensemble            # 正常安装
$ pip install --upgrade imbalanced-ensemble  # 升级安装
```
> IMBENS更新较为频繁，请确认安装的是最新版本以规避可能的问题。

或者从Github克隆到本地安装：
```shell
$ git clone https://github.com/ZhiningLiu1998/imbalanced-ensemble.git
$ cd imbalanced-ensemble
$ pip install .
```
imbalanced-ensemble 具有以下依赖项:
- [Python](https://www.python.org/) (>=3.6)
- [numpy](https://numpy.org/) (>=1.16.0)
- [pandas](https://pandas.pydata.org/) (>=1.1.3)
- [scipy](https://www.scipy.org/) (>=0.19.1)
- [joblib](https://pypi.org/project/joblib/) (>=0.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.24)
- [matplotlib](https://matplotlib.org/) (>=3.3.2)
- [seaborn](https://seaborn.pydata.org/) (>=0.11.0)
- [tqdm](https://tqdm.github.io/) (>=4.50.2)

## 已实现的方法

**目前，IMBENS实现了16种集成学习方法（点击类名可跳转至文档页面）：**

- **基于重采样的方法**
  - *降采样 + 集成*
    1. **[`SelfPacedEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html) [1] ([in Github](https://github.com/ZhiningLiu1998/self-paced-ensemble)) ([in 知乎/Zhihu](https://zhuanlan.zhihu.com/p/86891438))**
    2. **[`BalanceCascadeClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalanceCascadeClassifier.html) [2]**
    3. **[`BalancedRandomForestClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.BalancedRandomForestClassifier.html) [3] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html))**
    4. **[`EasyEnsembleClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.EasyEnsembleClassifier.html) [2] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html))**
    5. **[`RUSBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.RUSBoostClassifier.html) [4] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.RUSBoostClassifier.html))**
    6. **[`UnderBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.UnderBaggingClassifier.html) [5] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
  - *过采样 + 集成*
    1. **[`OverBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBoostClassifier.html)**
    2. **[`SMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBoostClassifier.html) [6]**
    3. **[`KmeansSMOTEBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.KmeansSMOTEBoostClassifier.html)**
    4. **[`OverBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.OverBaggingClassifier.html) [5] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
    5. **[`SMOTEBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.over_sampling.SMOTEBaggingClassifier.html) [7] ([in imblearn](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html))**
- **基于重加权的方法**
  - *代价敏感学习 + 集成*
    1. **[`AdaCostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaCostClassifier.html) [8]**
    2. **[`AdaUBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AdaUBoostClassifier.html) [9]**
    3. **[`AsymBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.reweighting.AsymBoostClassifier.html) [10]**
- **兼容方法**
  - **[`CompatibleAdaBoostClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleAdaBoostClassifier.html) [11]**
  - **[`CompatibleBaggingClassifier`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.compatible.CompatibleBaggingClassifier.html) [12]**

## 5分钟快速上手IMBENS

#### 基础示例

一个可运行的示例：以 SPE[1] 为例，仅需少于10行的代码就可以部署它：
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

#### 对集成分类器进行可视化

[`imbalanced_ensemble.visualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/api.html)子模块提供了一个可视化器类[`ImbalancedEnsembleVisualizer`](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html)。它可对集成分类器进行直观的可视化来获取更多信息或比较不同方法的性能。请阅读 [可视化工具的文档](https://imbalanced-ensemble.readthedocs.io/en/latest/api/visualizer/_autosummary/imbalanced_ensemble.visualizer.ImbalancedEnsembleVisualizer.html) 以及 [使用示例](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html) 以获取更详细的信息。

拟合一个可视化器
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
使用可视化器展示不同方法的性能曲线（performance curve）
```python
fig, axes = visualizer.performance_lineplot()
```
![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/examples/visualize_performance_example.png)

使用可视化器展示不同方法的混淆矩阵（confusion matrices）
```python
fig, axes = visualizer.confusion_matrix_heatmap()
```
![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/imbalanced-ensemble/examples/visualize_confusion_matrix_example.png)

#### 自定义训练日志

IMBENS 中实现的所有集成分类器都支持打印可自定义的训练日志。训练日志由 fit() 方法的 eval_datasets、eval_metrics 和 training_verbose 3 个参数控制。请阅读 [fit() 方法的文档](https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html#imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.fit) 来获得更详细的使用方法。

启用训练日志
```python
clf.fit(..., train_verbose=True)
```

自定义训练日志的粒度和内容
```python
clf.fit(...,
        train_verbose={
            'granularity': 10,
            'print_distribution': False,
            'print_metrics': True,
        })
```

增加验证集（可以有多个名字不重复的验证集）
```python
clf.fit(..., 
        eval_datasets={
            'valid': (X_valid, y_valid)
        })
```

自定义所使用的评价指标
```python
from sklearn.metrics import accuracy_score, f1_score
clf.fit(..., 
        eval_metrics={
            'acc': (accuracy_score, {}),
            'weighted_f1': (f1_score, {'average':'weighted'}),
        })
```

## 有关类别不平衡学习

“类别不平衡”指一个分类任务的数据中来自不同类别的样本数目相差悬殊。传统的机器学习模型假设数据的边缘分布P(Y)是大致均匀的，因此它们通常被设计为优化分类的准确率(accuracy)，并未考虑不同类别的样本数量差异。在类别不平衡的情况下，样本数量少的类别对分类准确率的影响很小，因此直接优化分类准确率的模型会难以学习到少数类的模式，导致对于少数类的预测结果较差。尽管少数类的样本个数更少，表示的质量也更差，但其通常会携带更重要的信息，因此**一般我们更关注模型正确分类少数类样本的能力**。因此我们希望能够使用某些手段修正不平衡数据给模型带来的偏见，得到一个无偏的预测模型。从类别不平衡数据中学习无偏模型的问题通常被称为**不平衡学习**，在多类别场景下也被称为**长尾学习**。

> 更多有关不平衡学习的背景、定义、评价准则等，请参考：[极端类别不平衡数据下的分类问题S02：问题概述，模型选择及人生经验 - 知乎 (zhihu.com) ](https://zhuanlan.zhihu.com/p/66373943)。

我们可以大致对常见的不平衡学习技术做出如下分类：

1. **重采样** (re-sampling): 直接更改训练集中不同类别样本的数量
   1. **欠采样** (under-sampling): 丢弃多数类中的样本
   2. **过采样** (over-sampling): 为少数类生成新的样本
   3. **数据清洁** (cleaning): 根据特定的规则清除一些样本
   4. **混合采样** (hybrid-sampling): 结合上述方法，常见组合为过采样+数据清洁
2. **重加权** (re-weighting): 更改不同样本在模型训练中的权重
   1. **类别重加权** (class-wise reweighting): 为不同类别的样本分配不同权重，如代价敏感学习 (cost-sensitive learning) 类方法
   2. **样本重加权** (instance-wise reweighting): 为不同的样本分配不同权重，如难例挖掘 (hard example mining) 类方法
3. **其他方法**，如后验概率调整 (posterior probability adjustment) 等。

> 若对相关的研究论文以及子领域划分感兴趣，请参考[有关类别不平衡(长尾)机器学习的一切：论文，代码，框架与库 -知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/111460698) 以及 [ZhiningLiu1998/awesome-imbalanced-learning: A curated list of awesome imbalanced learning papers, codes, frameworks, and libraries. (github.com) ](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning)。

## 参考文献

| #   | Reference |
|-----|-------|
| [1] | Liu, Z., Cao, W., Gao, Z., Bian, J., Chen, H., Chang, Y., & Liu, T. Y. (2020, April). Self-paced ensemble for highly imbalanced massive data classification. In 2020 IEEE 36th International Conference on Data Engineering (ICDE) (pp. 841-852). IEEE. |
| [2] | Liu, X. Y., Wu, J., & Zhou, Z. H. (2008). Exploratory undersampling for class-imbalance learning. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39(2), 539-550. |
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