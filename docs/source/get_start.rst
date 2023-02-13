Getting Started
***************

Background
====================================

Class-imbalance (also known as the long-tail problem) is the fact that the 
classes are not represented equally in a classification problem, which is 
quite common in practice. For instance, fraud detection, prediction of 
rare adverse drug reactions and prediction gene families. Failure to account 
for the class imbalance often causes inaccurate and decreased predictive 
performance of many classification algorithms. 

Imbalanced learning (IL) aims 
to tackle the class imbalance problem to learn an unbiased model from 
imbalanced data. This is usually achieved by changing the training data 
distribution by resampling or reweighting. However, naive resampling or 
reweighting may introduce bias/variance to the training data, especially 
when the data has class-overlapping or contains noise.

Ensemble imbalanced learning (EIL) is known to effectively improve typical 
IL solutions by combining the outputs of multiple classifiers, thereby 
reducing the variance introduce by resampling/reweighting. 

About ``imbens``
====================================

``imbens`` aims to provide users with easy-to-use EIL methods 
and related utilities, so that everyone can quickly deploy EIL algorithms 
to their tasks. The EIL methods implemented in this package have 
unified APIs and are compatible with other popular Python machine-learning 
packages such as `scikit-learn <https://scikit-learn.org/stable/index.html>`__
and `imbalanced-learn <https://imbalanced-learn.org/stable/>`__.

``imbens`` is an early version software and is under development.
Any kinds of contributions are welcome!

> Note: *many resampling algorithms and utilities are adapted from* 
`imbalanced-learn <https://imbalanced-learn.org/>`__, *which is an amazing 
project!*