"""Test utilities for plot."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


import numpy as np

from imbens.utils._plot import *

X = np.array(
    [
        [2.45166, 1.86760],
        [1.34450, -1.30331],
        [1.02989, 2.89408],
        [-1.94577, -1.75057],
        [1.21726, 1.90146],
        [2.00194, 1.25316],
        [2.31968, 2.33574],
        [1.14769, 1.41303],
        [1.32018, 2.17595],
        [-1.74686, -1.66665],
        [-2.17373, -1.91466],
        [2.41436, 1.83542],
        [1.97295, 2.55534],
        [-2.12126, -2.43786],
        [1.20494, 3.20696],
        [-2.30158, -2.39903],
        [1.76006, 1.94323],
        [2.35825, 1.77962],
        [-2.06578, -2.07671],
        [0.00245, -0.99528],
    ]
)
y = np.array([2, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 0])


def test_plot_scatter():
    plot_scatter(X, y)


def test_plot_class_distribution():
    plot_class_distribution(y)


def test_plot_2Dprojection_and_cardinality():
    plot_2Dprojection_and_cardinality(X, y)
