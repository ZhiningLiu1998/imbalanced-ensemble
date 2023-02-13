"""Toolbox for ensemble learning on class-imbalanced dataset.

``imbalanced-ensemble`` is a set of python-based ensemble learning methods for 
dealing with class-imbalanced classification problems in machine learning.

Subpackages
-----------
ensemble
    Module which provides ensemble imbalanced learning methods.
sampler
    Module which provides samplers for resampling class-imbalanced data.
visualizer
    Module which provides a visualizer for convenient visualization of 
    ensemble learning process and results.
metrics
    Module which provides metrics to quantified the classification performance
    with imbalanced dataset.
utils
    Module including various utilities.
exceptions
    Module including custom warnings and error classes used across
    imbalanced-learn.
pipeline
    Module which allowing to create pipeline with scikit-learn estimators.
"""

from . import ensemble
from . import sampler
from . import visualizer
from . import metrics
from . import utils
from . import exceptions
from . import pipeline
from . import datasets

from .sampler.base import FunctionSampler

from ._version import __version__

__all__ = [
    "ensemble",
    "sampler",
    "visualizer",
    "metrics",
    "utils",
    "exceptions",
    "pipeline",
    "datasets",
    "FunctionSampler",
    "__version__",
]
