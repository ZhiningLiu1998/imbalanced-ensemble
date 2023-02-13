"""
The :mod:`imbens.datasets` provides methods to generate
imbalanced data.
"""

from ._imbalance import make_imbalance
from ._imbalance import generate_imbalance_data

from ._zenodo import fetch_datasets

__all__ = [
    "make_imbalance",
    "generate_imbalance_data",
    "fetch_datasets",
]
