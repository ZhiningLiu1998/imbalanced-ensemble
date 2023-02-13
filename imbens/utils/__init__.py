"""
The :mod:`imbens.utils` module includes various utilities.
"""

from ._docstring import Substitution

from ._evaluate import evaluate_print

from ._validation import check_neighbors_object
from ._validation import check_target_type
from ._validation import check_sampling_strategy

from ._validation_data import check_eval_datasets

from ._validation_param import check_eval_metrics
from ._validation_param import check_target_label_and_n_target_samples
from ._validation_param import check_balancing_schedule

__all__ = [
    "evaluate_print",
    "check_neighbors_object",
    "check_sampling_strategy",
    "check_target_type",
    "check_eval_datasets",
    "check_eval_metrics",
    "check_target_label_and_n_target_samples",
    "check_balancing_schedule",
    "Substitution",
]
