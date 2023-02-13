"""
The :mod:`imbens.exceptions` module includes all custom warnings and error
classes and functions used across imbalanced-learn.
"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
# License: MIT


def raise_isinstance_error(variable_name, possible_type, variable):
    raise ValueError(
        f"{variable_name} has to be one of {possible_type}. "
        f"Got {type(variable)} instead."
    )
