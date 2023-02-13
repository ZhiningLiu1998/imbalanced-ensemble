"""Utilities for data validation."""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT


from collections import OrderedDict

from sklearn.utils import check_X_y

VALID_DATA_INFO = (
    "'eval_datasets' should be a `dict` of validation data,"
    + " e.g., {..., dataset_name : (X_valid, y_valid), ...}."
)

TRAIN_DATA_NAME = "train"


def _check_eval_datasets_name(data_name):
    if not isinstance(data_name, str):
        raise TypeError(
            VALID_DATA_INFO
            + f" The keys must be `string`, got {type(data_name)}, "
            + f" please check your usage."
        )
    if data_name == TRAIN_DATA_NAME:
        raise ValueError(
            f"The name {TRAIN_DATA_NAME} is reserved for the training"
            f" data (it will automatically add into the 'eval_datasets_'"
            f" attribute after calling `fit`), please use another name"
            f" for your evaluation dataset."
        )
    return data_name


def _check_eval_datasets_tuple(data_tuple, data_name, **check_x_y_kwargs):
    if not isinstance(data_tuple, tuple):
        raise TypeError(
            VALID_DATA_INFO
            + f" The value of '{data_name}' is {type(data_tuple)} (should be tuple),"
            + f" please check your usage."
        )
    elif len(data_tuple) != 2:
        raise ValueError(
            VALID_DATA_INFO
            + f" The data tuple of '{data_name}' has {len(data_tuple)} element(s)"
            + f" (should be 2), please check your usage."
        )
    else:
        X, y = check_X_y(data_tuple[0], data_tuple[1], **check_x_y_kwargs)
        return (X, y)


def _check_eval_datasets_dict(eval_datasets_dict, **check_x_y_kwargs):

    if TRAIN_DATA_NAME in eval_datasets_dict.keys():
        raise ValueError(
            f"The name '{TRAIN_DATA_NAME}' could not be used"
            f" for the validation datasets. Please use another name."
        )

    eval_datasets_dict_ = {}
    for data_name, data_tuple in eval_datasets_dict.items():
        data_name_ = _check_eval_datasets_name(data_name)
        data_tuple_ = _check_eval_datasets_tuple(
            data_tuple, data_name_, **check_x_y_kwargs
        )
        eval_datasets_dict_[data_name_] = data_tuple_

    return eval_datasets_dict_


def _all_elements_equal(list_to_check: list) -> bool:
    if len(list_to_check) == 1:
        return True
    return all(
        [
            (list_to_check[i] == list_to_check[i + 1])
            for i in range(len(list_to_check) - 1)
        ]
    )


def check_eval_datasets(
    eval_datasets, X_train_=None, y_train_=None, **check_x_y_kwargs
):
    """Check `eval_datasets` parameter."""
    # Whether to add training data in to returned data dictionary
    if X_train_ is None and y_train_ is None:
        result_datasets = OrderedDict({})
    else:
        result_datasets = OrderedDict({TRAIN_DATA_NAME: (X_train_, y_train_)})

    # If eval_datasets is None
    #   return data dictionary
    if eval_datasets == None:
        return result_datasets

    # If eval_datasets is dict
    elif isinstance(eval_datasets, dict):

        # Check dict and validate all names (keys) and data tuples (values)
        eval_datasets_ = _check_eval_datasets_dict(eval_datasets, **check_x_y_kwargs)

        # Combine train_datasets and eval_datasets_
        result_datasets.update(eval_datasets_)

        # Ensure all datasets have the same number of features
        if not _all_elements_equal(
            [data_tuple[0].shape[1] for data_tuple in result_datasets.values()]
        ):
            raise ValueError(
                f"The train + evaluation datasets have inconsistent number of"
                f" features. Make sure that the data given in 'eval_datasets'"
                f" and the training data ('X', 'y') are sampled from the same"
                f" task/distribution."
            )
        return result_datasets

    # Else raise TypeError
    else:
        raise TypeError(
            VALID_DATA_INFO + f" Got {type(eval_datasets)}, please check your usage."
        )
