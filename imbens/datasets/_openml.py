# %%
"""Collection of imbalanced datasets from OpenML.

This module provides functions to fetch and preprocess datasets from OpenML,
including handling categorical features and saving/loading datasets locally.
The datasets are preprocessed to handle missing values, categorical features,
and standardize numerical features.
"""

# Author: Zhining Liu
# License: MIT

import pandas as pd
import numpy as np
import openml
from pandas.api.types import is_any_real_numeric_dtype
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pathlib import Path
from platformdirs import user_cache_dir
from typing import OrderedDict
import tqdm


def _fetch_openml_data(openml_id, cat_preprocess="onehot", data_home=None):
    """
    Fetches a dataset from OpenML, preprocesses it, and caches it locally.

    Parameters
    ----------
    openml_id : int
        The OpenML dataset ID.

    cat_preprocess : {'drop', 'onehot', 'ordinal'}, optional (default='onehot')
        The method for preprocessing categorical features:
        - 'drop': Drop categorical features.
        - 'onehot': One-hot encode categorical features.
        - 'ordinal': Ordinal encode categorical features.

    data_home : str or Path or None, optional (default=None)
        The directory where the dataset should be stored. If None, it will use the default user cache directory.

    Returns
    -------
    X : pandas.DataFrame
        The features of the dataset.

    y : pandas.Series
        The target variable.

    Notes
    -----
    If the dataset is already cached locally, it will be loaded from the cache.
    Otherwise, the dataset will be downloaded, preprocessed, and saved locally.
    """

    def get_data_path(data_home):
        if data_home is None:
            data_home = Path(user_cache_dir("imbens")) / "datasets"
            data_home.mkdir(parents=True, exist_ok=True)
            return data_home
        else:
            # check if store_path is a valid path
            if not isinstance(data_home, Path):
                data_home = Path(data_home)
            if not data_home.is_dir():
                raise ValueError(f"data_home {data_home} is not a directory")
            data_home.mkdir(parents=True, exist_ok=True)
            return data_home

    def save_data(X, y, data_home, file_name):
        np.savez_compressed(data_home / file_name, X=X, y=y)
        # print(f"Data saved to {data_home / file_name}")

    def load_data(data_home, file_name):
        data = np.load(data_home / file_name, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        return X, y

    assert isinstance(openml_id, int), "openml_id must be an integer"
    assert cat_preprocess in [
        "drop",
        "onehot",
        "ordinal",
    ], "cat_preprocess must be one of ['drop', 'onehot', 'ordinal']"
    assert isinstance(
        data_home, (str, Path, type(None))
    ), "data_home must be a string, Path or None"

    data_home = get_data_path(data_home)
    dataset = openml.datasets.get_dataset(openml_id)
    file_name = f"{dataset.id}_{cat_preprocess}_{dataset.name}.npz"

    # check if data is already cached
    try:
        X, y = load_data(data_home, file_name)
        # print(f"Data loaded from {data_home / file_name}")
        return X, y
    except FileNotFoundError:
        # print(f"Data not cached in {data_home}, processing from scratch.")
        pass

    X, y, cat_ind, feat_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # target encoding, smaller classes aer assigned larger values
    y_map = y.value_counts().sort_values()[::-1].index
    y_map = {v: i for i, v in enumerate(y_map)}
    y = y.map(y_map)

    feats_type = {"num": [], "bin_cat": [], "multi_cat": [], "drop": []}
    # preprocessing
    for feat in X.columns:
        if is_any_real_numeric_dtype(X[feat]):
            feats_type["num"].append(feat)
            if X[feat].isnull().sum() > 0:
                # for numerical columns, fill nan with mean
                X[feat] = X[feat].fillna(X[feat].mean())
        else:  # categorical column
            if X[feat].isnull().sum() > 0:
                # for categorical columns, fill nan with most frequent value
                X[feat] = X[feat].fillna(X[feat].mode().iloc[0])
            n_unique = len(X[feat].unique())
            if n_unique > 2:
                # try to convert to numeric
                try:
                    X[feat] = pd.to_numeric(X[feat])
                except ValueError:
                    if n_unique <= 50:
                        feats_type["multi_cat"].append(feat)
                    else:
                        feats_type["drop"].append(feat)
            else:
                feats_type["bin_cat"].append(feat)

    ord_encoder = OrdinalEncoder()
    X = X.drop(columns=feats_type["drop"])
    # encode categorical columns
    if cat_preprocess == "drop":
        X = X.drop(columns=feats_type["multi_cat"])
        X = X.drop(columns=feats_type["bin_cat"])
    elif cat_preprocess == "onehot":
        X[feats_type["bin_cat"]] = ord_encoder.fit_transform(X[feats_type["bin_cat"]])
        X = pd.get_dummies(X, columns=feats_type["multi_cat"])
    elif cat_preprocess == "ordinal":
        # ordinal encoding for multi categorical columns
        X[feats_type["bin_cat"]] = ord_encoder.fit_transform(X[feats_type["bin_cat"]])
        X[feats_type["multi_cat"]] = ord_encoder.fit_transform(
            X[feats_type["multi_cat"]]
        )
    else:
        raise ValueError(f"Unknown cat_preprocess: {cat_preprocess}")

    # standardize numerical columns
    scaler = StandardScaler()
    X[feats_type["num"]] = scaler.fit_transform(X[feats_type["num"]])

    # save data
    save_data(X, y, data_home, file_name)

    return X, y


def fetch_openml_datasets(
    target_type="all", imalance_type="all", cat_preprocess="onehot", data_home=None
):
    """
    Fetches multiple datasets from OpenML [1]_ based on the specified criteria and preprocesses them.

    .. versionadded:: 0.3.0

    Parameters
    ----------
    target_type : {'all', 'binary', 'multiclass'}, optional (default='all')
        The type of target variable:

        - ``all``: No filter on target variable type.
        - ``binary``: Datasets with binary target variables.
        - ``multiclass``: Datasets with multiclass target variables.

    imalance_type : {'all', 'low', 'medium', 'high', 'extreme'}, optional (default='all')
        The imbalance type of the dataset:

        - ``all``: No filter on imbalance level.
        - ``low``: Datasets with low imbalance ratio (less than 5:1).
        - ``medium``: Datasets with medium imbalance ratio (between 5:1 and 10:1).
        - ``high``: Datasets with high imbalance ratio (between 10:1 and 50:1).
        - ``extreme``: Datasets with extreme imbalance ratio (greater than 50:1).

    cat_preprocess : {'drop', 'onehot', 'ordinal'}, optional (default='onehot')
        The method for preprocessing categorical features:

        - ``drop``: Drop categorical features.
        - ``onehot``: One-hot encode categorical features.
        - ``ordinal``: Ordinal encode categorical features.

    data_home : str or Path or None, optional (default=None)
        The directory where the datasets should be stored. If None, it will use the default user cache directory.

    Returns
    -------
    datasets : OrderedDict of Bunch object,
        The ordered is ranked by the imbalance ratio of the dataset. Each Bunch object ---
        referred as dataset --- have the following attributes:

    dataset.data : ndarray of shape (n_samples, n_features)

    dataset.target : ndarray of shape (n_samples,)

    dataset.openml_id : int (OpenML dataset ID)

    dataset.IR : float (imbalance ratio of the dataset)

    References
    ----------
    .. [1] Vanschoren, J., Van Rijn, J. N., Bischl, B., & Torgo, L. (2014).
       OpenML: networked science in machine learning. ACM SIGKDD Explorations
       Newsletter, 15(2), 49-60.

    Notes
    -----
    This function fetches datasets based on target and imbalance type filters, preprocesses them,
    and caches the datasets locally. The characteristics of the available datasets are presented
    in the table below.

    +------+-----------------------------------------------+--------------+----------+------------+-------+
    | ID   | Name                                          | OpenML ID    | Ratio    |         #S |    #F |
    +======+===============================================+==============+==========+============+=======+
    |    1 | bwin_amlb                                     | 45717        | 2.01:1   |        530 |    13 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    2 | mozilla4                                      | 1046         | 2.04:1   |     15,545 |     5 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    3 | mc2                                           | 1054         | 2.10:1   |        161 |    39 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    4 | wholesale-customers                           | 1511         | 2.10:1   |        440 |     8 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    5 | vertebra-column                               | 1524         | 2.10:1   |        310 |     6 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    6 | law-school-admission-bianry                   | 43904        | 2.11:1   |     20,800 |    14 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    7 | bank32nh                                      | 833          | 2.22:1   |      8,192 |    32 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    8 | elevators                                     | 846          | 2.24:1   |     16,599 |    18 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |    9 | cpu_small                                     | 735          | 2.31:1   |      8,192 |    12 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   10 | Credit_Approval_Classification                | 46503        | 2.33:1   |      1,000 |    50 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   11 | house_8L                                      | 843          | 2.38:1   |     22,784 |     8 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   12 | house_16H                                     | 821          | 2.38:1   |     22,784 |    16 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   13 | phoneme                                       | 1489         | 2.41:1   |      5,404 |     5 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   14 | ilpd-numeric                                  | 41945        | 2.49:1   |        583 |    10 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   15 | planning-relax                                | 1490         | 2.50:1   |        182 |    12 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   16 | MiniBooNE                                     | 41150        | 2.56:1   |    130,064 |    50 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   17 | machine_cpu                                   | 733          | 2.73:1   |        209 |     6 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   18 | telco-customer-churn                          | 42178        | 2.77:1   |      7,043 |    39 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   19 | haberman                                      | 43           | 2.78:1   |        306 |     3 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   20 | vehicle                                       | 994          | 2.88:1   |        846 |    18 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   21 | cpu                                           | 796          | 2.94:1   |        209 |    36 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   22 | ada                                           | 41156        | 3.03:1   |      4,147 |    48 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   23 | adult                                         | 45068        | 3.18:1   |     48,842 |   107 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   24 | blood-transfusion-service-center              | 1464         | 3.20:1   |        748 |     4 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   25 | default-of-credit-card-clients                | 42477        | 3.52:1   |     30,000 |    23 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   26 | Customer_Churn_Classification                 | 46362        | 3.74:1   |    175,028 |    24 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   27 | SPECTF                                        | 1600         | 3.85:1   |        267 |    44 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   28 | Medical-Appointment-No-Shows                  | 43439        | 3.95:1   |    110,527 |    36 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   29 | JapaneseVowels                                | 976          | 5.17:1   |      9,961 |    14 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   30 | ibm-employee-attrition                        | 43893        | 5.20:1   |      1,470 |    53 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   31 | first-order-theorem-proving                   | 1475         | 5.26:1   |      6,118 |    51 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   32 | user-knowledge                                | 1508         | 5.38:1   |        403 |     5 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   33 | online-shoppers-intention                     | 45560        | 5.46:1   |     12,330 |    28 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   34 | kc1                                           | 1067         | 5.47:1   |      2,109 |    21 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   35 | thoracic-surgery                              | 1506         | 5.71:1   |        470 |    16 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   36 | UCI_churn                                     | 44232        | 5.90:1   |      3,333 |    18 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   37 | arsenic-female-bladder                        | 949          | 5.99:1   |        559 |     4 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   38 | okcupid_stem                                  | 45067        | 6.83:1   |     26,677 |   117 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   39 | ecoli                                         | 40671        | 7.15:1   |        327 |     7 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   40 | pc4                                           | 1049         | 7.19:1   |      1,458 |    37 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   41 | bank-marketing                                | 1558         | 7.68:1   |      4,521 |    48 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   42 | Diabetes-130-Hospitals_(Fairlearn)            | 43903        | 7.96:1   |    101,766 |    50 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   43 | Otto-Group-Product-Classification-Challenge   | 45548        | 8.36:1   |     61,878 |    93 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   44 | eucalyptus                                    | 43925        | 8.54:1   |      4,331 |    26 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   45 | pendigits                                     | 1019         | 8.61:1   |     10,992 |    16 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   46 | pc3                                           | 1050         | 8.77:1   |      1,563 |    37 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   47 | page-blocks-bin                               | 1021         | 8.77:1   |      5,473 |    10 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   48 | optdigits                                     | 980          | 8.83:1   |      5,620 |    64 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   49 | mfeat-karhunen                                | 1020         | 9.00:1   |      2,000 |    64 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   50 | mfeat-fourier                                 | 971          | 9.00:1   |      2,000 |    76 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   51 | mfeat-zernike                                 | 995          | 9.00:1   |      2,000 |    47 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   52 | Pulsar-Dataset-HTRU2                          | 45558        | 9.92:1   |     17,898 |     8 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   53 | vowel                                         | 1016         | 10.00:1  |        990 |    26 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   54 | heart-h                                       | 1565         | 12.53:1  |        294 |    13 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   55 | pc1                                           | 1068         | 13.40:1  |      1,109 |    21 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   56 | seismic-bumps                                 | 45562        | 14.20:1  |      2,584 |    22 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   57 | ozone-level-8hr                               | 1487         | 14.84:1  |      2,534 |    72 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   58 | microaggregation2                             | 41671        | 15.02:1  |     20,000 |    20 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   59 | Sick_numeric                                  | 41946        | 15.33:1  |      3,772 |    29 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   60 | insurance_company                             | 46281        | 15.76:1  |      9,822 |    85 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   61 | wilt                                          | 40983        | 17.54:1  |      4,839 |     5 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   62 | Click_prediction_small                        | 1217         | 21.37:1  |    149,639 |    11 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   63 | jannis                                        | 41168        | 22.83:1  |     83,733 |    54 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   64 | letter                                        | 977          | 23.60:1  |     20,000 |    16 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   65 | walking-activity                              | 1509         | 24.14:1  |    149,332 |     4 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   66 | helena                                        | 41169        | 36.08:1  |     65,196 |    27 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   67 | mammography                                   | 310          | 42.01:1  |     11,183 |     6 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   68 | dis                                           | 40713        | 64.03:1  |      3,772 |    29 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   69 | Satellite                                     | 40900        | 67.00:1  |      5,100 |    36 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   70 | Employee-Turnover-at-TECHCO                   | 43551        | 68.74:1  |     34,452 |     9 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   71 | page-blocks                                   | 30           | 175.46:1 |      5,473 |    10 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   72 | allbp                                         | 40707        | 257.79:1 |      3,772 |    29 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    |   73 | CreditCardFraudDetection                      | 42397        | 577.88:1 |    284,807 |    30 |
    +------+-----------------------------------------------+--------------+----------+------------+-------+
    """
    assert target_type in [
        "all",
        "binary",
        "multiclass",
    ], "target_type must be one of ['all', 'binary', 'multiclass']"
    assert imalance_type in [
        "all",
        "low",
        "medium",
        "high",
        "extreme",
    ], "imalance_type must be one of ['all', 'low', 'medium', 'high', 'extreme']"
    assert cat_preprocess in [
        "drop",
        "onehot",
        "ordinal",
    ], "cat_preprocess must be one of ['drop', 'onehot', 'ordinal']"
    assert isinstance(
        data_home, (str, Path, type(None))
    ), "data_home must be a string, Path or None"

    data_stats = pd.read_csv("./_openml_datainfo.csv")
    if target_type != "all":
        data_stats = data_stats[data_stats["target_type"] == target_type]
    if imalance_type != "all":
        data_stats = data_stats[data_stats["imalance_type"] == imalance_type]

    datasets = OrderedDict()
    iterator = tqdm.tqdm(
        data_stats.iterrows(), total=len(data_stats), desc="Fetching datasets"
    )
    for _, row in iterator:
        openml_id = row["openml_id"]
        dataset_name = row["dataset_name"]
        X, y = _fetch_openml_data(
            openml_id, cat_preprocess=cat_preprocess, data_home=data_home
        )
        # dataset is a Bunch object
        datasets[dataset_name] = {
            "data": X,
            "target": y,
            "openml_id": openml_id,
            "IR": row["IR"],
        }
        iterator.set_postfix_str(
            f"Loaded {dataset_name} (OpenML ID: {openml_id})", refresh=True
        )
    return datasets


# %%

if __name__ == "__main__":  # pragma: no cover
    """
    Example usage of fetching datasets from OpenML.
    """
    datasets = fetch_openml_datasets(
        target_type="all", imalance_type="all", cat_preprocess="onehot", data_home=None
    )
    print(datasets.keys())

# %%
