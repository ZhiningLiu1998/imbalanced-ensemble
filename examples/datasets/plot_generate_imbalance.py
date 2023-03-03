"""
===============================
Generate an imbalanced dataset
===============================

An illustration of using the 
:func:`~imbens.datasets.generate_imbalance_data` 
function to create an imbalanced dataset. 
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
print(__doc__)

from imbens.datasets import generate_imbalance_data
from imbens.utils._plot import plot_2Dprojection_and_cardinality
from collections import Counter

# %% [markdown]
# Generate the dataset
# --------------------
#

# %%
X_train, X_test, y_train, y_test = generate_imbalance_data(
    n_samples=1000,
    weights=[0.7, 0.2, 0.1],
    test_size=0.5,
    kwargs={'n_informative': 3},
)

print("Train class distribution: ", Counter(y_train))
print("Test class distribution:  ", Counter(y_test))

# %% [markdown]
# Plot the generated (training) data
# ----------------------------------
#

plot_2Dprojection_and_cardinality(X_train, y_train)
