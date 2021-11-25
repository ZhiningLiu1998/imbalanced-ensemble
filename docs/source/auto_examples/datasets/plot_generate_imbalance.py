"""
===============================
Generate an imbalanced dataset
===============================

An illustration of using the 
:func:`~imbalanced_ensemble.datasets.generate_imbalance_data` 
function to create an imbalanced dataset. 
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
print(__doc__)

from imbalanced_ensemble.datasets import generate_imbalance_data
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality
from collections import Counter

# %% [markdown]
# Generate the dataset
# --------------------
#

# %%
X_train, X_test, y_train, y_test = generate_imbalance_data(
    n_samples=1000, weights=[.7,.2,.1], test_size=.5,
    kwargs={'n_informative': 3},
)

print ("Train class distribution: ", Counter(y_train))
print ("Test class distribution:  ", Counter(y_test))

# %% [markdown]
# Plot the generated (training) data
# ----------------------------------
#

plot_2Dprojection_and_cardinality(X_train, y_train)
