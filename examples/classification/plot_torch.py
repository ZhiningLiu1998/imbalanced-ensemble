"""
=========================================================
Classification with PyTorch Neural Network
=========================================================

An example of how to cooperate imbens with deep learning frameworks like PyTorch.
In this example, we first define a simple MLP classifier using PyTorch, and pack 
it into a scikit-learn-style wrapper class `TorchMLPClassifier`. We then use it as 
the base estimator of :class:`imbens.ensemble.SelfPacedEnsembleClassifier` and 
train the ensemble on a toy imbalanced dataset.

This example uses:
    
    - :class:`imbens.ensemble.SelfPacedEnsembleClassifier`
"""

# Authors: Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
print(__doc__)

# Import imbalanced-ensemble
import imbens

# Import pytorch and numpy
import torch
import torch.nn as nn
import numpy as np

# Import utilities
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imbens.datasets import make_imbalance

# Import plot utilities
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RANDOM_STATE = 42

# %% [markdown]
# PyTorch MLPClassifier
# ---------------------
# **Define a simple 3-layer perceptron with PyTorch.**


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# %% [markdown]
# **Wrap the MLP into a scikit-learn-style ``TorchMLPClassifier`` class.**


class TorchMLPClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=32,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = MLP(input_size, hidden_size, output_size)

    def _validate_input(self, X, y):
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            dtype=(np.float64, np.float32),
            reset=True,
        )
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        return X, y

    def fit(self, X, y):

        X, y = self._validate_input(X, y)

        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for i in range(0, len(X), self.batch_size):
                # Forward pass
                outputs = self.model(X_tensor[i : i + self.batch_size])

                # Compute loss
                loss = criterion(outputs, y_tensor[i : i + self.batch_size])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass and get predictions
        outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)

        # Convert predictions to numpy array and return
        return predicted.numpy()

    def predict_proba(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Forward pass and get softmax probabilities
        outputs = self.model(X_tensor)
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs).detach().numpy()

        # Return probabilities
        return probabilities


# %% [markdown]
# Classification and Visualization
# --------------------------------
# **Prepare the class-imbalanced toy dataset.**

# imbalanced moons dataset
distribution = {0: 100, 1: 50}
X, y = make_moons(200, noise=0.2, random_state=RANDOM_STATE)
imb_moons_dataset = make_imbalance(
    X, y, sampling_strategy=distribution, random_state=RANDOM_STATE
)
classes = sklearn.utils.multiclass.unique_labels(y)

# %% [markdown]
# **Use the ``TorchMLPClassifier`` as the ensemble base estimator.**

torch_spe = imbens.ensemble.SelfPacedEnsembleClassifier(
    estimator=TorchMLPClassifier(
        input_size=X.shape[1], hidden_size=64, output_size=classes.shape[0]
    ),
    n_estimators=10,
)


# %% [markdown]
# **Visualize function.**


def plot_classification_result(dataset, clf, **axset_kwargs):
    h = 0.01  # step size in the mesh
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Normalize and split the dataset
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Prepare the meshgrid for plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    clf.fit(X_train, y_train)
    score = sklearn.metrics.average_precision_score(y_test, clf.predict(X_test))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax = plt.gca()
    ax.imshow(
        -Z, extent=(xx.min(), xx.max(), yy.max(), yy.min()), cmap='bwr', alpha=0.8
    )

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(
        0.95,
        0.06,
        ('%.2f' % score).lstrip('0'),
        size=15,
        bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
        transform=ax.transAxes,
        horizontalalignment='right',
    )
    ax.set(**axset_kwargs)
    return ax


# %% [markdown]
# **Visualize the classification result.**

ax = plot_classification_result(
    imb_moons_dataset, torch_spe, title='SPE with PyTorch MLP base classifier'
)
