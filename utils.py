import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


def load_mnist(normalize: bool = True, test_size: float = 0.2, random_state: int = 42):
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.values.T  # shape (784, 70000)
    Y = mnist.target.astype(int).values  # shape (70000,)

    if normalize:
        X = X / 255.0

    # train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.T, Y, test_size=test_size, random_state=random_state
    )
    return X_train.T, Y_train, X_test.T, Y_test


def one_hot(Y, num_classes: int):
    Y = Y.astype(int)
    one_hot = np.zeros((num_classes, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot


def compute_accuracy(preds, labels):
    return np.mean(preds == labels)
