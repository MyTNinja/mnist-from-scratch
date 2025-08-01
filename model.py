import numpy as np
from utils import one_hot


def init_params(dim_input: int = 784, hidden_units: int = 128, num_classes: int = 10):
    W1 = np.random.randn(hidden_units, dim_input) * np.sqrt(2.0 / dim_input)
    b1 = np.zeros((hidden_units, 1))
    W2 = np.random.randn(num_classes, hidden_units) * np.sqrt(2.0 / hidden_units)
    b2 = np.zeros((num_classes, 1))
    return W1, b1, W2, b2


def relu(Z):
    return np.maximum(0, Z)


def relu_deriv(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_pass(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache


def backward_pass(cache, W2, Y):
    X, Z1, A1, Z2, A2 = cache
    m = X.shape[1]
    Y_oh = one_hot(Y, A2.shape[0])

    dZ2 = A2 - Y_oh
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(params, grads, lr):
    W1, b1, W2, b2 = params
    dW1, db1, dW2, db2 = grads
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2
