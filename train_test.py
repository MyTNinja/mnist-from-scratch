import numpy as np
from model import init_params, forward_pass, backward_pass, update_params
from utils import compute_accuracy
import matplotlib.pyplot as plt


def train(X_train, Y_train, X_test, Y_test, epochs=30, lr=0.1, print_every=1):
    params = init_params(dim_input=X_train.shape[0])
    W1, b1, W2, b2 = params

    for epoch in range(1, epochs + 1):
        A2, cache = forward_pass(X_train, W1, b1, W2, b2)
        grads = backward_pass(cache, W2, Y_train)
        W1, b1, W2, b2 = update_params((W1, b1, W2, b2), grads, lr)

        if epoch % print_every == 0:
            train_preds = np.argmax(A2, axis=0)
            train_acc = compute_accuracy(train_preds, Y_train)
            dev_preds, _ = forward_pass(X_test, W1, b1, W2, b2)
            dev_preds = np.argmax(dev_preds, axis=0)
            dev_acc = compute_accuracy(dev_preds, Y_test)
            print(
                f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {dev_acc:.4f}"
            )

    return W1, b1, W2, b2


def predict(X, params):
    W1, b1, W2, b2 = params
    A2, _ = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)


def show_example(X, Y, params, index):
    pred = predict(X[:, index : index + 1], params)[0]
    label = Y[index]
    img = X[:, index].reshape(28, 28)
    print(f"Pred: {pred}, True: {label}")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
