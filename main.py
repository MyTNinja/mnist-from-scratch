from utils import compute_accuracy, load_mnist
from train_test import predict, train, show_example

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist()
    params = train(X_train, Y_train, X_test, Y_test, epochs=50, lr=0.05)
    show_example(X_test, Y_test, params, index=0)

    # Evaluate and print final accuracy on dev set
    test_pred = predict(X_test, params)
    test_acc = compute_accuracy(test_pred, Y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")
