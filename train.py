# train.py
import logging
import numpy as np
from data import MNISTDataLoader
from network import CNN

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def cross_entropy_loss(probs, labels):
    N = probs.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    loss = -np.sum(one_hot * np.log(probs + 1e-9)) / N
    return loss


def cross_entropy_grad(probs, labels):
    N = probs.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    grad = (probs - one_hot) / N
    return grad


def accuracy(probs, labels):
    pred_classes = np.argmax(probs, axis=1)
    return np.mean(pred_classes == labels)


if __name__ == "__main__":
    # Load data
    data_loader = MNISTDataLoader(dataset_path="./data")
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()

    layers_config = [
        {"type": "conv", "num_filters": 8, "filter_size": 3, "stride": 1, "padding": 1},
        {"type": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2},
        {
            "type": "conv",
            "num_filters": 16,
            "filter_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {"type": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 10},
        {"type": "softmax"},
    ]

    cnn = CNN(layers_config, input_shape=(28, 28, 1))

    # Training parameters
    epochs = 1  # Adjust as needed
    batch_size = 16  # Start small to see faster progress
    learning_rate = 0.01

    num_samples = X_train.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        total_loss = 0.0
        total_acc = 0.0
        num_batches = num_samples // batch_size

        for i in range(num_batches):
            batch_x = X_train_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_y = y_train_shuffled[i * batch_size : (i + 1) * batch_size]

            # Forward
            probs = cnn.forward(batch_x)
            loss = cross_entropy_loss(probs, batch_y)
            total_loss += loss
            acc = accuracy(probs, batch_y)
            total_acc += acc

            # Backward
            d_out = cross_entropy_grad(probs, batch_y)
            cnn.backward(d_out)
            cnn.update_params(learning_rate)

            # Log every 10 batches
            if i % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss:.4f}, Acc: {acc:.4f}"
                )

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        logging.info(
            f"Completed epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}"
        )

    # Evaluate on test set
    test_probs = cnn.forward(X_test)
    test_acc = accuracy(test_probs, y_test)
    logging.info(f"Test Accuracy: {test_acc:.4f}")
