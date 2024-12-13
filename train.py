# train.py
import logging
import numpy as np
import time
import threading

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


class Trainer:
    def __init__(self, cnn, X_train, y_train, X_test, y_test, update_queue=None):
        self.cnn = cnn
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.update_queue = update_queue

        self.running = False
        self.paused = False
        self.stopped = False

    def start_training(self, epochs=1, batch_size=16, lr=0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.running = True
        self.paused = False
        self.stopped = False
        thread = threading.Thread(target=self.run_training_loop)
        thread.start()

    def pause_training(self):
        self.paused = not self.paused

    def stop_training(self):
        self.stopped = True

    def run_training_loop(self):
        X_train = self.X_train
        y_train = self.y_train
        epochs = self.epochs
        batch_size = self.batch_size
        lr = self.lr

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
                if self.stopped:
                    self.running = False
                    return

                while self.paused and not self.stopped:
                    time.sleep(0.1)

                if self.stopped:
                    self.running = False
                    return

                batch_x = X_train_shuffled[i * batch_size : (i + 1) * batch_size]
                batch_y = y_train_shuffled[i * batch_size : (i + 1) * batch_size]

                probs = self.cnn.forward(batch_x)
                loss = cross_entropy_loss(probs, batch_y)
                total_loss += loss
                acc = accuracy(probs, batch_y)
                total_acc += acc

                d_out = cross_entropy_grad(probs, batch_y)
                self.cnn.backward(d_out)
                self.cnn.update_params(lr)

                if i % 10 == 0:
                    logging.info(
                        f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss:.4f}, Acc: {acc:.4f}"
                    )

                # After each batch, send updates
                if self.update_queue:
                    layers_data = self.get_layer_data()
                    self.update_queue.put(("all_updates", layers_data))

            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            logging.info(
                f"Completed epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}"
            )

        # Evaluate on test
        test_probs = self.cnn.forward(self.X_test)
        test_acc = accuracy(test_probs, self.y_test)
        logging.info(f"Test Accuracy: {test_acc:.4f}")
        self.running = False

    def get_layer_data(self):
        layers_data = []
        for layer in self.cnn.layers:
            layer_type = layer.__class__.__name__.replace("Layer", "").lower()

            if hasattr(layer, "out") and layer.out is not None:
                out_shape = layer.out.shape
                activation_mean = np.mean(layer.out)
                activation_std = np.std(layer.out)
                activation_info = (
                    f"A:{out_shape},Am={activation_mean:.2f},As={activation_std:.2f}"
                )
            else:
                # If no out stored (e.g. input layer), use a placeholder
                out_shape = (28, 28, 1)
                activation_info = ""

            param_summary = ""
            detail_info = ""

            if layer_type == "input":
                # No parameters
                param_summary = ""
                detail_info = ""

            elif layer_type == "conv":
                filters = layer.filters
                biases = layer.biases
                f_flat = filters.flatten()
                b_flat = biases.flatten()
                Wmean, Wstd = f_flat.mean(), f_flat.std()
                Bmean, Bstd = b_flat.mean(), b_flat.std()
                param_summary = f"Wmean={Wmean:.2f},Bmean={Bmean:.2f}"
                detail_info = f"W:{filters.shape}, B:{biases.shape}"

                # Transpose filters to (num_filters, in_channels, kernel_size, kernel_size)
                transposed_filters = filters.transpose(0, 3, 1, 2)

                kernel_size = layer.filter_size
                in_channels = layer.input_shape[2] if layer.input_shape else 1
                num_filters = layer.num_filters

                layer_dict = {
                    "type": layer_type,
                    "shape": out_shape,
                    "param_summary": param_summary,
                    "detail_info": detail_info,
                    "activation_info": activation_info,
                    "kernel_size": kernel_size,
                    "in_channels": in_channels,
                    "num_filters": num_filters,
                    "filters": transposed_filters,
                    "biases": biases,
                }

            elif layer_type == "relu":
                # No parameters, just activation info
                layer_dict = {
                    "type": layer_type,
                    "shape": out_shape,
                    "param_summary": param_summary,
                    "detail_info": detail_info,
                    "activation_info": activation_info,
                }

            else:
                # If other layers appear later, handle similarly
                layer_dict = {
                    "type": layer_type,
                    "shape": out_shape,
                    "param_summary": param_summary,
                    "detail_info": detail_info,
                    "activation_info": activation_info,
                }

            layers_data.append(layer_dict)

        return layers_data
