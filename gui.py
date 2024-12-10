import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import numpy as np
from data import MNISTDataLoader
from network import CNN


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


class TrainingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CNN Training GUI")

        # Load data
        self.data_loader = MNISTDataLoader(dataset_path="./data")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = (
            self.data_loader.load_data()
        )

        # Define network configuration
        self.layers_config = [
            {
                "type": "conv",
                "num_filters": 8,
                "filter_size": 3,
                "stride": 1,
                "padding": 1,
            },
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

        # Create CNN and store layer output shapes
        self.cnn = CNN(self.layers_config, input_shape=(28, 28, 1))
        # We will gather shapes from the CNN initialization logs or from the code.
        # The CNN initialization code already computes shapes for each layer.
        # Let's replicate that logic here to have a list of output shapes:
        self.layer_shapes = self.get_layer_output_shapes()

        # Display network configuration
        config_label = tk.Label(self.master, text="Network Configuration:")
        config_label.pack(pady=5)

        config_text = "\n".join([str(layer) for layer in self.layers_config])
        config_display = tk.Label(
            self.master, text=config_text, justify="left", anchor="w"
        )
        config_display.pack(pady=5)

        # Canvas for architecture visualization
        # Increase size to ensure full network is visible
        self.arch_canvas = tk.Canvas(self.master, width=1200, height=300, bg="white")
        self.arch_canvas.pack(pady=5)
        self.draw_network_architecture()

        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=5)

        self.train_button = tk.Button(
            button_frame, text="Train", command=self.start_training_thread
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.test_button = tk.Button(button_frame, text="Test", command=self.run_test)
        self.test_button.pack(side=tk.LEFT, padx=5)

        # Text area for logs
        self.log_area = scrolledtext.ScrolledText(self.master, width=60, height=15)
        self.log_area.pack(pady=5)

        # Variables for training
        self.training_running = False
        self.batch_size = 16
        self.learning_rate = 0.01
        self.epochs = 1
        self.log_lines = []

    def get_layer_output_shapes(self):
        """
        Recompute the shapes of each layer's output, similar to what's done in CNN __init__.
        """
        input_shape = (28, 28, 1)
        shapes = []
        current_shape = input_shape
        for config in self.layers_config:
            layer_type = config["type"]
            if layer_type == "conv":
                padding = config.get("padding", 1)
                stride = config.get("stride", 1)
                filter_size = config.get("filter_size", 3)
                num_filters = config.get("num_filters", 8)
                H, W, C = current_shape
                H_out = (H + 2 * padding - filter_size) // stride + 1
                W_out = (W + 2 * padding - filter_size) // stride + 1
                current_shape = (H_out, W_out, num_filters)

            elif layer_type == "relu":
                # Same shape
                pass

            elif layer_type == "pool":
                pool_size = config.get("pool_size", 2)
                stride = config.get("stride", 2)
                H, W, C = current_shape
                H_out = (H - pool_size) // stride + 1
                W_out = (W - pool_size) // stride + 1
                current_shape = (H_out, W_out, C)

            elif layer_type == "flatten":
                H, W, C = current_shape
                current_shape = (H * W * C,)

            elif layer_type == "dense":
                output_dim = config.get("units")
                current_shape = (output_dim,)

            elif layer_type == "softmax":
                # Shape stays same after softmax
                pass

            shapes.append((layer_type, current_shape))
        return shapes

    def draw_network_architecture(self):
        """
        Draw the network architecture as boxes and arrows on the canvas.
        Display layer type and output shape inside the box.
        """
        x_start = 50
        y_start = 50
        box_width = 140
        box_height = 70
        x_gap = 180  # increase gap to handle more text

        for i, (layer_type, shape) in enumerate(self.layer_shapes):
            # Coordinates for the box
            x1 = x_start + i * x_gap
            y1 = y_start
            x2 = x1 + box_width
            y2 = y1 + box_height

            # Draw rectangle
            self.arch_canvas.create_rectangle(
                x1, y1, x2, y2, fill="lightblue", outline="black"
            )

            # Prepare text: layer type on top line, shape on next line(s)
            shape_str = str(shape)
            # We'll show a two-line text:
            # Line 1: Layer type
            # Line 2: Output shape
            self.arch_canvas.create_text(
                (x1 + x2) / 2,
                y1 + 20,
                text=layer_type,
                font=("Helvetica", 10),
                anchor="center",
            )
            self.arch_canvas.create_text(
                (x1 + x2) / 2,
                y1 + 45,
                text=shape_str,
                font=("Helvetica", 9),
                anchor="center",
            )

            # Draw arrow from previous to current (except for the first layer)
            if i > 0:
                prev_x2 = x_start + (i - 1) * x_gap + box_width
                prev_y_mid = (y_start + y_start + box_height) / 2
                curr_x1 = x1
                # Arrow line
                self.arch_canvas.create_line(
                    prev_x2, prev_y_mid, curr_x1, prev_y_mid, arrow=tk.LAST
                )

    def log(self, message):
        self.log_lines.append(message)
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]
        self.log_area.delete("1.0", tk.END)
        self.log_area.insert(tk.END, "\n".join(self.log_lines))
        self.log_area.see(tk.END)

    def start_training_thread(self):
        if not self.training_running:
            self.training_running = True
            self.log("Starting training...")
            thread = threading.Thread(target=self.train_network)
            thread.start()
        else:
            self.log("Training is already running.")

    def train_network(self):
        X_train = self.X_train
        y_train = self.y_train
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(self.epochs):
            self.log(f"Starting epoch {epoch+1}/{self.epochs}")
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            total_loss = 0.0
            total_acc = 0.0
            num_batches = num_samples // self.batch_size

            for i in range(num_batches):
                if not self.training_running:
                    self.log("Training stopped prematurely.")
                    return
                batch_x = X_train_shuffled[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                batch_y = y_train_shuffled[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]

                probs = self.cnn.forward(batch_x)
                loss = cross_entropy_loss(probs, batch_y)
                acc = accuracy(probs, batch_y)

                total_loss += loss
                total_acc += acc

                d_out = cross_entropy_grad(probs, batch_y)
                self.cnn.backward(d_out)
                self.cnn.update_params(self.learning_rate)

                if i % 10 == 0:
                    self.log(
                        f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss:.4f}, Acc: {acc:.4f}"
                    )

                time.sleep(0.001)

            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            self.log(
                f"Completed epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}"
            )

        self.log("Training complete.")
        self.training_running = False

    def run_test(self):
        if self.training_running:
            self.log("Cannot test while training is in progress.")
            return

        self.log("Testing the model...")
        probs = self.cnn.forward(self.X_test)
        test_acc = accuracy(probs, self.y_test)
        self.log(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
