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

        self.cnn = CNN(self.layers_config, input_shape=(28, 28, 1))
        self.layer_shapes = self.get_layer_output_shapes()

        # Display network configuration
        config_label = tk.Label(self.master, text="Network Configuration:")
        config_label.pack(pady=5)

        config_text = "\n".join([str(layer) for layer in self.layers_config])
        config_display = tk.Label(
            self.master, text=config_text, justify="left", anchor="w"
        )
        config_display.pack(pady=5)

        # Frame to hold architecture and details
        top_frame = tk.Frame(self.master)
        top_frame.pack(pady=5, fill="x")

        # Canvas for architecture visualization
        self.arch_canvas = tk.Canvas(top_frame, width=1200, height=300, bg="white")
        self.arch_canvas.pack(side=tk.LEFT, padx=5)

        # Will store the item IDs of layer boxes for highlighting and a mapping from box_id to layer_idx
        self.layer_boxes = []
        self.box_to_layer_idx = {}
        self.draw_network_architecture()

        # Bind mouse click event on canvas
        self.arch_canvas.bind("<Button-1>", self.on_canvas_click)

        # Details panel on the right to show weights, biases, activations
        self.details_text = scrolledtext.ScrolledText(top_frame, width=50, height=20)
        self.details_text.pack(side=tk.LEFT, padx=5)

        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=5)

        self.train_button = tk.Button(
            button_frame, text="Train", command=self.start_training_thread
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.test_button = tk.Button(button_frame, text="Test", command=self.run_test)
        self.test_button.pack(side=tk.LEFT, padx=5)

        # Text area for logs at bottom
        self.log_area = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.log_area.pack(pady=5)

        # Variables for training
        self.training_running = False
        self.batch_size = 16
        self.learning_rate = 0.01
        self.epochs = 1
        self.log_lines = []

        self.selected_layer_idx = None
        # We'll store intermediate activations from forward passes
        self.last_activations = [None] * (
            len(self.cnn.layers)
        )  # Store output of each layer in forward pass

    def get_layer_output_shapes(self):
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
                pass

            shapes.append((layer_type, current_shape))
        return shapes

    def draw_network_architecture(self):
        x_start = 50
        y_start = 50
        box_width = 140
        box_height = 70
        x_gap = 180

        for i, (layer_type, shape) in enumerate(self.layer_shapes):
            x1 = x_start + i * x_gap
            y1 = y_start
            x2 = x1 + box_width
            y2 = y1 + box_height

            box_id = self.arch_canvas.create_rectangle(
                x1, y1, x2, y2, fill="lightblue", outline="black"
            )
            self.layer_boxes.append(box_id)
            self.box_to_layer_idx[box_id] = i

            shape_str = str(shape)
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

            if i > 0:
                prev_x2 = 50 + (i - 1) * x_gap + box_width
                prev_y_mid = (y_start + y_start + box_height) / 2
                curr_x1 = x1
                self.arch_canvas.create_line(
                    prev_x2, prev_y_mid, curr_x1, prev_y_mid, arrow=tk.LAST
                )

    def on_canvas_click(self, event):
        # Determine if a layer box was clicked
        item = self.arch_canvas.find_closest(event.x, event.y)
        item_id = item[0]
        if item_id in self.box_to_layer_idx:
            self.selected_layer_idx = self.box_to_layer_idx[item_id]
            self.log(f"Selected layer {self.selected_layer_idx}")
            self.update_details_panel()

    def highlight_layer(self, idx, color):
        self.arch_canvas.itemconfig(self.layer_boxes[idx], fill=color)
        self.master.update_idletasks()

    def reset_layer_color(self, idx):
        self.arch_canvas.itemconfig(self.layer_boxes[idx], fill="lightblue")
        self.master.update_idletasks()

    def forward_step_by_step(self, x):
        for i, layer in enumerate(self.cnn.layers):
            self.highlight_layer(i, "yellow")
            self.master.update()
            time.sleep(0.01)

            x = layer.forward(x)
            self.last_activations[i] = x  # store activations

            self.reset_layer_color(i)
        return x

    def backward_step_by_step(self, d_out):
        for i in reversed(range(len(self.cnn.layers))):
            layer = self.cnn.layers[i]
            self.highlight_layer(i, "orange")
            self.master.update()
            time.sleep(0.01)
            d_out = layer.backward(d_out)
            self.reset_layer_color(i)
        return d_out

    def update_details_panel(self):
        # Display details of the selected layer
        self.details_text.delete("1.0", tk.END)
        if self.selected_layer_idx is None:
            self.details_text.insert(tk.END, "No layer selected.")
            return

        layer = self.cnn.layers[self.selected_layer_idx]
        layer_type = self.layers_config[self.selected_layer_idx]["type"]
        self.details_text.insert(
            tk.END, f"Layer {self.selected_layer_idx}: {layer_type}\n\n"
        )

        # Show weights and biases if available
        if hasattr(layer, "weights"):
            w = layer.weights
            self.details_text.insert(tk.END, f"Weights shape: {w.shape}\n")
            self.details_text.insert(
                tk.END, f"Weights mean: {w.mean():.4f}, std: {w.std():.4f}\n"
            )
        if hasattr(layer, "filters"):
            f = layer.filters
            self.details_text.insert(tk.END, f"Filters shape: {f.shape}\n")
            self.details_text.insert(
                tk.END, f"Filters mean: {f.mean():.4f}, std: {f.std():.4f}\n"
            )
        if hasattr(layer, "biases"):
            b = layer.biases
            self.details_text.insert(tk.END, f"Biases shape: {b.shape}\n")
            self.details_text.insert(
                tk.END, f"Biases mean: {b.mean():.4f}, std: {b.std():.4f}\n"
            )

        # Show intermediate activations if available (from last batch)
        act = self.last_activations[self.selected_layer_idx]
        if act is not None:
            self.details_text.insert(tk.END, "\nActivations (last forward pass):\n")
            self.details_text.insert(tk.END, f"Shape: {act.shape}\n")
            # Just show mean and std
            self.details_text.insert(
                tk.END, f"Mean: {act.mean():.4f}, Std: {act.std():.4f}\n"
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

                probs = self.forward_step_by_step(batch_x)
                loss = cross_entropy_loss(probs, batch_y)
                acc = accuracy(probs, batch_y)
                total_loss += loss
                total_acc += acc

                d_out = cross_entropy_grad(probs, batch_y)
                self.backward_step_by_step(d_out)
                self.cnn.update_params(self.learning_rate)

                # Update details panel if a layer is selected
                if self.selected_layer_idx is not None:
                    self.update_details_panel()

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
