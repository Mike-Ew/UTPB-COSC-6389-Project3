import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import math
import ast
import numpy as np
import queue

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

        # Default network configuration
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

        self.build_model()  # Build CNN from self.layers_config

        # Load data
        self.data_loader = MNISTDataLoader(dataset_path="./data")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = (
            self.data_loader.load_data()
        )

        # Display network configuration
        config_label = tk.Label(self.master, text="Network Configuration:")
        config_label.pack(pady=5)

        self.config_display = tk.Label(
            self.master, text=str(self.layers_config), justify="left", anchor="w"
        )
        self.config_display.pack(pady=5)

        top_frame = tk.Frame(self.master)
        top_frame.pack(pady=5, fill="x")

        canvas_frame = tk.Frame(top_frame)
        canvas_frame.pack(side=tk.LEFT, padx=5)

        # Large canvas size
        self.arch_canvas = tk.Canvas(canvas_frame, width=1200, height=400, bg="white")
        h_scroll = tk.Scrollbar(
            canvas_frame, orient="horizontal", command=self.arch_canvas.xview
        )
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.arch_canvas.configure(xscrollcommand=h_scroll.set)
        self.arch_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        self.layer_boxes = []
        self.box_to_layer_idx = {}

        # Store text item IDs for each layer line
        self.p_summary_text_ids = {}
        self.detail_text_ids = {}
        self.activation_text_ids = {}

        self.draw_network_architecture()

        # Removed side panel and details, no on_canvas_click needed
        # No selection logic either

        # Controls frame
        controls_frame = tk.Frame(self.master)
        controls_frame.pack(pady=5)

        tk.Label(controls_frame, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        self.batch_size_var = tk.StringVar(value="16")
        tk.Entry(controls_frame, textvariable=self.batch_size_var, width=5).pack(
            side=tk.LEFT
        )

        tk.Label(controls_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=5)
        self.learning_rate_var = tk.StringVar(value="0.01")
        tk.Entry(controls_frame, textvariable=self.learning_rate_var, width=5).pack(
            side=tk.LEFT
        )

        self.train_button = tk.Button(
            controls_frame, text="Train", command=self.start_training_thread
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(
            controls_frame, text="Pause", command=self.pause_training
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            controls_frame, text="Stop", command=self.stop_training
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.test_button = tk.Button(controls_frame, text="Test", command=self.run_test)
        self.test_button.pack(side=tk.LEFT, padx=5)

        self.configure_button = tk.Button(
            controls_frame, text="Configure Network", command=self.open_config_window
        )
        self.configure_button.pack(side=tk.LEFT, padx=5)

        self.log_area = scrolledtext.ScrolledText(self.master, width=60, height=10)
        self.log_area.pack(pady=5)

        # Training control variables
        self.training_running = False
        self.training_paused = False
        self.training_stopped = False
        self.epochs = 1
        self.log_lines = []

        # Store last activations for each layer
        self.last_activations = [None] * (len(self.cnn.layers))

        # Queue for thread communication (training -> UI)
        self.update_queue = queue.Queue()

        # Poll the queue periodically
        self.master.after(100, self.update_ui_from_queue)

    def build_model(self):
        self.cnn = CNN(self.layers_config, input_shape=(28, 28, 1))
        self.layer_shapes = self.get_layer_output_shapes()
        self.param_summaries = self.get_param_summaries()

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

    def get_param_summaries(self):
        summaries = []
        for layer in self.cnn.layers:
            summary = ""
            if hasattr(layer, "weights"):
                w = layer.weights
                summary += f"Wmean={w.mean():.2f},Wstd={w.std():.2f} "
            if hasattr(layer, "filters"):
                f = layer.filters
                summary += f"Fmean={f.mean():.2f},Fstd={f.std():.2f} "
            if hasattr(layer, "biases"):
                b = layer.biases
                summary += f"Bmean={b.mean():.2f},Bstd={b.std():.2f}"
            summaries.append(summary.strip())
        return summaries

    def get_box_size_for_shape(self, shape):
        width_per_unit = 5.0
        height_per_unit = 2.5

        if len(shape) == 3:
            H, W, C = shape
            w = max(60, width_per_unit * W)
            h = max(30, height_per_unit * H)
        elif len(shape) == 1:
            D = shape[0]
            pseudo_dim = int(math.sqrt(D)) if D > 0 else 1
            w = max(60, width_per_unit * pseudo_dim)
            h = max(30, height_per_unit * pseudo_dim)
        else:
            w = 140
            h = 70
        return int(w), int(h)

    def draw_network_architecture(self):
        self.arch_canvas.delete("all")
        self.layer_boxes.clear()
        self.box_to_layer_idx.clear()
        self.p_summary_text_ids.clear()
        self.detail_text_ids.clear()
        self.activation_text_ids.clear()

        x_start = 50
        y_start = 50
        x_gap = 180

        current_x = x_start
        for i, ((layer_type, shape), p_summary) in enumerate(
            zip(self.layer_shapes, self.param_summaries)
        ):
            box_w, box_h = self.get_box_size_for_shape(shape)
            x1 = current_x
            x2 = x1 + box_w
            y1 = y_start
            y2 = y1 + box_h

            box_id = self.arch_canvas.create_rectangle(
                x1, y1, x2, y2, fill="lightblue", outline="black"
            )
            self.layer_boxes.append(box_id)
            self.box_to_layer_idx[box_id] = i

            shape_str = str(shape)
            self.arch_canvas.create_text(
                (x1 + x2) / 2,
                y1 + (box_h * 0.3),
                text=layer_type,
                font=("Helvetica", 10),
                anchor="center",
            )
            self.arch_canvas.create_text(
                (x1 + x2) / 2,
                y1 + (box_h * 0.6),
                text=shape_str,
                font=("Helvetica", 9),
                anchor="center",
            )

            # Parameter summary text below the box
            line_y = y2 + 20
            if p_summary:
                tid_p = self.arch_canvas.create_text(
                    (x1 + x2) / 2,
                    line_y,
                    text=p_summary,
                    font=("Helvetica", 8),
                    anchor="center",
                )
                self.p_summary_text_ids[i] = tid_p
            else:
                self.p_summary_text_ids[i] = None

            # Add another line for detailed parameters (W/F/B shapes + means)
            line_y += 20
            tid_detail = self.arch_canvas.create_text(
                (x1 + x2) / 2, line_y, text="", font=("Helvetica", 8), anchor="center"
            )
            self.detail_text_ids[i] = tid_detail

            # Another line for activations
            line_y += 20
            tid_act = self.arch_canvas.create_text(
                (x1 + x2) / 2, line_y, text="", font=("Helvetica", 8), anchor="center"
            )
            self.activation_text_ids[i] = tid_act

            if i > 0:
                prev_box_id = self.layer_boxes[i - 1]
                prev_coords = self.arch_canvas.coords(prev_box_id)
                prev_x_center = (prev_coords[0] + prev_coords[2]) / 2
                prev_y_center = (prev_coords[1] + prev_coords[3]) / 2
                curr_y_center = (y1 + y2) / 2
                self.arch_canvas.create_line(
                    prev_coords[2], prev_y_center, x1, curr_y_center, arrow=tk.LAST
                )

            current_x = x2 + x_gap

        total_width = current_x + 50
        self.arch_canvas.config(scrollregion=(0, 0, total_width, 400))

    def log(self, message):
        self.log_lines.append(message)
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]
        self.log_area.delete("1.0", tk.END)
        self.log_area.insert(tk.END, "\n".join(self.log_lines))
        self.log_area.see(tk.END)

    def start_training_thread(self):
        if not self.training_running:
            try:
                self.batch_size = int(self.batch_size_var.get())
            except ValueError:
                self.log("Invalid batch size, using default 16.")
                self.batch_size = 16

            try:
                self.learning_rate = float(self.learning_rate_var.get())
            except ValueError:
                self.log("Invalid learning rate, using default 0.01.")
                self.learning_rate = 0.01

            self.training_running = True
            self.training_paused = False
            self.training_stopped = False
            self.log("Starting training...")
            thread = threading.Thread(target=self.train_network)
            thread.start()
        else:
            self.log("Training is already running.")

    def pause_training(self):
        if self.training_running and not self.training_stopped:
            self.training_paused = not self.training_paused
            state = "paused" if self.training_paused else "resumed"
            self.log(f"Training {state}.")

    def stop_training(self):
        if self.training_running:
            self.training_stopped = True
            self.log("Stop requested. Training will stop after current batch.")

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
                if self.training_stopped:
                    self.log("Training stopped prematurely.")
                    self.training_running = False
                    return

                while self.training_paused and not self.training_stopped:
                    time.sleep(0.1)

                if self.training_stopped:
                    self.log("Training stopped prematurely.")
                    self.training_running = False
                    return

                batch_x = X_train_shuffled[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                batch_y = y_train_shuffled[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]

                # Forward
                x = batch_x
                for idx, layer in enumerate(self.cnn.layers):
                    x = layer.forward(x)
                    self.last_activations[idx] = x

                # Compute loss/acc
                probs = x
                loss = cross_entropy_loss(probs, batch_y)
                acc = accuracy(probs, batch_y)
                total_loss += loss
                total_acc += acc

                # Backward
                d_out = cross_entropy_grad(probs, batch_y)
                for idx in reversed(range(len(self.cnn.layers))):
                    d_out = self.cnn.layers[idx].backward(d_out)

                self.cnn.update_params(self.learning_rate)

                # After update, compute new param_summaries
                new_summaries = self.get_param_summaries()

                # Also gather detailed stats for each layer
                detail_info = self.gather_detailed_info()

                # Gather activation info
                activation_info = self.gather_activation_info()

                # Send to queue
                self.update_queue.put(
                    ("all_updates", (new_summaries, detail_info, activation_info))
                )

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
        if self.training_running and not self.training_stopped:
            self.log("Cannot test while training is in progress.")
            return

        self.log("Testing the model...")
        probs = self.cnn.forward(self.X_test)
        test_acc = accuracy(probs, self.y_test)
        self.log(f"Test Accuracy: {test_acc:.4f}")

        pred_classes = np.argmax(probs, axis=1)
        print("Expected vs Predicted (first 10 samples):")
        for i in range(min(10, len(self.y_test))):
            print(f"Index {i}: Expected={self.y_test[i]}, Predicted={pred_classes[i]}")

    def open_config_window(self):
        config_window = tk.Toplevel(self.master)
        config_window.title("Configure Network")

        tk.Label(config_window, text="Edit the layers_config (Python literal):").pack(
            pady=5
        )
        config_text = scrolledtext.ScrolledText(config_window, width=60, height=15)
        config_text.pack(pady=5)
        config_text.insert(tk.END, str(self.layers_config))

        def apply_config():
            new_config_str = config_text.get("1.0", tk.END).strip()
            try:
                new_config = ast.literal_eval(new_config_str)
                if not isinstance(new_config, list):
                    raise ValueError("Configuration must be a list of dicts.")
                for layer in new_config:
                    if not isinstance(layer, dict) or "type" not in layer:
                        raise ValueError(
                            "Each layer config must be a dict with a 'type' key."
                        )
                self.layers_config = new_config
                self.build_model()
                self.draw_network_architecture()
                self.config_display.config(text=str(self.layers_config))
                self.log("Network configuration updated.")
                config_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse config: {e}")

        apply_button = tk.Button(config_window, text="Apply", command=apply_config)
        apply_button.pack(pady=5)

    def update_ui_from_queue(self):
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                if msg_type == "all_updates":
                    new_summaries, detail_info, activation_info = data
                    self.param_summaries = new_summaries
                    self.update_all_info_ui(detail_info, activation_info)
        except queue.Empty:
            pass

        self.master.after(100, self.update_ui_from_queue)

    def gather_detailed_info(self):
        """
        Gather shape/mean/std info for weights, filters, biases
        """
        info = []
        for layer in self.cnn.layers:
            lines = []
            if hasattr(layer, "weights"):
                w = layer.weights
                lines.append(f"W:{w.shape}, Wm={w.mean():.2f},Ws={w.std():.2f}")
            if hasattr(layer, "filters"):
                f = layer.filters
                lines.append(f"F:{f.shape}, Fm={f.mean():.2f},Fs={f.std():.2f}")
            if hasattr(layer, "biases"):
                b = layer.biases
                lines.append(f"B:{b.shape}, Bm={b.mean():.2f},Bs={b.std():.2f}")
            info.append(" | ".join(lines))
        return info

    def gather_activation_info(self):
        """
        Gather shape/mean/std info for activations
        """
        act_info = []
        for act in self.last_activations:
            if act is not None:
                act_info.append(
                    f"A:{act.shape}, Am={act.mean():.2f},As={act.std():.2f}"
                )
            else:
                act_info.append("")
        return act_info

    def update_all_info_ui(self, detail_info, activation_info):
        # Update param_summaries line
        for i, summary in enumerate(self.param_summaries):
            tid_p = self.p_summary_text_ids.get(i, None)
            if tid_p is not None:
                self.arch_canvas.itemconfig(tid_p, text=summary)

        # Update detail line
        for i, det in enumerate(detail_info):
            tid_d = self.detail_text_ids.get(i, None)
            if tid_d is not None:
                self.arch_canvas.itemconfig(tid_d, text=det)

        # Update activation line
        for i, act in enumerate(activation_info):
            tid_a = self.activation_text_ids.get(i, None)
            if tid_a is not None:
                self.arch_canvas.itemconfig(tid_a, text=act)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
