# visualization.py
import tkinter as tk


class Visualization:
    def __init__(self, root):
        self.root = root
        # A large canvas to draw the entire network
        self.canvas = tk.Canvas(root, width=1500, height=1000, bg="white")
        self.canvas.pack(fill="both", expand=True)

    def generate_dummy_conv_params(self, num_filters, kernel_size, in_channels):
        # This function is only a fallback if filters/biases aren't provided by layers_data.
        # In a real scenario, you will receive real filters from layers_data.
        weights = []
        for f in range(num_filters):
            w_f = []
            for ic in range(in_channels):
                w_ch = []
                for r in range(kernel_size):
                    row_vals = []
                    for c in range(kernel_size):
                        val = 0.1 * (
                            f * kernel_size * kernel_size
                            + r * kernel_size
                            + c
                            + ic * kernel_size * kernel_size
                        )
                        row_vals.append(val)
                    w_ch.append(row_vals)
                w_f.append(w_ch)
            weights.append(w_f)
        biases = [0.01 * f for f in range(num_filters)]
        return weights, biases

    def draw_layer(self, layer, x_offset, y_offset):
        layer_type = layer["type"]
        shape = layer["shape"]
        param_summary = layer.get("param_summary", "")
        detail_info = layer.get("detail_info", "")
        activation_info = layer.get("activation_info", "")

        label_text = (
            f"{layer_type.upper()}\n"
            f"Shape: {shape}\n"
            f"ParamSummary: {param_summary}\n"
            f"Details: {detail_info}\n"
            f"Activations: {activation_info}"
        )

        if layer_type in ["input", "conv", "relu", "pool"]:
            w, h, d = shape
            scale = 2
            scaled_w = w * scale
            scaled_h = h * scale
            depth_offset = 3

            if layer_type == "input":
                color = "#aaaaaa"
            elif layer_type == "conv":
                color = "#4DA6FF"
            elif layer_type == "relu":
                color = "#4DFF4D"
            else:
                color = "#B366FF"

            # Draw feature maps as stacked rectangles
            for i in range(d):
                x1 = x_offset + i * depth_offset
                y1 = y_offset - i * depth_offset
                x2 = x1 + scaled_w
                y2 = y1 + scaled_h
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="black"
                )

            self.canvas.create_text(
                x_offset + scaled_w / 2,
                y_offset + scaled_h + 20,
                text=label_text,
                font=("Helvetica", 10),
                anchor="n",
            )

            final_width = scaled_w + 50
            final_height = scaled_h + 80
            center_x = x_offset + scaled_w / 2
            center_y = y_offset + scaled_h / 2

            # If convolutional layer, show filters and biases
            if layer_type == "conv":
                num_filters = layer.get("num_filters", 8)
                kernel_size = layer.get("kernel_size", 3)
                in_channels = layer.get("in_channels", 1)
                filters = layer.get("filters", None)
                biases = layer.get("biases", None)

                if filters is None or biases is None:
                    # If no real filters provided, use dummy values
                    filters, biases = self.generate_dummy_conv_params(
                        num_filters, kernel_size, in_channels
                    )

                filter_x = x_offset + scaled_w + 100
                filter_y = y_offset
                filter_gap = 150
                cell_size = 20

                for f_idx in range(num_filters):
                    fx_start = filter_x + (f_idx * filter_gap)
                    fy_start = filter_y
                    self.canvas.create_text(
                        fx_start + (kernel_size * cell_size) / 2,
                        fy_start - 20,
                        text=f"Filter {f_idx+1}",
                        font=("Helvetica", 10, "bold"),
                        anchor="s",
                    )

                    # Draw each filter channel as a grid
                    for ic in range(in_channels):
                        for r in range(kernel_size):
                            for c in range(kernel_size):
                                val = filters[f_idx][ic][r][c]
                                x1 = fx_start + c * cell_size
                                y1 = (
                                    fy_start
                                    + r * cell_size
                                    + ic * (kernel_size * cell_size + 10)
                                )
                                x2 = x1 + cell_size
                                y2 = y1 + cell_size
                                self.canvas.create_rectangle(
                                    x1, y1, x2, y2, fill="#e6e6e6", outline="black"
                                )
                                self.canvas.create_text(
                                    (x1 + x2) / 2,
                                    (y1 + y2) / 2,
                                    text=f"{val:.2f}",
                                    font=("Helvetica", 8),
                                )

                    # Draw bias as a small circle below
                    bias_x = fx_start + (kernel_size * cell_size) / 2
                    bias_y = fy_start + in_channels * kernel_size * cell_size + 30
                    b_val = biases[f_idx]
                    r = 10
                    self.canvas.create_oval(
                        bias_x - r,
                        bias_y - r,
                        bias_x + r,
                        bias_y + r,
                        fill="#ffcc66",
                        outline="black",
                    )
                    self.canvas.create_text(
                        bias_x,
                        bias_y + 20,
                        text=f"Bias: {b_val:.2f}",
                        font=("Helvetica", 8),
                        anchor="n",
                    )

                final_width = (num_filters * filter_gap) + scaled_w + 150

            return final_width, final_height, center_x, center_y

        elif layer_type == "flatten":
            num_nodes = shape[0]
            display_nodes = min(num_nodes, 20)
            radius = 2
            node_gap = 5
            for i in range(display_nodes):
                cx = x_offset + i * (2 * radius + node_gap)
                cy = y_offset
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill="#FFB84D",
                    outline="black",
                )

            self.canvas.create_text(
                x_offset + display_nodes * (2 * radius + node_gap) / 2,
                y_offset + 20,
                text=label_text,
                font=("Helvetica", 10),
                anchor="n",
            )
            final_width = display_nodes * (2 * radius + node_gap) + 50
            final_height = 40
            center_x = x_offset + final_width / 2
            center_y = y_offset
            return final_width, final_height, center_x, center_y

        elif layer_type in ["dense", "softmax"]:
            num_nodes = shape[0]
            display_nodes = min(num_nodes, 10)
            radius = 5
            node_gap = 30
            color = "#FF6666" if layer_type == "dense" else "#FFFF66"
            for i in range(display_nodes):
                cx = x_offset + i * node_gap
                cy = y_offset
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=color,
                    outline="black",
                )

            self.canvas.create_text(
                x_offset + display_nodes * node_gap / 2,
                y_offset + 20,
                text=label_text,
                font=("Helvetica", 10),
                anchor="n",
            )
            final_width = display_nodes * node_gap + 50
            final_height = 40
            center_x = x_offset + final_width / 2
            center_y = y_offset
            return final_width, final_height, center_x, center_y

        else:
            final_width = 50
            final_height = 50
            center_x = x_offset
            center_y = y_offset
            return final_width, final_height, center_x, center_y

    def draw_connections(self, prev_center, current_center, prev_width, prev_height):
        x1, y1 = prev_center
        x2, y2 = current_center
        self.canvas.create_line(
            x1, y1 + prev_height / 2, x2, y2 - prev_height / 2, arrow=tk.LAST
        )

    def update_network(self, layers_data):
        # Clear and redraw the entire network layout from scratch
        self.canvas.delete("all")

        x_offset = 50
        y_offset = 100
        prev_center = None
        prev_width = 0
        prev_height = 0

        for i, layer in enumerate(layers_data):
            w_used, h_used, center_x, center_y = self.draw_layer(
                layer, x_offset, y_offset
            )
            if prev_center is not None:
                self.draw_connections(
                    prev_center, (center_x, center_y), prev_width, h_used
                )

            prev_center = (center_x, center_y)
            prev_width = w_used
            prev_height = h_used
            y_offset += h_used + 100
