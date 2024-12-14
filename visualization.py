# visualization.py
import tkinter as tk


class Visualization:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=1500, height=1000, bg="white")
        self.canvas.pack(fill="both", expand=True)

    def draw_layer(self, layer, x_offset, y_offset):
        # Ensure layer has a 'type'
        # If it doesn't, it's probably a progress dict or something else. Skip drawing.
        if "type" not in layer:
            # Return minimal values so caller can continue
            return 0, 0, x_offset, y_offset

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
            if shape and len(shape) == 4:
                N, H, W, C = shape
                w, h, d = W, H, C
            else:
                w, h, d = 0, 0, 0

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

            if w > 0 and h > 0 and d > 0:
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
            else:
                self.canvas.create_text(
                    x_offset,
                    y_offset,
                    text=label_text,
                    font=("Helvetica", 10),
                    anchor="nw",
                )
                final_width = 200
                final_height = 80
                center_x = x_offset + 100
                center_y = y_offset + 40

            # If convolutional layer, show filters and biases if provided
            if layer_type == "conv" and w > 0 and h > 0 and d > 0:
                num_filters = layer.get("num_filters", None)
                kernel_size = layer.get("kernel_size", None)
                in_channels = layer.get("in_channels", None)
                filters = layer.get("filters", None)
                biases = layer.get("biases", None)

                if (
                    (filters is not None)
                    and (biases is not None)
                    and (num_filters is not None)
                    and (kernel_size is not None)
                    and (in_channels is not None)
                ):
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
            # Flatten layers have shape (N,D)
            if shape and len(shape) == 2:
                N, D = shape
                num_nodes = D
            else:
                num_nodes = 0
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
            # Dense/Softmax: shape (N, units)
            if shape and len(shape) == 2:
                N, D = shape
                num_nodes = D
            else:
                num_nodes = 0
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
            # If no 'type' or unknown type, just skip or handle gracefully
            self.canvas.create_text(
                x_offset, y_offset, text=label_text, font=("Helvetica", 10), anchor="nw"
            )
            final_width = 200
            final_height = 80
            center_x = x_offset + 100
            center_y = y_offset + 40
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

        # Filter out the progress dictionary before drawing
        drawable_layers = [l for l in layers_data if "type" in l]

        for i, layer in enumerate(drawable_layers):
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
