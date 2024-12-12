import tkinter as tk

root = tk.Tk()
root.title("CNN Visualization - Column Layout with Integrated Conv Parameter View")

canvas = tk.Canvas(root, width=1200, height=2000, bg="white")
canvas.pack(fill="both", expand=True)

# Example CNN configuration with a conv layer and other layers
layers_info = [
    {
        "type": "input",
        "shape": (28, 28, 1),
        "params": 0,
        "activation_shape": (28, 28, 1),
    },
    {
        "type": "conv",
        "shape": (28, 28, 8),
        "params": 80,  # as previously calculated
        "activation_shape": (28, 28, 8),
        "kernel_size": 3,
        "in_channels": 1,
        "num_filters": 8,
    },
    {
        "type": "relu",
        "shape": (28, 28, 8),
        "params": 0,
        "activation_shape": (28, 28, 8),
    },
    {
        "type": "pool",
        "shape": (14, 14, 8),
        "params": 0,
        "activation_shape": (14, 14, 8),
    },
    {"type": "flatten", "shape": (784,), "params": 0, "activation_shape": (784,)},
    {
        "type": "dense",
        "shape": (10,),
        "params": 7850,  # example: 784*10 + 10
        "activation_shape": (10,),
    },
    {"type": "softmax", "shape": (10,), "params": 0, "activation_shape": (10,)},
]


# Dummy parameters for the conv layer filters
def generate_dummy_conv_params(num_filters, kernel_size, in_channels):
    # Each filter: kernel_size*kernel_size*in_channels weights + 1 bias
    # We'll just fill them with some dummy values
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


def draw_layer(canvas, layer, x_offset, y_offset):
    layer_type = layer["type"]
    shape = layer["shape"]
    params = layer["params"]
    activation_shape = layer.get("activation_shape", shape)

    label_text = (
        f"{layer_type.upper()}\nShape: {shape}\n"
        f"Params: {params}\nActivation: {activation_shape}"
    )
    width_used = 200  # base width allocation for column

    # Draw different layers
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

        # Draw the stack of feature maps
        for i in range(d):
            x1 = x_offset + i * depth_offset
            y1 = y_offset - i * depth_offset
            x2 = x1 + scaled_w
            y2 = y1 + scaled_h
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

        # Label under the layer
        canvas.create_text(
            x_offset + scaled_w / 2,
            y_offset + scaled_h + 20,
            text=label_text,
            font=("Helvetica", 10),
            anchor="n",
        )

        # If it's a conv layer, show the parameter detail view to the right
        if layer_type == "conv":
            num_filters = layer["num_filters"]
            kernel_size = layer["kernel_size"]
            in_channels = layer["in_channels"]
            weights, biases = generate_dummy_conv_params(
                num_filters, kernel_size, in_channels
            )

            # Draw filters to the right
            # Start drawing filters to the right of main block
            filter_x = x_offset + scaled_w + 100
            filter_y = y_offset
            filter_gap = 150
            cell_size = 20

            # For simplicity, if multiple in_channels > 1, stack them vertically
            # each filter is in_channels * (kernel_size x kernel_size)
            for f_idx in range(num_filters):
                fx_start = filter_x + (f_idx * filter_gap)
                fy_start = filter_y
                canvas.create_text(
                    fx_start + (kernel_size * cell_size) / 2,
                    fy_start - 20,
                    text=f"Filter {f_idx+1}",
                    font=("Helvetica", 10, "bold"),
                    anchor="s",
                )

                # Draw each input channel as a separate grid stacked vertically
                for ic in range(in_channels):
                    for r in range(kernel_size):
                        for c in range(kernel_size):
                            val = weights[f_idx][ic][r][c]
                            x1 = fx_start + c * cell_size
                            y1 = (
                                fy_start
                                + r * cell_size
                                + ic * (kernel_size * cell_size + 10)
                            )
                            x2 = x1 + cell_size
                            y2 = y1 + cell_size
                            canvas.create_rectangle(
                                x1, y1, x2, y2, fill="#e6e6e6", outline="black"
                            )
                            canvas.create_text(
                                (x1 + x2) / 2,
                                (y1 + y2) / 2,
                                text=f"{val:.2f}",
                                font=("Helvetica", 8),
                            )

                # Bias next to the filter(s)
                bias_x = fx_start + (kernel_size * cell_size) / 2
                bias_y = fy_start + in_channels * kernel_size * cell_size + 30
                b_val = biases[f_idx]
                r = 10
                canvas.create_oval(
                    bias_x - r,
                    bias_y - r,
                    bias_x + r,
                    bias_y + r,
                    fill="#ffcc66",
                    outline="black",
                )
                canvas.create_text(
                    bias_x,
                    bias_y + 20,
                    text=f"Bias: {b_val:.2f}",
                    font=("Helvetica", 8),
                    anchor="n",
                )

            # Update width_used considering the filters drawn
            width_used = (num_filters * filter_gap) + scaled_w + 150

        else:
            # No extra detail view for non-conv layers
            width_used = scaled_w + 50

        final_width = width_used
        final_height = scaled_h + 80
        center_x = x_offset + scaled_w / 2
        center_y = y_offset + scaled_h / 2

    elif layer_type == "flatten":
        num_nodes = shape[0]
        display_nodes = min(num_nodes, 20)
        radius = 2
        node_gap = 5
        for i in range(display_nodes):
            cx = x_offset + i * (2 * radius + node_gap)
            cy = y_offset
            canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#FFB84D",
                outline="black",
            )

        canvas.create_text(
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

    elif layer_type in ["dense", "softmax"]:
        num_nodes = shape[0]
        display_nodes = min(num_nodes, 10)
        radius = 5
        node_gap = 30
        for i in range(display_nodes):
            cx = x_offset + i * node_gap
            cy = y_offset
            if layer_type == "dense":
                color = "#FF6666"
            else:
                color = "#FFFF66"
            canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=color,
                outline="black",
            )

        canvas.create_text(
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

    else:
        # Just in case
        final_width = 50
        final_height = 50
        center_x = x_offset
        center_y = y_offset

    return final_width, final_height, center_x, center_y


def draw_connections(canvas, prev_center, current_center, prev_width, prev_height):
    x1, y1 = prev_center
    x2, y2 = current_center
    # Since we are drawing vertically, we connect from the bottom center of the previous to the top of the current
    canvas.create_line(
        x1, y1 + prev_height / 2, x2, y2 - prev_height / 2, arrow=tk.LAST
    )


# Layout the layers in a vertical column
x_offset = 50
y_offset = 100
prev_center = None
prev_width = 0
prev_height = 0

for i, layer in enumerate(layers_info):
    w_used, h_used, center_x, center_y = draw_layer(canvas, layer, x_offset, y_offset)
    # Connect from previous layer
    if prev_center is not None:
        draw_connections(canvas, prev_center, (center_x, center_y), prev_width, h_used)

    prev_center = (center_x, center_y)
    prev_width = w_used
    prev_height = h_used
    y_offset += h_used + 100  # space between layers

root.mainloop()
