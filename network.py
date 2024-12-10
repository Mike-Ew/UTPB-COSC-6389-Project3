# network.py
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ConvLayer:
    """
    A simple convolutional layer implementation (forward only for demonstration).
    Parameters:
        num_filters: int, number of filters (output channels)
        filter_size: int, size of each filter (assume square, filter_size x filter_size)
        stride: int, stride for convolution
        padding: int, zero-padding on each side of the input
        input_shape: tuple (H, W, C)
    """

    def __init__(self, num_filters, filter_size, stride=1, padding=0, input_shape=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        if input_shape is not None:
            self._initialize_weights()

    def _initialize_weights(self):
        # Xavier initialization
        C = self.input_shape[2]
        scale = np.sqrt(2.0 / (C * self.filter_size * self.filter_size))
        self.filters = (
            np.random.randn(self.num_filters, self.filter_size, self.filter_size, C)
            * scale
        )
        self.biases = np.zeros((self.num_filters,))

    def set_input_shape(self, input_shape):
        """Set input shape and initialize weights if not done yet."""
        if self.input_shape is None:
            self.input_shape = input_shape
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass of convolution:
        x shape: (N, H, W, C)
        output: (N, H_out, W_out, num_filters)
        """
        N, H, W, C = x.shape
        assert C == self.input_shape[2], "Input channel dimension must match filters."

        # Compute output size
        H_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1

        # Pad input
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
                mode="constant",
            )
        else:
            x_padded = x

        # Perform convolution
        out = np.zeros((N, H_out, W_out, self.num_filters))
        for i in range(H_out):
            for j in range(W_out):
                # Region to apply filter
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size

                # Extract the patch
                patch = x_padded[
                    :, h_start:h_end, w_start:w_end, :
                ]  # shape (N, filter_size, filter_size, C)
                # Convolve with filters
                # Reshape patch and filters for vectorized multiply
                patch_reshaped = patch.reshape(N, -1)
                filters_reshaped = self.filters.reshape(
                    self.num_filters, -1
                )  # (num_filters, filter_size*filter_size*C)

                # out for this position: (N, num_filters)
                out[:, i, j, :] = patch_reshaped.dot(filters_reshaped.T) + self.biases

        return out


class ReLULayer:
    def forward(self, x):
        return np.maximum(0, x)


class MaxPoolLayer:
    """
    Max pooling layer.
    Parameters:
        pool_size: int, size of the pooling window
        stride: int, stride of pooling
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        N, H, W, C = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        out = np.zeros((N, H_out, W_out, C))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                patch = x[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(patch, axis=(1, 2))

        return out


class FlattenLayer:
    def forward(self, x):
        N = x.shape[0]
        return x.reshape(N, -1)


class DenseLayer:
    """
    Fully connected layer.
    Parameters:
        input_dim: int, number of input units
        output_dim: int, number of output units
    """

    def __init__(self, input_dim, output_dim):
        # Initialize weights
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.biases = np.zeros((output_dim,))

    def forward(self, x):
        return x.dot(self.weights) + self.biases


class SoftmaxLayer:
    def forward(self, x):
        # For demonstration, just return softmax output
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class CNN:
    """
    A configurable CNN model that performs forward pass.
    layers_config: list of layer configurations (dicts) e.g.:
        [
            {"type": "conv", "num_filters": 8, "filter_size": 3, "stride":1, "padding":1},
            {"type": "relu"},
            {"type": "pool", "pool_size":2, "stride":2},
            {"type": "flatten"},
            {"type": "dense", "units": 10}
        ]
    """

    def __init__(self, layers_config, input_shape):
        self.layers = []
        current_shape = input_shape  # (H, W, C) for conv layers
        logging.info(f"Initializing CNN with input shape {current_shape}.")

        for config in layers_config:
            layer_type = config["type"]
            if layer_type == "conv":
                layer = ConvLayer(
                    num_filters=config.get("num_filters", 8),
                    filter_size=config.get("filter_size", 3),
                    stride=config.get("stride", 1),
                    padding=config.get("padding", 1),
                    input_shape=current_shape,
                )
                # Update shape:
                H, W, C = current_shape
                H_out = (H + 2 * layer.padding - layer.filter_size) // layer.stride + 1
                W_out = (W + 2 * layer.padding - layer.filter_size) // layer.stride + 1
                current_shape = (H_out, W_out, layer.num_filters)
            elif layer_type == "relu":
                layer = ReLULayer()
                # Shape doesn't change
            elif layer_type == "pool":
                layer = MaxPoolLayer(
                    pool_size=config.get("pool_size", 2), stride=config.get("stride", 2)
                )
                H, W, C = current_shape
                H_out = (H - layer.pool_size) // layer.stride + 1
                W_out = (W - layer.pool_size) // layer.stride + 1
                current_shape = (H_out, W_out, C)
            elif layer_type == "flatten":
                layer = FlattenLayer()
                # Flatten shape: (N, H*W*C)
                H, W, C = current_shape
                current_shape = (H * W * C,)  # for the next dense layer
            elif layer_type == "dense":
                input_dim = current_shape[0]
                output_dim = config.get("units")
                layer = DenseLayer(input_dim, output_dim)
                current_shape = (output_dim,)
            elif layer_type == "softmax":
                layer = SoftmaxLayer()
                # Shape does not change
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(layer)
            logging.info(f"Added layer {layer_type} with shape {current_shape}.")

    def forward(self, x):
        logging.info("Starting forward pass:")
        for layer in self.layers:
            x = layer.forward(x)
            logging.info(f"Layer {layer.__class__.__name__} output shape: {x.shape}")
        return x


if __name__ == "__main__":
    # Example usage:
    # We'll load MNIST data using the data.py we created before.
    # Assume data.py is in the same directory.
    from data import MNISTDataLoader

    # Load data
    data_loader = MNISTDataLoader(dataset_path="./data")
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()

    # Let's create a small CNN configuration:
    # Input shape: (28, 28, 1)
    # Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax
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

    # Instantiate the CNN
    cnn = CNN(layers_config, input_shape=(28, 28, 1))

    # Test a forward pass with a small batch of the training data
    batch_size = 5
    batch_x = X_train[:batch_size]  # shape (5, 28, 28, 1)
    logging.info(f"Testing forward pass with batch size {batch_size}")
    output = cnn.forward(batch_x)
    logging.info(f"Output shape: {output.shape}")
    logging.info(f"Output (first sample): {output[0]}")
    logging.info("Forward pass test complete.")

    # Expected output:
    # - Logging info for each layer initialization and forward pass
    # - Final output shape should be (5, 10) since we have 10 classes.
