# network.py
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class ConvLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding=0, input_shape=None):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape
        if input_shape is not None:
            self._initialize_weights()
        self.x = None
        self.dW = None
        self.db = None

    def _initialize_weights(self):
        C = self.input_shape[2]
        scale = np.sqrt(2.0 / (C * self.filter_size * self.filter_size))
        self.filters = (
            np.random.randn(self.num_filters, self.filter_size, self.filter_size, C)
            * scale
        )
        self.biases = np.zeros((self.num_filters,))

    def set_input_shape(self, input_shape):
        if self.input_shape is None:
            self.input_shape = input_shape
            self._initialize_weights()

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        H_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1

        x_padded = x
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
                "constant",
            )

        out = np.zeros((N, H_out, W_out, self.num_filters))
        filters_reshaped = self.filters.reshape(self.num_filters, -1)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                patch = x_padded[:, h_start:h_end, w_start:w_end, :]
                patch_reshaped = patch.reshape(N, -1)
                out[:, i, j, :] = patch_reshaped.dot(filters_reshaped.T) + self.biases
        return out

    def backward(self, d_out):
        N, H_in, W_in, C_in = self.x.shape
        H_out, W_out = d_out.shape[1], d_out.shape[2]

        if self.padding > 0:
            x_padded = np.pad(
                self.x,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
                "constant",
            )
        else:
            x_padded = self.x

        dX_padded = np.zeros_like(x_padded)
        self.dW = np.zeros_like(self.filters)
        self.db = np.zeros_like(self.biases)

        filters_reshaped = self.filters.reshape(self.num_filters, -1)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size

                patch = x_padded[:, h_start:h_end, w_start:w_end, :]
                patch_reshaped = patch.reshape(N, -1)

                # db
                self.db += np.sum(d_out[:, i, j, :], axis=0)

                # dW
                for f in range(self.num_filters):
                    d_out_f = d_out[:, i, j, f][:, np.newaxis]
                    self.dW[f] += (
                        (patch_reshaped * d_out_f)
                        .sum(axis=0)
                        .reshape(self.filter_size, self.filter_size, C_in)
                    )

                # dX
                for n in range(N):
                    grad_in = d_out[n, i, j, :].dot(filters_reshaped)
                    dX_padded[n, h_start:h_end, w_start:w_end, :] += grad_in.reshape(
                        self.filter_size, self.filter_size, C_in
                    )

        if self.padding > 0:
            dX = dX_padded[
                :, self.padding : -self.padding, self.padding : -self.padding, :
            ]
        else:
            dX = dX_padded
        return dX

    def update_params(self, lr=0.01):
        self.filters -= lr * self.dW
        self.biases -= lr * self.db


class ReLULayer:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.x > 0)

    def update_params(self, lr=0.01):
        pass


class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        self.x = x
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

    def backward(self, d_out):
        N, H, W, C = self.x.shape
        H_out = d_out.shape[1]
        W_out = d_out.shape[2]
        dX = np.zeros_like(self.x)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                patch = self.x[:, h_start:h_end, w_start:w_end, :]
                max_vals = np.max(patch, axis=(1, 2), keepdims=True)
                mask = patch == max_vals
                dX[:, h_start:h_end, w_start:w_end, :] += (
                    mask * d_out[:, i, j, :][:, np.newaxis, np.newaxis, :]
                )
        return dX

    def update_params(self, lr=0.01):
        pass


class FlattenLayer:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.x_shape)

    def update_params(self, lr=0.01):
        pass


class DenseLayer:
    def __init__(self, input_dim, output_dim):
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.biases = np.zeros((output_dim,))
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x.dot(self.weights) + self.biases

    def backward(self, d_out):
        N = self.x.shape[0]
        self.dW = self.x.T.dot(d_out)
        self.db = np.sum(d_out, axis=0)
        dX = d_out.dot(self.weights.T)
        return dX

    def update_params(self, lr=0.01):
        self.weights -= lr * self.dW
        self.biases -= lr * self.db


class SoftmaxLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, d_out):
        # Here d_out is expected to be dL/dY from cross-entropy
        return d_out

    def update_params(self, lr=0.01):
        pass


class CNN:
    def __init__(self, layers_config, input_shape):
        self.layers = []
        logging.info(f"Initializing CNN with input shape {input_shape}.")
        current_shape = input_shape

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
                H, W, C = current_shape
                H_out = (H + 2 * layer.padding - layer.filter_size) // layer.stride + 1
                W_out = (W + 2 * layer.padding - layer.filter_size) // layer.stride + 1
                current_shape = (H_out, W_out, layer.num_filters)

            elif layer_type == "relu":
                layer = ReLULayer()

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
                H, W, C = current_shape
                current_shape = (H * W * C,)

            elif layer_type == "dense":
                input_dim = current_shape[0]
                output_dim = config.get("units")
                layer = DenseLayer(input_dim, output_dim)
                current_shape = (output_dim,)

            elif layer_type == "softmax":
                layer = SoftmaxLayer()
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            self.layers.append(layer)
            logging.info(f"Added layer {layer_type} with shape {current_shape}.")

    def forward(self, x):
        # Reduced logging: only log at start and final shape
        # Start forward pass
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def update_params(self, lr=0.01):
        for layer in self.layers:
            if hasattr(layer, "update_params"):
                layer.update_params(lr)


if __name__ == "__main__":
    from data import MNISTDataLoader

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

    # Test forward and backward on a small batch
    batch_size = 5
    batch_x = X_train[:batch_size]
    out = cnn.forward(batch_x)
    logging.info(f"Forward pass output shape: {out.shape}")
    d_out = np.random.randn(*out.shape)
    cnn.backward(d_out)
    cnn.update_params(lr=0.01)
    logging.info("Backward pass and parameter update test complete.")
