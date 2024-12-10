# data.py
import os
import numpy as np


class MNISTDataLoader:
    """
    A class to handle loading and preprocessing of the MNIST dataset.
    It expects the MNIST files in their original IDX format (uncompressed):
        - train-images-idx3-ubyte
        - train-labels-idx1-ubyte
        - t10k-images-idx3-ubyte
        - t10k-labels-idx1-ubyte
    """

    def __init__(self, dataset_path):
        """
        Initialize the loader with a path where the MNIST dataset files reside.
        :param dataset_path: Path to the directory containing the MNIST .ubyte files.
        """
        self.dataset_path = dataset_path

    def _load_images(self, filename):
        """
        Load image data from MNIST .ubyte file.
        :param filename: The name of the .ubyte file containing image data.
        :return: A numpy array of shape (num_samples, 28, 28) with pixel values.
        """
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, "rb") as f:
            # According to the MNIST file format:
            # The first 16 bytes are:
            # - Magic number (4 bytes)
            # - Number of images (4 bytes)
            # - Rows (4 bytes)
            # - Columns (4 bytes)
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise ValueError(
                    f"Invalid magic number {magic} for images file: {filename}"
                )
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")

            # Read the image data
            data = f.read(num_images * rows * cols)
            images = np.frombuffer(data, dtype=np.uint8).reshape(num_images, rows, cols)
            return images

    def _load_labels(self, filename):
        """
        Load label data from MNIST .ubyte file.
        :param filename: The name of the .ubyte file containing label data.
        :return: A numpy array of shape (num_samples,) with labels.
        """
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, "rb") as f:
            # According to the MNIST file format:
            # The first 8 bytes are:
            # - Magic number (4 bytes)
            # - Number of labels (4 bytes)
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise ValueError(
                    f"Invalid magic number {magic} for labels file: {filename}"
                )
            num_labels = int.from_bytes(f.read(4), "big")

            # Read the label data
            labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
            return labels

    def load_data(self, normalize=True):
        """
        Load and preprocess the MNIST dataset from the given path.

        :param normalize: If True, normalize the images to [0, 1].
        :return: (X_train, y_train), (X_test, y_test)
                 X arrays have shape (num_samples, 28, 28, 1)
                 y arrays have shape (num_samples,)
        """
        # Training set
        X_train = self._load_images("train-images.idx3-ubyte")
        y_train = self._load_labels("train-labels.idx1-ubyte")

        # Test set
        X_test = self._load_images("t10k-images.idx3-ubyte")
        y_test = self._load_labels("t10k-labels.idx1-ubyte")

        # Optionally normalize the pixel values
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0

        # Reshape data to include channel dimension for CNNs: (H, W) -> (H, W, 1)
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # Example usage:
    # Ensure that 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
    # 't10k-images-idx3-ubyte', and 't10k-labels-idx1-ubyte'
    # are present in ./data directory.
    data_loader = MNISTDataLoader(dataset_path="./data")
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()

    print("Training set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)
