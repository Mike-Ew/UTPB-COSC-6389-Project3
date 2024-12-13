# data.py
import os
import numpy as np


class MNISTDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def _load_images(self, filename):
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise ValueError(
                    f"Invalid magic number {magic} for images file: {filename}"
                )
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")

            data = f.read(num_images * rows * cols)
            images = np.frombuffer(data, dtype=np.uint8).reshape(num_images, rows, cols)
            return images

    def _load_labels(self, filename):
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, "rb") as f:
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise ValueError(
                    f"Invalid magic number {magic} for labels file: {filename}"
                )
            num_labels = int.from_bytes(f.read(4), "big")
            labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
            return labels

    def load_data(self, normalize=True):
        X_train = self._load_images("train-images.idx3-ubyte")
        y_train = self._load_labels("train-labels.idx1-ubyte")
        X_test = self._load_images("t10k-images.idx3-ubyte")
        y_test = self._load_labels("t10k-labels.idx1-ubyte")

        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        return (X_train, y_train), (X_test, y_test)
