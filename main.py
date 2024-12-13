# main.py
import numpy as np
import tkinter as tk
import queue
from data import MNISTDataLoader
from network import CNN
from train import Trainer
from gui import GUIApp


def main():
    data_loader = MNISTDataLoader(dataset_path="./data")
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()
    print("Training set:", X_train.shape, y_train.shape)
    print("Test set:", X_test.shape, y_test.shape)

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

    update_queue = queue.Queue()
    trainer = Trainer(cnn, X_train, y_train, X_test, y_test, update_queue=update_queue)

    root = tk.Tk()

    def start_training():
        trainer.start_training(epochs=1, batch_size=16, lr=0.01)

    def pause_training():
        trainer.pause_training()

    def stop_training():
        trainer.stop_training()

    app = GUIApp(root, update_queue, start_training, pause_training, stop_training)

    # Run a small forward pass to populate self.out for each layer:
    dummy_x = X_train[:1]
    _ = cnn.forward(dummy_x)  # Now self.out is set in each layer.

    # Now get real layer data based on actual model parameters/activations
    initial_data = trainer.get_layer_data()
    app.draw_initial_network(initial_data)

    root.mainloop()


if __name__ == "__main__":
    main()
