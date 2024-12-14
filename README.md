# CNN Visualization and Training Tool

This project provides a visual and interactive tool for training and inspecting a Convolutional Neural Network (CNN) on the MNIST dataset. It integrates data loading, model training, live updates of parameters and activations, and interactive controls for tuning training parameters and examining test results.

## Key Features

1. **Data Loading and Model Initialization**:  
   The code loads MNIST data from `.idx` files, normalizes it, and sets up a CNN defined in `network.py`. The network typically consists of layers like `Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax`.

2. **Real-Time Visualization**:  
   As the network trains, the GUI (`gui.py` and `visualization.py`) displays the CNN's architecture, including:
   - Convolutional layers as grids of feature maps.
   - Detailed views of convolutional filters and their biases.
   - Flatten, Dense, and Softmax layers represented as nodes or shapes, scaled to their dimensions.

3. **Interactive GUI**:  
   The `GUIApp` class in `gui.py` creates a Tkinter-based GUI with:
   - Buttons to Start, Pause, and Stop training.
   - Entry fields to input the desired number of epochs and batch size before starting training.
   - A scrollable canvas with both vertical and horizontal scrollbars to navigate large or detailed network visualizations.
   - A progress label that updates to show the current epoch and batch during training.
   - A “Test” button that, once training is done, runs the model on the test set and shows overall accuracy and a small sample of predicted vs. expected results.

4. **Queue-Based Updates and Threading**:  
   Training runs in a background thread (`train.py`), sending updates (layer data, progress info) via a `Queue` to the main GUI thread. The GUI periodically checks the queue and updates the display accordingly, ensuring the interface remains responsive.

## How It Works

1. **Loading Data**:  
   `data.py` loads the MNIST dataset, normalizes it, and returns `(X_train, y_train), (X_test, y_test)` arrays.

2. **Model Definition** (`network.py`):  
   The CNN layers are defined, each setting `self.out` after a forward pass to store activations. The `ConvLayer` also holds `filters` and `biases`. The `DenseLayer` has `weights` and `biases`. At the end of `CNN.forward(x)`, each layer's outputs (`self.out`) are available for inspection.

3. **Training Loop** (`train.py`):  
   `Trainer` runs the training in a separate thread, performing forward and backward passes. After each batch, it calls `get_layer_data()` to extract:
   - Activation stats (mean, std) for each layer.
   - Parameter stats (mean, std of weights and biases) for conv layers.
   - The current epoch and batch number for the progress label.

   This data is pushed into a queue.

4. **GUI Updates** (`gui.py` and `visualization.py`):  
   The GUI polls the queue every 200ms, processes one message at a time, and redraws the network using the data provided.  
   - `visualization.py` translates layer data into graphical boxes, nodes, and grids.
   - If the layer is convolutional and filter data is provided, it draws each filter’s weights and bias.

   Horizontal and vertical scrolling is enabled, and the GUI displays training progress and test results upon clicking “Test.”

## Explanation of Values

- **Am (Activation Mean)** and **As (Activation Std)**:  
  For each layer, `Am` and `As` represent the mean and standard deviation of that layer’s output activations (`self.out`), computed over the batch currently fed into the network.  
  A stable or final snapshot means you are not feeding new inputs or re-running forward passes. If these values change, it indicates the model saw a new input or a new forward pass was triggered.

- **Wmean/Wstd, Bmean/Bstd**:  
  For convolutional layers, these show the mean and standard deviation of the filter weights (`Wmean/Wstd`) and biases (`Bmean/Bstd`). These values change as training updates the parameters. After training finishes, these values reflect the final learned distribution of parameters.

- **Epoch/Batch Progress**:  
  The progress label at the top shows `Epoch: X, Batch: Y`, indicating the current training epoch and batch number being processed. As training proceeds, these numbers increment. Once training is done, they no longer update.

- **Test Accuracy and Predictions**:  
  After training, clicking “Test” runs the model on `X_test`. The GUI displays the test accuracy and a small sample of predictions vs. expected labels. This helps validate the model’s performance and ensure the network learned meaningful patterns from the data.

## Usage

1. **Run `main.py`**:  

   ```bash
   python main.py
