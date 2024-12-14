# gui.py
import tkinter as tk
from queue import Empty
from visualization import Visualization


class GUIApp:
    def __init__(
        self,
        root,
        update_queue,
        start_training_callback,
        pause_callback,
        stop_callback,
        trainer=None,
    ):
        self.root = root
        self.root.title("CNN Training GUI")
        self.update_queue = update_queue
        self.trainer = (
            trainer  # We'll store trainer reference to call run_test if needed
        )

        # Control frame at the top
        control_frame = tk.Frame(root)
        control_frame.pack(side="top", fill="x", pady=5)

        self.train_button = tk.Button(
            control_frame,
            text="Train",
            command=self.start_training_wrapper(start_training_callback),
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(
            control_frame, text="Pause", command=pause_callback
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="Stop", command=stop_callback)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Entries for epochs and batch size
        tk.Label(control_frame, text="Epochs:").pack(side=tk.LEFT, padx=5)
        self.epoch_entry = tk.Entry(control_frame, width=5)
        self.epoch_entry.insert(0, "1")
        self.epoch_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        self.batch_entry = tk.Entry(control_frame, width=5)
        self.batch_entry.insert(0, "16")
        self.batch_entry.pack(side=tk.LEFT, padx=5)

        # Progress label
        self.progress_label = tk.Label(control_frame, text="Epoch: 0, Batch: 0")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Test button and result label
        self.test_button = tk.Button(control_frame, text="Test", command=self.run_test)
        self.test_button.pack(side=tk.LEFT, padx=5)

        self.test_result_label = tk.Label(control_frame, text="")
        self.test_result_label.pack(side=tk.LEFT, padx=5)

        # Frame for the canvas and scrollbars
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side="top", fill="both", expand=True)

        v_scrollbar = tk.Scrollbar(canvas_frame, orient="vertical")
        v_scrollbar.pack(side="right", fill="y")

        h_scrollbar = tk.Scrollbar(canvas_frame, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")

        self.viz = Visualization(canvas_frame)
        # Configure both scrollbars
        self.viz.canvas.config(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )
        v_scrollbar.config(command=self.viz.canvas.yview)
        h_scrollbar.config(command=self.viz.canvas.xview)

        self.root.after(200, self.update_ui_from_queue)

    def start_training_wrapper(self, start_training_callback):
        def wrapper():
            epochs = int(self.epoch_entry.get())
            batch_size = int(self.batch_entry.get())
            start_training_callback(epochs=epochs, batch_size=batch_size, lr=0.01)

        return wrapper

    def update_ui_from_queue(self):
        # Attempt to process just one message per cycle to keep the UI responsive
        try:
            msg_type, data = self.update_queue.get_nowait()
            if msg_type == "all_updates":
                layers_data = data
                # Extract progress info if present
                progress_info = next(
                    (l for l in layers_data if l.get("info_type") == "progress"), None
                )
                if progress_info:
                    self.progress_label.config(
                        text=f"Epoch: {progress_info['epoch']}, Batch: {progress_info['batch']}"
                    )

                self.viz.update_network(layers_data)
                # Update scrollregion so scrollbar knows the size
                self.viz.canvas.config(scrollregion=self.viz.canvas.bbox("all"))
        except Empty:
            # No more messages in the queue right now
            pass
        except Exception as e:
            # Log unexpected exceptions
            print(f"Unexpected error in update_ui_from_queue: {e}")

        # Schedule the next update
        self.root.after(200, self.update_ui_from_queue)

    def draw_initial_network(self, layers_data):
        self.viz.update_network(layers_data)
        # Update scrollregion after initial draw
        self.viz.canvas.config(scrollregion=self.viz.canvas.bbox("all"))

    def run_test(self):
        if self.trainer is not None:
            test_acc, predicted, expected = self.trainer.test_network()
            self.test_result_label.config(text=f"Test Acc: {test_acc:.4f}")

            sample_count = 5
            samples_str = "\n".join(
                f"Expected: {exp}, Pred: {pred}"
                for exp, pred in zip(expected[:sample_count], predicted[:sample_count])
            )
            # Show this in a popup
            top = tk.Toplevel(self.root)
            top.title("Test Samples")
            tk.Label(top, text=samples_str, font=("Helvetica", 10)).pack(
                padx=10, pady=10
            )
        else:
            print("Trainer not provided, cannot run test.")
