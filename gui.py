# gui.py
import tkinter as tk
from queue import Empty
from visualization import Visualization


class GUIApp:
    def __init__(
        self, root, update_queue, start_training_callback, pause_callback, stop_callback
    ):
        self.root = root
        self.root.title("CNN Training GUI")
        self.update_queue = update_queue

        # Control frame at the top
        control_frame = tk.Frame(root)
        control_frame.pack(side="top", fill="x", pady=5)

        self.train_button = tk.Button(
            control_frame, text="Train", command=start_training_callback
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(
            control_frame, text="Pause", command=pause_callback
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="Stop", command=stop_callback)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Frame for the canvas and scrollbar
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side="top", fill="both", expand=True)

        v_scrollbar = tk.Scrollbar(canvas_frame, orient="vertical")
        v_scrollbar.pack(side="right", fill="y")

        # Create visualization with parent frame as canvas_frame
        self.viz = Visualization(canvas_frame)
        # Configure the canvas scrollbar
        self.viz.canvas.config(yscrollcommand=v_scrollbar.set)
        v_scrollbar.config(command=self.viz.canvas.yview)

        self.root.after(100, self.update_ui_from_queue)

    def update_ui_from_queue(self):
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                if msg_type == "all_updates":
                    layers_data = data
                    self.viz.update_network(layers_data)
                    # Update scrollregion so scrollbar knows the size
                    self.viz.canvas.config(scrollregion=self.viz.canvas.bbox("all"))
        except Empty:
            # No more messages in the queue right now
            pass
        except Exception as e:
            # Log unexpected exceptions
            print(f"Unexpected error in update_ui_from_queue: {e}")

        self.root.after(100, self.update_ui_from_queue)

    def draw_initial_network(self, layers_data):
        self.viz.update_network(layers_data)
        # Update scrollregion after initial draw
        self.viz.canvas.config(scrollregion=self.viz.canvas.bbox("all"))
