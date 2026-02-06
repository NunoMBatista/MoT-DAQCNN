#!/usr/bin/env python3
"""
Interactive TUI for DAQCNN Model Testing
Navigate through saved models and perform inference on datasets or individual images.
"""

import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.append(_project_root)

import subprocess
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Static,
    Tree,
)

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

from src.config import DATA_DIR
from src.utils.data import (
    check_model_dataset_compatibility,
    get_dataloaders,
    get_dataset_channels,
)
from src.utils.evaluate import compute_metrics
from src.utils.model_cache_manager import load_model_from_checkpoint, scan_all_outputs
from src.utils.quantum_dataset_cache import (
    find_cached_quantum_dataset,
    load_cached_quantum_dataset,
)


class ConfigModal(ModalScreen):
    """Modal to display model configuration."""

    def __init__(self, config_info: str):
        super().__init__()
        self.config_info = config_info

    def compose(self) -> ComposeResult:
        with Container(id="config-modal"):
            with VerticalScroll(id="config-scroll"):
                yield Static(self.config_info, id="config-text")
            yield Label("[↑↓/Mouse] Scroll | [i/ESC] Close", id="config-help")

    async def on_key(self, event) -> None:
        """Handle all key presses."""
        if event.key == "i" or event.key == "escape":
            self.dismiss()
        elif event.key == "up":
            scroll = self.query_one("#config-scroll", VerticalScroll)
            scroll.scroll_relative(y=-3, animate=False)
        elif event.key == "down":
            scroll = self.query_one("#config-scroll", VerticalScroll)
            scroll.scroll_relative(y=3, animate=False)
        elif event.key == "pageup":
            scroll = self.query_one("#config-scroll", VerticalScroll)
            scroll.scroll_page_up()
        elif event.key == "pagedown":
            scroll = self.query_one("#config-scroll", VerticalScroll)
            scroll.scroll_page_down()


class DatasetSelectionModal(ModalScreen):
    """Modal to select dataset for testing."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self):
        super().__init__()
        self.selected_dataset = None

    def compose(self) -> ComposeResult:
        with Container(id="dataset-modal"):
            yield Label("Select Dataset:", id="dataset-title")
            yield Button("Pneumonia MNIST", id="pneumonia_mnist", variant="primary")
            yield Button("Breast MNIST", id="breast_mnist", variant="primary")
            yield Button("Path MNIST", id="path_mnist", variant="primary")
            yield Button("Derma MNIST", id="derma_mnist", variant="primary")
            yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        else:
            self.dismiss(event.button.id)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class ImageSelectionModal(ModalScreen):
    """Modal to select and preview images from dataset."""

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
        ("i", "preview_image", "Preview"),
        ("enter", "classify_current", "Classify"),
    ]

    def __init__(self, dataset_name: str, model_device: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_device = model_device
        self.images = []
        self.labels = []
        self.current_idx = 0

    def compose(self) -> ComposeResult:
        with Container(id="image-selection-modal"):
            yield Label(f"Loading {self.dataset_name} images...", id="image-info")
            yield ListView(id="image-list")
            yield Label(
                "[↑↓/Click] Select | [i] Preview | [Enter] Classify | [ESC] Cancel",
                id="image-help",
            )
            yield Button("Classify Selected", id="classify", variant="primary")
            yield Button("Cancel", id="cancel", variant="error")

    def on_mount(self) -> None:
        """Load images when modal opens."""
        self.load_images()

    def on_key(self, event) -> None:
        """Handle key presses for navigation."""
        # ListView handles arrow keys automatically, but we need to track for preview
        pass

    def load_images(self) -> None:
        """Load test images from dataset."""
        try:
            # Create minimal config for data loading
            cfg = {
                "dataset": {
                    "name": self.dataset_name,
                    "data_root": str(DATA_DIR),
                    "batch_size": 32,
                    "num_workers": 0,
                    "download": True,
                }
            }

            train_loader, _, _, _ = get_dataloaders(cfg)

            # Load 10 images from each class (positive and negative)
            class_0_images = []
            class_0_labels = []
            class_1_images = []
            class_1_labels = []

            for imgs, lbls in train_loader:
                for img, lbl in zip(imgs, lbls):
                    label_val = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)

                    if label_val == 0 and len(class_0_images) < 10:
                        class_0_images.append(img)
                        class_0_labels.append(lbl)
                    elif label_val == 1 and len(class_1_images) < 10:
                        class_1_images.append(img)
                        class_1_labels.append(lbl)

                    # Stop when we have 10 from each class
                    if len(class_0_images) >= 10 and len(class_1_images) >= 10:
                        break

                if len(class_0_images) >= 10 and len(class_1_images) >= 10:
                    break

            # Combine images from both classes
            self.images = class_0_images + class_1_images
            self.labels = class_0_labels + class_1_labels

            self.update_image_list()
            self.query_one("#image-info", Label).update(
                f"Loaded {len(self.images)} images ({len(class_0_images)} class 0, {len(class_1_images)} class 1) from {self.dataset_name}"
            )

        except Exception as e:
            self.query_one("#image-info", Label).update(
                f"Error loading images: {str(e)}"
            )

    def update_image_list(self) -> None:
        """Update the displayed image list."""
        if not self.images:
            return

        list_view = self.query_one("#image-list", ListView)
        list_view.clear()

        for i, label in enumerate(self.labels):
            # Handle both tensor and scalar labels
            label_val = int(label.item()) if hasattr(label, "item") else int(label)
            list_item = ListItem(Label(f"Image {i + 1}: Label={label_val}"))
            list_view.append(list_item)

        # Set initial selection
        if self.current_idx < len(self.images):
            list_view.index = self.current_idx

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection (mouse click or arrow keys)."""
        list_view = self.query_one("#image-list", ListView)
        self.current_idx = list_view.index if list_view.index is not None else 0
        self.query_one("#image-info", Label).update(
            f"Dataset: {self.dataset_name} | Selected: Image {self.current_idx + 1}/{len(self.images)}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "classify":
            if self.images:
                # Get current selection from ListView
                list_view = self.query_one("#image-list", ListView)
                self.current_idx = list_view.index if list_view.index is not None else 0

                img = self.images[self.current_idx]
                # Ensure img is a tensor
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)

                label = self.labels[self.current_idx]
                label_val = int(label.item()) if hasattr(label, "item") else int(label)

                self.dismiss(
                    {
                        "image": img,
                        "label": label_val,
                        "index": self.current_idx,
                    }
                )

    def action_classify_current(self) -> None:
        """Classify the currently selected image."""
        if self.images:
            # Get current selection from ListView
            list_view = self.query_one("#image-list", ListView)
            self.current_idx = list_view.index if list_view.index is not None else 0

            img = self.images[self.current_idx]
            # Ensure img is a tensor
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)

            label = self.labels[self.current_idx]
            label_val = int(label.item()) if hasattr(label, "item") else int(label)

            self.dismiss(
                {
                    "image": img,
                    "label": label_val,
                    "index": self.current_idx,
                }
            )

    def action_preview_image(self) -> None:
        """Show image preview with matplotlib."""
        if not self.images:
            return

        # Get current selection from ListView
        list_view = self.query_one("#image-list", ListView)
        self.current_idx = list_view.index if list_view.index is not None else 0

        img = self.images[self.current_idx]
        # Ensure img is a tensor
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        img_np = img.squeeze().cpu().numpy()

        label = self.labels[self.current_idx]
        label_val = int(label.item()) if hasattr(label, "item") else int(label)

        # Plot the image using matplotlib and save to temp file
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_path = temp_file.name
            temp_file.close()

            # Create and save the plot
            plt.figure(figsize=(8, 8))
            plt.imshow(img_np, cmap="gray")
            plt.title(
                f"Image {self.current_idx + 1} - Label: {label_val}\nShape: {img_np.shape}",
                fontsize=14,
            )
            plt.colorbar(label="Intensity")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(temp_path, dpi=100, bbox_inches="tight")
            plt.close()

            # Open with system default viewer (non-blocking)
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", temp_path])
            elif sys.platform == "win32":  # Windows
                subprocess.Popen(["start", temp_path], shell=True)
            else:  # Linux and others
                subprocess.Popen(["xdg-open", temp_path])

            # Show stats in notification
            info = f"Image {self.current_idx + 1}\n"
            info += f"Label: {label_val}\n"
            info += f"Min: {img_np.min():.3f}, Max: {img_np.max():.3f}, Mean: {img_np.mean():.3f}"
            self.app.notify(info, title="Image Preview Opened")
        except Exception as e:
            self.app.notify(f"Error displaying image: {str(e)}", severity="error")

    def action_dismiss(self) -> None:
        self.dismiss(None)


class ErrorModal(ModalScreen):
    """Modal to display error messages."""

    BINDINGS = [("escape", "dismiss", "Close"), ("enter", "dismiss", "Close")]

    def __init__(self, error_message: str, title: str = "Error"):
        super().__init__()
        self.error_message = error_message
        self.title = title

    def compose(self) -> ComposeResult:
        with Container(id="error-modal"):
            yield Label(f"[bold red]{self.title}[/bold red]", id="error-title")
            with VerticalScroll(id="error-scroll"):
                yield Static(self.error_message, id="error-text")
            yield Button("OK", id="ok-button", variant="error")
            yield Label("[Enter/ESC] Close", id="error-help")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_dismiss(self) -> None:
        self.dismiss()


class ResultsModal(ModalScreen):
    """Modal to display evaluation results."""

    BINDINGS = [("escape", "dismiss", "Close"), ("q", "dismiss", "Close")]

    def __init__(self, results_text: str, confusion_matrix=None, predictions_data=None):
        super().__init__()
        self.results_text = results_text
        self.confusion_matrix = confusion_matrix
        self.predictions_data = predictions_data

    def compose(self) -> ComposeResult:
        with Container(id="results-modal"):
            with VerticalScroll(id="results-scroll"):
                yield Static(self.results_text, id="results-text")
            if self.predictions_data is not None:
                with Horizontal():
                    yield Button(
                        "View Correct Predictions", id="view-correct", variant="success"
                    )
                    yield Button(
                        "View Incorrect Predictions",
                        id="view-incorrect",
                        variant="error",
                    )
            yield Label("[ESC/q] Close", id="results-help")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "view-correct":
            self.dismiss({"action": "view-correct"})
        elif event.button.id == "view-incorrect":
            self.dismiss({"action": "view-incorrect"})

    def action_dismiss(self) -> None:
        self.dismiss(None)


class PredictionResultsModal(ModalScreen):
    """Modal to display prediction results with option to view images."""

    BINDINGS = [("escape", "dismiss", "Close"), ("i", "preview_selected", "Preview")]

    def __init__(self, title: str, predictions: list):
        super().__init__()
        self.title = title
        self.predictions = predictions  # List of (image, true_label, pred_label, prob)
        self.current_idx = 0

    def compose(self) -> ComposeResult:
        with Container(id="predictions-modal"):
            yield Label(self.title, id="predictions-title")
            yield ListView(id="predictions-list")
            yield Label(
                "[↑↓/Click] Select | [i] Preview Image | [ESC] Close",
                id="predictions-help",
            )
            yield Button("Close", id="close", variant="primary")

    def on_mount(self) -> None:
        """Populate the predictions list."""
        list_view = self.query_one("#predictions-list", ListView)

        for i, (img, true_label, pred_label, prob) in enumerate(self.predictions):
            list_item = ListItem(
                Label(
                    f"Image {i + 1}: True={true_label}, Pred={pred_label}, Prob={prob:.3f}"
                )
            )
            list_view.append(list_item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        list_view = self.query_one("#predictions-list", ListView)
        self.current_idx = list_view.index if list_view.index is not None else 0

    def action_preview_selected(self) -> None:
        """Preview the selected image."""
        if not self.predictions:
            return

        list_view = self.query_one("#predictions-list", ListView)
        self.current_idx = list_view.index if list_view.index is not None else 0

        img, true_label, pred_label, prob = self.predictions[self.current_idx]

        # Ensure img is a tensor and convert to numpy
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        img_np = img.squeeze().cpu().numpy()

        # Plot the image
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_path = temp_file.name
            temp_file.close()

            plt.figure(figsize=(8, 8))
            plt.imshow(img_np, cmap="gray")
            title = f"Image {self.current_idx + 1}\nTrue: {true_label}, Predicted: {pred_label}\nProbability: {prob:.3f}"
            plt.title(title, fontsize=14)
            plt.colorbar(label="Intensity")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(temp_path, dpi=100, bbox_inches="tight")
            plt.close()

            # Open with system viewer
            if sys.platform == "darwin":
                subprocess.Popen(["open", temp_path])
            elif sys.platform == "win32":
                subprocess.Popen(["start", temp_path], shell=True)
            else:
                subprocess.Popen(["xdg-open", temp_path])

            self.app.notify(f"Viewing image {self.current_idx + 1}", title="Preview")
        except Exception as e:
            self.app.notify(f"Error displaying image: {str(e)}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss()

    def action_dismiss(self) -> None:
        self.dismiss()


class ProgressModal(ModalScreen):
    """Modal to display progress during inference."""

    def __init__(self, title: str, total: int):
        super().__init__()
        self.title = title
        self.total = total
        self.current = 0

    def compose(self) -> ComposeResult:
        with Container(id="progress-modal"):
            yield Label(self.title, id="progress-title")
            yield Static("", id="progress-bar")
            yield Label("Processing...", id="progress-status")

    def update_progress(self, current: int, status: str = "") -> None:
        """Update the progress bar."""
        self.current = current
        percent = (current / self.total * 100) if self.total > 0 else 0
        filled = int(percent / 2)  # 50 chars = 100%
        bar = "█" * filled + "░" * (50 - filled)

        progress_text = f"[{bar}] {percent:.1f}% ({current}/{self.total})"

        try:
            self.query_one("#progress-bar", Static).update(progress_text)

            if status:
                self.query_one("#progress-status", Label).update(status)
        except Exception:
            # Modal not mounted yet, skip update
            pass


class DAQCNNTestUI(App):
    """Interactive TUI for DAQCNN model testing."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    #model-tree {
        width: 50%;
        border: solid $primary;
        height: 100%;
    }

    #info-panel {
        width: 50%;
        border: solid $accent;
        height: 100%;
        padding: 1;
    }

    #config-modal, #dataset-modal, #image-selection-modal, #results-modal, #progress-modal, #predictions-modal, #error-modal {
        align: center middle;
        background: $surface;
        border: thick $primary;
        width: 80%;
        height: 80%;
        padding: 2;
    }

    #dataset-modal, #progress-modal, #error-modal {
        height: auto;
        width: 50%;
    }

    #predictions-modal {
        width: 70%;
        height: 70%;
    }

    #predictions-list {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        margin: 1;
    }

    #predictions-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #config-scroll {
        width: 100%;
        height: 1fr;
        border: solid $accent;
    }

    #image-list {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #results-scroll {
        width: 100%;
        height: 1fr;
        border: solid $accent;
    }

    #config-text, #results-text {
        width: 100%;
        height: auto;
    }

    Button {
        margin: 1;
    }

    #dataset-title, #image-info {
        text-align: center;
        margin-bottom: 1;
    }

    #progress-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #progress-bar {
        width: 100%;
        margin: 1;
    }

    #progress-status {
        text-align: center;
        margin-top: 1;
    }

    #error-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #error-scroll {
        width: 100%;
        height: 1fr;
        border: solid $error;
        margin: 1;
    }

    #error-text {
        width: 100%;
        height: auto;
        padding: 1;
    }

    #error-help {
        text-align: center;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        Binding("i", "show_info", "Show Info", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.runs = []
        self.selected_checkpoint = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            yield Tree("Models", id="model-tree")
            with VerticalScroll(id="info-panel"):
                yield Label("Select a model to view details", id="info-label")
                yield Button("Test on Dataset", id="test-dataset", disabled=True)
                yield Button("Test on Single Image", id="test-image", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app."""
        self.title = "DAQCNN Model Testing Interface"
        self.sub_title = f"Device: {self.device}"
        self.load_models()

    def load_models(self) -> None:
        """Scan and load available models."""
        tree = self.query_one("#model-tree", Tree)
        tree.clear()
        tree.root.expand()

        self.runs = scan_all_outputs()

        if not self.runs:
            tree.root.add_leaf("No models found in outputs/")
            return

        for run in self.runs:
            run_node = tree.root.add(run["run_name"])

            if run["checkpoints"]["best"]:
                best_node = run_node.add("Best Models")
                for ckpt in run["checkpoints"]["best"]:
                    best_node.add_leaf(
                        os.path.basename(ckpt), data={"path": ckpt, "run": run}
                    )

            if run["checkpoints"]["final"]:
                final_node = run_node.add("Final Models")
                for ckpt in run["checkpoints"]["final"]:
                    final_node.add_leaf(
                        os.path.basename(ckpt), data={"path": ckpt, "run": run}
                    )

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Handle tree node selection."""
        if event.node.data:
            self.selected_checkpoint = event.node.data
            self.update_info_panel()
            self.query_one("#test-dataset", Button).disabled = False
            self.query_one("#test-image", Button).disabled = False

    def update_info_panel(self) -> None:
        """Update info panel with selected model details."""
        if not self.selected_checkpoint:
            return

        info_label = self.query_one("#info-label", Label)
        ckpt_path = self.selected_checkpoint["path"]
        run_info = self.selected_checkpoint["run"]

        info_text = f"[bold]Selected:[/bold] {os.path.basename(ckpt_path)}\n"
        info_text += f"[bold]Run:[/bold] {run_info['run_name']}\n\n"

        if run_info["config"]:
            dataset = run_info["config"].get("dataset", {}).get("name", "Unknown")
            epochs = run_info["config"].get("optim", {}).get("epochs", "Unknown")
            lr = run_info["config"].get("optim", {}).get("lr", "Unknown")
            info_text += f"[bold]Dataset:[/bold] {dataset}\n"
            info_text += f"[bold]Epochs:[/bold] {epochs}\n"
            info_text += f"[bold]Learning Rate:[/bold] {lr}\n\n"

        # Display model metadata if available
        if run_info.get("model_metadata"):
            metadata = run_info["model_metadata"]
            info_text += f"[bold yellow]Model Architecture:[/bold yellow]\n"
            total_params = metadata.get("total_params", "N/A")
            trainable_params = metadata.get("trainable_params", "N/A")
            if isinstance(total_params, int):
                info_text += f"  Total params: {total_params:,}\n"
                info_text += f"  Trainable params: {trainable_params:,}\n"
            else:
                info_text += f"  Total params: {total_params}\n"
                info_text += f"  Trainable params: {trainable_params}\n"

            # Display quantum kernel information
            kernel_size = metadata.get("quantum_kernel_size")
            kernel_topologies = metadata.get("quantum_kernel_topologies", [])
            if kernel_size is not None:
                info_text += f"  Quantum kernel: {kernel_size}×{kernel_size}\n"
            if kernel_topologies:
                topologies_str = ", ".join(kernel_topologies)
                info_text += f"  Topologies: {topologies_str}\n"

            # Display CNN head structure summary
            head_structure = metadata.get("head_structure", [])
            if head_structure:
                conv_layers = [l for l in head_structure if l["type"] == "Conv2d"]
                linear_layers = [
                    l for l in head_structure if l["type"] in ["Linear", "LazyLinear"]
                ]
                info_text += f"  CNN layers: {len(conv_layers)}\n"
                info_text += f"  FC layers: {len(linear_layers)}\n"

            # Show compatible datasets
            model_type = (
                "RGB"
                if total_params != "N/A" and isinstance(total_params, int)
                else None
            )
            if run_info.get("config"):
                model_in_channels = (
                    run_info["config"].get("model", {}).get("in_channels", 1)
                )
                model_type_str = "RGB" if model_in_channels == 3 else "Grayscale"
                compatible_datasets = []
                all_datasets = [
                    "pneumonia_mnist",
                    "breast_mnist",
                    "path_mnist",
                    "derma_mnist",
                ]
                for ds in all_datasets:
                    is_compat, _ = check_model_dataset_compatibility(
                        model_in_channels, ds
                    )
                    if is_compat:
                        compatible_datasets.append(ds.replace("_", " ").title())

                if compatible_datasets:
                    info_text += f"\n[bold green]Compatible datasets:[/bold green]\n"
                    info_text += f"  {', '.join(compatible_datasets)}\n"

            info_text += "\n"

        # Display test metrics if available
        if run_info.get("metrics"):
            metrics = run_info["metrics"]
            info_text += f"[bold cyan]Test Metrics:[/bold cyan]\n"
            if "test_acc" in metrics:
                acc = metrics["test_acc"].get("mean", "N/A")
                info_text += (
                    f"  Accuracy: {acc:.4f}\n"
                    if isinstance(acc, float)
                    else f"  Accuracy: {acc}\n"
                )
            if "test_auc" in metrics:
                auc = metrics["test_auc"].get("mean", "N/A")
                info_text += (
                    f"  AUC/ROC: {auc:.4f}\n"
                    if isinstance(auc, float)
                    else f"  AUC/ROC: {auc}\n"
                )
            if "test_f1" in metrics:
                f1 = metrics["test_f1"].get("mean", "N/A")
                info_text += (
                    f"  F1 Score: {f1:.4f}\n"
                    if isinstance(f1, float)
                    else f"  F1 Score: {f1}\n"
                )
            if "test_recall" in metrics:
                recall = metrics["test_recall"].get("mean", "N/A")
                info_text += (
                    f"  Recall: {recall:.4f}\n"
                    if isinstance(recall, float)
                    else f"  Recall: {recall}\n"
                )
            info_text += "\n"

        info_text += "[dim]Press 'i' for full config[/dim]"
        info_label.update(info_text)

    def action_show_info(self) -> None:
        """Show detailed config in modal."""
        if not self.selected_checkpoint:
            return

        ckpt_path = self.selected_checkpoint["path"]
        run_info = self.selected_checkpoint["run"]

        # Load checkpoint to get the complete config including seed
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            full_config = checkpoint.get("config", {})
            seed = checkpoint.get("seed", "N/A")

            # Build comprehensive config display
            config_text = "[bold cyan]Model Configuration[/bold cyan]\n"
            config_text += "=" * 60 + "\n\n"
            config_text += f"[bold]Seed:[/bold] {seed}\n\n"

            # Display full config as YAML
            config_str = yaml.dump(full_config, default_flow_style=False, indent=2)
            config_text += config_str

            self.push_screen(ConfigModal(config_text))
        except Exception as e:
            # Fallback to just the config file if checkpoint loading fails
            if run_info["config"]:
                config_str = yaml.dump(
                    run_info["config"], default_flow_style=False, indent=2
                )
                self.push_screen(ConfigModal(config_str))
            else:
                self.notify(f"Error loading config: {str(e)}", severity="error")

    def action_refresh(self) -> None:
        """Refresh the model list."""
        self.load_models()
        self.notify("Models refreshed")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "test-dataset":
            self.show_dataset_selection()
        elif event.button.id == "test-image":
            self.show_image_selection()

    def show_dataset_selection(self) -> None:
        """Show dataset selection modal."""

        def handle_selection(dataset_name):
            if dataset_name:
                self.test_on_dataset(dataset_name)

        self.push_screen(DatasetSelectionModal(), handle_selection)

    def show_image_selection(self) -> None:
        """Show image selection modal."""

        def handle_dataset_choice(dataset_name):
            if dataset_name:

                def handle_image_selection(image_data):
                    if image_data:
                        self.test_single_image(dataset_name, image_data)

                self.push_screen(
                    ImageSelectionModal(dataset_name, self.device),
                    handle_image_selection,
                )

        self.push_screen(DatasetSelectionModal(), handle_dataset_choice)

    def test_on_dataset(self, dataset_name: str) -> None:
        """Test model on full dataset."""
        if not self.selected_checkpoint:
            return

        try:
            # Load model
            ckpt_path = self.selected_checkpoint["path"]
            model, checkpoint = load_model_from_checkpoint(
                ckpt_path, device=self.device
            )

            # Check model/dataset compatibility
            model_in_channels = (
                checkpoint.get("config", {}).get("model", {}).get("in_channels", 1)
            )
            is_compatible, error_msg = check_model_dataset_compatibility(
                model_in_channels, dataset_name
            )

            if not is_compatible:
                self.push_screen(ErrorModal(error_msg, "Incompatible Dataset"))
                return

            self.notify(f"Loading model and testing on {dataset_name}...")

            # Load dataset (check for cached quantum features first)
            cfg = checkpoint["config"]
            cfg["dataset"]["name"] = dataset_name

            cached_path = find_cached_quantum_dataset(cfg)
            if cached_path is not None:
                self.notify("Using cached quantum features", severity="information")
                model.bypass_quantum = True
                batch_size = cfg["dataset"].get("batch_size", 32)
                num_workers = cfg["dataset"].get("num_workers", 2)
                _, _, test_loader, n_classes, _ = load_cached_quantum_dataset(
                    cached_path, batch_size, num_workers
                )
            else:
                _, _, test_loader, n_classes = get_dataloaders(cfg)

            # Create progress modal
            total_batches = len(test_loader)
            progress_modal = ProgressModal(f"Testing on {dataset_name}", total_batches)
            self.push_screen(progress_modal)

            # Evaluate with progress tracking
            model.eval()
            all_logits, all_labels = [], []
            all_images = []  # Store images for later viewing

            with torch.no_grad():
                for batch_idx, (features, labels) in enumerate(test_loader):
                    features = features.to(self.device)
                    labels = labels.squeeze().long().to(self.device)
                    logits = model(features)
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())

                    # Update progress
                    progress_modal.update_progress(
                        batch_idx + 1,
                        f"Processing batch {batch_idx + 1}/{total_batches}",
                    )
                    self.refresh()

            # Load classical images for preview (cheap for 28x28 MedMNIST)
            _, _, classical_loader, _ = get_dataloaders(cfg)
            for imgs, _ in classical_loader:
                all_images.extend(list(imgs))

            # Close progress modal
            self.pop_screen()

            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            metrics = compute_metrics(all_logits, all_labels)
            cm = metrics["confusion_matrix"]

            # Get predictions and probabilities
            preds = all_logits.argmax(dim=1).numpy()
            probs = torch.softmax(all_logits, dim=1)
            labels_np = all_labels.numpy()

            # Prepare predictions data
            predictions_data = {
                "images": all_images,
                "true_labels": labels_np,
                "pred_labels": preds,
                "probs": probs.numpy(),
            }

            # Format results with confusion matrix
            results = (
                f"[bold cyan]Results on {dataset_name} (test set):[/bold cyan]\n\n"
            )
            results += f"[bold]Accuracy:[/bold] {metrics['accuracy']:.4f}\n"
            results += f"[bold]AUC:[/bold]      {metrics['auc']:.4f}\n"
            results += f"[bold]F1:[/bold]       {metrics['f1']:.4f}\n"
            results += f"[bold]Recall:[/bold]   {metrics['recall']:.4f}\n\n"

            # Add confusion matrix
            results += "[bold]Confusion Matrix:[/bold]\n"
            results += "          Predicted\n"
            results += "         Class 0  Class 1\n"
            results += f"Actual 0   {cm[0][0]:4d}     {cm[0][1]:4d}\n"
            results += f"       1   {cm[1][0]:4d}     {cm[1][1]:4d}\n\n"

            # Add summary
            correct = (preds == labels_np).sum()
            incorrect = (preds != labels_np).sum()
            results += f"[bold]Correct predictions:[/bold] {correct}/{len(labels_np)}\n"
            results += (
                f"[bold]Incorrect predictions:[/bold] {incorrect}/{len(labels_np)}\n\n"
            )
            results += "[dim]Click buttons below to view predictions[/dim]"

            # Show results and handle button clicks
            def handle_results(action):
                if action and isinstance(action, dict):
                    if action["action"] == "view-correct":
                        self.show_predictions(predictions_data, correct_only=True)
                    elif action["action"] == "view-incorrect":
                        self.show_predictions(predictions_data, correct_only=False)

            self.push_screen(
                ResultsModal(
                    results, confusion_matrix=cm, predictions_data=predictions_data
                ),
                handle_results,
            )
            self.notify("Evaluation complete!", severity="information")

        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")

    def show_predictions(self, predictions_data: dict, correct_only: bool) -> None:
        """Show a modal with prediction results."""
        images = predictions_data["images"]
        true_labels = predictions_data["true_labels"]
        pred_labels = predictions_data["pred_labels"]
        probs = predictions_data["probs"]

        # Filter predictions
        predictions = []
        for i in range(len(images)):
            is_correct = true_labels[i] == pred_labels[i]
            if correct_only == is_correct:
                # Get max probability for predicted class
                max_prob = probs[i][pred_labels[i]]
                predictions.append(
                    (
                        images[i],
                        int(true_labels[i]),
                        int(pred_labels[i]),
                        float(max_prob),
                    )
                )

        title = "Correct Predictions" if correct_only else "Incorrect Predictions"
        title += f" ({len(predictions)} images)"

        self.push_screen(PredictionResultsModal(title, predictions))

    def test_single_image(self, dataset_name: str, image_data: dict) -> None:
        """Test model on a single image."""
        if not self.selected_checkpoint:
            return

        try:
            # Load model
            ckpt_path = self.selected_checkpoint["path"]
            model, checkpoint = load_model_from_checkpoint(
                ckpt_path, device=self.device
            )

            # Check model/dataset compatibility
            model_in_channels = (
                checkpoint.get("config", {}).get("model", {}).get("in_channels", 1)
            )
            is_compatible, error_msg = check_model_dataset_compatibility(
                model_in_channels, dataset_name
            )

            if not is_compatible:
                self.push_screen(ErrorModal(error_msg, "Incompatible Dataset"))
                return

            self.notify("Loading model for inference...")
            self.notify("Running inference...", timeout=1)

            # Get image
            img = image_data["image"].unsqueeze(0).to(self.device)
            true_label = image_data["label"]

            # Predict
            model.eval()
            with torch.no_grad():
                logits = model(img)
                probs = torch.softmax(logits, dim=1)
                pred_label = logits.argmax(dim=1).item()

            # Format results
            results = f"[bold cyan]Single Image Classification[/bold cyan]\n\n"
            results += f"[bold]Dataset:[/bold] {dataset_name}\n"
            results += f"[bold]Image:[/bold] #{image_data['index'] + 1}\n\n"
            results += f"[bold]True Label:[/bold] {true_label}\n"
            results += f"[bold]Predicted:[/bold] {pred_label}\n\n"
            results += f"[bold]Probabilities:[/bold]\n"
            for i, prob in enumerate(probs[0]):
                results += f"  Class {i}: {prob.item():.4f}\n"

            if pred_label == true_label:
                results += "\n[bold green]✓ Correct prediction![/bold green]"
            else:
                results += "\n[bold red]✗ Incorrect prediction[/bold red]"

            self.push_screen(ResultsModal(results))

        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")


def main():
    app = DAQCNNTestUI()
    app.run()


if __name__ == "__main__":
    main()
