"""
Analyze quantization errors for activation values.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted

from utils import get_quantizers, HadamardTransform

import matplotlib.pyplot as plt
import numpy as np


def process_activation_file(act_file, quantizers, rotater_group, save_root, layer_name):
    """Process a single activation file and generate its 3x3 plot."""
    # print(f"Processing: {act_file.name}")

    # # DEBUG:
    # if '_3_' not in act_file.stem:
    #     return

    activation = torch.load(act_file, map_location="cuda").float()
    # layer_idx = int(act_file.stem.split("_")[2])

    # Get the last dimension size (hidden size)
    hidden_size = activation.shape[-1]
    rotater_channel = HadamardTransform(group_size=hidden_size)

    for quantizer_name, quantizer in quantizers.items():
        quantization_errors = defaultdict(dict)  # Store errors for this specific file
        for rotater_name, rotater in [
            ("Group", rotater_group),
            ("Channel", rotater_channel),
        ]:
            rot_act = rotater(activation)

            # Test both original and rotated activations
            for act_type, test_act in [
                ("Original", activation),
                ("Rotated", rot_act),
            ]:
                # Skip Channel_Original to avoid duplication with Group_Original
                if rotater_name == "Channel" and act_type == "Original":
                    continue

                qdq = quantizer.fake_quant_act_dynamic(test_act, args={})
                err = F.mse_loss(qdq, test_act, reduction="none").abs().view(-1, 16)
                err += 1e-16    # avoid log scale issues

                # Calculate error statistics
                err_max = err.amax(-1)
                err_min = err.amin(-1)
                err_mean = err.mean(-1)

                if act_type == "Original":
                    act_condition = "Original"
                else:
                    act_condition = rotater_name

                quantization_errors[act_condition] = {
                    "max": err_max,
                    "min": err_min,
                    "mean": err_mean,
                    # "error_tensor": err.cpu(),
                }

        save_fig_path = (
            save_root / "per_file_plots" / f"{act_file.stem}_{quantizer_name}.png"
        )
        plot_single_file_3x3(quantization_errors, save_fig_path)


def get_activations(act_root, save_root, layer_name, ignore_qunatizers):
    """Test different quantization and rotation combinations on activations."""
    print(f"Loading activations from {act_root}")

    quantizers = get_quantizers(ignore_qunatizers)
    rotater_group = HadamardTransform(group_size=16)

    act_dir = Path(act_root) / layer_name

    for act_file in tqdm(
        natsorted(act_dir.glob("*_input.pt")), desc="Processing activation files"
    ):
        process_activation_file(
            act_file, quantizers, rotater_group, save_root, layer_name
        )
        # layer_info.append(layer_name_str)


def plot_single_file_3x3(file_errors, save_fig_path, nbins=500):
    """Create 3x3 grid plot for a single activation file."""

    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = ["Original", "Group", "Channel"]
    metrics = ["max", "min", "mean"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Quantization Errors - {save_fig_path.stem}", fontsize=16, fontweight="bold"
    )

    for j, act_type in enumerate(act_types):
        if act_type in file_errors:
            error_stats = file_errors[act_type]

            # Create histogram for each metric
            for k, metric in enumerate(metrics):
                ax = axes[j, k]  # rows are act_types, columns are metrics
                assert torch.all(error_stats[metric] > 0), "Error tensor contains non-positive values"
                error_tensor = error_stats[metric].cpu().flatten().numpy()

                # Calculate log-spaced bins for equal width on log x-scale
                log_min = np.log10(error_tensor.min())
                log_max = np.log10(error_tensor.max())
                bins = np.logspace(log_min, log_max, nbins)

                # Plot histogram
                ax.hist(
                    error_tensor,
                    bins=bins,
                    alpha=0.7,
                    color=f"C{j}",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.set_yscale("log")
                ax.set_title(f"{act_type} - {metric.upper()}", fontsize=10)
                ax.grid(True, alpha=0.3)

                # Add vertical line for mean value
                mean_val = np.mean(error_tensor)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
                ax.legend(loc='upper right', fontsize=8)

                # Use log scale for x-axis since values are always positive
                ax.set_xscale('log')

                # # Add the specific metric value as text
                # ax.text(
                #     0.7,
                #     0.95,
                #     # f"{metric.upper()}: {error_tensor.sum():.2e}",
                #     f"sum: {error_tensor.sum():.2e}",
                #     transform=ax.transAxes,
                #     verticalalignment="top",
                #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                #     fontsize=8,
                # )

        else:
            # Empty subplot if data not available
            for k in range(3):
                axes[j, k].text(
                    0.5,
                    0.5,
                    f"No data\n{act_type}",
                    ha="center",
                    va="center",
                    transform=axes[j, k].transAxes,
                )
                axes[j, k].set_title(f"{act_type} - {metrics[k].upper()}", fontsize=10)

    # Set column labels
    for k, metric in enumerate(metrics):
        axes[0, k].set_xlabel("")
        axes[2, k].set_xlabel(f"{metric.upper()} Error Distribution", fontsize=12)

    # Set row labels
    for j, act_type in enumerate(act_types):
        axes[j, 0].set_ylabel(f"{act_type}\nFrequency (log)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # print(f"✓ Generated 3x3 plot for {save_fig_path.stem}")


if __name__ == "__main__":
    act_root = Path("figs/group_rotate/act/picked")
    save_root = Path("figs/group_rotate/act")
    layer_name = "down_proj"
    ignore_qunatizers = ["INT4"]

    error_data, layer_info = get_activations(
        act_root, save_root, layer_name, ignore_qunatizers
    )

    # Per-file 3x3 plots are generated on-the-fly in process_activation_file()
    print("Activation analysis completed!")
