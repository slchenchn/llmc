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
import gc


def get_activations(act_root, save_root, layer_name, ignore_qunatizers):
    """Test different quantization and rotation combinations on activations."""
    print(f"Loading activations from {act_root}")

    quantizers = get_quantizers(ignore_qunatizers)
    rotater_group = HadamardTransform(group_size=16)

    act_dir = Path(act_root) / layer_name
    layer_info = []

    for act_file in tqdm(
        natsorted(act_dir.glob("*_input.pt")), desc="Processing activation files"
    ):
        process_activation_file(
            act_file, quantizers, rotater_group, save_root, layer_name
        )
        # layer_info.append(layer_name_str)


def process_activation_file(act_file, quantizers, rotater_group, save_root, layer_name):
    """Process a single activation file and generate its comparison plot."""
    # print(f"Processing: {act_file.name}")

    activation = torch.load(act_file, map_location="cuda").float()
    # layer_idx = int(act_file.stem.split("_")[2])

    # Get the last dimension size (hidden size)
    hidden_size = activation.shape[-1]
    rotater_channel = HadamardTransform(group_size=hidden_size)

    # Store results for all quantizers
    all_quantizers_errors = {}

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

                # qdq = quantizer.fake_quant_act_dynamic(test_act, args={})
                quantized_act, global_scale, local_scales = quantizer.real_quant_act_dynamic(
                    test_act, args={}
                )

                if act_type == "Original":
                    act_condition = "Original"
                else:
                    act_condition = rotater_name

                quantized_act = quantized_act
                quantization_errors[act_condition] = {"quantized": quantized_act}

        all_quantizers_errors[quantizer_name] = quantization_errors

    # # Generate bar plot with all quantizers
    # save_fig_path_bar = save_root / "realquant" / f"{act_file.stem}_comparison_bar.png"
    # plot_single_file_3x3(all_quantizers_errors, save_fig_path_bar)
    
    # Generate line plot with all quantizers
    save_fig_path_line = save_root / "realquant" / f"{act_file.stem}_comparison_line.png"
    plot_line_comparison(all_quantizers_errors, save_fig_path_line)
    
    del all_quantizers_errors
    del activation
    torch.cuda.empty_cache()
    gc.collect()


def plot_single_file_3x3(all_quantizers_errors, save_fig_path, nbins=100):
    """Create 2x3 grid plot comparing quantized activation histograms across quantizers.

    Layout: 2 rows (one per quantizer) x 3 columns (Original, Group, Channel)
    """

    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = ["Original", "Group", "Channel"]
    quantizer_names = list(all_quantizers_errors.keys())
    num_quantizers = len(quantizer_names)

    # Create figure with num_quantizers rows and 3 columns
    fig, axes = plt.subplots(num_quantizers, 3, figsize=(18, 6 * num_quantizers))

    # Handle case where there's only one quantizer (axes won't be 2D)
    if num_quantizers == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Quantized Activation Distribution - {save_fig_path.stem}",
        fontsize=18,
        fontweight="bold",
    )

    # Plot each quantizer in its own row
    for i, quantizer_name in enumerate(quantizer_names):
        file_errors = all_quantizers_errors[quantizer_name]

        for j, act_type in enumerate(act_types):
            ax = axes[i, j]

            if act_type in file_errors:
                quantized_act_tensor = file_errors[act_type]["quantized"]

                # Use torch.unique for faster histogram computation
                unique_vals_tensor, counts_tensor = torch.unique(
                    quantized_act_tensor.flatten(), return_counts=True
                )
                unique_vals_np = unique_vals_tensor.cpu().numpy()
                counts_np = counts_tensor.cpu().numpy()

                # Plot bar chart instead of histogram
                ax.bar(
                    unique_vals_np,
                    counts_np,
                    width=0.8 if quantizer_name == "NVFP4" else 0.7,
                    alpha=0.7,
                    color=f"C{j}",
                    edgecolor="black",
                    linewidth=0.5,
                )

                ax.set_yscale("log")
                ax.set_title(
                    f"{quantizer_name} - {act_type}",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.grid(True, which="major", alpha=0.4, axis="y", linewidth=1)
                ax.grid(True, which="minor", alpha=0.2, axis="y", linewidth=0.5)
                
                # Set denser y-axis ticks for log scale
                from matplotlib.ticker import LogLocator
                ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
                ax.minorticks_on()

                # Only add y-label to leftmost column
                if j == 0:
                    ax.set_ylabel("Frequency (log)", fontsize=11)

                # # Add statistics text box (compute on CPU for stats)
                # quantized_act_np = quantized_act_tensor.cpu().float().numpy()
                # mean_val = np.mean(quantized_act_np)
                # median_val = np.median(quantized_act_np)
                # std_val = np.std(quantized_act_np)
                # num_unique_vals = len(unique_vals_np)

                # stats_text = (
                #     f"Mean: {mean_val:.2f}\n"
                #     f"Median: {median_val:.2f}\n"
                #     f"Std: {std_val:.2f}\n"
                #     f"Unique: {num_unique_vals}"
                # )
                # ax.text(
                #     0.98,
                #     0.97,
                #     stats_text,
                #     transform=ax.transAxes,
                #     verticalalignment="top",
                #     horizontalalignment="right",
                #     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                #     fontsize=9,
                #     family="monospace",
                # )

                # # Add vertical line for mean value
                # ax.axvline(
                #     mean_val,
                #     color="red",
                #     linestyle="--",
                #     linewidth=2,
                #     alpha=0.8,
                # )

            else:
                # Empty subplot if data not available
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{act_type}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_title(
                    f"{quantizer_name} - {act_type}",
                    fontsize=12,
                    fontweight="bold",
                )

            # Set x-label only for bottom row
            if i == num_quantizers - 1:
                ax.set_xlabel("Quantized Integer Value", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # print(f"✓ Generated quantized activation histogram for {save_fig_path.stem}")


def plot_line_comparison(all_quantizers_errors, save_fig_path):
    """Create 1x2 line plot comparing quantized activation distributions.
    
    Layout: 1 row x 2 columns (one per quantizer)
    Each subplot shows 3 lines (Original, Group, Channel)
    """

    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = ["Original", "Group", "Channel"]
    quantizer_names = list(all_quantizers_errors.keys())
    num_quantizers = len(quantizer_names)
    
    # Create figure with 1 row and num_quantizers columns
    fig, axes = plt.subplots(1, num_quantizers, figsize=(10 * num_quantizers, 8))
    
    # Handle case where there's only one quantizer (axes won't be an array)
    if num_quantizers == 1:
        axes = [axes]
    
    fig.suptitle(
        f"Quantized Activation Distribution (Line) - {save_fig_path.stem}",
        fontsize=18,
        fontweight="bold",
    )

    # Color and marker for each activation type
    colors = ["C0", "C1", "C2"]
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    # Plot each quantizer in its own column
    for i, quantizer_name in enumerate(quantizer_names):
        file_errors = all_quantizers_errors[quantizer_name]
        ax = axes[i]
        
        for j, act_type in enumerate(act_types):
            if act_type in file_errors:
                quantized_act_tensor = file_errors[act_type]["quantized"]
                
                # Use torch.unique for faster histogram computation
                unique_vals_tensor, counts_tensor = torch.unique(
                    quantized_act_tensor.flatten(), return_counts=True
                )
                unique_vals_np = unique_vals_tensor.cpu().numpy()
                counts_np = counts_tensor.cpu().numpy()
                
                # Plot line
                ax.plot(
                    unique_vals_np,
                    counts_np,
                    color=colors[j],
                    marker=markers[j],
                    linestyle=linestyles[j],
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                    label=act_type,
                )

        ax.set_yscale("log")
        ax.set_title(
            f"{quantizer_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, which="major", alpha=0.4, linewidth=1)
        ax.grid(True, which="minor", alpha=0.2, linewidth=0.5)
        
        # Set denser y-axis ticks for log scale
        from matplotlib.ticker import LogLocator
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.minorticks_on()
        
        ax.set_xlabel("Quantized Value", fontsize=12)
        
        # Only add y-label to leftmost column
        if i == 0:
            ax.set_ylabel("Frequency (log)", fontsize=12)
        
        # Add legend
        ax.legend(loc="best", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # print(f"✓ Generated line comparison plot for {save_fig_path.stem}")


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
