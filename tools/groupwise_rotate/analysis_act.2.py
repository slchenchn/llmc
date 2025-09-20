"""
Analyze quantization errors for activation values.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

try:
    from .utils import (
        get_quantizers_for_activations,
        HadamardTransform,
    )
except ImportError:
    from utils import (
        get_quantizers_for_activations,
        HadamardTransform,
    )


def load_activations(act_dir):
    """Load activation values from pickle files."""
    activations = {}
    act_dir = Path(act_dir)

    for act_file in tqdm(act_dir.glob("*_input.pt"), desc="Loading activations"):
        # Extract layer name from filename
        # e.g., "model_layers_0_mlp_down_proj_input.pkl" -> "model.layers.0.mlp.down_proj"
        filename = act_file.stem  # remove .pt extension
        layer_name = filename.replace("_input", "").replace("_", ".")

        act = torch.load(act_file)
        activations[layer_name] = act

    return activations


def process_activation_file(act_file, save_dir_mean, save_dir_kurtosis, rotater_group):
    """Process a single activation file and generate histograms."""
    # Extract layer name from filename
    # e.g., "model_layers_0_mlp_down_proj_input.pkl" -> "model.layers.0.mlp.down_proj"
    filename = act_file.stem  # remove .pt extension
    layer_name = filename.replace("_input", "").replace("_", ".")

    # Load activation
    activation = torch.load(act_file)

    # Move to GPU
    activation = activation.cuda()

    # Get the last dimension size (hidden size)
    hidden_size = activation.shape[-1]

    # For activations, we use group size 16 regardless of hidden size
    # Skip if too small
    if hidden_size < 16:
        print(f"Skipping {layer_name}: hidden_size {hidden_size} < 16")
        return layer_name

    rotater_channel = HadamardTransform(group_size=hidden_size)

    ori_act = activation
    rot_act_group = rotater_group(activation)
    rot_act_channel = rotater_channel(activation)

    # Define conditions: (activation, name_suffix, color)
    conditions = [
        (ori_act, "Original", "#1f77b4"),
        (rot_act_group, "Group_Rotated", "#ff7f0e"),
        (rot_act_channel, "Channel_Rotated", "#2ca02c"),
    ]

    # Create figure with 3 subplots for mean histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Layer: {layer_name}", fontsize=16, fontweight="bold")

    for i, (act, act_name, color) in enumerate(conditions):
        act_reshaped = act.view(-1, 16)

        # Calculate statistics for each group of 16 values
        group_means = act_reshaped.mean(dim=-1)
        group_stds = act_reshaped.std(dim=-1)

        # Calculate kurtosis (4th moment / variance^2 - 3)
        # kurtosis = mean((x - mean)^4) / std^4 - 3
        means = act_reshaped.mean(dim=-1, keepdim=True)
        stds = (
            act_reshaped.std(dim=-1, keepdim=True) + 1e-8
        )  # Add small epsilon to avoid division by zero
        centered = act_reshaped - means
        fourth_moment = (centered**4).mean(dim=-1)
        group_kurtosis = fourth_moment / (stds.squeeze(-1) ** 4) - 3

        # Convert to CPU for plotting and get numpy array
        group_means_np = group_means.cpu().float().numpy()
        group_stds_np = group_stds.cpu().float().numpy()
        group_kurtosis_np = group_kurtosis.cpu().float().numpy()

        # Plot mean histogram
        axes[i].hist(
            group_means_np,
            bins=200,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[i].set_yscale("log")
        axes[i].set_title(f"{act_name}", fontsize=14)
        axes[i].set_xlabel("Mean Value", fontsize=12)
        axes[i].set_ylabel("Frequency (log scale)", fontsize=12)
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        mean_val = group_means_np.mean()
        std_val = group_means_np.std()
        kurtosis_val = group_kurtosis_np.mean()
        stats_text = ".4f"
        axes[i].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        save_dir_mean
        / f"layer_{layer_name.replace('.', '_').replace('/', '_')}_mean_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create figure for kurtosis histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Layer: {layer_name} - Kurtosis", fontsize=16, fontweight="bold")

    for i, (act, act_name, color) in enumerate(conditions):
        act_reshaped = act.view(-1, 16)

        # Calculate kurtosis
        means = act_reshaped.mean(dim=-1, keepdim=True)
        stds = act_reshaped.std(dim=-1, keepdim=True) + 1e-8
        centered = act_reshaped - means
        fourth_moment = (centered**4).mean(dim=-1)
        group_kurtosis = fourth_moment / (stds.squeeze(-1) ** 4) - 3

        group_kurtosis_np = group_kurtosis.cpu().float().numpy()

        # Plot kurtosis histogram
        axes[i].hist(
            group_kurtosis_np,
            bins=200,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[i].set_yscale("log")
        axes[i].set_title(f"{act_name}", fontsize=14)
        axes[i].set_xlabel("Kurtosis Value", fontsize=12)
        axes[i].set_ylabel("Frequency (log scale)", fontsize=12)
        axes[i].grid(True, alpha=0.3)

        # Add kurtosis statistics text
        kurtosis_mean = group_kurtosis_np.mean()
        kurtosis_std = group_kurtosis_np.std()
        stats_text = ".4f"
        axes[i].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        save_dir_kurtosis
        / f"layer_{layer_name.replace('.', '_').replace('/', '_')}_kurtosis_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Clean up GPU memory
    del ori_act, rot_act_group, rot_act_channel, activation
    torch.cuda.empty_cache()

    return layer_name


if __name__ == "__main__":
    act_dir = Path("figs/group_rotate/act/picked")
    layer_suffix = "q_proj"
    save_dir = Path("figs/group_rotate/act/histograms_v3")

    act_dir = act_dir / layer_suffix
    save_dir = save_dir / layer_suffix

    # Create separate directories for mean and kurtosis histograms
    save_dir_mean = save_dir / "mean"
    save_dir_kurtosis = save_dir / "kurtosis"
    save_dir_mean.mkdir(parents=True, exist_ok=True)
    save_dir_kurtosis.mkdir(parents=True, exist_ok=True)

    # Initialize Hadamard transform (shared across files)
    rotater_group = HadamardTransform(group_size=16)

    layer_info = []

    # Process each activation file on the fly
    for act_file in tqdm(
        act_dir.glob("*_input.pt"), desc="Processing activation files"
    ):
        try:
            layer_name = process_activation_file(act_file, save_dir_mean, save_dir_kurtosis, rotater_group)
            layer_info.append(layer_name)
            print(f"✓ Processed: {layer_name}")
        except Exception as e:
            print(f"✗ Failed to process {act_file.name}: {e}")
            continue

    print(
        f"\nActivation analysis completed! Successfully processed {len(layer_info)} layers."
    )
    print(f"Mean histogram plots saved to: {save_dir_mean}")
    print(f"Kurtosis histogram plots saved to: {save_dir_kurtosis}")

    # Print summary of processed layers
    print(f"\nProcessed layers: {layer_info}")
