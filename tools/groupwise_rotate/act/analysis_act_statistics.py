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
from natsort import natsorted
import pandas as pd
from scipy.stats import kurtosis

from utils import HadamardTransform


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


def process_activation_file(act_file, save_dir_mean, save_dir_kurtosis, save_dir_absmax, rotater_group):
    """Process a single activation file and generate histograms."""
    # Extract layer name from filename
    # e.g., "model_layers_0_mlp_down_proj_input.pkl" -> "model.layers.0.mlp.down_proj"
    filename = act_file.stem  # remove .pt extension
    layer_name = filename.replace("_input", "").replace("_", ".")

    # Load activation
    activation = torch.load(act_file).cuda()

    # Get the last dimension size (hidden size)
    hidden_size = activation.shape[-1]

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

    # # Create figure with 3 subplots for mean histograms
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle(f"Layer: {layer_name}", fontsize=16, fontweight="bold")

    # for i, (act, act_name, color) in enumerate(conditions):
    #     act_reshaped = act.view(-1, 16)

    #     # Calculate statistics for each group of 16 values
    #     group_means = act_reshaped.mean(dim=-1).cpu().float().numpy()
    #     axes[i].hist(
    #         group_means,
    #         bins=200,
    #         alpha=0.7,
    #         color=color,
    #         edgecolor="black",
    #         linewidth=0.5,
    #     )

    #     # Calculate statistics
    #     mean_val = group_means.mean()
    #     q85 = np.percentile(group_means, 85)
    #     q80 = np.percentile(group_means, 80)
        
    #     # Add text with statistics
    #     stats_text = f"Mean: {mean_val:.4f}\n85%: {q85:.4f}\n80%: {q80:.4f}"
    #     axes[i].text(
    #         0.95, 0.95, stats_text,
    #         transform=axes[i].transAxes,
    #         fontsize=10,
    #         verticalalignment='top',
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    #     )

    #     axes[i].set_yscale("log")
    #     axes[i].set_title(f"{act_name}", fontsize=14)
    #     axes[i].set_xlabel("Mean Value", fontsize=12)
    #     axes[i].set_ylabel("Frequency (log scale)", fontsize=12)
    #     axes[i].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(
    #     save_dir_mean
    #     / f"layer_{layer_name.replace('.', '_').replace('/', '_')}_mean_histogram.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # # Create figure for kurtosis histograms
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig.suptitle(f"Layer: {layer_name} - Kurtosis", fontsize=16, fontweight="bold")

    # for i, (act, act_name, color) in enumerate(conditions):
    #     act_reshaped = act.view(-1, 16)

    #     # Calculate kurtosis using scipy (unbiased, Fisher=True, bias=False)
    #     group_kurtosis = kurtosis(
    #         act_reshaped.cpu().float().numpy(), axis=-1, fisher=True, bias=False
    #     )

    #     axes[i].hist(
    #         group_kurtosis,
    #         bins=200,
    #         alpha=0.7,
    #         color=color,
    #         edgecolor="black",
    #         linewidth=0.5,
    #     )

    #     # Calculate statistics
    #     mean_kurt = np.nanmean(group_kurtosis)
    #     q85 = np.nanpercentile(group_kurtosis, 85)
    #     q80 = np.nanpercentile(group_kurtosis, 80)
        
    #     # Add text with statistics
    #     stats_text = f"Mean: {mean_kurt:.4f}\n85%: {q85:.4f}\n80%: {q80:.4f}"
    #     axes[i].text(
    #         0.95, 0.95, stats_text,
    #         transform=axes[i].transAxes,
    #         fontsize=10,
    #         verticalalignment='top',
    #         horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    #     )

    #     axes[i].set_yscale("log")
    #     axes[i].set_title(f"{act_name}", fontsize=14)
    #     axes[i].set_xlabel("Kurtosis Value", fontsize=12)
    #     axes[i].set_ylabel("Frequency (log scale)", fontsize=12)
    #     axes[i].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(
    #     save_dir_kurtosis
    #     / f"layer_{layer_name.replace('.', '_').replace('/', '_')}_kurtosis_histogram.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.close()

    # Create figure for absmax histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Layer: {layer_name} - Absmax", fontsize=16, fontweight="bold")

    for i, (act, act_name, color) in enumerate(conditions):
        act_reshaped = act.view(-1, 16)

        # Calculate absmax (absolute maximum) for each group of 16 values
        group_absmax = act_reshaped.abs().max(dim=-1).values.cpu().float().numpy()

        # Create log-spaced bins for equal width on log scale
        log_bins = np.logspace(np.log10(group_absmax.min()), np.log10(group_absmax.max()), 200)
        
        axes[i].hist(
            group_absmax,
            bins=log_bins,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

        # Calculate statistics
        mean_absmax = group_absmax.mean()
        q85 = np.percentile(group_absmax, 85)
        q80 = np.percentile(group_absmax, 80)
        
        # Add text with statistics
        stats_text = f"Mean: {mean_absmax:.4f}\n85%: {q85:.4f}\n80%: {q80:.4f}"
        axes[i].text(
            0.95, 0.95, stats_text,
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].set_title(f"{act_name}", fontsize=14)
        axes[i].set_xlabel("Absmax Value (log scale)", fontsize=12)
        axes[i].set_ylabel("Frequency (log scale)", fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir_absmax
        / f"layer_{layer_name.replace('.', '_').replace('/', '_')}_absmax_histogram.png",
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
    layer_suffix = "down_proj"
    save_dir = Path("figs/group_rotate/act/histograms")

    act_dir = act_dir / layer_suffix
    save_dir = save_dir / layer_suffix

    # Create separate directories for mean, kurtosis, and absmax histograms
    save_dir_mean = save_dir / "mean"
    save_dir_kurtosis = save_dir / "kurtosis"
    save_dir_absmax = save_dir / "absmax"
    save_dir_mean.mkdir(parents=True, exist_ok=True)
    save_dir_kurtosis.mkdir(parents=True, exist_ok=True)
    save_dir_absmax.mkdir(parents=True, exist_ok=True)

    # Initialize Hadamard transform (shared across files)
    rotater_group = HadamardTransform(group_size=16)

    layer_info = []

    # Process each activation file on the fly
    for act_file in tqdm(
        natsorted(act_dir.glob("*_input.pt")), desc="Processing activation files"
    ):
        layer_name = process_activation_file(
            act_file, save_dir_mean, save_dir_kurtosis, save_dir_absmax, rotater_group
        )
        layer_info.append(layer_name)
        # print(f"✓ Processed: {layer_name}")

    print(
        f"\nActivation analysis completed! Successfully processed {len(layer_info)} layers."
    )
    print(f"Mean histogram plots saved to: {save_dir_mean}")
    print(f"Kurtosis histogram plots saved to: {save_dir_kurtosis}")
    print(f"Absmax histogram plots saved to: {save_dir_absmax}")
