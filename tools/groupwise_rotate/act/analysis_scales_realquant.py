"""
Analyze quantization scales for activation values.
"""

import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from natsort import natsorted
import re

from utils import HadamardTransform

import matplotlib.pyplot as plt
import gc


def get_scales(act_root, save_root, layer_name):
    """Test different quantization and rotation combinations on activations."""
    print(f"Loading activations from {act_root}")

    # Only use NVFP4 quantizer
    from llmc.compression.quantization.quant_nvfp4 import NVFP4Quantizer
    quantizer_nvfp4 = NVFP4Quantizer(
        bit=4, symmetric=True, granularity="per_group", group_size=16
    )
    quantizers = {"NVFP4": quantizer_nvfp4}

    rotater_group = HadamardTransform(group_size=16)

    act_dir = Path(act_root) / layer_name

    # Collect data from all activation files
    all_scales_data = collect_all_scales_data(act_dir, quantizers, rotater_group)

    # Generate overall comparison plot
    save_fig_path = save_root / "scales_realquant" / "all_blocks_scales_comparison.png"
    plot_scales_vs_blocks(all_scales_data, save_fig_path)


def collect_all_scales_data(act_dir, quantizers, rotater_group):
    """Collect scale data from all activation files."""
    all_scales_data = defaultdict(lambda: defaultdict(list))

    for act_file in tqdm(
        natsorted(act_dir.glob("*_input.pt")), desc="Processing activation files"
    ):
        # Extract block index from filename
        match = re.search(r'model_layers_(\d+)_', act_file.name)
        if not match:
            print(f"Warning: Could not extract block index from {act_file.name}")
            continue
        block_idx = int(match.group(1))

        activation = torch.load(act_file, map_location="cuda").float()

        # Get the last dimension size (hidden size)
        hidden_size = activation.shape[-1]
        rotater_channel = HadamardTransform(group_size=hidden_size)

        for quantizer_name, quantizer in quantizers.items():
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

                    quantized_act, global_scale, local_scales = quantizer.real_quant_act_dynamic(
                        test_act, args={}
                    )

                    if act_type == "Original":
                        act_condition = "Original"
                    else:
                        act_condition = rotater_name

                    # Calculate activation statistics: test_act.reshape(-1, 16).abs().max(-1)
                    act_stats = test_act.reshape(-1, 16).abs().max(-1).values.float()
                    act_max = torch.max(act_stats).item()
                    act_min = torch.min(act_stats).item()
                    act_mean = torch.mean(act_stats).item()

                    # Store data for this block and condition
                    all_scales_data[block_idx][act_condition] = {
                        "global_scale": global_scale.item(),
                        "local_scales": local_scales,
                        "act_max": act_max,
                        "act_min": act_min,
                        "act_mean": act_mean
                    }

        del activation
        torch.cuda.empty_cache()
        gc.collect()

    return all_scales_data


def plot_scales_vs_blocks(all_scales_data, save_fig_path):
    """Create 3x3 line plot showing scales vs block index.

    9 subplots: Global Scale, Local Scale Mean, Local Scale Min,
               Local/Global Mean, Local/Global Min, Scale/Activation Ratio,
               Activation Max (per 16), Activation Min (per 16), Activation Mean (per 16)
    Each subplot shows 3 lines: Original, Group, Channel
    """

    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = ["Original", "Group", "Channel"]
    colors = ["C0", "C1", "C2"]
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    # Get sorted block indices
    block_indices = sorted(all_scales_data.keys())

    # Create figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    fig.suptitle(
        "Scale Analysis Across Blocks",
        fontsize=18,
        fontweight="bold",
    )

    # Define subplot configurations
    subplot_configs = [
        ("Global Scale", lambda data: data["global_scale"]),
        ("Local Scale Mean", lambda data: torch.mean(data["local_scales"].float()).item()),
        ("Local Scale Min", lambda data: torch.min(data["local_scales"].float()).item()),
        ("Local/Global Mean", lambda data: torch.mean(data["local_scales"].float()).item() / data["global_scale"]),
        ("Local/Global Min", lambda data: torch.min(data["local_scales"].float()).item() / data["global_scale"]),
        ("Scale/Activation Ratio", lambda data: data["global_scale"] / data["act_mean"]),
        ("Activation Max (per 16)", lambda data: data["act_max"]),
        ("Activation Min (per 16)", lambda data: data["act_min"]),
        ("Activation Mean (per 16)", lambda data: data["act_mean"]),
    ]

    # Plot each subplot
    for idx, (title, value_func) in enumerate(subplot_configs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        for j, act_type in enumerate(act_types):
            values = []
            valid_blocks = []
            for block_idx in block_indices:
                if act_type in all_scales_data[block_idx]:
                    data = all_scales_data[block_idx][act_type]
                    val = value_func(data)
                    values.append(val)
                    valid_blocks.append(block_idx)

            if values:
                ax.plot(
                    valid_blocks,
                    values,
                    color=colors[j],
                    marker=markers[j],
                    linestyle=linestyles[j],
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                    label=act_type,
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Block Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Generated scales vs blocks plot: {save_fig_path}")


if __name__ == "__main__":
    act_root = Path("figs/group_rotate/act/picked")
    save_root = Path("figs/group_rotate/act")
    layer_name = "down_proj"

    get_scales(act_root, save_root, layer_name)

    # Per-file plots are generated on-the-fly in process_activation_file()
    print("Scale analysis completed!")
