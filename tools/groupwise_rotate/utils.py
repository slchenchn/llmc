"""
Common utilities for quantization analysis.
"""

import sys

sys.path.append("/nfs/FM/chenshuailin/code/llmc")
from safetensors.torch import load_file
import math
import torch
from llmc.compression.quantization.hadamard_utils import random_hadamard_matrix
from llmc.compression.quantization.quant_nvfp4 import NVFP4Quantizer
from llmc.compression.quantization.quant import IntegerQuantizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def load_state_dict(model_dir):
    """Load state dict from safetensors files."""
    state_dict = {}
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor))
    return state_dict


def is_power_of_2(num):
    """Check if a number is a power of 2."""
    return num > 0 and (num & (num - 1)) == 0


class HadamardTransform:
    """Hadamard transform for quantization analysis."""

    def __init__(self, group_size: int = 128):
        self.group_size = group_size
        self.scale = 1 / math.sqrt(self.group_size)

    def __call__(self, x: torch.Tensor):
        # Hadamard transform is its own inverse
        x_shape = x.shape
        # Apply Hadamard transform to the last dimension
        H = random_hadamard_matrix(self.group_size, x.device).to(x.dtype)
        return torch.matmul(x.view(-1, self.group_size), H).view(x_shape)


def get_quantizers():
    """Get the three quantizers used in analysis."""
    quantizer_fp4 = NVFP4Quantizer(
        bit=4, symmetric=True, granularity="per_group", group_size=16
    )
    quantizer_int4 = IntegerQuantizer(bit=4, symmetric=True, granularity="per_channel")
    quantizer_int4_group16 = IntegerQuantizer(
        bit=4, symmetric=True, granularity="per_group", group_size=16
    )
    return {
        "NVFP4": quantizer_fp4,
        "INT4": quantizer_int4,
        "INT4_G16": quantizer_int4_group16,
    }


def get_quantizers_for_activations():
    """Get quantizers that support activation quantization."""
    quantizer_fp4 = NVFP4Quantizer(
        bit=4, symmetric=True, granularity="per_group", group_size=16
    )
    quantizer_int4 = IntegerQuantizer(bit=4, symmetric=True, granularity="per_channel")
    quantizer_int4_group16 = IntegerQuantizer(
        bit=4, symmetric=True, granularity="per_group", group_size=16
    )
    return {
        "NVFP4": quantizer_fp4,
        "INT4": quantizer_int4,
        "INT4_G16": quantizer_int4_group16,
    }


def plot_quantization_errors(
    error_data, layer_info, save_dir, title_prefix="Quantization"
):
    """Plot quantization errors for different conditions."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for easier plotting
    df_data = []
    for condition, errors in error_data.items():
        for layer_idx, (layer, error) in enumerate(
            zip(layer_info[: len(errors)], errors)
        ):
            parts = condition.split("_")

            # Handle different quantizer formats
            if condition.startswith("NVFP4"):
                quantizer = "NVFP4"
                remaining_parts = parts[1:]
            elif condition.startswith("INT4_G16"):
                quantizer = "INT4_G16"
                remaining_parts = parts[2:]
            else:
                quantizer = "INT4"
                remaining_parts = parts[1:]

            if len(remaining_parts) == 2:
                # quantizer_rotater_weight_type format
                rotater, weight_type = remaining_parts[0], remaining_parts[1]
            else:
                # quantizer_weight_type or quantizer_rotater format
                if "Original" in condition:
                    rotater = "Group"  # Default for Original
                    weight_type = remaining_parts[0]
                else:
                    rotater = remaining_parts[0]
                    weight_type = "Rotated"

            df_data.append(
                {
                    "Layer": layer_idx,
                    "Layer_Name": layer,
                    "Quantizer": quantizer,
                    "Rotater": rotater,
                    "Weight_Type": weight_type,
                    "Error": error,
                    "Condition": condition,
                }
            )

    df = pd.DataFrame(df_data)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Get unique quantizers
    quantizers = df["Quantizer"].unique()
    print(f"\nFound quantizers: {list(quantizers)}")

    # 1. Create separate box plots for each quantizer
    for quantizer in quantizers:
        quantizer_df = df[df["Quantizer"] == quantizer]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=quantizer_df, x="Condition", y="Error", ax=ax)
        ax.set_yscale("log")
        ax.set_title(f"{title_prefix} Error Distribution - {quantizer}", fontsize=14)
        ax.set_xlabel("Condition (Rotater_Weight_Type)", fontsize=12)
        ax.set_ylabel("MSE Error (log scale)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"error_distribution_boxplot_{quantizer.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 2. Create separate violin plots for each quantizer
    for quantizer in quantizers:
        quantizer_df = df[df["Quantizer"] == quantizer]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=quantizer_df, x="Condition", y="Error", ax=ax)
        ax.set_yscale("log")
        ax.set_title(
            f"{title_prefix} Error Distribution (Violin) - {quantizer}", fontsize=14
        )
        ax.set_xlabel("Condition (Rotater_Weight_Type)", fontsize=12)
        ax.set_ylabel("MSE Error (log scale)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            save_dir / f"error_distribution_violin_{quantizer.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 3. Create separate line plots for each quantizer (all conditions in one plot with different colors)
    for quantizer in quantizers:
        quantizer_df = df[df["Quantizer"] == quantizer]
        quantizer_conditions = quantizer_df["Condition"].unique()

        fig, ax = plt.subplots(figsize=(15, 8))

        # Use a more distinctive color palette
        if len(quantizer_conditions) <= 8:
            # For 8 or fewer conditions, use Set1 (very distinctive colors)
            colors = plt.cm.Set1(np.linspace(0, 1, len(quantizer_conditions)))
        elif len(quantizer_conditions) <= 12:
            # For 9-12 conditions, use tab20
            colors = plt.cm.tab20(np.linspace(0, 1, len(quantizer_conditions)))
        else:
            # For more conditions, use a custom high-contrast palette
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
                "#aec7e8",
                "#ffbb78",
                "#98df8a",
                "#ff9896",
                "#c5b0d5",
            ]

        for i, condition in enumerate(quantizer_conditions):
            condition_data = quantizer_df[quantizer_df["Condition"] == condition]
            ax.plot(
                condition_data["Layer"],
                condition_data["Error"],
                "o-",
                linewidth=2,
                markersize=4,
                color=colors[i % len(colors)],
                label=condition,
                alpha=0.8,
            )

        ax.set_yscale("log")
        ax.set_title(
            f"{quantizer} {title_prefix} Errors Across Layers",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("MSE Error (log scale)", fontsize=12)

        # Create denser grid
        ax.grid(True, which="major", alpha=0.5, linewidth=1)
        ax.grid(True, which="minor", alpha=0.3, linewidth=0.5)

        # Set minor locators for denser grid
        ax.xaxis.set_minor_locator(
            plt.MultipleLocator(5)
        )  # minor ticks every 5 units on x
        ax.yaxis.set_minor_locator(plt.LogLocator(subs="all"))

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            save_dir / f"error_across_layers_{quantizer.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Create separate comparison plots for each quantizer
    rotaters = ["Group", "Channel"]

    for quantizer in quantizers:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, rotater in enumerate(rotaters):
            subset = df[(df["Quantizer"] == quantizer) & (df["Rotater"] == rotater)]
            if len(subset) > 0:
                sns.boxplot(data=subset, x="Weight_Type", y="Error", ax=axes[i])
                axes[i].set_yscale("log")
                axes[i].set_title(f"{quantizer} + {rotater} Rotation", fontsize=12)
                axes[i].set_xlabel("Weight Type", fontsize=10)
                axes[i].set_ylabel("MSE Error (log scale)", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            save_dir / f"quantizer_rotater_comparison_{quantizer.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 5. Statistical summary table
    summary_stats = (
        df.groupby("Condition")["Error"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(6)
    )

    # Print summary in separate markdown tables for each quantizer
    print(f"\n## {title_prefix} Error Summary")
    print("")

    for quantizer in quantizers:
        print(f"\n### {quantizer}")
        print("")

        # Filter data for this quantizer
        quantizer_mask = df["Quantizer"] == quantizer
        quantizer_df_summary = df[quantizer_mask]
        quantizer_summary_stats = (
            quantizer_df_summary.groupby("Condition")["Error"]
            .agg(["mean", "median", "std", "min", "max"])
            .round(9)
        )

        print("| Condition | Mean | Median | Std | Min | Max |")
        print("|-----------|------|--------|-----|-----|-----|")

        for condition in quantizer_summary_stats.index:
            stats = quantizer_summary_stats.loc[condition]
            print(
                f"| {condition} | {stats['mean']:.4e} | {stats['median']:.4e} | {stats['std']:.4e} | {stats['min']:.4e} | {stats['max']:.4e} |"
            )

        print("")  # Add empty line after each table

    print(f"\nPlots and data saved to: {save_dir}")
    print(f"\nGenerated separate plots for each quantizer:")
    for quantizer in quantizers:
        print(
            f"  - {quantizer}: boxplot, violin plot, rotater comparison, and error across layers"
        )
