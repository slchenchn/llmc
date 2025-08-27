from pathlib import Path
from safetensors.torch import load_file
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_state_dict(path: Path) -> Dict:
    """Load state dict from safetensors, bin file, or a directory of safetensors."""
    state_dict = {}
    if path.is_dir():
        for f in path.glob("*.safetensors"):
            state_dict.update(load_file(f, device="cpu"))
    elif path.name.endswith(".safetensors"):
        state_dict = load_file(path, device="cpu")
    elif path.name.endswith(".bin"):
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file type or path: {path}")

    # Filter out embedding and head weights, and non-2D weights
    filtered_state_dict = {
        name: weight
        for name, weight in state_dict.items()
        if "lm_head" not in name
        and "embed" not in name
        and "buf_" not in name
        and weight.dim() == 2
    }
    return filtered_state_dict


def extract_layer_num(key: str) -> int:
    """Extract layer number from weight key."""
    patterns = [
        r"layer\.(\d+)",
        r"layers\.(\d+)",
        r"h\.(\d+)",
        r"transformer\.h\.(\d+)",
        r"model\.layers\.(\d+)",
        r"decoder\.layers\.(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, key)
        if match:
            return int(match.group(1))

    return -1  # Non-layer weights


def process_weight_differences(
    method_state_dict: Dict, nearest_state_dict: Dict, method_name: str
) -> Tuple[np.ndarray, Dict, Dict]:
    """Calculate weight differences and group by layer."""
    all_diffs = []
    layer_diffs = defaultdict(list)
    layer_weights = defaultdict(
        list
    )  # Store original weights for relative calculations

    for key, method_weight in method_state_dict.items():
        nearest_weight = nearest_state_dict[key]

        # Calculate differences
        diff = method_weight.float() - nearest_weight.float()
        all_diffs.append(diff.flatten())

        # Group by layer
        layer_num = extract_layer_num(key)
        if layer_num >= 0:
            layer_diffs[layer_num].append(diff.flatten())
            layer_weights[layer_num].append(
                nearest_weight.float().flatten()
            )  # Store reference weights

    # Concatenate all differences
    all_diffs_array = torch.cat(all_diffs).numpy() if all_diffs else np.array([])

    return all_diffs_array, layer_diffs, layer_weights


def calculate_layer_statistics(layer_diffs: Dict, layer_weights: Dict) -> Dict:
    """Calculate statistics for each layer."""
    layer_stats = {}

    for layer_num in sorted(layer_diffs.keys()):
        layer_diff = torch.cat(layer_diffs[layer_num]).numpy()
        layer_weight = torch.cat(layer_weights[layer_num]).numpy()

        # Calculate relative differences
        epsilon = 1e-5
        rel_diff = np.abs(layer_diff) / (np.abs(layer_weight) + epsilon)

        layer_stats[layer_num] = {
            "mean": np.mean(layer_diff),
            "abs_mean": np.mean(np.abs(layer_diff)),
            "abs_max": np.max(np.abs(layer_diff)),
            "rel_mean": np.mean(rel_diff),
            "std": np.std(layer_diff),
            "count": len(layer_diff),
        }

    return layer_stats


def print_overall_statistics(method_name: str, all_diffs: np.ndarray):
    """Print overall statistics for a method."""
    if len(all_diffs) == 0:
        print(f"\n=== {method_name} - No data available ===")
        return

    abs_mean_diff = np.mean(np.abs(all_diffs))
    print(f"\n=== {method_name} - Nearest Statistics ===")
    print(f"Mean: {np.mean(all_diffs):.6f}")
    print(f"Abs mean: {abs_mean_diff:.6f}")
    print(f"Standard deviation: {np.std(all_diffs):.6f}")
    print(f"Min: {np.min(all_diffs):.6f}")
    print(f"Max: {np.max(all_diffs):.6f}")
    print(f"Total number of weights: {len(all_diffs)}")


def create_histogram_subplot(ax, tesseraq_diffs: np.ndarray):
    """Create histogram comparison subplot."""
    ax.hist(
        tesseraq_diffs,
        bins=100,
        alpha=0.6,
        edgecolor="black",
        label="TesseraQ - Nearest",
        color="red",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Weight Difference")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of TesseraQ - Nearest Differences")
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_layer_comparison_subplot(
    ax, layers: List[int], tesseraq_values: List[float], ylabel: str, title: str
):
    """Create layer-wise comparison subplot."""
    ax.plot(
        layers,
        tesseraq_values,
        "r-s",
        linewidth=2,
        markersize=4,
        label="TesseraQ - Nearest",
    )
    ax.set_xlabel("Layer Number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if "Mean" in ylabel and "Absolute" not in ylabel:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)


def create_comparison_plots(
    tesseraq_diffs: np.ndarray, tesseraq_stats: Dict, save_fig_dir: Path
):
    """Create all comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get common layers
    layers = sorted(tesseraq_stats.keys()) if tesseraq_stats else []

    # Subplot 1: Histogram comparison
    create_histogram_subplot(axes[0, 0], tesseraq_diffs)

    if layers:
        # Extract values for each layer
        tesseraq_means = [tesseraq_stats[layer]["mean"] for layer in layers]
        tesseraq_max_diffs = [tesseraq_stats[layer]["abs_max"] for layer in layers]
        tesseraq_abs_means = [tesseraq_stats[layer]["abs_mean"] for layer in layers]
        tesseraq_rel_means = [tesseraq_stats[layer]["rel_mean"] for layer in layers]

        # create_layer_comparison_subplot(
        #     axes[0, 0],
        #     layers,
        #     tesseraq_max_diffs,
        #     "Max Absolute Difference",
        #     "Max Absolute Difference per Layer",
        # )

        create_layer_comparison_subplot(
            axes[0, 1],
            layers,
            tesseraq_means,
            "Mean Difference",
            "Mean Difference per Layer",
        )

        create_layer_comparison_subplot(
            axes[1, 0],
            layers,
            tesseraq_rel_means,
            "Relative Mean Difference",
            "Relative Mean Difference per Layer",
        )

        create_layer_comparison_subplot(
            axes[1, 1],
            layers,
            tesseraq_abs_means,
            "Absolute Mean Difference",
            "Absolute Mean Difference per Layer",
        )

    plt.tight_layout()
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_fig_dir / "tesseraq_nearest_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main function to run the comparison analysis."""
    # File paths
    nearest_path = Path("checkpoints/Qwen3-1.7B/rtn/sym_w4g128/fake_quant_model")
    tesseraq_path = Path(
        "checkpoints/Qwen3-1.7B/tesseraq/awq_tesseraq_w4a16g128_sym_s10_dynamic/fake_quant_model"
    )
    save_fig_dir = Path("figs/tesseraq/compare_nearest_tesseraq")

    # Load state dictionaries
    print("Loading state dictionaries...")
    tesseraq_state_dict = load_state_dict(tesseraq_path)
    nearest_state_dict = load_state_dict(nearest_path)

    # Process differences for TesseraQ
    print("Processing TesseraQ differences...")
    tesseraq_all_diffs, tesseraq_layer_diffs, tesseraq_layer_weights = (
        process_weight_differences(tesseraq_state_dict, nearest_state_dict, "TesseraQ")
    )

    # Calculate layer statistics
    print("Calculating statistics...")
    tesseraq_layer_stats = calculate_layer_statistics(
        tesseraq_layer_diffs, tesseraq_layer_weights
    )

    # Print verification info
    print(f"Found {len(tesseraq_layer_stats)} TesseraQ layers")
    print("First few layer keys for verification:")
    sample_keys = list(tesseraq_state_dict.keys())[:10]
    for key in sample_keys:
        layer_num = extract_layer_num(key)
        print(f"  {key} -> Layer {layer_num}")

    # Print statistics
    print_overall_statistics("TesseraQ", tesseraq_all_diffs)

    # Create plots
    print("Creating comparison plots...")
    create_comparison_plots(tesseraq_all_diffs, tesseraq_layer_stats, save_fig_dir)

    print("Analysis complete! Check 'tesseraq_nearest_comparison.png' for results.")


if __name__ == "__main__":
    main()
