"""
Combined analysis for NVFP4 quantization errors on activations with different group sizes,
covering both absolute (MSE-based) and relative error metrics.

Outputs:
- Per-file histograms for absolute and relative errors (max/min/mean over NVFP4 groups of 16)
- Violin plots comparing group sizes for absolute and relative errors
- Across-blocks plots for absolute and relative errors including mean, max, and min curves
"""

import re
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from tqdm import tqdm
from utils import HadamardTransform, NVFP4Quantizer


def extract_layer_idx(filename: str) -> int:
    """Extract layer index from activation filename.

    Supports patterns like: model_layers_{idx}_mlp_{name}_input.pt
    Fallback: first number in the filename.
    """
    match = re.search(r"model_layers_(\d+)_mlp_", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def create_quantizer_and_rotater(group_size, hidden_size: int | None = None):
    """Create NVFP4 quantizer and rotater for a given group size.

    NVFP4 quantization uses fixed group_size=16 with optional arbitrary grouping support.
    The rotater applies a Hadamard transform when group_size > 1.
    """
    quantizer_nvfp4 = NVFP4Quantizer(
        bit=4,
        symmetric=True,
        granularity="per_group",
        group_size=16,
        arbitrary_group_size=True,
    )
    quantizers = {"NVFP4": quantizer_nvfp4}

    if group_size == 1:
        rotater = None
    elif isinstance(group_size, str) and group_size == "channelwise":
        if hidden_size is None:
            raise ValueError("hidden_size must be provided for channelwise rotation")
        rotater = HadamardTransform(group_size=hidden_size)
    else:
        rotater = HadamardTransform(group_size=group_size)
    return quantizers, rotater


def compute_relative_error(qdq: torch.Tensor, x: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """Compute element-wise relative error abs(qdq - x) / (abs(x) + eps)."""
    return (qdq - x).abs() / (x.abs() + eps)


def load_all_errors_from_csv(csv_path: Path):
    """Load absolute and relative errors from combined CSV and reconstruct structures.

    Returns: (all_errors_abs, all_errors_rel, group_sizes, layer_indices_sorted)
    """
    from collections import defaultdict as _dd

    temp_abs = _dd(lambda: _dd(dict))  # group_size -> key -> {layer_idx: value}
    temp_rel = _dd(lambda: _dd(dict))
    layer_idx_set = set()
    group_sizes_found = set()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                layer_idx = int(row["layer_index"])  # stored as int in CSV
            except Exception:
                continue
            gs_raw = row["group_size"]
            group_size = "channelwise" if str(gs_raw) == "channelwise" else int(gs_raw)
            error_kind = row["error_kind"]  # "absolute" or "relative"
            key_prefix = row["type"]  # "Original" or "Group_Rotated"
            metric = row["metric"]  # "max" | "min" | "mean"
            value = float(row["value"]) if row["value"] != "" else 0.0

            key = f"{key_prefix}_{metric}"
            layer_idx_set.add(layer_idx)
            group_sizes_found.add(group_size)

            if error_kind == "absolute":
                temp_abs[group_size][key][layer_idx] = value
            elif error_kind == "relative":
                temp_rel[group_size][key][layer_idx] = value

    layer_indices_sorted = sorted(layer_idx_set)

    all_errors_abs = {}
    for gs, key_to_map in temp_abs.items():
        all_errors_abs[gs] = {}
        for key, li_to_val in key_to_map.items():
            all_errors_abs[gs][key] = [li_to_val.get(li, 0.0) for li in layer_indices_sorted]

    all_errors_rel = {}
    for gs, key_to_map in temp_rel.items():
        all_errors_rel[gs] = {}
        for key, li_to_val in key_to_map.items():
            all_errors_rel[gs][key] = [li_to_val.get(li, 0.0) for li in layer_indices_sorted]

    numeric_group_sizes = sorted([gs for gs in group_sizes_found if isinstance(gs, int)])
    group_sizes = numeric_group_sizes + (["channelwise"] if "channelwise" in group_sizes_found else [])

    return all_errors_abs, all_errors_rel, group_sizes, layer_indices_sorted


def plot_single_file_abs_errors(file_errors: dict, save_fig_path: Path, nbins: int = 500) -> None:
    """Plot per-file absolute error distributions (1x3: max/min/mean)."""
    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = list(file_errors.keys())
    metrics = ["max", "min", "mean"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    act_type = act_types[0] if act_types else "Unknown"
    fig.suptitle(
        f"NVFP4 {act_type} Quantization Errors - {save_fig_path.stem}", fontsize=16, fontweight="bold"
    )

    for act_type in act_types:
        error_stats = file_errors[act_type]
        for k, metric in enumerate(metrics):
            ax = axes[k]
            # Ensure strictly positive for log scale
            assert torch.all(error_stats[metric] > 0), "Absolute error must be strictly positive for log scale"
            error_tensor = error_stats[metric].cpu().flatten().numpy()

            log_min = np.log10(error_tensor.min())
            log_max = np.log10(error_tensor.max())
            bins = np.logspace(log_min, log_max, nbins)

            ax.hist(
                error_tensor, bins=bins, alpha=0.7, color="C0", edgecolor="black", linewidth=0.5
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{metric.upper()} Error Distribution", fontsize=12)
            ax.grid(True, alpha=0.3)

            mean_val = np.mean(error_tensor)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2e}")
            ax.legend(loc="upper right", fontsize=8)

    for k, metric in enumerate(metrics):
        axes[k].set_xlabel(f"{metric.upper()} Error Distribution", fontsize=12)
    axes[0].set_ylabel("Frequency (log)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {save_fig_path}")
    plt.close()


def plot_single_file_rel_errors(file_errors: dict, save_fig_path: Path, nbins: int = 500) -> None:
    """Plot per-file relative error distributions (1x3: max/min/mean)."""
    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    act_types = list(file_errors.keys())
    metrics = ["max", "min", "mean"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    act_type = act_types[0] if act_types else "Unknown"
    fig.suptitle(
        f"NVFP4 {act_type} Relative Error - {save_fig_path.stem}", fontsize=16, fontweight="bold"
    )

    for act_type in act_types:
        error_stats = file_errors[act_type]
        for k, metric in enumerate(metrics):
            ax = axes[k]
            error_tensor = error_stats[metric].cpu().flatten().numpy()
            error_tensor = np.clip(error_tensor, 1e-16, None)

            log_min = np.log10(error_tensor.min())
            log_max = np.log10(error_tensor.max())
            bins = np.logspace(log_min, log_max, nbins)

            ax.hist(
                error_tensor, bins=bins, alpha=0.7, color="C0", edgecolor="black", linewidth=0.5
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(f"{metric.upper()} Relative Error", fontsize=12)
            ax.grid(True, alpha=0.3)

            mean_val = np.mean(error_tensor)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2e}")
            ax.legend(loc="upper right", fontsize=8)

    for k, metric in enumerate(metrics):
        axes[k].set_xlabel(f"{metric.upper()} Relative Error (log)", fontsize=12)
    axes[0].set_ylabel("Frequency (log)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {save_fig_path}")
    plt.close()


def _plot_group_size_comparison_core(
    all_errors: dict,
    save_root: Path,
    group_sizes: list,
    *,
    relative: bool,
    title: str,
    ylabel: str,
    out_filename: str,
    summary_title: str,
) -> None:
    print(f"Creating group size comparison plots ({'relative' if relative else 'absolute'})...")
    save_root.mkdir(parents=True, exist_ok=True)
    metrics = ["max", "min", "mean"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for k, metric in enumerate(metrics):
        ax = axes[k]
        x_labels = []
        all_values = []

        for group_size in group_sizes:
            if group_size == 1:
                key = f"Original_{metric}"
                label = "Original (G1)"
            else:
                key = f"Group_Rotated_{metric}"
                label = "Channelwise" if group_size == "channelwise" else f"G{group_size}"

            if group_size in all_errors and key in all_errors[group_size]:
                values = np.array(all_errors[group_size][key])
                if len(values) > 0:
                    x_labels.append(label)
                    all_values.append(np.clip(values, 1e-16, None))

        if all_values:
            ax.violinplot(all_values, showmeans=True, showmedians=False)
            ax.set_yscale("log")
            ax.set_title(f"{metric.upper()} {'Relative ' if relative else ''}Error", fontsize=12)
            ax.set_xlabel("Group Size", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(range(1, len(x_labels) + 1))
            ax.set_xticklabels(x_labels)
            ax.grid(True, alpha=0.3)

            for i, values in enumerate(all_values):
                mean_val = np.mean(values)
                ax.text(i + 1, mean_val * 1.2, f"{mean_val:.2e}", ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout()
    out_path = save_root / out_filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close()

    print(f"\n{summary_title}")
    print("| Group | Metric | Mean | Median | Std | Min | Max |")
    print("|-------|--------|------|--------|-----|-----|-----|")
    for group_size in group_sizes:
        for metric in metrics:
            if group_size == 1:
                key = f"Original_{metric}"
                label = f"Original (G{group_size})"
            else:
                key = f"Group_Rotated_{metric}"
                label = f"Rotated (G{group_size})" if group_size != "channelwise" else "Rotated (channelwise)"
            if group_size in all_errors and key in all_errors[group_size]:
                values = np.array(all_errors[group_size][key])
                if len(values) == 0:
                    continue
                mean_val = np.mean(values)
                median_val = np.median(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"| {label} | {metric} | {mean_val:.4e} | {median_val:.4e} | {std_val:.4e} | {min_val:.4e} | {max_val:.4e} |")

    print(f"\nPlots saved to: {save_root}")


def plot_group_size_comparison_abs(all_errors: dict, save_root: Path, group_sizes: list) -> None:
    return _plot_group_size_comparison_core(
        all_errors,
        save_root,
        group_sizes,
        relative=False,
        title="NVFP4 Group Size Comparison - Group Rotated (Absolute)",
        ylabel="MSE Error (log scale)",
        out_filename="group_size_comparison.png",
        summary_title="=== NVFP4 Group Size Comparison Summary (Absolute) ===",
    )


def plot_group_size_comparison_rel(all_errors: dict, save_root: Path, group_sizes: list) -> None:
    return _plot_group_size_comparison_core(
        all_errors,
        save_root,
        group_sizes,
        relative=True,
        title="NVFP4 Group Size Comparison - Relative Error",
        ylabel="Relative Error (log scale)",
        out_filename="group_size_comparison_relative.png",
        summary_title="=== NVFP4 Relative Error Summary ===",
    )


def plot_error_across_blocks_multi(
    all_errors: dict,
    save_root: Path,
    group_sizes: list,
    *,
    relative: bool,
    layer_indices: list[int] | None = None,
) -> None:
    """Create line plots for mean/max/min error across blocks per group size.

    Plots mean, max, and min as SEPARATE IMAGES. When relative=True, saves to
    error_across_blocks_relative_mean.png, _max.png, _min.png. Otherwise,
    saves to error_across_blocks_mean.png, _max.png, _min.png for absolute error.
    """
    print(f"Creating error across blocks plots ({'relative' if relative else 'absolute'})...")

    # X-axis: layer indices. Use provided if available, otherwise infer from files
    if layer_indices is None:
        act_root = Path("figs/group_rotate/act/picked")
        layer_name = "down_proj"
        act_dir = Path(act_root) / layer_name
        act_files = natsorted(act_dir.glob("*_input.pt"))
        layer_indices = [extract_layer_idx(p.name) for p in act_files]

    metrics = ("mean", "max", "min")
    metric_titles = {"mean": "Mean", "max": "Max", "min": "Min"}
    ylabel = "Relative Error (log scale)" if relative else "MSE Error (log scale)"
    suffix = "_relative" if relative else ""

    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        for gi, group_size in enumerate(group_sizes):
            key = ("Original_" + metric) if group_size == 1 else ("Group_Rotated_" + metric)
            if group_size in all_errors and key in all_errors[group_size]:
                values = np.array(all_errors[group_size][key])
                if len(values) > 0:
                    values = np.clip(values, 1e-16, None)
                    if group_size == 1:
                        label = "Original (G1)"
                    else:
                        label = "Rotated (Channelwise)" if group_size == "channelwise" else f"Rotated (G{group_size})"
                    ax.plot(
                        layer_indices,
                        values,
                        linestyle="-",
                        color=f"C{gi % 10}",
                        linewidth=2,
                        markersize=4,
                        marker="o",
                        alpha=0.9,
                        label=label,
                    )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, title="Group Size")

        title = f"NVFP4 {metric_titles[metric]} Error Across Blocks{' (Relative)' if relative else ''}"
        ax.set_title(title, fontsize=16, fontweight="bold")

        plt.tight_layout()
        out_name = f"error_across_blocks{suffix}_{metric}.png"
        out_path = save_root / out_name
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_path}")
        plt.close()


def generate_aggregate_plots(
    all_errors_abs: dict,
    all_errors_rel: dict,
    save_root: Path,
    group_sizes: list,
    *,
    layer_indices: list[int] | None = None,
) -> None:
    """Generate group-size comparison and across-block plots from aggregated errors."""
    abs_comparison_save_root = save_root / "nvfp4_groupsize_comparison"
    rel_comparison_save_root = save_root / "nvfp4_relative_groupsize_comparison"
    abs_comparison_save_root.mkdir(parents=True, exist_ok=True)
    rel_comparison_save_root.mkdir(parents=True, exist_ok=True)

    plot_group_size_comparison_abs(all_errors_abs, abs_comparison_save_root, group_sizes)
    plot_group_size_comparison_rel(all_errors_rel, rel_comparison_save_root, group_sizes)
    plot_error_across_blocks_multi(
        all_errors_abs,
        abs_comparison_save_root,
        group_sizes,
        relative=False,
        layer_indices=layer_indices,
    )
    plot_error_across_blocks_multi(
        all_errors_rel,
        rel_comparison_save_root,
        group_sizes,
        relative=True,
        layer_indices=layer_indices,
    )


def compute_all_errors_from_activations(
    act_root: Path,
    save_root: Path,
    layer_name: str,
    group_sizes_input: list | None = None,
):
    """Compute absolute and relative errors from activation tensors on disk.

    Returns: (all_errors_abs, all_errors_rel, group_sizes, layer_indices)
    """
    # Inputs and defaults
    act_dir = Path(act_root) / layer_name
    act_files = natsorted(act_dir.glob("*_input.pt"))

    abs_comparison_save_root = save_root / "nvfp4_groupsize_comparison"
    rel_comparison_save_root = save_root / "nvfp4_relative_groupsize_comparison"
    abs_comparison_save_root.mkdir(parents=True, exist_ok=True)
    rel_comparison_save_root.mkdir(parents=True, exist_ok=True)

    group_sizes = group_sizes_input or [1, 4, 8, 16, 32, 512, "channelwise"]

    # Error collectors
    all_errors_abs: dict[int, dict[str, list[float]]] = {}
    all_errors_rel: dict[int, dict[str, list[float]]] = {}
    for group_size in group_sizes:
        all_errors_abs[group_size] = defaultdict(list)
        all_errors_rel[group_size] = defaultdict(list)

    # Layer indices for alignment and CSV
    layer_indices = [extract_layer_idx(p.name) for p in act_files]

    for act_file in tqdm(act_files, desc="Processing activation files (combined)"):
        activation = torch.load(act_file, map_location="cuda").float()
        hidden_size = activation.shape[-1]
        print(f'{activation.shape=}, {hidden_size=}')

        for group_size in group_sizes:
            # Per-group save directories for per-file plots
            abs_group_save_root = abs_comparison_save_root / f"nvfp4_group{group_size}"
            rel_group_save_root = rel_comparison_save_root / f"nvfp4_group{group_size}"
            abs_group_save_root.mkdir(parents=True, exist_ok=True)
            rel_group_save_root.mkdir(parents=True, exist_ok=True)

            # Build quantizer and rotater
            quantizers, rotater = create_quantizer_and_rotater(group_size, hidden_size=hidden_size)

            # Rotate (if needed) and quantize
            rotated = activation if rotater is None else rotater(activation)
            qdq = quantizers["NVFP4"].fake_quant_act_dynamic(rotated, args={})

            # Absolute error (ensure strictly positive for log plots)
            abs_err = F.mse_loss(qdq, rotated, reduction="none").abs() + 1e-16
            abs_err = abs_err.view(-1, 16)
            abs_err_max = abs_err.amax(-1)
            abs_err_min = abs_err.amin(-1)
            abs_err_mean = abs_err.mean(-1)

            # Relative error
            rel_err = compute_relative_error(qdq, rotated)
            rel_err = rel_err.view(-1, 16)
            rel_err_max = rel_err.amax(-1)
            rel_err_min = rel_err.amin(-1)
            rel_err_mean = rel_err.mean(-1)

            key_prefix = "Original" if group_size == 1 else "Group_Rotated"

            # Store per-file aggregate scalars
            all_errors_abs[group_size][f"{key_prefix}_max"].append(abs_err_max.mean().item())
            all_errors_abs[group_size][f"{key_prefix}_min"].append(abs_err_min.mean().item())
            all_errors_abs[group_size][f"{key_prefix}_mean"].append(abs_err_mean.mean().item())

            all_errors_rel[group_size][f"{key_prefix}_max"].append(rel_err_max.mean().item())
            all_errors_rel[group_size][f"{key_prefix}_min"].append(rel_err_min.mean().item())
            all_errors_rel[group_size][f"{key_prefix}_mean"].append(rel_err_mean.mean().item())

            # Per-file hist plots: absolute
            file_errors_abs = {
                key_prefix: {
                    "max": abs_err_max,
                    "min": abs_err_min,
                    "mean": abs_err_mean,
                }
            }
            abs_fig_path = abs_group_save_root / "per_file_plots" / f"{act_file.stem}_NVFP4.png"
            plot_single_file_abs_errors(file_errors_abs, abs_fig_path)

            # Per-file hist plots: relative
            file_errors_rel = {
                key_prefix: {
                    "max": rel_err_max,
                    "min": rel_err_min,
                    "mean": rel_err_mean,
                }
            }
            rel_fig_path = rel_group_save_root / "per_file_plots" / f"{act_file.stem}_NVFP4_relative.png"
            plot_single_file_rel_errors(file_errors_rel, rel_fig_path)

    return all_errors_abs, all_errors_rel, group_sizes, layer_indices


def export_all_errors_both_to_csv(
    all_errors_abs: dict,
    all_errors_rel: dict,
    group_sizes: list[int],
    layer_indices: list[int],
    save_path: Path,
) -> None:
    """Export absolute and relative errors into a single CSV.

    Columns: layer_index, group_size, error_kind, type, metric, value
    where error_kind in {absolute, relative} and type in {Original, Group_Rotated}.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["max", "min", "mean"]
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_index", "group_size", "error_kind", "type", "metric", "value"])
        for error_kind, all_errors in (("absolute", all_errors_abs), ("relative", all_errors_rel)):
            for group_size in group_sizes:
                key_prefix = "Original" if group_size == 1 else "Group_Rotated"
                for metric in metrics:
                    key = f"{key_prefix}_{metric}"
                    values = all_errors.get(group_size, {}).get(key, [])
                    for i, value in enumerate(values):
                        layer_idx = layer_indices[i] if i < len(layer_indices) else i
                        writer.writerow([layer_idx, group_size, error_kind, key_prefix, metric, value])
    print(f"Saved CSV: {save_path}")


def main() -> None:
    """Run combined NVFP4 absolute and relative error group size comparison analysis."""
    act_root = Path("figs/group_rotate/act/picked")
    act_root = Path("figs/group_rotate/act/picked")
    # save_root = Path("figs/group_rotate/act")
    save_root = act_root.parent
    layer_name = "down_proj"

    # 1) Load from CSV if available, else 2) compute
    csv_combined_path = save_root / "all_errors_combined.csv"
    if csv_combined_path.exists():
        print(f"Found existing CSV: {csv_combined_path}. Loading...")
        all_errors_abs, all_errors_rel, group_sizes, layer_indices = load_all_errors_from_csv(csv_combined_path)
    else:
        all_errors_abs, all_errors_rel, group_sizes, layer_indices = compute_all_errors_from_activations(
            act_root,
            save_root,
            layer_name,
        )
        export_all_errors_both_to_csv(
            all_errors_abs,
            all_errors_rel,
            group_sizes,
            layer_indices,
            csv_combined_path,
        )

    # 3) Unified plotting
    generate_aggregate_plots(
        all_errors_abs,
        all_errors_rel,
        save_root,
        group_sizes,
        layer_indices=layer_indices,
    )

    print("NVFP4 combined (absolute + relative) group size comparison analysis completed!")


if __name__ == "__main__":
    main()


