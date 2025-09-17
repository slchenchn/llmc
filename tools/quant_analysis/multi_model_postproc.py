import re
from collections import defaultdict
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


def read_log_v2(log_path: Path):
    with open(log_path, "r") as f:
        logs = f.readlines()

    kurtosis_pat = re.compile(r"The kurtosis of (\w+) is: (\d+\.\d+)")
    absmax_pat = re.compile(r"The absmax of (\w+) is: (\d+\.\d+)")
    outlier_degree_pat = re.compile(
        r"The (outlier_degree\..+) of (\w+) is: (\d+\.?\d*)"
    )
    weight_name_pat = re.compile(r"block_\d+\..*")

    # Use nested defaultdict structure to avoid the need for manual initialization
    res = defaultdict(lambda: defaultdict(dict))
    weight_name = None
    block_idx = None

    for cur_line in tqdm(logs):
        # Match weight name pattern first
        weight_match = weight_name_pat.search(cur_line)
        if weight_match:
            weight_name_full = weight_match.group(0)
            block_idx_str, weight_name = weight_name_full.split(".", 1)
            block_idx = int(block_idx_str.split("_")[-1])
            continue

        if block_idx is None or weight_name is None:
            continue

        # Process outlier degree metrics
        outlier_match = outlier_degree_pat.search(cur_line)
        if outlier_match:
            outlier_metric, tensor_name, value_str = outlier_match.groups()
            value = int(value_str) if value_str.isdigit() else float(value_str)
            res[block_idx][weight_name][f"{tensor_name}.{outlier_metric}"] = value
            continue

        # Process kurtosis metrics
        kurtosis_match = kurtosis_pat.search(cur_line)
        if kurtosis_match:
            tensor_name, value_str = kurtosis_match.groups()
            value = float(value_str)
            res[block_idx][weight_name][f"{tensor_name}.kurtosis"] = value
            continue

        # Process absmax metrics
        absmax_match = absmax_pat.search(cur_line)
        if absmax_match:
            tensor_name, value_str = absmax_match.groups()
            value = float(value_str)
            res[block_idx][weight_name][f"{tensor_name}.absmax"] = value
            continue

    # Check if any data was parsed
    if not res:
        raise ValueError(f"No data found in {log_path}")

    # Convert nested defaultdict to list of dicts, sorted by block_idx
    result_list = []
    for block_idx in sorted(res.keys()):
        for weight_name, metrics in res[block_idx].items():
            result_dict = {
                "block_idx": block_idx,
                "weight_name": weight_name,
                **metrics,
            }
            result_list.append(result_dict)

    print(f"read {len(result_list)} records from {log_path}")
    return result_list


def calculate_curve_distance(
    ref_data: dict, target_data: dict, metric: str = "kurtosis"
) -> dict:
    """
    Calculate various distance metrics between reference curve and target curve.

    Args:
        ref_data: Reference model layer data
        target_data: Target model layer data
        metric: Metric to compare (e.g., "kurtosis")

    Returns:
        Dictionary containing various distance metrics
    """
    # Find common layers between reference and target
    common_layers = set(ref_data.keys()) & set(target_data.keys())

    if not common_layers:
        return {"error": "No common layers found"}

    # Extract values for common layers
    ref_values = np.array([ref_data[layer][metric] for layer in sorted(common_layers)])
    target_values = np.array(
        [target_data[layer][metric] for layer in sorted(common_layers)]
    )

    # Calculate various distance metrics
    distances = {
        "l1_distance": np.mean(np.abs(target_values - ref_values)),
        "l2_distance": np.sqrt(np.mean((target_values - ref_values) ** 2)),
        "max_distance": np.max(np.abs(target_values - ref_values)),
        "relative_l1": np.mean(
            np.abs((target_values - ref_values) / (ref_values + 1e-8))
        ),
        "relative_l2": np.sqrt(
            np.mean(((target_values - ref_values) / (ref_values + 1e-8)) ** 2)
        ),
        "common_layers": len(common_layers),
        "layers": sorted(common_layers),
    }

    return distances


def plot_weight_across_layers(
    data_dict: dict,
    weight_pattern: str,
    save_dir: Path,
    metric: str = "kurtosis",
    opacity: float = 0.7,
    ref_model: str = None,
    distance_metric: str = "l2_distance",
    smoothing_alpha: float = 0.0,
) -> None:
    """
    Plot a specific metric across layers for weights matching a pattern across multiple models.
    Supports multi-row/multi-column plots using '|' as a separator in weight_pattern and metric.

    Args:
        data_dict: Dictionary mapping model names to their data
        weight_pattern: Pattern(s) to match weight names. Use '|' to separate multiple patterns for multi-row plots.
        save_dir: Directory to save the plot
        metric: Metric(s) to plot. Use '|' to separate multiple metrics for multi-column plots.
        opacity: Opacity of plot lines
        ref_model: Reference model name for distance calculation
        distance_metric: Distance metric to show in legend (l1_distance, l2_distance, max_distance, relative_l1, relative_l2)
        smoothing_alpha: Smoothing factor (0 for no smoothing).
    """
    metric_list = metric.split("|")
    pattern_list = weight_pattern.split("|")
    num_rows = len(pattern_list)
    num_cols = len(metric_list)

    # Create figure with subplots
    subplot_titles = [
        f"{m} of '{p}'" for p in pattern_list for m in metric_list
    ]
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(7 * num_cols, 4 * num_rows),
        sharex=False,
        sharey=False
    )
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Set subplot titles
    for i, title in enumerate(subplot_titles):
        if i < len(axes):
            axes[i].set_title(title, fontsize=14)

    # Visualization configuration
    model_colors = [
        "blue", "red", "green", "purple", "orange", "cyan", "magenta", "yellow",
    ]
    dash_style = "solid"

    for row_idx, current_pattern in enumerate(pattern_list, 1):
        for col_idx, current_metric in enumerate(metric_list, 1):
            # First pass: collect all layer data and indices for the current subplot
            model_layer_data = {}
            for model_name, data in data_dict.items():
                filtered_data = [
                    item
                    for item in data
                    if current_pattern in item["weight_name"]
                    and current_metric in item
                ]
                if not filtered_data:
                    print(
                        f"No data for pattern '{current_pattern}' and metric '{current_metric}' in model '{model_name}'"
                    )
                    continue

                model_layers = {}
                for item in filtered_data:
                    layer_idx = item["block_idx"]
                    if (
                        layer_idx not in model_layers
                        or item[current_metric] > model_layers[layer_idx][current_metric]
                    ):
                        model_layers[layer_idx] = item
                model_layer_data[model_name] = model_layers

            if not model_layer_data:
                print(
                    f"No data to plot for pattern '{current_pattern}' and metric '{current_metric}'"
                )
                continue

            min_layers = min(
                (len(d) for d in model_layer_data.values() if d), default=0
            )
            if min_layers == 0:
                continue

            # Calculate distances relative to reference model if specified
            distance_results = {}
            if ref_model and ref_model in model_layer_data:
                ref_data = model_layer_data[ref_model]
                if row_idx == 1 and col_idx == 1: # Print header only once
                    print(f"\nCalculating distances relative to reference model: {ref_model}")
                    print("-" * 60)
                print(f"Pattern: {current_pattern}, Metric: {current_metric}")

                for model_name, layer_data in model_layer_data.items():
                    if model_name == ref_model:
                        continue
                    distances = calculate_curve_distance(
                        ref_data, layer_data, current_metric
                    )
                    distance_results[model_name] = distances
                    if "error" not in distances:
                        print(
                            f"  {model_name:20s} | L1: {distances['l1_distance']:.4f} | L2: {distances['l2_distance']:.4f} | "
                            f"Max: {distances['max_distance']:.4f} | RelL1: {distances['relative_l1']:.4f} | "
                            f"RelL2: {distances['relative_l2']:.4f} | Layers: {distances['common_layers']}"
                        )
                    else:
                        print(f"  {model_name:20s} | {distances['error']}")


            # Second pass: plot data for each model in the current subplot
            for i, (model_name, layer_data) in enumerate(model_layer_data.items()):
                if not layer_data:
                    continue

                sorted_layers = sorted(layer_data.keys())[:min_layers]
                x_values = sorted_layers
                y_values = [layer_data[idx][current_metric] for idx in sorted_layers]

                legend_name = model_name
                show_legend_for_trace = (row_idx == 1 and col_idx == 1)

                if (
                    model_name in distance_results
                    and "error" not in distance_results[model_name]
                ):
                    dist_info = distance_results[model_name]
                    if distance_metric in dist_info:
                        metric_labels = {
                            "l1_distance": "L1", "l2_distance": "L2", "max_distance": "Max",
                            "relative_l1": "RelL1", "relative_l2": "RelL2",
                        }
                        label = metric_labels.get(distance_metric, distance_metric)
                        legend_name = f"{model_name} ({label}: {dist_info[distance_metric]:.3f})"

                # Calculate subplot index
                subplot_idx = (row_idx - 1) * num_cols + (col_idx - 1)
                ax = axes[subplot_idx]
                
                y_values_to_plot = y_values
                if smoothing_alpha > 0 and len(y_values) > 1:
                    # Plot original data with low opacity
                    ax.plot(
                        x_values, y_values, 
                        color=model_colors[i % len(model_colors)],
                        linewidth=2, linestyle=':', alpha=0.3
                    )
                    smoothed_y = []
                    last_val = y_values[0]
                    for val in y_values:
                        last_val = (
                            smoothing_alpha * val + (1 - smoothing_alpha) * last_val
                        )
                        smoothed_y.append(last_val)
                    y_values_to_plot = smoothed_y
                
                # Plot main data
                ax.plot(
                    x_values, y_values_to_plot, 
                    color=model_colors[i % len(model_colors)],
                    linewidth=3, linestyle=dash_style,
                    marker='o', markersize=8, alpha=opacity,
                    label=legend_name if show_legend_for_trace else None
                )

    # Configure layout
    title_text = "Comparison Across Layers"
    if ref_model:
        title_text += f" (Distances relative to {ref_model})"
    if smoothing_alpha > 0:
        title_text += f" (Smoothed: {smoothing_alpha})"

    fig.suptitle(title_text, fontsize=24, y=0.98)
    
    # Configure subplots
    for r in range(num_rows):
        for c in range(num_cols):
            subplot_idx = r * num_cols + c
            if subplot_idx < len(axes):
                ax = axes[subplot_idx]
                ax.set_xlabel("Layer Index", fontsize=14)
                ax.set_ylabel(metric_list[c], fontsize=14)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=12)
    
    # Configure legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                  ncol=min(len(handles), 4), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


    # Save the figure
    safe_pattern_name = weight_pattern.replace("|", "_")
    safe_metric_name = metric.replace("|", "_").replace(".", "-")
    per_weight_dir = save_dir / "per_weight" / safe_pattern_name
    per_weight_dir.mkdir(parents=True, exist_ok=True)
    output_path = per_weight_dir / f"{safe_metric_name}_across_layers.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"Figure saved as {output_path}")

    # Note: Distance results are not saved to a file in the multi-plot version for simplicity.
    # A consolidated JSON would require more complex data structuring.


if __name__ == "__main__":
    log_paths = {
        # "RTN-NVFP4":"analysis_model/Qwen3-1.7B/RTN-NVFP4/outlier_degree_kurtosis_absmax_20250814_032127.log",
        # "QAT-post-NVFP4": "analysis_model/Qwen3-1.7B/QAT-post-NVFP4/outlier_degree_kurtosis_absmax_20250814_024803.log",

        "bf16": "analysis_model/Qwen2.5-3B-Instruct/bf16/outlier_degree_kurtosis_absmax_20250917_093429.log",
        # "QAT-pre-NVFP4": "analysis_model/Qwen3-1.7B/QAT-pre-NVFP4/outlier_degree_kurtosis_absmax_20250814_023237.log",
    }
    ref = None
    smoothing_alpha = 0
    save_dir = Path("analysis_model/Qwen2.5-3B-Instruct/compare_bf16_quarot")

    vis_metrics = [
        "weight.absmax|weight.kurtosis",
        "act.absmax|act.kurtosis",
        # "weight.absmax",
        # "weight.kurtosis",
        # "act.absmax",
        # "act.kurtosis",
        # "act.outlier_degree.#upper outliers",
        # "act.outlier_degree.#lower outliers",
        # "act.outlier_degree.#total outliers",
        # "act.outlier_degree.max outlier degree",
        # "act.outlier_degree.min outlier degree",
    ]
    vis_weight_patterns = [
        "down_proj|gate_proj",
        "q_proj|v_proj",
        # "down_proj",
        # "up_proj",
        # "gate_proj",
        # "input_layernorm",
        # "q_proj",
    ]


    # Remove the assertion to support any number of models
    save_dir /= "--".join(log_paths.keys())
    save_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for abbr, log_path in log_paths.items():
        data = read_log_v2(log_path)
        res[abbr] = data

    # Plot weight comparison for specific patterns
    for metric in vis_metrics:
        for weight_pattern in vis_weight_patterns:
            plot_weight_across_layers(
                res,
                weight_pattern,
                save_dir,
                metric=metric,
                ref_model=ref,
                distance_metric="relative_l2",
                smoothing_alpha=smoothing_alpha,
            )

    print("All comparison plots generated")
