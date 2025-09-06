from pathlib import Path
import json_repair
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm


def remove_from_list_no_err(lst, rms):
    for rm in rms:
        try:
            lst.remove(rm)
        except ValueError:
            pass


def read_gptq_log(log_path: Path):
    def _update_log():
        assert len(errors) == len(layers), f"{len(errors)} != {len(layers)}"
        if len(errors) == 0:
            return
        log.append(dict(zip(layers, errors)))
        layers.clear()
        errors.clear()

    with open(log_path, "r") as f:
        lines = f.readlines()
    log = []
    errors = []
    layers = []
    for line in tqdm(lines, desc=f"reading {log_path}"):
        if "subset: " in line:
            _update_log()
            layers_dict = json_repair.repair_json(line, return_objects=True)
            layers = list(layers_dict["layers"].keys())
            remove_from_list_no_err(
                layers, ["prev_op", "input", "inspect", "has_kwargs", "is_mlp"]
            )
        elif "- error " in line:
            errors.append(float(line.rsplit(" ", 1)[1]))
    _update_log()
    return log


def plot_gptq_err(datas: dict, module_names: list = None, save_fig_dir: Path = None):
    """
    Plot GPTQ errors using matplotlib

    Args:
        datas: Dictionary with legend as key and list of layer error dictionaries as value
        module_names: List of module names to plot. If None, plot all modules found
    """
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    if not datas:
        print("No data to plot")
        return

    # Collect all unique module names if not specified
    if module_names is None:
        all_modules = set()
        for data_list in datas.values():
            for layer_dict in data_list:
                all_modules.update(layer_dict.keys())
        module_names = sorted(list(all_modules))

    plot_data = {}
    for legend, data_list in datas.items():
        plot_data[legend] = {}
        for layer_idx, layer_dict in enumerate(data_list):
            for key, value in layer_dict.items():
                if module_names is not None and key not in module_names:
                    continue
                if key not in plot_data[legend]:
                    plot_data[legend][key] = []
                plot_data[legend][key].append(value)

    if not plot_data:
        print("No matching data found for specified modules")
        return

    # Get all unique modules
    unique_modules = set()
    for config_data in plot_data.values():
        unique_modules.update(config_data.keys())
    unique_modules = sorted(list(unique_modules))

    colors = plt.cm.get_cmap("Set1").colors
    legends = "_".join(datas.keys())

    # Create subfolder based on legends
    legends_dir = save_fig_dir / legends
    legends_dir.mkdir(parents=True, exist_ok=True)

    # Create individual plots for each module
    for module in unique_modules:
        fig, ax = plt.subplots(figsize=(10, 6))

        for j, (config, config_data) in enumerate(plot_data.items()):
            if module in config_data:
                layer_indices = list(range(len(config_data[module])))
                ax.plot(
                    layer_indices,
                    config_data[module],
                    "o-",
                    label=config,
                    color=colors[j % len(colors)],
                )

        ax.set_title(f"GPTQ Quantization Errors - {module}")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Quantization Error")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_yscale("log")

        # Save individual module plot in subfolder
        safe_module_name = module.replace(".", "_")
        output_file = legends_dir / f"gptq_errors_{safe_module_name}.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved as {output_file}")

    # Also create a heatmap view
    create_heatmap(plot_data, legends_dir)


def create_heatmap(plot_data, legends_dir: Path):
    """Create a heatmap view of errors across layers and modules"""

    # Create individual heatmaps for each configuration
    for config, config_data in plot_data.items():
        # Get all modules and max layer count
        modules = sorted(list(config_data.keys()))
        if not modules:
            continue
        max_layers = (
            max(len(errors) for errors in config_data.values()) if config_data else 0
        )

        # Create matrix for heatmap
        z_matrix = []
        for layer_idx in range(max_layers):
            layer_errors = []
            for module in modules:
                if layer_idx < len(config_data[module]):
                    layer_errors.append(config_data[module][layer_idx])
                else:
                    layer_errors.append(np.nan)  # Fill missing values with NaN
            z_matrix.append(layer_errors)
        z_matrix = np.array(z_matrix)
        fig, ax = plt.subplots(
            figsize=(max(8, len(modules) * 0.8), max(6, max_layers * 0.3))
        )

        non_nan_values = z_matrix[~np.isnan(z_matrix)]
        median_val = np.median(non_nan_values) if len(non_nan_values) > 0 else None

        # Check if all values are positive for log scale
        min_val = np.nanmin(z_matrix) if len(non_nan_values) > 0 else 0
        use_log_norm = min_val > 0

        sns.heatmap(
            z_matrix,
            ax=ax,
            xticklabels=modules,
            yticklabels=list(range(max_layers)),
            cmap="viridis",
            annot=False,
            fmt=".4f",
            center=median_val,
            norm=LogNorm() if use_log_norm else None,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"GPTQ Errors Heatmap - {config}")
        ax.set_xlabel("Module")
        ax.set_ylabel("Layer Index")

        # Save individual heatmap in the same subfolder
        safe_config_name = config.replace(".", "_").replace("/", "_")
        heatmap_file = legends_dir / f"gptq_errors_heatmap_{safe_config_name}.png"
        plt.savefig(heatmap_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved as {heatmap_file}")


if __name__ == "__main__":
    save_fig_dir = Path("figs/gptq_errs/qwen2.5-3B-it")
    log_paths = [
        "logs/step2_gptq_qwen2.5-3B-it_sym_w4a8_20250906_143211.log",
        "logs/step2_gptq_qwen2.5-3B-it_sym_w4a8_force_dtype_20250906_174959.log",
        # "logs/step2_gptq_qwen2.5-3B-it_sym_w4a8_fp32_20250906_163150.log",
        "logs/step2_gptq_qwen2.5-3B-it_sym_w4a8_force_dtype_ultrachat_20250906_192030.log",
    ]
    datas = {}
    for log_path in log_paths:
        if Path(log_path).exists():
            legend = "_".join(Path(log_path).stem.split("_")[:-2])
            data = read_gptq_log(log_path)
            datas[legend] = data
        else:
            print(f"Warning: {log_path} does not exist, skipping...")

    module_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    module_names = None  # Use this to plot all modules found in data

    plot_gptq_err(datas, module_names, save_fig_dir)
