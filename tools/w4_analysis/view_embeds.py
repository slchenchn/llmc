from pathlib import Path
from safetensors.torch import load_file
from natsort import natsorted
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis
import re
import torch.nn as nn


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def calculate_statistics(llm_dirs, keys):
    stats = {}
    all_weights = {}
    model_names_list = []

    print("Calculating statistics...")
    for llm_dir in llm_dirs:
        llm_dir_path = Path(llm_dir)
        model_name = re.search(r"(Qwen3-\d+\.?\d*B)", str(llm_dir_path)).group(1)
        model_names_list.append(model_name)

        print(f"Processing {model_name}...")
        safetensor_files = natsorted(list(llm_dir_path.glob("*.safetensors")))
        bin_file = llm_dir_path / "pytorch_model.bin"

        if safetensor_files:
            model_path = safetensor_files[0]
            state_dict = load_file(model_path)
        elif bin_file.is_file():
            model_path = bin_file
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            print(
                f"No model file (.safetensors or pytorch_model.bin) found in {llm_dir_path}"
            )
            continue
        param = state_dict[keys["embed"]].cuda()
        norm = Qwen3RMSNorm(param.shape[1]).cuda()
        norm.load_state_dict({"weight": state_dict[keys["norm"]]})
        act = norm(param)

        all_weights[model_name] = act.cpu().numpy().flatten()

        min_val = torch.min(act)
        max_val = torch.max(act)
        var_val = torch.var(act)
        kurtosis_val = kurtosis(act.flatten().numpy())
        cov_mat = torch.matmul(act.T, act) ** 2
        cov_mean = cov_mat.mean()
        cov_max = cov_mat.max()
        cov_sum = cov_mat.sum()

        stats[model_name] = {
            "min": min_val.item(),
            "max": max_val.item(),
            "var": var_val.item(),
            "kurtosis": kurtosis_val.item(),
            "cov_mean": cov_mean.item(),
            "cov_max": cov_max.item(),
            "cov_sum": cov_sum.item(),
        }
        print("-" * 20)
    return stats, all_weights, model_names_list


def plot_primary_statistics(stats, model_names_list, key, save_dir):
    all_primary_stats = ["min", "max", "var", "kurtosis"]
    main_primary_stats = ["min", "max", "kurtosis"]
    var_stat_name = "var"
    x_indices = {stat: i for i, stat in enumerate(all_primary_stats)}
    width = 0.2
    n_models = len(model_names_list)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax_var = ax1.twinx()
    fig.suptitle(f'Primary Statistics of "{key}"', fontsize=16)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i, model_name in enumerate(model_names_list):
        color = colors[i % len(colors)]
        offset = width * (i - (n_models - 1) / 2)
        main_values = [stats[model_name][stat] for stat in main_primary_stats]
        main_x_pos = [x_indices[stat] for stat in main_primary_stats]
        ax1.bar(
            np.array(main_x_pos) + offset,
            main_values,
            width,
            label=model_name,
            color=color,
        )
        var_value = [stats[model_name][var_stat_name]]
        var_x_pos = x_indices[var_stat_name]
        ax_var.bar(var_x_pos + offset, var_value, width, color=color)

    ax1.set_ylabel("Value")
    ax1.set_xticks(range(len(all_primary_stats)))
    ax1.set_xticklabels(all_primary_stats)
    ax1.grid(axis="y", linestyle="--")
    ax_var.set_ylabel("Variance")

    fig.legend(*ax1.get_legend_handles_labels(), loc="upper right", title="Model")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = save_dir / f"{key}_primary.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Primary statistics plot saved as {save_path}")


def plot_covariance_statistics(stats, model_names_list, key, save_dir):
    width = 0.25
    x_secondary = np.arange(len(model_names_list))
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f'Covariance Statistics of "{key}"', fontsize=16)
    ax2 = ax.twinx()
    ax3 = ax.twinx()

    ax3.spines.right.set_position(("axes", 1.2))

    cov_means = [stats[model_name]["cov_mean"] for model_name in model_names_list]
    cov_maxs = [stats[model_name]["cov_max"] for model_name in model_names_list]
    cov_sums = [stats[model_name]["cov_sum"] for model_name in model_names_list]

    ax.bar(x_secondary - width, cov_means, width, color="tab:blue")
    ax2.bar(x_secondary, cov_maxs, width, color="tab:red")
    ax3.bar(x_secondary + width, cov_sums, width, color="tab:green")

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Covariance (log scale)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_yscale("log")

    ax2.set_ylabel("Max Covariance (log scale)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_yscale("log")

    ax3.set_ylabel("Sum Covariance (log scale)", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")
    ax3.set_yscale("log")

    ax.set_xticks(x_secondary)
    ax.set_xticklabels(model_names_list, rotation=45, ha="right")
    fig.tight_layout(rect=[0, 0.05, 0.85, 0.95])

    save_path = save_dir / f"{key}_cov.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Covariance statistics plot saved as {save_path}")


def plot_weights_histogram(all_weights, key, save_dir):
    model_names_list = list(all_weights.keys())
    n_models = len(model_names_list)

    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), sharex=True, sharey=True
    )
    fig.suptitle(f'Histogram of "{key}" Weights', fontsize=16)

    axes_flat = axes.flatten()

    for i, model_name in enumerate(model_names_list):
        ax = axes_flat[i]
        weights = all_weights[model_name]
        ax.hist(weights, bins=256, density=True)
        ax.set_title(model_name)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    for i in range(n_models, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.supxlabel("Weight Value")
    fig.supylabel("Density (log scale)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = save_dir / f"{key}_hist.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Weights histogram plot saved as {save_path}")


@torch.inference_mode()
def main():
    llm_dirs = [
        # "/data/chenshuailin/checkpoints/Qwen/Qwen3-0.6B",
        # "/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B",
        # "/data/chenshuailin/checkpoints/Qwen/Qwen3-8B",
        # "/data/chenshuailin/checkpoints/Qwen/Qwen3-32B",
        "checkpoints/Qwen3-0.6B/quarot/sym_w4_a8-dynamic/transformed_model",
        "checkpoints/Qwen3-1.7B/quarot/sym_w8_a8-dynamic/transformed_model",
        "checkpoints/Qwen3-8B/quarot/sym_w8_a8-dynamic/transformed_model",
        "checkpoints/Qwen3-32B/quarot/sym_w8_a8-dynamic/transformed_model",
    ]
    keys = {
        "embed": "model.embed_tokens.weight",
        "norm": "model.layers.0.input_layernorm.weight",
    }
    save_dir = Path("analysis_model/embeds/normed/quarot")
    save_dir.mkdir(parents=True, exist_ok=True)

    statistics, weights, models = calculate_statistics(llm_dirs, keys)

    plot_primary_statistics(statistics, models, key, save_dir)
    plot_covariance_statistics(statistics, models, key, save_dir)
    plot_weights_histogram(weights, key, save_dir)


if __name__ == "__main__":
    main()
