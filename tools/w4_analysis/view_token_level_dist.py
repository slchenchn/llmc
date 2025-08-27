from pathlib import Path
from safetensors.torch import load_file
from natsort import natsorted
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import random


def load_parameter(llm_dir, key):
    llm_dir_path = Path(llm_dir)
    model_name = llm_dir_path.name

    print(f"Processing {model_name}...")
    safetensor = natsorted(llm_dir_path.glob("*.safetensors"))[0]
    state_dict = load_file(safetensor)
    param = state_dict[key].float()
    print("-" * 20)
    return param, model_name


def calculate_token_level_stats(param, model_name):
    l2_norms = torch.linalg.norm(param, ord=2, dim=1).cpu().numpy()
    variances = torch.var(param, dim=1).cpu().numpy()
    means = torch.mean(param, dim=1).cpu().numpy()

    stats_df = pd.DataFrame(
        {
            "Model": model_name,
            "L2 Norm": l2_norms,
            "Variance": variances,
            "Mean": means,
        }
    )
    return stats_df


def plot_token_level_distributions(stats_df, key, save_dir):
    model_name = stats_df["Model"].unique()[0]
    print(f"Generating token-level distribution plot for {model_name}...")

    stat_metrics = ["L2 Norm", "Variance", "Mean"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f'Token-Level Embedding Statistics for "{key}" in {model_name}', fontsize=16
    )

    for i, metric in enumerate(stat_metrics):
        ax = axes[i]
        ax.violinplot(stats_df[metric])
        ax.set_title(f"Distribution of Token {metric}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = save_dir / f"{model_name}_{key}_token_dist.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Token-level distribution plot saved as {save_path}")


def plot_tsne_representation(param, model_name, key, save_dir, sample_size=20000):
    print(f"Generating t-SNE plot for {model_name}...")

    data = param.cpu().numpy()

    if data.shape[0] > sample_size:
        print(f"Sampling {sample_size} tokens out of {data.shape[0]} for t-SNE.")
        random.seed(42)
        indices = np.random.choice(data.shape[0], sample_size, replace=False)
        data = data[indices]

    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, s=5)
    ax.set_title(f't-SNE Representation of "{key}" in {model_name}')
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()

    save_path = save_dir / f"{model_name}_{key}_tsne.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"t-SNE plot saved as {save_path}")


if __name__ == "__main__":
    llm_dirs = [
        "/data/chenshuailin/checkpoints/Qwen/Qwen3-0.6B",
        "/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B",
        "/data/chenshuailin/checkpoints/Qwen/Qwen3-8B",
        "/data/chenshuailin/checkpoints/Qwen/Qwen3-32B",
    ]
    key = "model.embed_tokens.weight"
    save_dir = Path("analysis_model/tokens")
    save_dir.mkdir(parents=True, exist_ok=True)

    for llm_dir in llm_dirs:
        param, model_name = load_parameter(llm_dir, key)
        # token_stats_df = calculate_token_level_stats(param, model_name)
        # plot_token_level_distributions(token_stats_df, key, save_dir)
        plot_tsne_representation(param, model_name, key, save_dir)
