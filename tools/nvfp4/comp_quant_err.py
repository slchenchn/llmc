"""Analyze NVFP4 quantization loss for selected layers across blocks."""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.append("/ms/FM/chenshuailin/code/llmc")
from llmc.compression.quantization.quant_nvfp4 import NVFP4Quantizer


def load_state_dict(model_dir):
    """Load state dict from safetensors files."""
    state_dict = {}
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor))
    for bin_file in model_dir.glob("*.bin"):
        loaded = torch.load(bin_file, map_location="cpu")
        state_dict.update(loaded.get("state_dict", loaded))
    return state_dict


def extract_layer_idx_from_weight_name(name: str) -> int:
    """Extract layer index from weight parameter name.

    Supports patterns like: model.layers.{idx}.mlp.down_proj.weight
    Fallback: first number in the name.
    """
    match = re.search(r"model\.layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def compute_quant_loss_by_layer(
    model_dir: Path,
    vis_layers: list[str],
    quantizer: NVFP4Quantizer,
):
    """Compute NVFP4 quantization loss (MSE) for selected layers across blocks."""
    state_dict = load_state_dict(model_dir)
    if not state_dict:
        print(f"No weights found in {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_layer_losses: dict[str, dict[int, list[float]]] = {
        vis_layer: defaultdict(list) for vis_layer in vis_layers
    }

    for name, weight in tqdm(state_dict.items(), desc="Processing weights"):
        if "embed" in name or "head" in name or "norm" in name:
            continue
        if weight.ndim != 2:
            continue

        # Only process weights that have a compatible second dimension
        if not any(vis_layer in name for vis_layer in vis_layers):
            continue

        weight = weight.to(device).float()

        qdq = quantizer.fake_quant_weight_dynamic(weight, args={})
        loss = F.mse_loss(qdq, weight, reduction="mean").item()

        layer_idx = extract_layer_idx_from_weight_name(name)
        for vis_layer in vis_layers:
            if vis_layer in name:
                per_layer_losses[vis_layer][layer_idx].append(loss)

    per_layer_means: dict[str, tuple[list[int], list[float]]] = {}
    for vis_layer, layer_losses in per_layer_losses.items():
        sorted_indices = sorted(layer_losses.keys())
        means = []
        for idx in sorted_indices:
            values = layer_losses[idx]
            means.append(sum(values) / len(values))
        per_layer_means[vis_layer] = (sorted_indices, means)

    return per_layer_means


def plot_quant_loss(
    all_model_losses: dict[str, dict[str, tuple[list[int], list[float]]]],
    save_root: Path,
    vis_layers: list[str],
) -> None:
    """Plot NVFP4 quantization loss across blocks for each model and layer."""
    save_root.mkdir(parents=True, exist_ok=True)
    nrows = len(vis_layers)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 4 * nrows))
    if nrows == 1:
        axes = [axes]

    for ax, vis_layer in zip(axes, vis_layers):
        for idx, (model_name, model_losses) in enumerate(all_model_losses.items()):
            if vis_layer not in model_losses:
                continue
            layer_indices, losses = model_losses[vis_layer]
            (line,) = ax.plot(
                layer_indices, losses, marker="o", linewidth=1.5, label=model_name
            )
            if losses:
                avg_loss = sum(losses) / len(losses)
                ax.axhline(
                    avg_loss, linestyle="--", linewidth=1.0, alpha=0.6, color=line.get_color()
                )
                ax.text(
                    0.01,
                    0.98 - 0.06 * idx,
                    f"{model_name} avg: {avg_loss:.3e}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color=line.get_color(),
                )
        ax.set_title(f"NVFP4 Quantization Loss - {vis_layer}")
        ax.set_xlabel("Block Index")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out_path = save_root / "nvfp4_quant_loss_lineplot.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def main() -> None:
    """Run NVFP4 quantization loss analysis for selected layers across blocks."""
    models = {
        "official": "/ms/FM/chenshuailin/checkpoints/Qwen/Qwen3-8B-Base/",
        "fp8": "/ms/FM/chenshuailin/projects/pretrain/Megatron-LM/checkpoints/Qwen/Qwen3-8B-Base-bridge-mcore/TP1_PP1_data_aihub_te_ce/iter_0007152/hf_model",
        "bf16": "/ms/FM/checkpoints/Qwen-Zoo/Qwen3-8B-Base-bridge-mcore-iter7152/iter_0007152/hf_model"
    }
    vis_layers = ["down_proj", "q_proj"]
    save_root = Path("figs/comp_nvf4_quant_err/weight")

    quantizer = NVFP4Quantizer(
        bit=4,
        symmetric=True,
        granularity="per_group",
        group_size=16,
    )
    all_model_losses: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for model_name, model_path in models.items():
        model_dir = Path(model_path)
        all_model_losses[model_name] = compute_quant_loss_by_layer(
            model_dir=model_dir,
            vis_layers=vis_layers,
            quantizer=quantizer,
        )

    plot_quant_loss(all_model_losses, save_root, vis_layers)


if __name__ == "__main__":
    main()
