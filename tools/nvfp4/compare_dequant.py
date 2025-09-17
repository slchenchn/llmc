from pathlib import Path
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import torch


def load_safetensors_from_dir(dirpath: Path):
    """Load all safetensors files from a directory into a single state dict."""
    safetensors = list(dirpath.rglob("*.safetensors"))
    state_dict = {}
    for safetensor in safetensors:
        state_dict.update(load_file(safetensor))
    return state_dict


def compare_keys(model, ref, name):
    """Compare keys between two models and print differences."""
    print(f"\n---- comparing keys of {name} ----")
    print(f"{len(model.keys())=}")
    print(f"{len(ref.keys())=}")
    print(f"keys that only in model: {sorted(model.keys() - ref.keys())}")
    print(f"keys that only in ref: {sorted(ref.keys() - model.keys())}")


def parse_layer_name(key):
    """Parse layer name from model key."""
    # Handle different key formats
    if "layers" in key:
        # Format: model.layers.X.xxx.layer_name.xxx
        parts = key.split(".")
        if len(parts) >= 4:
            # Find the layer name part (usually after the block index)
            for i, part in enumerate(parts[3:], 3):
                if part in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]:
                    return part
                elif part in ["input_layernorm", "post_attention_layernorm"]:
                    return part
    return "other"


def analyze_model_diffs(model_llmc, model_llmcompressor, ref_model):
    """
    Analyze differences between quantized models and reference model per block and per layer.

    Args:
        model_llmc: LLMC quantized model state dict
        model_llmcompressor: LLMCompressor quantized model state dict
        ref_model: Reference model state dict

    Returns:
        dict: Dictionary with layer names as keys, each containing block analysis results
    """
    # 存储按layer name分组的block diff数据
    layer_data = defaultdict(
        lambda: {
            "block_diffs_llmc": defaultdict(list),
            "block_diffs_llmcompressor": defaultdict(list),
        }
    )

    for key, param_ref in tqdm(ref_model.items(), desc="Analyzing model differences"):
        # 解析key获取block_idx
        parts = key.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            block_idx = int(parts[2])
        else:
            continue  # 跳过非layer参数

        # 跳过norm和bias参数（用户要求提取layernorm，但这里我们跳过它们，因为要分析权重）
        if "norm" in key or ".bias" in key:
            continue

        # 解析layer name
        layer_name = parse_layer_name(key)
        if layer_name == "other":
            continue

        param_llmc = model_llmc[key]
        param_llmcompressor = model_llmcompressor[key]

        diff_llmc = param_ref - param_llmc
        diff_llmcompressor = param_ref - param_llmcompressor

        # 将diff添加到对应layer和block
        layer_data[layer_name]["block_diffs_llmc"][block_idx].append(
            diff_llmc.flatten()
        )
        layer_data[layer_name]["block_diffs_llmcompressor"][block_idx].append(
            diff_llmcompressor.flatten()
        )

    # 计算每个layer的分析结果
    results = {}
    for layer_name, data in layer_data.items():
        block_diffs_llmc = data["block_diffs_llmc"]
        block_diffs_llmcompressor = data["block_diffs_llmcompressor"]

        # 获取所有block indices
        block_indices = sorted(
            set(block_diffs_llmc.keys()) | set(block_diffs_llmcompressor.keys())
        )

        absmax_llmc = []
        absmean_llmc = []
        absmax_llmcompressor = []
        absmean_llmcompressor = []

        for block_idx in block_indices:
            # 处理LLMC数据
            if block_idx in block_diffs_llmc and block_diffs_llmc[block_idx]:
                diffs_llmc = torch.cat(block_diffs_llmc[block_idx])
                absmax_llmc.append(diffs_llmc.abs().max().cpu().numpy())
                absmean_llmc.append(diffs_llmc.abs().mean().cpu().numpy())
            else:
                absmax_llmc.append(0.0)
                absmean_llmc.append(0.0)

            # 处理LLMCompressor数据
            if (
                block_idx in block_diffs_llmcompressor
                and block_diffs_llmcompressor[block_idx]
            ):
                diffs_llmcompressor = torch.cat(block_diffs_llmcompressor[block_idx])
                absmax_llmcompressor.append(
                    diffs_llmcompressor.abs().max().cpu().numpy()
                )
                absmean_llmcompressor.append(
                    diffs_llmcompressor.abs().mean().cpu().numpy()
                )
            else:
                absmax_llmcompressor.append(0.0)
                absmean_llmcompressor.append(0.0)

        results[layer_name] = {
            "block_indices": block_indices,
            "absmax_llmc": absmax_llmc,
            "absmean_llmc": absmean_llmc,
            "absmax_llmcompressor": absmax_llmcompressor,
            "absmean_llmcompressor": absmean_llmcompressor,
        }

    return results


def plot_diff_analysis(
    block_indices,
    absmax_llmc,
    absmean_llmc,
    absmax_llmcompressor,
    absmean_llmcompressor,
    save_path,
    layer_name="",
):
    """
    Plot the difference analysis results with LLMC and LLMCompressor on the same axes for comparison.

    Args:
        block_indices: List of block indices
        absmax_llmc: Absolute max differences for LLMC
        absmean_llmc: Absolute mean differences for LLMC
        absmax_llmcompressor: Absolute max differences for LLMCompressor
        absmean_llmcompressor: Absolute mean differences for LLMCompressor
        save_path: Path to save the plot
        layer_name: Name of the layer being analyzed (for title)
    """
    # 绘制图形 - 2x1布局，每个图对比两种模型
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Absolute Max Difference comparison
    ax1.plot(
        block_indices,
        absmax_llmc,
        "b-",
        marker="o",
        linewidth=2,
        markersize=6,
        label="LLMC",
    )
    ax1.plot(
        block_indices,
        absmax_llmcompressor,
        "r--",
        marker="s",
        linewidth=2,
        markersize=6,
        label="LLMCompressor",
    )
    title_suffix = f" - {layer_name}" if layer_name else ""
    ax1.set_title(
        f"Absolute Max Difference per Block - LLMC vs LLMCompressor{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Block Index", fontsize=12)
    ax1.set_ylabel("Absolute Max Difference", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    # Absolute Mean Difference comparison
    ax2.plot(
        block_indices,
        absmean_llmc,
        "b-",
        marker="o",
        linewidth=2,
        markersize=6,
        label="LLMC",
    )
    ax2.plot(
        block_indices,
        absmean_llmcompressor,
        "r--",
        marker="s",
        linewidth=2,
        markersize=6,
        label="LLMCompressor",
    )
    ax2.set_title(
        f"Absolute Mean Difference per Block - LLMC vs LLMCompressor{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Block Index", fontsize=12)
    ax2.set_ylabel("Absolute Mean Difference", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭figure以释放内存


if __name__ == "__main__":
    # Define paths
    ref_dir = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct")
    llmc_dir = Path(
        "checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16/vllm_nvfp4_quant_model_dequant"
    )
    llmcompressor_dir = Path(
        "checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16/llmcompressor_vllm_nvfp4_quant_model_dequant"
    )
    save_fig_dir = Path("figs/nvfp4/compare_dequant")

    # Load models
    print("Loading models...")
    model_llmc = load_safetensors_from_dir(llmc_dir)
    model_llmcompressor = load_safetensors_from_dir(llmcompressor_dir)
    ref_model = load_safetensors_from_dir(ref_dir)

    # Compare keys
    compare_keys(model_llmc, ref_model, "llmc")
    compare_keys(model_llmcompressor, ref_model, "llmcompressor")

    # Analyze differences
    layer_results = analyze_model_diffs(model_llmc, model_llmcompressor, ref_model)

    # Plot and save results for each layer
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(layer_results)} layers to analyze:")
    for layer_name in sorted(layer_results.keys()):
        print(f"  - {layer_name}")

    print("\nGenerating plots...")
    for layer_name, data in layer_results.items():
        print(f"  Processing {layer_name}...")
        save_path = save_fig_dir / f"diff_analysis_{layer_name}.png"
        plot_diff_analysis(
            data["block_indices"],
            data["absmax_llmc"],
            data["absmean_llmc"],
            data["absmax_llmcompressor"],
            data["absmean_llmcompressor"],
            save_path,
            layer_name,
        )
        print(f"Saved {save_path}")

    print(f"\nAnalysis complete!")
    print(f"Processed {len(layer_results)} layers")
    print(f"Results saved to {save_fig_dir}")
