import argparse
import json
import shutil
from pathlib import Path

import torch
from natsort import natsorted
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def svd_reconstruction(weight: torch.Tensor, rank_or_ratio: float) -> torch.Tensor:
    """
    使用SVD分解对权重矩阵进行低秩重构

    Args:
        weight: 输入的权重矩阵 (2D tensor)
        rank_or_ratio: 如果大于1，表示保留的rank数量；如果小于等于1，表示保留的比例

    Returns:
        重构后的权重矩阵
    """
    # 确保输入是2D矩阵
    assert weight.ndim == 2, f"Expected 2D tensor, got {weight.ndim}D"

    # 获取原始矩阵的形状
    m, n = weight.shape
    max_rank = min(m, n)

    # 计算实际要保留的rank数量
    if rank_or_ratio > 1:
        # 如果大于1，直接作为rank数量
        rank = int(rank_or_ratio)
        rank = min(rank, max_rank)  # 确保不超过矩阵的最大rank
    else:
        # 如果小于等于1，按比例计算
        rank = int(max_rank * rank_or_ratio)
        rank = max(1, rank)  # 至少保留1个rank

    # 执行SVD分解
    # full_matrices=False可以节省内存，只计算需要的部分
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    # 保留前rank个奇异值进行重构
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    Vh_truncated = Vh[:rank, :]

    # 重构矩阵
    reconstructed = U_truncated @ torch.diag(S_truncated) @ Vh_truncated

    return reconstructed.to(weight.dtype)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--rank_or_ratio", type=float, default=0.25)
    args = parser.parse_args()

    args.model_path = Path(args.model_path)
    if args.output_path is None:
        args.output_path = args.model_path.parent / "dequant_model"

    return args


def copy_meta_files(src_dir: Path, dst_dir: Path):
    for meta_file in src_dir.iterdir():
        if "model.safetensors.index.json" in meta_file.name:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            new_weight_map = {}
            for k, v in meta_data["weight_map"].items():
                # Handle VLLM format
                if "weight_scale" in k:
                    continue
                # Handle AWQ format
                if "scales" in k or "qzeros" in k:
                    continue
                if "qweight" in k:
                    k = k.replace(".qweight", ".weight")

                new_weight_map[k] = v
            meta_data["weight_map"] = new_weight_map
            with open(dst_dir / meta_file.name, "w") as f:
                json.dump(meta_data, f, indent=4)
        elif "config.json" == meta_file.name:
            with open(meta_file, "r") as f:
                config = json.load(f)
            config.pop("quantization_config", None)
            config.pop("compression_config", None)
            with open(dst_dir / meta_file.name, "w") as f:
                json.dump(config, f, indent=4)
        elif not meta_file.name.endswith(".safetensors"):
            shutil.copy(meta_file, dst_dir / meta_file.name)


def dequant_bf16_weights(src_dir, output_dir, rank_or_ratio=0.25):
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors = natsorted(list(src_dir.glob("*.safetensors")))
    for safetensor in tqdm(safetensors, desc="Reconstructing weights"):
        state_dict = load_file(safetensor, device="cuda")
        new_state_dict = {}
        keys = list(state_dict.keys())
        for weight_name in keys:
            weight = state_dict[weight_name]
            if (
                "lm_head" in weight_name
                or "embed_tokens" in weight_name
                or weight.ndim != 2
            ):
                new_state_dict[weight_name] = state_dict[weight_name]
                continue

            new_weight = svd_reconstruction(weight, rank_or_ratio)
            new_state_dict[weight_name] = new_weight

        save_file(new_state_dict, output_dir / safetensor.name)


def dequant_autoawq_weights(src_dir, output_dir, rank_or_ratio=0.25):
    raise NotImplementedError("AWQ is not supported for SVD reconstruction")
    import awq_ext

    # try:
    #     import awq_ext
    # except ImportError as e:
    #     raise ImportError("Please install awq_ext: pip install autoawq-kernels")
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors = natsorted(list(src_dir.glob("*.safetensors")))

    for safetensor in tqdm(safetensors, desc="Dequantizing"):
        state_dict = load_file(safetensor, device="cuda")
        new_state_dict = {}

        for param_name, param in state_dict.items():
            # Skip quantized weight components
            if "qweight" in param_name or "qzeros" in param_name:
                continue

            # Handle scales (AWQ format)
            if "scales" in param_name:
                param_name = param_name.replace(".scales", "")
                scales = param.to(torch.float16)
                qweight = state_dict[param_name + ".qweight"]
                qzeros = state_dict[param_name + ".qzeros"]
                param_name += ".weight"

                # Use AWQ's CUDA dequantization
                dequant_weight = awq_ext.dequantize_weights_cuda(
                    qweight, scales, qzeros, 0, 0, 0, False
                )
                param = dequant_weight
                param = param.transpose(0, 1)
            else:
                # Handle non-quantized parameters
                param = param

            param = param.contiguous().clone()
            new_state_dict[param_name] = param

        save_file(new_state_dict, output_dir / safetensor.name)


def reconstruct_weights(src_dir, output_dir, rank_or_ratio=0.25):
    if src_dir.name == "vllm_quant_model":
        dequant_bf16_weights(src_dir, output_dir, rank_or_ratio)
    elif src_dir.name == "autoawq_quant_model":
        dequant_autoawq_weights(src_dir, output_dir, rank_or_ratio)
    else:
        dequant_bf16_weights(src_dir, output_dir, rank_or_ratio)


if __name__ == "__main__":
    args = get_args()
    reconstruct_weights(args.model_path, args.output_path, args.rank_or_ratio)
    copy_meta_files(args.model_path, args.output_path)
