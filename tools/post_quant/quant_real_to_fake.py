import argparse
import json
import shutil
from pathlib import Path

import torch
from natsort import natsorted
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/vllm_quant_model",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--w_only", action="store_true")
    parser.add_argument("--w_asym", action="store_true")
    parser.add_argument("--w_granularity", type=str, default="per_group")
    args = parser.parse_args()

    args.model_path = Path(args.model_path)
    if args.output_path is None:
        args.output_path = args.model_path.parent / "dequant_model"

    return args


def dequantize_vllm_param(
    param: torch.Tensor,
    scale: torch.Tensor,
    zeros: int = 0,
    granularity: str = "per_channel",
):
    assert param.dtype in (torch.int8, torch.int32)
    if granularity == "per_channel":
        assert param.ndim == scale.ndim == 2
        assert param.shape[0] == scale.shape[0]
        assert scale.shape[1] == 1
        return (param * scale) + zeros
    elif granularity == "per_group":
        assert param.ndim == scale.ndim == 2
        assert param.shape[0] == scale.shape[0]
        assert param.shape[1] % scale.shape[1] == 0
        group_size = param.shape[1] // scale.shape[1]
        scale = scale.repeat_interleave(group_size, dim=1)
        return (param * scale) + zeros
    else:
        raise NotImplementedError


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


def dequant_vllm_weights(src_dir, output_dir, w_asym=False, w_granularity="per_group"):
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors = natsorted(list(src_dir.glob("*.safetensors")))
    for safetensor in tqdm(safetensors, desc="Dequantizing"):
        state_dict = load_file(safetensor, device="cuda")
        new_state_dict = {}
        keys = [k for k in state_dict.keys() if ".weight_scale" not in k]
        for weight_name in keys:
            weight = state_dict[weight_name]
            if weight_name + "_scale" in state_dict:
                scale_name = weight_name + "_scale"
            elif weight_name + "_scale_inv" in state_dict:
                scale_name = weight_name + "_scale_inv"
            else:
                assert weight.dtype in (torch.bfloat16, torch.float16, torch.float32), (
                    f"{weight_name} is not quantized"
                )
                new_state_dict[weight_name] = weight
                continue

            scale = state_dict[scale_name]
            if w_asym:
                raise NotImplementedError("Asymmetric weight quantization")

            weight = dequantize_vllm_param(weight, scale, granularity=w_granularity)
            new_state_dict[weight_name] = weight

        save_file(new_state_dict, output_dir / safetensor.name)


def dequant_autoawq_weights(
    src_dir, output_dir, w_asym=False, w_granularity="per_group"
):
    import awq_ext
    # try:
    #     import awq_ext
    # except ImportError as e:
    #     raise ImportError("Please install awq_ext: pip install autoawq-kernels")

    assert w_asym is True, "for AWQ, only support asymmetric weight quantization"
    assert w_granularity == "per_group", "for AWQ, only support per_group granularity"
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


def dequant_weights(src_dir, output_dir, w_asym=False, w_granularity="per_group"):
    if src_dir.name == "vllm_quant_model":
        dequant_vllm_weights(src_dir, output_dir, w_asym, w_granularity)
    elif src_dir.name == "autoawq_quant_model":
        dequant_autoawq_weights(src_dir, output_dir, w_asym, w_granularity)
    else:
        raise ValueError(f"Unknown model type: {src_dir.name}")


if __name__ == "__main__":
    args = get_args()
    dequant_weights(args.model_path, args.output_path, args.w_asym, args.w_granularity)
    copy_meta_files(args.model_path, args.output_path)
