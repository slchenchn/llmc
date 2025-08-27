from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from collections import defaultdict
from tqdm import tqdm


def get_model_index_path(model_dir: Path):
    safetensors_index_path = model_dir / "model.safetensors.index.json"
    bin_index_path = model_dir / "pytorch_model.bin.index.json"
    if safetensors_index_path.exists():
        index_path = safetensors_index_path
    elif bin_index_path.exists():
        index_path = bin_index_path
    else:
        raise FileNotFoundError(f"No index file found in {model_dir}")
    return index_path


def load_shard(shard_path: Path):
    if shard_path.suffix == ".safetensors":
        shard_ckpt = load_file(shard_path)
    else:
        shard_ckpt = torch.load(str(shard_path), map_location="cpu")
    return shard_ckpt


def get_name_to_shape(model_dir: Path, param_pat: str):
    index_path = get_model_index_path(model_dir)
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files_to_load = set()
    for k, v in weight_map.items():
        if param_pat in k:
            shard_files_to_load.add(v)

    name_to_shape = defaultdict(dict)
    for shard_file in shard_files_to_load:
        shard_path = model_dir / shard_file
        state_dict = load_shard(shard_path)
        for name, param in state_dict.items():
            if param_pat in name:
                name_to_shape[name]["shape"] = param.shape
                name_to_shape[name]["dtype"] = param.dtype

    return name_to_shape



if __name__ == "__main__":
    ref_dir = Path("/ms/FM/gongoubo/checkpoints/DeepSeek-R1-Channel-INT8")
    mtp_pat = "model.layers.61."
    model_dirs = [
        "/ms/FM/chenshuailin/code/llmc/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp_v5",
        # "/ms/FM/chenshuailin/code/llmc/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp_v4",
        # "/ms/FM/chenshuailin/code/llmc/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp_v3",
        # "/ms/FM/chenshuailin/code/llmc/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp_v2",
        # "/ms/FM/chenshuailin/code/llmc/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp_v1",
    ]
    ref_name_to_shape = get_name_to_shape(ref_dir, mtp_pat)
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        model_name_to_shape = get_name_to_shape(model_dir, mtp_pat)
        for name, ref_info in ref_name_to_shape.items():
            if name in model_name_to_shape:
                if model_name_to_shape[name]["shape"] != ref_info["shape"]:
                    print(
                        f"{name} shape mismatch: {model_name_to_shape[name]['shape']} != {ref_info['shape']}"
                    )

                if (
                    model_name_to_shape[name]["dtype"] == torch.bfloat16
                    and ref_info["dtype"] == torch.float32
                ):
                    continue

                if model_name_to_shape[name]["dtype"] != ref_info["dtype"]:
                    print(
                        f"{name} dtype mismatch: {model_name_to_shape[name]['dtype']} != {ref_info['dtype']}"
                    )
            else:
                print(f"{name} not found in model")

    print(
        f"keys that are in ref but not in model: {ref_name_to_shape.keys() - model_name_to_shape.keys()}"
    )
    print(
        f"keys that are in model but not in ref: {model_name_to_shape.keys() - ref_name_to_shape.keys()}"
    )
    print("Done")
