from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file, save_file
import torch
from pathlib import Path
import json
import shutil
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm
import re
import os


def copy_meta_files(src_dir: Path, dst_dir: Path):
    for meta_file in src_dir.iterdir():
        if "model.safetensors.index.json" in meta_file.name:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            new_weight_map = {}
            for k, v in meta_data["weight_map"].items():
                if "weight_scale" in k:
                    continue

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


def load_state_dict_by_pats(model_dir, layer_idx_pat, load_shard=True):
    with open(model_dir / "model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_weights = defaultdict(list)
    for name in weight_map.keys():
        if "_scale" not in name and layer_idx_pat.match(name):
            shard_file = weight_map[name]
            shard_to_weights[shard_file].append(name)

    state_dict = {}
    if load_shard:
        for shard_file, weights in shard_to_weights.items():
            shard_path = model_dir / shard_file
            shard_state_dict = load_file(shard_path, device="cpu")
            for name in weights:
                state_dict[name] = shard_state_dict[name]
    return state_dict, shard_to_weights


def replace_layers(src_dir, ref_dir, layer_idx_pat):
    _, shard_to_weights = load_state_dict_by_pats(
        src_dir, layer_idx_pat, load_shard=False
    )
    ref_state_dict, _ = load_state_dict_by_pats(ref_dir, layer_idx_pat, load_shard=True)

    for shard_file, weights in shard_to_weights.items():
        shard_path = src_dir / shard_file
        src_state_dict = load_file(shard_path, device="cpu")
        for name in weights:
            src_state_dict[name] = ref_state_dict[name]
        os.remove(shard_path)
        save_file(src_state_dict, shard_path)


if __name__ == "__main__":
    layer_idx_pat = re.compile(r"model\.layers\.61\.")
    src_dir = Path(
        "checkpoints/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp.int8_fake_int4_v2"
    )
    ref_dir = Path(
        "checkpoints/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp"
    )

    replace_layers(src_dir, ref_dir, layer_idx_pat)
