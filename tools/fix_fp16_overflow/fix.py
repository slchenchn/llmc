import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from natsort import natsorted
from safetensors.torch import load_file, save_file


def is_target_weight(key: str) -> bool:
    if ("model.layers.60.mlp." in key) and (".up_proj.scales" in key):
        return True

    return False


if __name__ == "__main__":
    src_model_dir = Path(
        "/ms/FM/checkpoints/llmc/DeepSeek-R1/quarot/asym_w4a16_5/autoawq_quant_model"
    )
    dst_model_dir = src_model_dir.with_suffix(".fix-fp16-overflow")
    scale = 10

    print(f"{src_model_dir=}\n{dst_model_dir=}\n{scale=}")

    with open(src_model_dir / "model.safetensors.index.json", "r") as f:
        weight_map = json.load(f)["weight_map"]

    target_safetensors = set()
    for name, safetensor in weight_map.items():
        if is_target_weight(name):
            target_safetensors.add(safetensor)
    print(f"target_safetensors:\n{target_safetensors}")

    cnt = defaultdict(int)
    for safetensor in target_safetensors:
        print(f"\nprocessing {safetensor}")
        src_safetensor = src_model_dir / safetensor
        dst_safetensor = dst_model_dir / safetensor
        if dst_safetensor.exists():
            os.remove(dst_safetensor)

        src_state_dict = load_file(src_safetensor)
        # src_state_dict = dict(natsorted(src_state_dict.items()))
        dst_state_dict = {}
        for name, weight in src_state_dict.items():
            if is_target_weight(name):
                print(f"{name} {weight.dtype}")
                # Check if the operation might cause numerical overflow
                min_val = torch.finfo(weight.dtype).min
                if torch.any(weight < min_val * scale):
                    # print(f"WARNING: Potential underflow detected in {name}")
                    raise ValueError(f"Potential underflow detected in {name}")
                weight /= scale
                cnt["processed"] += 1
            dst_state_dict[name] = weight

        save_file(dst_state_dict, dst_safetensor)

    for src_file in src_model_dir.iterdir():
        if src_file.is_dir():
            continue

        dst_file = dst_model_dir / src_file.name
        if not dst_file.exists():
            os.symlink(src_file, dst_file)

    print(f"done")
    print(f"cnt: {cnt}")
