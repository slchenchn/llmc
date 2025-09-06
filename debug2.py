import torch

import argparse
import json
import shutil
from pathlib import Path

import torch
from natsort import natsorted
from safetensors.torch import load_file, save_file
from tqdm import tqdm


p = Path("checkpoints/Qwen2.5-3B-Instruct/gptqv2/sym_w4a8_olrotate/vllm_quant_model/model.safetensors")
state_dict = load_file(p)

for weight_key in [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.layers.0.self_attn.v_proj.weight",
]:
    if weight_key not in state_dict:
        continue
    print("\n" + weight_key)
    weight = state_dict[weight_key]
    if weight.dtype == torch.int8:
        print(f"{weight.unique().shape=}")
    else:
        print(f"{weight.dtype=}")

    scale_key = weight_key + "_scale"
    if scale_key in state_dict:
        scale = state_dict[scale_key]
        print(f"{scale.dtype=}")
