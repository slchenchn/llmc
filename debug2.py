import torch

import argparse
import json
import shutil
from pathlib import Path

import torch
from natsort import natsorted
from safetensors.torch import load_file, save_file
from tqdm import tqdm


p = Path(
    "checkpoints/Qwen3-32B/gptq/quarot_gptq_w4a8_sym_dynamic/vllm_quant_model/model-00004-of-00007.safetensors"
)
state_dict = load_file(p)
key = "model.layers.32.self_attn.v_proj.weight"
weight = state_dict[key]
print()
