"""
将int8量化模型转换为fake_intn量化模型，也即用int8来模拟intn
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def round_int8_to_fake_intn(value: torch.Tensor, n_bits: int) -> torch.Tensor:
    scale = 2 ** (8 - n_bits)
    intn_fp = value / scale
    rounded_value = torch.round(intn_fp)
    max_val = 2 ** (n_bits - 1) - 1
    min_val = -(2 ** (n_bits - 1))
    rounded_value = torch.clamp(rounded_value, min_val, max_val).to(torch.int8) * scale
    return rounded_value


def load_state_dict(path):
    if path.name.endswith(".safetensors"):
        return load_file(path, device="cuda")
    elif path.name.endswith(".bin"):
        return torch.load(path, weights_only=True, map_location="cuda")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_state_dict(state_dict: dict, path: Path):
    if path.name.endswith(".safetensors"):
        save_file(state_dict, path)
    elif path.name.endswith(".bin"):
        torch.save(state_dict, path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def convert_model_int8_to_intn(src_model_dir: Path, dst_model_dir: Path, n_bits: int):
    dst_model_dir.mkdir(exist_ok=True, parents=True)
    for file in tqdm(src_model_dir.iterdir()):
        if file.is_dir():
            continue
        if file.suffix in [".safetensors", ".bin"]:
            state_dict = load_state_dict(file)
            for key, value in state_dict.items():
                if value.dtype == torch.int8:
                    state_dict[key] = round_int8_to_fake_intn(value, n_bits)
            save_state_dict(state_dict, dst_model_dir / file.name)
        else:
            shutil.copy(file, dst_model_dir / file.name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert int8 model to fake intn model"
    )
    parser.add_argument(
        "--n_bits", type=int, default=4, help="Target bit precision (default: 4)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    src_model_dirs = [
        "checkpoints/Qwen3-1.7B/quarot/sym_w8_a8-dynamic/vllm_quant_model",
    ]

    for src_model_dir in src_model_dirs:
        src_model_dir = Path(src_model_dir)
        dst_model_dir = src_model_dir.with_suffix(f".int8_fake_int{args.n_bits}")
        print(f"Converting {src_model_dir} to {dst_model_dir} with {args.n_bits} bits")
        convert_model_int8_to_intn(src_model_dir, dst_model_dir, args.n_bits)

    print("Done")
