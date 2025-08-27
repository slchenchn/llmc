import torch

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
        default="checkpoints/Qwen2.5-3B-Instruct/gptq/quarot_gptq_w4a16_sym/fake_quant_model",
    )
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    args.model_path = Path(args.model_path)
    assert args.model_path.name == "fake_quant_model", (
        "model_path must be a fake_quant_model"
    )
    if args.output_path is None:
        args.output_path = args.model_path.parent / "fake_quant_model_cleaned"

    return args


def copy_meta_files(src_dir: Path, dst_dir: Path, pats_to_remove):
    for meta_file in src_dir.iterdir():
        if "model.safetensors.index.json" in meta_file.name:
            print("-" * 100)
            print("cleaning model.safetensors.index.json")
            print("-" * 100)
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
            for k, v in meta_data["weight_map"].copy().items():
                for pat in pats_to_remove:
                    if pat in k:
                        del meta_data["weight_map"][k]
                        print(f"removed {k}")
                        break
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


def clean_weights(src_dir, output_dir, pats_to_remove):
    print("-" * 100)
    print("cleaning weights")
    print("-" * 100)
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors = natsorted(list(src_dir.glob("*.safetensors")))
    for safetensor in safetensors:
        state_dict = load_file(safetensor, device="cuda")
        keys = list(state_dict.keys())
        for key in keys:
            for pat in pats_to_remove:
                if pat in key:
                    del state_dict[key]
                    print(f"removed {key}")
                    break
        save_file(state_dict, output_dir / safetensor.name)


if __name__ == "__main__":
    args = get_args()

    pats_to_remove = [".buf_"]
    clean_weights(args.model_path, args.output_path, pats_to_remove)
    copy_meta_files(args.model_path, args.output_path, pats_to_remove)
