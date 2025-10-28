from safetensors.torch import load_file
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig
import torch
import json
import argparse


def check_shared_scales(state_dict, num_hidden_layers):
    print("\n-----------------------------------------------")
    print("start checking shared scales...")
    for layer in trange(num_hidden_layers):
        for scale_name in ("weight_global_scale", "input_global_scale"):
            # qkv
            q_scale_key = f"model.layers.{layer}.self_attn.q_proj.{scale_name}"
            q_scale = state_dict[q_scale_key]
            k_scale = state_dict[q_scale_key.replace("q_proj", "k_proj")]
            v_scale = state_dict[q_scale_key.replace("q_proj", "v_proj")]
            assert q_scale == k_scale == v_scale
            # print(f"{q_scale_key} is the same")

            # up/gate
            up_scale_key = f"model.layers.{layer}.mlp.up_proj.{scale_name}"
            up_scale = state_dict[up_scale_key]
            gate_scale = state_dict[up_scale_key.replace("up_proj", "gate_proj")]
            assert up_scale == gate_scale
            # print(f"{up_scale_key} is the same")

    print("check shared scales done")


def check_dtype(state_dict):
    print("\n-----------------------------------------------")
    print("start checking dtype...")
    for name, weight in state_dict.items():
        if "embed" in name or "head" in name or "norm" in name:
            print(f"{name}: {weight.dtype}")
            continue

        if ".bias" in name:
            print(f"{name}: {weight.dtype}")
            continue

        if "weight_packed" in name:  # packed nvfp4
            assert weight.dtype == torch.uint8, (
                f"name: {name}, expect uint8, but got {weight.dtype}"
            )
        elif "weight_scale" in name:
            assert weight.dtype == torch.float8_e4m3fn, (
                f"name: {name}, expect float8_e4m3fn, but got {weight.dtype}"
            )
        elif "global_scale" in name:
            assert weight.dtype == torch.float32, (
                f"name: {name}, expect fp32, but got {weight.dtype}"
            )
        else:
            raise NotImplementedError(f"{name} is not supported")
    print("check dtype done")


def check_quant_group_completeness(state_dict, require_input_global_scale):
    print("\n-----------------------------------------------")
    print("start checking quant group completeness...")

    required_suffixes = (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
    )
    if require_input_global_scale:
        required_suffixes = required_suffixes + ("input_global_scale",)

    # base_name -> set(found_suffixes)
    base_to_found = {}
    for name in state_dict.keys():
        # Only consider the four quantization-related suffixes
        for suffix in required_suffixes:
            if name.endswith(suffix):
                base = name.rsplit(".", 1)[0]
                found = base_to_found.get(base)
                if found is None:
                    found = set()
                    base_to_found[base] = found
                found.add(suffix)
                break

    # Validate that each base either has all 4 or none
    for base, found in base_to_found.items():
        if len(found) == 0:
            continue
        if len(found) != len(required_suffixes):
            missing = [s for s in required_suffixes if s not in found]
            raise AssertionError(
                f"Quant group incomplete for '{base}': found {sorted(found)}, missing {missing}"
            )

    # If input scales are not required by config, assert they do not exist at all
    if not require_input_global_scale:
        for name in state_dict.keys():
            if name.endswith("input_global_scale"):
                raise AssertionError(
                    f"Found unexpected input_global_scale '{name}' while config has no input_activations"
                )

    print("check quant group completeness done")


def _should_require_input_global_scale(model_dir: Path) -> bool:
    cfg_path = model_dir / "config.json"

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    group0 = cfg["quantization_config"]["config_groups"]["group_0"]

    return "input_activations" in group0 and group0["input_activations"] is not None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_dir = Path(args.model_dir)

    state_dict = {}
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor, device="cpu"))

    cfg = AutoConfig.from_pretrained(model_dir)
    check_shared_scales(state_dict, cfg.num_hidden_layers)
    require_input_scale = _should_require_input_global_scale(model_dir)
    check_quant_group_completeness(state_dict, require_input_scale)
    check_dtype(state_dict)
    # for layer in trange(cfg.num_hidden_layers):

    print("\nAll check done")
