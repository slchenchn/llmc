import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm, trange
from transformers import AutoConfig


def check_shared_scales(state_dict, cfg, require_input_scale):
    print("\n-----------------------------------------------")
    print("start checking shared scales...")
    mlp_names = {
        "qwen3_moe": {
            "gate_proj": "model.layers.{}.mlp.experts.{}.gate_proj",
            "up_proj": "model.layers.{}.mlp.experts.{}.up_proj",
        },
        "minimax_m2": {
            "gate_proj": "model.layers.{}.block_sparse_moe.experts.{}.w1",
            "up_proj": "model.layers.{}.block_sparse_moe.experts.{}.w3",
        },
        "default": {
            "gate_proj": "model.layers.{}.mlp.gate",
            "up_proj": "model.layers.{}.mlp.up_proj",
        },
    }
    cur_mlp_names = mlp_names.get(cfg.model_type, mlp_names["default"])
    for layer in trange(cfg.num_hidden_layers):
        scale_names = ("weight_global_scale",)
        if require_input_scale:
            scale_names = scale_names + ("input_global_scale",)
        for scale_name in scale_names:
            # qkv
            q_scale_key = f"model.layers.{layer}.self_attn.q_proj.{scale_name}"
            q_scale = state_dict[q_scale_key]
            k_scale = state_dict[q_scale_key.replace("q_proj", "k_proj")]
            v_scale = state_dict[q_scale_key.replace("q_proj", "v_proj")]
            assert q_scale == k_scale == v_scale, (
                f"q_scale ({q_scale}) != k_scale ({k_scale}) != v_scale ({v_scale})"
            )
            # print(f"{q_scale_key} is the same")

            # up/gate
            n_experts = getattr(cfg, "num_experts", 1)
            if n_experts > 1:
                for i in range(n_experts):
                    up_scale_key = cur_mlp_names["up_proj"].format(layer, i)
                    gate_scale_key = cur_mlp_names["gate_proj"].format(layer, i)
                    up_scale = state_dict[f"{up_scale_key}.{scale_name}"]
                    gate_scale = state_dict[f"{gate_scale_key}.{scale_name}"]
                    assert up_scale == gate_scale, (
                        f"up_scale ({up_scale}) != gate_scale ({gate_scale})"
                    )
            else:
                up_scale_key = cur_mlp_names["up_proj"].format(layer, 0)
                gate_scale_key = cur_mlp_names["gate_proj"].format(layer, 0)
                up_scale = state_dict[f"{up_scale_key}.{scale_name}"]
                gate_scale = state_dict[f"{gate_scale_key}.{scale_name}"]
                assert up_scale == gate_scale, (
                    f"up_scale ({up_scale}) != gate_scale ({gate_scale})"
                )
            # print(f"{up_scale_key} is the same")

    print("check shared scales done")


def check_dtype(state_dict):
    print("\n-----------------------------------------------")
    print("start checking dtype...")
    for name, weight in state_dict.items():
        if "embed" in name or "head" in name or "norm" in name or ".gate." in name:
            # print(f"{name}: {weight.dtype}")
            continue

        if ".bias" in name or ".e_score_correction_bias" in name:
            # print(f"{name}: {weight.dtype}")
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


def check_scale_value(state_dict):
    print("\n-----------------------------------------------")
    print("start checking scale value...")

    for name, value in state_dict.items():
        if "scale" in name:
            # Convert to float for comparison (weight_scale is float8_e4m3fn)
            value_float = value.float()
            if not (value_float > 0).all():
                min_val = value_float.min().item()
                raise AssertionError(
                    f"Scale '{name}' contains non-positive values, min value: {min_val}"
                )
            # weight_global_scale should be greater than 100
            if "weight_global_scale" in name:
                if not (value_float > 100).all():
                    min_val = value_float.min().item()
                    raise AssertionError(
                        f"weight_global_scale '{name}' should be > 100, but min value: {min_val}"
                    )
            # input_global_scale should be greater than 1e-3
            if "input_global_scale" in name:
                if not (value_float > 1e-3).all():
                    min_val = value_float.min().item()
                    raise AssertionError(
                        f"input_global_scale '{name}' should be > 1e-3, but min value: {min_val}"
                    )

    print("check scale value done")


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


def print_scale_statistics(state_dict, require_input_global_scale):
    print("\n-----------------------------------------------")
    print("start printing scale statistics...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect all weight_global_scale values
    weight_scales = []
    for name, value in state_dict.items():
        if name.endswith("weight_global_scale"):
            weight_scales.append(value.flatten())

    if weight_scales:
        weight_tensor = torch.cat(weight_scales).to(device)
        print("\nweight_global_scale statistics:")
        print(f"  Max: {weight_tensor.max().item():.6f}")
        print(f"  Min: {weight_tensor.min().item():.6f}")
        print(f"  Mean: {weight_tensor.mean().item():.6f}")
        print(f"  Std: {weight_tensor.std().item():.6f}")
        print(f"  Count: {weight_tensor.numel()}")

    # Collect all weight_scale (local_scale) values
    local_scales = []
    for name, value in state_dict.items():
        if name.endswith("weight_scale"):
            local_scales.append(value.float().flatten())

    if local_scales:
        local_tensor = torch.cat(local_scales).to(device)
        print("\nlocal_scale (weight_scale) statistics:")
        print(f"  Max: {local_tensor.max().item():.6f}")
        print(f"  Min: {local_tensor.min().item():.6f}")
        print(f"  Mean: {local_tensor.mean().item():.6f}")
        print(f"  Std: {local_tensor.std().item():.6f}")
        print(f"  Count: {local_tensor.numel()}")

    # Collect all input_global_scale values if required
    if require_input_global_scale:
        input_scales = []
        for name, value in state_dict.items():
            if name.endswith("input_global_scale"):
                input_scales.append(value.flatten())

        if input_scales:
            input_tensor = torch.cat(input_scales).to(device)
            print("\ninput_global_scale statistics:")
            print(f"  Max: {input_tensor.max().item():.6f}")
            print(f"  Min: {input_tensor.min().item():.6f}")
            print(f"  Mean: {input_tensor.mean().item():.6f}")
            print(f"  Std: {input_tensor.std().item():.6f}")
            print(f"  Count: {input_tensor.numel()}")

    print("scale statistics printed")


def _should_require_input_global_scale(model_dir: Path) -> bool:
    cfg_path = model_dir / "config.json"

    with cfg_path.open("r") as f:
        cfg = json.load(f)

    group0 = cfg["quantization_config"]["config_groups"]["group_0"]

    return "input_activations" in group0 and group0["input_activations"] is not None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_dir = Path(args.model_dir)

    state_dict = {}
    for safetensor in tqdm(model_dir.glob("*.safetensors"), desc="loading state dicts"):
        state_dict.update(load_file(safetensor, device="cpu"))

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    require_input_scale = _should_require_input_global_scale(model_dir)
    check_shared_scales(state_dict, cfg, require_input_scale)
    check_quant_group_completeness(state_dict, require_input_scale)
    check_dtype(state_dict)
    check_scale_value(state_dict)
    # for layer in trange(cfg.num_hidden_layers):

    # Print scale statistics after all checks pass
    print_scale_statistics(state_dict, require_input_scale)

    print("\nAll check done")
