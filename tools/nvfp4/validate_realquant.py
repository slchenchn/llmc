from safetensors.torch import load_file
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig
import torch
import json
import argparse


def check_shared_scales(state_dict, num_hidden_layers, require_input_scale):
    print("\n-----------------------------------------------")
    print("start checking shared scales...")
    for layer in trange(num_hidden_layers):
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
            up_scale_key = f"model.layers.{layer}.mlp.up_proj.{scale_name}"
            up_scale = state_dict[up_scale_key]
            gate_scale = state_dict[up_scale_key.replace("up_proj", "gate_proj")]
            assert up_scale == gate_scale, (
                f"up_scale ({up_scale}) != gate_scale ({gate_scale})"
            )
            # print(f"{up_scale_key} is the same")

    print("check shared scales done")


def check_dtype(state_dict):
    print("\n-----------------------------------------------")
    print("start checking dtype...")
    for name, weight in state_dict.items():
        if "embed" in name or "head" in name or "norm" in name:
            # print(f"{name}: {weight.dtype}")
            continue

        if ".bias" in name:
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

    # Collect all weight_global_scale values
    weight_scales = []
    for name, value in state_dict.items():
        if name.endswith("weight_global_scale"):
            # Handle both scalar and tensor cases
            if value.numel() == 1:
                weight_scales.append(value.item())
            else:
                weight_scales.extend(value.flatten().tolist())

    if weight_scales:
        weight_tensor = torch.tensor(weight_scales)
        print("\nweight_global_scale statistics:")
        print(f"  Max: {weight_tensor.max().item():.6f}")
        print(f"  Min: {weight_tensor.min().item():.6f}")
        print(f"  Mean: {weight_tensor.mean().item():.6f}")
        print(f"  Std: {weight_tensor.std().item():.6f}")
        print(f"  Count: {len(weight_scales)}")

    # Collect all weight_scale (local_scale) values
    local_scales = []
    for name, value in state_dict.items():
        if name.endswith("weight_scale"):
            # Handle both scalar and tensor cases
            if value.numel() == 1:
                local_scales.append(value.float().item())
            else:
                local_scales.extend(value.float().flatten().tolist())

    if local_scales:
        local_tensor = torch.tensor(local_scales)
        print("\nlocal_scale (weight_scale) statistics:")
        print(f"  Max: {local_tensor.max().item():.6f}")
        print(f"  Min: {local_tensor.min().item():.6f}")
        print(f"  Mean: {local_tensor.mean().item():.6f}")
        print(f"  Std: {local_tensor.std().item():.6f}")
        print(f"  Count: {len(local_scales)}")

    # Collect all input_global_scale values if required
    if require_input_global_scale:
        input_scales = []
        for name, value in state_dict.items():
            if name.endswith("input_global_scale"):
                # Handle both scalar and tensor cases
                if value.numel() == 1:
                    input_scales.append(value.item())
                else:
                    input_scales.extend(value.flatten().tolist())

        if input_scales:
            input_tensor = torch.tensor(input_scales)
            print("\ninput_global_scale statistics:")
            print(f"  Max: {input_tensor.max().item():.6f}")
            print(f"  Min: {input_tensor.min().item():.6f}")
            print(f"  Mean: {input_tensor.mean().item():.6f}")
            print(f"  Std: {input_tensor.std().item():.6f}")
            print(f"  Count: {len(input_scales)}")

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
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor, device="cpu"))

    cfg = AutoConfig.from_pretrained(model_dir)
    require_input_scale = _should_require_input_global_scale(model_dir)
    check_shared_scales(state_dict, cfg.num_hidden_layers, require_input_scale)
    check_quant_group_completeness(state_dict, require_input_scale)
    check_dtype(state_dict)
    # for layer in trange(cfg.num_hidden_layers):

    # Print scale statistics after all checks pass
    print_scale_statistics(state_dict, require_input_scale)

    print("\nAll check done")
