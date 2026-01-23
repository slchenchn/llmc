import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm, trange
from transformers import AutoConfig

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_tokenizer import check_chat_template, TokenizerConfigError


def check_shared_scales(state_dict, cfg, require_input_scale):
    print("\n-----------------------------------------------")
    print("start checking shared scales...")
    moe_mlp_names = {
        "qwen3_moe": {
            "gate_proj": "model.layers.{}.mlp.experts.{}.gate_proj",
            "up_proj": "model.layers.{}.mlp.experts.{}.up_proj",
        },
        "minimax_m2": {
            "gate_proj": "model.layers.{}.block_sparse_moe.experts.{}.w1",
            "up_proj": "model.layers.{}.block_sparse_moe.experts.{}.w3",
        },
        "deepseek_v3": {
            "gate_proj": "model.layers.{}.mlp.experts.{}.gate_proj",
            "up_proj": "model.layers.{}.mlp.experts.{}.up_proj",
        },
    }
    dense_mlp_names = {
        "deepseek_v3": {
            "gate_proj": "model.layers.{}.mlp.gate_proj",
            "up_proj": "model.layers.{}.mlp.up_proj",
        },
        "default": {
            "gate_proj": "model.layers.{}.mlp.gate_proj",
            "up_proj": "model.layers.{}.mlp.up_proj",
        },
    }
    shared_mlp_names = {
        "deepseek_v3": {
            "gate_proj": "model.layers.{}.mlp.shared_experts.gate_proj",
            "up_proj": "model.layers.{}.mlp.shared_experts.up_proj",
        }
    }
    attn_names = {
        "deepseek_v3": {
            "q_proj": "model.layers.{}.self_attn.q_a_proj",
            "k_proj": "model.layers.{}.self_attn.kv_a_proj_with_mqa",
            "v_proj": "model.layers.{}.self_attn.kv_a_proj_with_mqa",
        },
        "default": {
            "q_proj": "model.layers.{}.self_attn.q_proj",
            "k_proj": "model.layers.{}.self_attn.k_proj",
            "v_proj": "model.layers.{}.self_attn.v_proj",
        },
    }
    cur_attn_names = attn_names.get(cfg.model_type, attn_names["default"])
    cur_moe_names = moe_mlp_names.get(cfg.model_type)
    cur_dense_names = dense_mlp_names.get(cfg.model_type, dense_mlp_names["default"])
    cur_shared_names = shared_mlp_names.get(cfg.model_type)

    def _check_scale_pair(up_key, gate_key, scale_name):
        up_scale = state_dict[f"{up_key}.{scale_name}"]
        gate_scale = state_dict[f"{gate_key}.{scale_name}"]
        assert up_scale == gate_scale, (
            f"up_scale ({up_scale}) != gate_scale ({gate_scale})"
        )

    def _get_num_experts():
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise AttributeError(
            "No expert count found; expected one of num_experts, n_routed_experts, num_local_experts"
        )

    for layer in trange(cfg.num_hidden_layers):
        has_moe = False
        if cur_moe_names is not None:
            moe_probe = (
                cur_moe_names["gate_proj"].format(layer, 0) + ".weight_global_scale"
            )
            has_moe = moe_probe in state_dict

        scale_names = ("weight_global_scale",)
        if require_input_scale:
            scale_names = scale_names + ("input_global_scale",)
        for scale_name in scale_names:
            # qkv
            q_scale_key = cur_attn_names["q_proj"].format(layer) + f".{scale_name}"
            q_scale = state_dict[q_scale_key]
            k_scale = state_dict[
                cur_attn_names["k_proj"].format(layer) + f".{scale_name}"
            ]
            v_scale = state_dict[
                cur_attn_names["v_proj"].format(layer) + f".{scale_name}"
            ]
            assert q_scale == k_scale == v_scale, (
                f"q_scale ({q_scale}) != k_scale ({k_scale}) != v_scale ({v_scale})"
            )
            # print(f"{q_scale_key} is the same")

            # up/gate
            if has_moe:
                n_experts = _get_num_experts()
                for i in range(n_experts):
                    up_scale_key = cur_moe_names["up_proj"].format(layer, i)
                    gate_scale_key = cur_moe_names["gate_proj"].format(layer, i)
                    _check_scale_pair(up_scale_key, gate_scale_key, scale_name)
                if cur_shared_names is not None:
                    shared_up_key = cur_shared_names["up_proj"].format(layer)
                    shared_gate_key = cur_shared_names["gate_proj"].format(layer)
                    if (
                        f"{shared_up_key}.{scale_name}" in state_dict
                        and f"{shared_gate_key}.{scale_name}" in state_dict
                    ):
                        _check_scale_pair(shared_up_key, shared_gate_key, scale_name)
            else:
                up_scale_key = cur_dense_names["up_proj"].format(layer)
                gate_scale_key = cur_dense_names["gate_proj"].format(layer)
                _check_scale_pair(up_scale_key, gate_scale_key, scale_name)
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

    device = "cpu"

    def _init_stats():
        return {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
        }

    def _update_stats(stats, tensor):
        values = tensor.to(device=device).float().view(-1)
        stats["count"] += values.numel()
        stats["sum"] += values.sum().item()
        stats["sum_sq"] += (values * values).sum().item()
        stats["min"] = min(stats["min"], values.min().item())
        stats["max"] = max(stats["max"], values.max().item())

    def _print_stats(title, stats):
        if stats["count"] == 0:
            return
        mean = stats["sum"] / stats["count"]
        variance = stats["sum_sq"] / stats["count"] - mean * mean
        std = variance**0.5 if variance > 0 else 0.0
        print(f"\n{title} statistics:")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std: {std:.6f}")
        print(f"  Count: {stats['count']}")

    # Collect all weight_global_scale values
    weight_stats = _init_stats()
    for name, value in state_dict.items():
        if name.endswith("weight_global_scale"):
            _update_stats(weight_stats, value)

    _print_stats("weight_global_scale", weight_stats)

    # Collect all weight_scale (local_scale) values
    local_stats = _init_stats()
    for name, value in state_dict.items():
        if name.endswith("weight_scale"):
            _update_stats(local_stats, value)

    _print_stats("local_scale (weight_scale)", local_stats)

    # Collect all input_global_scale values if required
    if require_input_global_scale:
        input_stats = _init_stats()
        for name, value in state_dict.items():
            if name.endswith("input_global_scale"):
                _update_stats(input_stats, value)

        _print_stats("input_global_scale", input_stats)

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

    # Check tokenizer config has chat_template
    print("\n-----------------------------------------------")
    print("start checking tokenizer config...")
    try:
        check_chat_template(model_dir)
        print("check tokenizer config done (chat_template exists)")
    except TokenizerConfigError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

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
