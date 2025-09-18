from safetensors.torch import load_file
from tqdm import tqdm, trange
from pathlib import Path
from transformers import AutoConfig
import torch


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
            continue

        if ".bias" in name:
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


if __name__ == "__main__":
    model_dir = Path(
        # "checkpoints/Qwen2.5-3B-Instruct/gptqv3/nvfp4_w4a4_sgs.new/vllm_nvfp4_quant_model"
        "checkpoints/debug/vllm_nvfp4_quant_model"
    )

    state_dict = {}
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor, device="cpu"))

    cfg = AutoConfig.from_pretrained(model_dir)
    check_shared_scales(state_dict, cfg.num_hidden_layers)
    check_dtype(state_dict)
    # for layer in trange(cfg.num_hidden_layers):

    print("check done")
