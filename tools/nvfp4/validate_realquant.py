from safetensors.torch import load_file
from tqdm import tqdm, trange
from pathlib import Path
from transformers import AutoConfig

if __name__ == "__main__":
    model_dir = Path("checkpoints/debug/vllm_nvfp4_quant_model")

    state_dict = {}
    for safetensor in model_dir.glob("*.safetensors"):
        state_dict.update(load_file(safetensor, device="cpu"))

    cfg = AutoConfig.from_pretrained(model_dir)
    # for layer in trange(cfg.num_hidden_layers):
    for layer in range(cfg.num_hidden_layers):
        for scale_name in ("weight_global_scale", "input_global_scale"):
            # qkv
            q_scale_key = f"model.layers.{layer}.self_attn.q_proj.{scale_name}"
            q_scale = state_dict[q_scale_key]
            k_scale = state_dict[q_scale_key.replace("q_proj", "k_proj")]
            v_scale = state_dict[q_scale_key.replace("q_proj", "v_proj")]
            assert q_scale == k_scale == v_scale
            print(f"{q_scale_key} is the same")

            # up/gate
            up_scale_key = f"model.layers.{layer}.mlp.up_proj.{scale_name}"
            up_scale = state_dict[up_scale_key]
            gate_scale = state_dict[up_scale_key.replace("up_proj", "gate_proj")]
            assert up_scale == gate_scale
            print(f"{up_scale_key} is the same")

    print("check done")
