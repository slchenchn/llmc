from safetensors.torch import load_file
from tqdm import tqdm, trange


def weights(state_dict):
    for name, weight in state_dict.items():
        if "embed" in name or "head" in name:
            continue
        
        if weight.ndim != 2:
            continue

        

if __name__ == "__main__":
    safetensor = "checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a4_sgs/vllm_nvfp4_quant_model/model.safetensors"

    state_dict = load_file(safetensor)
    weights(state_dict)
    # for layer in trange(36):
    #     q_a_scale_key = f"model.layers.{layer}.self_attn.q_proj.input_global_scale"
    #     q_a_scale = state_dict[q_a_scale_key]
    #     k_a_scale = state_dict[q_a_scale_key.replace("q_proj", "k_proj")]
    #     v_a_scale = state_dict[q_a_scale_key.replace("q_proj", "v_proj")]

    #     assert q_a_scale == k_a_scale == v_a_scale

    #     up_a_scale_key = f"model.layers.{layer}.mlp.up_proj.input_global_scale"
    #     up_a_scale = state_dict[up_a_scale_key]
    #     gate_a_scale = state_dict[up_a_scale_key.replace("up_proj", "gate_proj")]

    #     assert up_a_scale == gate_a_scale

    print("check done")
