from pathlib import Path
import torch
from safetensors.torch import load_file
from dequant_nvfp4 import unpack_fp4_from_uint8


ref_path = Path(
    "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct/model-00001-of-00002.safetensors"
)
# root = Path("checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16")

# safetensors = list(root.rglob("model.safetensors"))
# safetensors = [
#     s
#     for s in safetensors
#     if "dequant" not in s.parent.name and ".bk" not in s.parent.name
# ]"
safetensors = [
    "checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16_sgs/vllm_nvfp4_quant_model/model.safetensors",
    "checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16/llmcompressor_vllm_nvfp4_quant_model/model.safetensors",
]
print(f"model_1: {safetensors[0]}")
print(f"model_2: {safetensors[1]}")
print(f"ref_model: {ref_path}")
model_1 = load_file(safetensors[0])
model_2 = load_file(safetensors[1])
ref_model = load_file(ref_path)

keys_1 = set(model_1.keys())
keys_2 = set(model_2.keys())
print(f"{len(keys_1)=}")
print(f"{len(keys_2)=}")
print(f"keys that only in model_1: {sorted(keys_1 - keys_2)}")
print(f"keys that only in model_2: {sorted(keys_2 - keys_1)}")


for key1, param1 in model_1.items():
    param2 = model_2[key1]
    if param1.numel() == param2.numel() == 1:
        ...
    else:
        assert param1.shape == param2.shape
    assert param1.dtype == param2.dtype
    # print(f"{key1}: dtype={param1.dtype}, shape={param1.shape}")

    if ".bias" in key1:
        if param1.equal(param2):
            continue
            print(f"{key1} is the same")
        else:
            raise ValueError(f"{key1} is different")
    elif "norm" in key1:
        if param1.equal(param2):
            continue
            print(f"{key1} is the same")
        else:
            raise ValueError(f"{key1} is different")
    elif "weight_global_scale" in key1:  # global scale
        param1 = param1.float()
        param2 = param2.float()
        if param1.equal(param2):
            continue
            print(f"{key1} is the same")
        else:
            raise ValueError(f"{key1} is different")
    elif "weight_scale" in key1:  # local scales
        param1 = param1.float()
        param2 = param2.float()

        diff = param1 - param2
        abs_diff = diff.abs()
        if abs_diff.max() <= 32:  # fp8's tolerance
            continue
            print(f"{key1} is the same")
        else:
            top_diff_values, top_diff_flat_indices = torch.topk(
                abs_diff.flatten(), min(10, abs_diff.numel())
            )
            top_diff_indices = torch.stack(
                torch.unravel_index(top_diff_flat_indices, abs_diff.shape), dim=-1
            )

            print(
                f"\n{key1} has differences at indices (top 10 by absolute difference):"
            )
            for idx in range(top_diff_indices.shape[0]):
                tensor_idx = top_diff_indices[idx]
                param1_val = param1[*tensor_idx].item()
                param2_val = param2[*tensor_idx].item()
                diff_val = diff[*tensor_idx].item()
                print(
                    f"  Index {tensor_idx.tolist()}: param1={param1_val:.2f}, param2={param2_val:.2f}, diff={diff_val:.2f}"
                )

            raise ValueError(f"{key1} is different")
    else:
        continue
        if not torch.allclose(param1, param2):
            global_scale1 = param1
            local_scales1 = model_1[key1.replace("_global_scale", "_scale")]
            weight1 = model_1[key1.replace("_global_scale", "_packed")]
            m, n = weight1.shape
            dweight1 = unpack_fp4_from_uint8(weight1, m, n * 2, torch.float)

            global_scale2 = param2
            local_scales2 = model_2[key1.replace("_global_scale", "_scale")]
            weight2 = model_2[key1.replace("_global_scale", "_packed")]
            dweight2 = unpack_fp4_from_uint8(weight2, m, n * 2, torch.float)

            ref_key = key1.replace("_global_scale", "")
            ref_param = ref_model[ref_key]
            print()


print("Models are the same")
