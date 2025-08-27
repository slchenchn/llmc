from pathlib import Path
import json


index_path = Path(
    "checkpoints/DeepSeek-R1/quarot/sym_w8a8_dynamic2/vllm_quant_model_mtp/model.safetensors.index.json"
)

with open(index_path, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]
for k, v in weight_map.copy().items():
    if "_scale" in k:
        weight_name = k.replace("_scale", "")
        if weight_name not in weight_map:
            print(f"Warning: {weight_name} not found in weight_map, add it")
            weight_map[weight_name] = v

index["weight_map"] = weight_map
with open(index_path, "w") as f:
    json.dump(index, f, indent=4)
