from tqdm import tqdm, trange
from llmc.compression.quantization.module_utils import Rotater
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

try:
    from .utils import load_state_dict, is_power_of_2, HadamardTransform, get_quantizers, plot_quantization_errors
except ImportError:
    from utils import load_state_dict, is_power_of_2, HadamardTransform, get_quantizers, plot_quantization_errors


def weights(state_dict, save_dir):
    """Test different quantization and rotation combinations and collect errors."""
    quantizers = get_quantizers()
    rotater_group = HadamardTransform(group_size=16)

    # Store errors for each condition
    error_data = defaultdict(list)
    layer_info = []

    for name, weight in tqdm(state_dict.items(), desc="Processing weights"):
        if "embed" in name or "head" in name:
            continue

        if weight.ndim != 2:
            continue

        group_size = weight.shape[1]
        if not is_power_of_2(group_size):
            continue

        weight = weight.cuda().float()
        layer_info.append(name)

        for quantizer_name, quantizer in quantizers.items():
            rotater_channel = HadamardTransform(group_size=weight.shape[1])

            # For Original weights, only use Group rotater to avoid duplication
            for rotater_name, rotater in [
                ("Group", rotater_group),
                ("Channel", rotater_channel),
            ]:
                rot_weight = rotater(weight)

                # Test both original and rotated weights
                for weight_type, test_weight in [
                    ("Original", weight),
                    ("Rotated", rot_weight),
                ]:
                    # Skip Channel_Original to avoid duplication with Group_Original
                    if rotater_name == "Channel" and weight_type == "Original":
                        continue

                    qdq = quantizer.fake_quant_weight_dynamic(test_weight, args={})
                    err = F.mse_loss(qdq, test_weight).item()

                    if weight_type == "Original":
                        condition = f"{quantizer_name}_{weight_type}"
                    else:
                        condition = f"{quantizer_name}_{rotater_name}"
                    error_data[condition].append(err)

    # Save error data
    save_dir.mkdir(parents=True, exist_ok=True)
    error_df = pd.DataFrame(error_data)
    error_df["layer"] = layer_info[: len(error_df)]

    return error_data, layer_info




if __name__ == "__main__":
    model_dir = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct")
    save_dir = Path("figs/group_rotate/weight")

    state_dict = load_state_dict(model_dir)
    error_data, layer_info = weights(state_dict, save_dir)
    plot_quantization_errors(error_data, layer_info, save_dir)

    print("Analysis completed!")
