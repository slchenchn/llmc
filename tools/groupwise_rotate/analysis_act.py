"""
Analyze quantization errors for activation values.
"""

import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

try:
    from .utils import get_quantizers_for_activations, HadamardTransform, plot_quantization_errors
except ImportError:
    from utils import get_quantizers_for_activations, HadamardTransform, plot_quantization_errors


def load_activations(act_dir):
    """Load activation values from pickle files."""
    activations = {}
    act_dir = Path(act_dir)

    for act_file in act_dir.glob("*_input.pkl"):
        # Extract layer name from filename
        # e.g., "model_layers_0_mlp_down_proj_input.pkl" -> "model.layers.0.mlp.down_proj"
        filename = act_file.stem  # remove .pkl extension
        layer_name = filename.replace("_input", "").replace("_", ".")

        with open(act_file, 'rb') as f:
            act = pickle.load(f)
        activations[layer_name] = act.float()

    return activations


def activations(act_dir, save_dir):
    """Test different quantization and rotation combinations on activations."""
    # Load activations
    print(f"Loading activations from {act_dir}")
    activations_dict = load_activations(act_dir)

    quantizers = get_quantizers_for_activations()
    rotater_group = HadamardTransform(group_size=16)

    # Store errors for each condition
    error_data = defaultdict(list)
    layer_info = []

    for layer_name, activation in tqdm(activations_dict.items(), desc="Processing activations"):
        # Move to GPU
        activation = activation.cuda()

        # Extract layer index from name for consistent ordering
        # e.g., "model.layers.0.mlp.down_proj" -> 0
        try:
            layer_idx = int(layer_name.split('.')[2])
        except (IndexError, ValueError):
            layer_idx = 0

        layer_info.append(layer_name)

        # Get the last dimension size (hidden size)
        hidden_size = activation.shape[-1]

        # For activations, we use group size 16 regardless of hidden size
        # Skip if too small
        if hidden_size < 16:
            continue

        for quantizer_name, quantizer in quantizers.items():
            rotater_channel = HadamardTransform(group_size=hidden_size)

            # For Original activations, only use Group rotater to avoid duplication
            for rotater_name, rotater in [
                ("Group", rotater_group),
                ("Channel", rotater_channel),
            ]:
                rot_act = rotater(activation)

                # Test both original and rotated activations
                for act_type, test_act in [
                    ("Original", activation),
                    ("Rotated", rot_act),
                ]:
                    # Skip Channel_Original to avoid duplication with Group_Original
                    if rotater_name == "Channel" and act_type == "Original":
                        continue

                    qdq = quantizer.fake_quant_act_dynamic(test_act, args={})
                    err = F.mse_loss(qdq, test_act).item()

                    if act_type == "Original":
                        condition = f"{quantizer_name}_{act_type}"
                    else:
                        condition = f"{quantizer_name}_{rotater_name}"
                    error_data[condition].append(err)

    # Save error data
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return error_data, layer_info


if __name__ == "__main__":
    act_dir = Path("figs/group_rotate/act/picked")
    save_dir = Path("figs/group_rotate/act")

    error_data, layer_info = activations(act_dir, save_dir)
    plot_quantization_errors(error_data, layer_info, save_dir, title_prefix="Activation Quantization")

    print("Activation analysis completed!")
