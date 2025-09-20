"""
Analyze quantization errors for activation values (merged version).

This script combines the functionality of get_acts.py and analysis_act.py
to collect activations and analyze quantization errors in a single run
without intermediate file I/O.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import random

import sys

sys.path.append("/nfs/FM/chenshuailin/code/llmc")

try:
    from .utils import (
        get_quantizers_for_activations,
        HadamardTransform,
        plot_quantization_errors,
    )
except ImportError:
    from utils import (
        get_quantizers_for_activations,
        HadamardTransform,
        plot_quantization_errors,
    )


# Copied and modified from get_acts.py
def hook_fn(module, input, output, layer_name, activations_dict):
    """Hook function to capture input activations"""
    if isinstance(input, tuple):
        input = input[0]
    activations_dict[layer_name].append(input.detach().cpu())


def wikitext2_gptq(calib_dataset, tokenizer, n_samples, seq_len, prefix_token_ids=None):
    assert prefix_token_ids is None, (
        "prefix_token_ids is not supported for wikitext2_gptq yet"
    )
    trainenc = tokenizer("\n\n".join(calib_dataset["text"]), return_tensors="pt")
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        samples.append(inp)
    return samples


def get_down_proj_activations(model_path, data_dir, n_samples=32, seq_len=2048):
    """Get input activations for all down_proj layers (modified to return in memory)"""

    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Dictionary to store activations
    activations = defaultdict(list)

    # Register hooks for all down_proj layers
    hooks = []
    for name, module in model.named_modules():
        if name.endswith("down_proj"):
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: hook_fn(m, inp, out, name, activations)
            )
            hooks.append(hook)
            print(f"Registered hook for {name}")

    # Prepare input data
    data = load_from_disk(data_dir)
    inputs = wikitext2_gptq(data, tokenizer, n_samples, seq_len)

    # Run forward pass
    with torch.no_grad():
        for cur_inp in tqdm(inputs, desc="Collecting activations"):
            _ = model(cur_inp.to(model.device))

    # Process activations (concatenate and return)
    processed_activations = {}
    for layer_name, activation_list in activations.items():
        activation = torch.cat(activation_list, dim=0).detach().cpu().bfloat16()
        processed_activations[layer_name] = activation
        print(f"Collected {layer_name} activation with shape {activation.shape}")

    # Remove hooks
    for hook in hooks:
        hook.remove()
    print(f"Total activations collected: {len(processed_activations)}")

    return processed_activations


# Modified from analysis_act.py
def analyze_activation_quantization(activations_dict, save_dir):
    """Test different quantization and rotation combinations on activations."""
    # Get quantizers
    quantizers = get_quantizers_for_activations()
    rotater_group = HadamardTransform(group_size=16)

    # Store errors for each condition
    error_data = defaultdict(list)
    layer_info = []

    for layer_name, activation in tqdm(
        activations_dict.items(), desc="Processing activations"
    ):
        # Move to GPU
        activation = activation.cuda()

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
                # Process in smaller batches to avoid OOM
                batch_size = 32  # Process 8 samples at a time
                batch_errors = []

                for start_idx in range(0, activation.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, activation.shape[0])
                    batch_act = activation[start_idx:end_idx]

                    rot_act_batch = rotater(batch_act)

                    # Test both original and rotated activations
                    for act_type, test_act in [
                        ("Original", batch_act),
                        ("Rotated", rot_act_batch),
                    ]:
                        # Skip Channel_Original to avoid duplication with Group_Original
                        if rotater_name == "Channel" and act_type == "Original":
                            continue

                        qdq = quantizer.fake_quant_act_dynamic(test_act, args={})
                        err = F.mse_loss(qdq, test_act).item()
                        batch_errors.append(err)

                # Average errors across batches for this rotater configuration
                avg_err = sum(batch_errors) / len(batch_errors)

                # Since we process both Original and Rotated in batches, we need to handle them separately
                # For Group rotater: we have both Original and Rotated
                # For Channel rotater: we only have Rotated (skip Original to avoid duplication)

                if rotater_name == "Group":
                    # Process both Original and Rotated for Group
                    original_errors = [
                        err for i, err in enumerate(batch_errors) if i % 2 == 0
                    ]
                    rotated_errors = [
                        err for i, err in enumerate(batch_errors) if i % 2 == 1
                    ]

                    if original_errors:
                        error_data[f"{quantizer_name}_Original"].append(
                            sum(original_errors) / len(original_errors)
                        )
                    if rotated_errors:
                        error_data[f"{quantizer_name}_Group"].append(
                            sum(rotated_errors) / len(rotated_errors)
                        )
                else:  # Channel rotater
                    # Only process Rotated (no Original to avoid duplication)
                    error_data[f"{quantizer_name}_Channel"].append(avg_err)

    # Save error data
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return error_data, layer_info


if __name__ == "__main__":
    model_path = "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct/"
    data_dir = "data/wikitext2"
    save_dir = "figs/group_rotate/act_v2"

    # Get activations directly in memory
    print("Collecting activations...")
    activations_dict = get_down_proj_activations(model_path, data_dir)

    # Analyze quantization errors
    print("Analyzing quantization errors...")
    error_data, layer_info = analyze_activation_quantization(activations_dict, save_dir)

    # Generate plots and summary
    plot_quantization_errors(
        error_data, layer_info, Path(save_dir), title_prefix="Activation Quantization"
    )

    print("Activation analysis completed!")
