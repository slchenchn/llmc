"""
Get input activations for specified layers in a transformer model.

This script loads a Qwen2.5-3B-Instruct model, registers forward hooks on all
layers ending with the specified suffix to capture their input activations,
runs a forward pass with a sample input, and saves the activations to pickle files.
"""

import torch
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import random
from tqdm import tqdm


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


def get_down_proj_activations(model_path, data_dir, save_dir, layer_suffix="down_proj"):
    """Get input activations for all layers ending with layer_suffix"""

    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store activations
    activations = defaultdict(list)

    # Register hooks for all layers ending with layer_suffix
    hooks = []
    for name, module in model.named_modules():
        if name.endswith(layer_suffix):
            hook = module.register_forward_hook(
                lambda m, inp, out, name=name: hook_fn(m, inp, out, name, activations)
            )
            hooks.append(hook)
            print(f"Registered hook for {name}")

    # Prepare input data
    data = load_from_disk(data_dir)
    inputs = wikitext2_gptq(data, tokenizer, 32, 2048)

    # Run forward pass
    with torch.no_grad():
        for cur_inp in tqdm(inputs):
            outputs = model(cur_inp.to(model.device))

    # Save activations
    print(f"Saving activations to {save_dir}")
    for layer_name, activation in activations.items():
        activation = torch.cat(activation, dim=0).detach().cpu()
        save_path = save_dir / f"{layer_name.replace('.', '_')}_input.pt"
        torch.save(activation, save_path)
        print(
            f"Saved {layer_name} activation with shape {activation.shape} to {save_path}"
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()
    print(f"Total activations saved: {len(activations)}")


if __name__ == "__main__":
    model_path = "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct/"
    data_dir = "data/wikitext2"
    save_dir = "figs/group_rotate/act/picked"
    layer_suffix = "q_proj"

    save_dir = save_dir + f"/{layer_suffix}"
    get_down_proj_activations(model_path, data_dir, save_dir, layer_suffix)
