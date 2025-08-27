"""
compare fake_quant_model and dequant_model
"""

import argparse
from pathlib import Path
from safetensors.torch import load_file
import torch


def dequant(param, scale, zeros=0):
    assert param.ndim == scale.ndim == 2
    assert param.shape[0] == scale.shape[0]
    assert param.shape[1] % scale.shape[1] == 0
    group_size = param.shape[1] // scale.shape[1]
    scale = scale.repeat_interleave(group_size, dim=1)
    return (param * scale) + zeros


if __name__ == "__main__":
    root = Path(
        "checkpoints/Qwen3-1.7B/tesseraq/quarot_awq_tesseraq_w4a16g64_sym_s10_dynamic_c4_noclip_seq8k"
    )

    state_fake = load_file(root / "fake_quant_model" / "model.safetensors")
    state_quant = load_file(root / "vllm_quant_model" / "model.safetensors")
    state_dequant = load_file(root / "dequant_model" / "model.safetensors")

    for k, v_dequant in state_dequant.items():
        v_fake = state_fake[k]
        v_quant = state_quant[k]
        if not torch.allclose(v_fake, v_dequant):
            d_fake_dequant = (v_fake - v_dequant).abs()
            print(
                f"{k} not equal, diff max: {d_fake_dequant.max()}, mean: {d_fake_dequant.mean()}"
            )
            quant_weight = state_quant[k]
            quant_scale = state_quant[k + "_scale"]
            redequant = dequant(quant_weight, quant_scale)
            print()

    print("done")
