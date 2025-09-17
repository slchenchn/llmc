"""
NOTE: untested
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
FLOAT_TO_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]


def unpack_fp4_from_uint8(a: torch.Tensor, m: int, n: int, dtype) -> torch.Tensor:
    """
    Unpacks uint8 values into fp4. Each uint8 consists of two fp4 values
    (i.e. first four bits correspond to one fp4 value, last four corresond to a consecutive
    fp4 value). The bits represent an index, which are mapped to an fp4 value.

    :param a: tensor to unpack
    :param m: original dim 0 size of the unpacked tensor
    :param n: original dim 1 size of the unpacked tensor
    :param dtype: dense dtype to cast the unpacked tensor to
    """
    assert a.dtype == torch.uint8

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n).to(dtype=dtype)


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor with values in the fp4 range into uint8.
    As there are 16 valid fp4 values, two fp4 values can be
    packed into one uint8. Each fp4 value is mapped to its
    particular index (e.g. 0.5 is mapped to index 1, 6.0 is mapped
    to index 7) which is then represented using 4 bits. Consecutive
    pairs of 4 bits are then packed into an uint8.

    :param x: tensor to pack
    returns: a packed tensor in uint8
    """

    m, n = x.shape
    device = x.device

    # Create lookup table for FP4 values to indices
    # Map the absolute values to 0-7 indices
    kE2M1 = torch.tensor(FLOAT_TO_E2M1, device=device, dtype=x.dtype)

    # Find closest valid FP4 value index for each element
    abs_x = torch.abs(x)
    abs_indices = torch.zeros_like(abs_x, dtype=torch.long)
    for i, val in enumerate(kE2M1):
        abs_indices = torch.where(torch.isclose(abs_x, val), i, abs_indices)

    # Apply sign bit (bit 3) to get final 4-bit representation
    indices = abs_indices + (torch.signbit(x) << 3).to(torch.long)

    # Reshape to prepare for packing pairs of values
    indices = indices.reshape(-1)

    # Handle odd length by padding if necessary
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.long, device=device)])

    # Reshape to pair consecutive elements
    indices = indices.reshape(-1, 2)

    # Pack pairs of 4-bit values into 8-bit values
    packed = (indices[:, 0] | (indices[:, 1] << 4)).to(torch.uint8)

    return packed.reshape(m, n // 2)


def parse_compressed_tensor(name, tensors_quant):
    m, n = tensors_quant[f"{name}.weight_packed"].shape
    n = n * 2
    fp4 = unpack_fp4_from_uint8(
        tensors_quant[f"{name}.weight_packed"], m, n, torch.float32
    )
    global_scale = tensors_quant[f"{name}.weight_global_scale"]
    scale = tensors_quant[f"{name}.weight_scale"]
    scale = scale.to(global_scale.dtype) / global_scale
    dequant = fp4.reshape(fp4.shape[0], fp4.shape[1] // 16, 16) * scale.unsqueeze(-1)
    dequant = dequant.reshape(m, n).to(torch.float16)
    return dequant


def main():
    parser = argparse.ArgumentParser(description="Dequantize a model.")
    parser.add_argument(
        "--quanted_model_id",
        type=str,
        help="checkpoints/Qwen2.5-3B-Instruct/RTN/nvfp4_w4a16/vllm_nvfp4_quant_model",
        default="",
    )
    parser.add_argument(
        "--ori_model_id",
        type=str,
        help="Path to the original model directory to copy non-safetensor files from.",
        default="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen2.5-3B-Instruct",
    )
    args = parser.parse_args()

    quanted_model_path = Path(args.quanted_model_id)
    ori_model_path = Path(args.ori_model_id)
    output_dir = Path(args.quanted_model_id + "_dequant")

    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying non-safetensor files from {ori_model_path}...")
    ori_files = list(ori_model_path.glob("*"))
    if not ori_files:
        print(f"Warning: No files found in {ori_model_path}")

    for file_path in ori_files:
        if file_path.is_dir():
            continue
        if file_path.suffix != ".safetensors":
            print(f"  Copying {file_path.name}")
            shutil.copy(file_path, output_dir)

    index_json_path = quanted_model_path / "model.safetensors.index.json"
    if index_json_path.exists():
        print(f"  Copying {index_json_path.name}")
        shutil.copy(index_json_path, output_dir)

    quanted_safetensors = sorted(quanted_model_path.glob("*.safetensors"))
    if not quanted_safetensors:
        print(f"Warning: No .safetensors files found in {quanted_model_path}")
        return

    print(f"Found {len(quanted_safetensors)} safetensors files to dequantize.")

    for quant_path in quanted_safetensors:
        print(f"Processing {quant_path.name}...")
        tensors_quant = load_file(quant_path, device="cpu")
        tensors_dequant = {}

        processed_keys = set()
        quantized_weight_keys = [
            k for k in tensors_quant.keys() if k.endswith(".weight_packed")
        ]

        for packed_key in quantized_weight_keys:
            name = packed_key.rsplit(".weight_packed", 1)[0]
            if (
                f"{name}.weight_global_scale" in tensors_quant
                and f"{name}.weight_scale" in tensors_quant
            ):
                print(f"  Dequantizing {name}.weight")
                dequant_tensor = parse_compressed_tensor(name, tensors_quant)
                tensors_dequant[f"{name}.weight"] = dequant_tensor

                processed_keys.add(packed_key)
                processed_keys.add(f"{name}.weight_global_scale")
                processed_keys.add(f"{name}.weight_scale")
            else:
                print(
                    f"  Warning: Missing scales for {packed_key}, skipping dequantization."
                )

        for key, tensor in tensors_quant.items():
            if key not in processed_keys:
                tensors_dequant[key] = tensor

        output_path = output_dir / quant_path.name
        print(f"  Saving dequantized tensors to {output_path.name}...")
        save_file(tensors_dequant, output_path)

    print("Dequantization finished successfully.")


if __name__ == "__main__":
    main()
