import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from duquant_quantizer import UniformAffineQuantizer

from llmc.compression.quantization.quarot import random_hadamard_matrix


def random_orthogonal_matrix(size, device):
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def show_statitics(input_act, output_file):

    # print("input_act:")
    print(
        f"abs max: {input_act.abs().max():,}, abs min: {input_act.abs().min():,}, abs mean: {input_act.abs().mean():,}"
    )

    per_channel_max = input_act.abs().max(dim=0)[0].float()
    per_token_max = input_act.abs().max(dim=1)[0].float()

    massive_index = torch.where(per_channel_max > 65_000)[0]
    print(f"massive_index (abs > 65,000) (len: {len(massive_index)}): {massive_index}")

    # Create figure with matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    
    # Plot per-channel max
    axes[0].plot(per_channel_max.cpu().numpy())
    axes[0].set_title("Per-Channel Maximum Absolute Values")
    axes[0].set_xlabel("Channel Index")
    axes[0].set_ylabel("Max Absolute Value")
    axes[0].set_yscale("log")
    
    # Plot per-token max
    axes[1].plot(per_token_max.cpu().numpy())
    axes[1].set_title("Per-Token Maximum Absolute Values")
    axes[1].set_xlabel("Token Index")
    axes[1].set_ylabel("Max Absolute Value")
    axes[1].set_yscale("log")
    
    # Save figure
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Plot saved to {output_file}")


def analysis_massive_outlier_lastlayer(res_path, n_try=10000):
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    input_act = res["block_61.input_layernorm"].float()
    print(f"shape: {input_act.shape}")
    show_statitics(input_act)

    # # random orthogonal
    # for i in range(n_try):
    #     Q = random_orthogonal_matrix(input_act.shape[-1], input_act.device).float()
    #     input_act_rotated = torch.matmul(input_act, Q)
    #     print("-" * 80)
    #     print(f"random orthogonal {i+1}/{n_try}")
    #     show_statitics(input_act_rotated)

    # random hadamard
    # for i in range(n_try):
    #     Q = random_hadamard_matrix(input_act.shape[-1], input_act.device).float()
    #     input_act_rotated = torch.matmul(input_act, Q)
    #     print("-" * 80)
    #     print(f"random hadamard {i+1}/{n_try}")
    #     show_statitics(input_act_rotated)
    #     input_act = input_act_rotated

    # duquant
    duquantizer = UniformAffineQuantizer(
        quant_method="duquant", max_rotation_step=256, block_size=1024
    )
    for i in range(n_try):
        input_act_duquant = duquantizer.init_duquant(input_act)
        print("-" * 80)
        print(f"duquant {i+1}/{n_try}")
        show_statitics(input_act_duquant)


def extract_act_by_pattern(state_dict, pattern):
    acts = []
    for key, value in state_dict.items():
        if re.search(pattern, key):
            if value.ndim == 3:
                value = value.squeeze()
                assert value.ndim == 2
            acts.append(value)
    return torch.cat(acts, dim=0)


def analysis_massive_outlier_penultimate_layer(res_path, n_try=10):
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    inspect_key = "down_proj"
    act = extract_act_by_pattern(res, inspect_key).float()
    print(f"shape: {act.shape}")
    save_dir = res_path.parent / "try_rotate"
    save_dir.mkdir(parents=True, exist_ok=True)
    show_statitics(act, save_dir / f"{Path(res_path).stem}_original.png")

    # random orthogonal
    for i in range(n_try):
        Q = random_orthogonal_matrix(act.shape[-1], act.device).float()
        act_rotated = torch.matmul(act, Q)
        print("-" * 80)
        print(f"random orthogonal {i+1}/{n_try}")
        show_statitics(
            act_rotated, save_dir / f"{Path(res_path).stem}_random_orthogonal_{i}.png"
        )

    # random hadamard
    for i in range(n_try):
        Q = random_hadamard_matrix(act.shape[-1], act.device).float()
        act_rotated = torch.matmul(act, Q)
        print("-" * 80)
        print(f"random hadamard {i+1}/{n_try}")
        show_statitics(
            act_rotated, save_dir / f"{Path(res_path).stem}_random_hadamard_{i}.png"
        )

    # duquant
    duquantizer = UniformAffineQuantizer(
        quant_method="duquant", max_rotation_step=32, block_size=512
    )
    for i in range(n_try):
        act_duquant = duquantizer.init_duquant(act)
        print("-" * 80)
        print(f"duquant {i+1}/{n_try}")
        show_statitics(act_duquant, save_dir / f"{Path(res_path).stem}_duquant_{i}.png")


def analysis_massive_outlier_penultimate_layer_2(res_path, n_try=10):
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    inspect_key = "down_proj"
    act = None
    for key, value in res.items():
        if re.search(inspect_key, key):
            if value.ndim == 3:
                value = value.squeeze()
                assert value.ndim == 2
            if act is None or act.abs().max() < value.abs().max():
                act = value
    per_token_max_id = act.abs().max(1)[0].max(0)[1]
    act = act[per_token_max_id].float().unsqueeze(0)
    print(f"shape: {act.shape}")
    save_dir = res_path.parent / "try_rotate_one_token"
    save_dir.mkdir(parents=True, exist_ok=True)
    show_statitics(act, save_dir / f"{Path(res_path).stem}_original.png")

    # random orthogonal
    for i in range(n_try):
        Q = random_orthogonal_matrix(act.shape[-1], act.device).float()
        act_rotated = torch.matmul(act, Q)
        print("-" * 80)
        print(f"random orthogonal {i+1}/{n_try}")
        show_statitics(
            act_rotated, save_dir / f"{Path(res_path).stem}_random_orthogonal_{i}.png"
        )

    # random hadamard
    for i in range(n_try):
        Q = random_hadamard_matrix(act.shape[-1], act.device).float()
        act_rotated = torch.matmul(act, Q)
        print("-" * 80)
        print(f"random hadamard {i+1}/{n_try}")
        show_statitics(
            act_rotated, save_dir / f"{Path(res_path).stem}_random_hadamard_{i}.png"
        )

    # # duquant
    # duquantizer = UniformAffineQuantizer(
    #     quant_method="duquant", max_rotation_step=32, block_size=512
    # )
    # for i in range(n_try):
    #     act_duquant = duquantizer.init_duquant(act)
    #     print("-" * 80)
    #     print(f"duquant {i+1}/{n_try}")
    #     show_statitics(act_duquant, save_dir / f"{Path(res_path).stem}_duquant_{i}.png")


if __name__ == "__main__":
    # pickled_res_paths = [
    #     "analysis_model/DeepSeek-R1/bf16/pickled/20250418_104623/block61.res.pkl",
    #     # "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled/20250418_104623/block61.res.pkl",
    # ]
    # for res_path in pickled_res_paths:
    #     # print("=" * 100)
    #     analysis_massive_outlier(Path(res_path))

    pickled_res_paths = [
        "analysis_model/DeepSeek-R1/bf16/pickled/20250423_041048/block60.res.pkl",
        "analysis_model/DeepSeek-R1/bf16/pickled/20250423_023357/block60.res.pkl",
    ]
    for res_path in pickled_res_paths:
        # print("=" * 100)
        # analysis_massive_outlier_penultimate_layer(Path(res_path))
        analysis_massive_outlier_penultimate_layer_2(Path(res_path))
    print("done")
