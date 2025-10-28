"""
Compare real-quantized activation distributions across different rotater group sizes
using only the NVFP4 quantizer.

This script mirrors the plotting style of line distribution plots from
`analysis_act_realquant.py` but overlays multiple lines, one for each rotater
group size.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from utils import HadamardTransform, NVFP4Quantizer


def create_nvfp4_quantizer():
    """Create NVFP4 quantizer for real quantization.

    Notes
    -----
    NVFP4 typically works with group_size=16 internally for activation
    quantization. We keep that behavior consistent here. The rotater group size
    (Hadamard) is varied independently from the quantizer's own grouping.
    """
    return NVFP4Quantizer(
        bit=4,
        symmetric=True,
        granularity="per_group",
        group_size=16,
        arbitrary_group_size=True,
    )


def apply_rotation(activation: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply Hadamard rotation per the requested group size.

    For group_size == 1, return the activation unchanged (no rotation).
    """
    if group_size == 1:
        return activation
    rotater = HadamardTransform(group_size=group_size)
    return rotater(activation)


def quantize_real_nvfp4(activation: torch.Tensor, quantizer: NVFP4Quantizer) -> torch.Tensor:
    """Perform real quantization using NVFP4 and return the quantized tensor.

    Returns
    -------
    torch.Tensor
        The integer-typed quantized activation tensor output by NVFP4.
    """
    quantized_act, _global_scale, _local_scales = quantizer.real_quant_act_dynamic(
        activation, args={}
    )
    return quantized_act


def plot_per_file_group_sizes(all_group_counts, save_fig_path):
    """Create a line plot comparing quantized distributions across group sizes.

    Parameters
    ----------
    all_group_counts : dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from group_size to (unique_values_np, counts_np) for the
        per-file quantized activation.
    save_fig_path : pathlib.Path
        Output path for the plot image.
    """
    save_fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Distinguishable colors and markers for up to ~8 lines
    colors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ]
    markers = ["o", "s", "^", "D", "v", "<", ">", "P"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        f"NVFP4 Real-Quantized Activation Distribution (Line) - {save_fig_path.stem}",
        fontsize=18,
        fontweight="bold",
    )

    for idx, (group_size, (unique_vals_np, counts_np)) in enumerate(sorted(all_group_counts.items())):
        ax.plot(
            unique_vals_np,
            counts_np,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            markersize=6,
            alpha=0.85,
            label=("Original (G1)" if group_size == 1 else f"Rotated (G{group_size})"),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Quantized Value", fontsize=12)
    ax.set_ylabel("Frequency (log)", fontsize=12)
    ax.grid(True, which="major", alpha=0.4, linewidth=1)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.5)

    # Denser log ticks on Y
    from matplotlib.ticker import LogLocator

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.minorticks_on()

    ax.legend(loc="best", fontsize=11, framealpha=0.9, title="Rotater Group Size")

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def _nvfp4_fixed_values(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return sorted NVFP4 value levels on the specified device/dtype."""
    return torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def nvfp4_histogram_counts_gpu(q_act: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed-bin histogram counts for NVFP4 values on GPU.

    Uses torch.bucketize against midpoints between the 15 NVFP4 levels to map each
    element to a bin in a single pass. Returns numpy arrays for plotting.
    """
    allowed = _nvfp4_fixed_values(q_act.device, q_act.dtype)  # [15]
    # Build 14 midpoints as edges
    edges = (allowed[:-1] + allowed[1:]) * 0.5  # [14]
    # Map each value to bin index in [0, 14]
    indices = torch.bucketize(q_act.view(-1), edges)  # int64 on GPU
    counts = torch.bincount(indices, minlength=allowed.numel())
    return allowed.detach().cpu().numpy(), counts.detach().cpu().numpy()


def process_activation_file(act_file: Path, quantizer: NVFP4Quantizer, save_root: Path, group_sizes: list[int], device: torch.device):
    """Process a single activation file and generate its comparison plot across group sizes."""
    activation = torch.load(act_file, map_location=device).float()

    # Compute counts for each rotater group size on CPU to avoid GPU OOM
    all_group_counts: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for group_size in group_sizes:
        rotated = apply_rotation(activation, group_size)
        q_act = quantize_real_nvfp4(rotated, quantizer)

        # Compute fixed-bin histogram on GPU, then move tiny results to CPU
        values_np, counts_np = nvfp4_histogram_counts_gpu(q_act)
        all_group_counts[group_size] = (values_np, counts_np)

        # Free per-iteration tensors
        del rotated
        del q_act
        torch.cuda.empty_cache()

    # Plot overlaid line plot across group sizes
    save_fig_path_line = save_root / f"{act_file.stem}_realquant_groupsize_line.png"
    plot_per_file_group_sizes(all_group_counts, save_fig_path_line)

    # Free VRAM
    del activation
    all_group_counts.clear()
    torch.cuda.empty_cache()


def main():
    """Main entry point for real-quant groupsize comparison using NVFP4 only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_root", type=str, default="figs/group_rotate/act/picked")
    parser.add_argument("--save_root", type=str, default="figs/group_rotate/act")
    parser.add_argument("--layer_name", type=str, default="down_proj")
    parser.add_argument("--device", type=int, default=1, help="CUDA device index to use")
    args = parser.parse_args()

    act_root = Path(args.act_root)
    save_root = Path(args.save_root) / "realquant_groupsize_comparison"
    layer_name = args.layer_name

    # Select device
    assert torch.cuda.is_available(), "CUDA is required for this script"
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    # Reference group sizes from NVFP4 group size comparison script
    group_sizes = [1, 4, 8, 16, 32, 512]

    # Prepare directories
    act_dir = act_root / layer_name
    act_files = natsorted(act_dir.glob("*_input.pt"))
    save_root.mkdir(parents=True, exist_ok=True)

    # NVFP4 quantizer (real quant)
    nvfp4 = create_nvfp4_quantizer()

    for act_file in tqdm(act_files, desc="Processing activation files"):
        process_activation_file(act_file, nvfp4, save_root, group_sizes, device)

    print("NVFP4 real-quant group size comparison completed!")


if __name__ == "__main__":
    main()


