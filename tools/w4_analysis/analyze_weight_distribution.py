from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import os
import argparse
import random


def plot_histogram(data, title, save_path, bins):
    """Create and save a histogram plot using Matplotlib."""
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.bfloat16, torch.float16]:
            data = data.float()
        data = data.detach().cpu().numpy().flatten()

    plt.figure(figsize=(10, 7))
    plt.hist(data, bins=bins, color="skyblue", edgecolor="black", log=True)

    # Pre-process title to handle <br> for matplotlib
    title = title.replace("<br>", "\n")

    plt.title(title, fontsize=14)
    plt.xlabel("Weight Value", fontsize=12)
    plt.ylabel("Frequency (log scale)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)

    stats_text = (
        f"Mean: {mean_val:.4f}\n"
        f"Std: {std_val:.4f}\n"
        f"Min: {min_val:.4f}\n"
        f"Max: {max_val:.4f}"
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.03,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()  # Close the figure to free memory
    print(f"Saved histogram plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weight distribution of a model."
    )
    parser.add_argument(
        "--llm_dir",
        type=Path,
        default="/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B",
        help="Directory containing the model files.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default="figs/view_weights_detail/qwen3-1.7B/bf16",
        help="Directory to save the analysis plots.",
    )
    parser.add_argument(
        "--view_weights",
        nargs="+",
        default=["q_proj"],
        help="List of weight names (substrings) to analyze.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=1,
        help="Number of rows to analyze for each weight. Set to -1 to analyze all rows.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="Size of chunks for analysis. Set to -1 to use the entire row as a single chunk.",
    )
    parser.add_argument(
        "--k_samples",
        type=int,
        default=8,
        help="Number of top divergent samples to plot.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir / f'chunk{args.chunk_size}'
    save_dir.mkdir(parents=True, exist_ok=True)

    model_file = None
    if (args.llm_dir / "model.safetensors").exists():
        model_file = args.llm_dir / "model.safetensors"
        file_type = "safetensors"
    else:
        safetensors_files = list(args.llm_dir.glob("*.safetensors"))
        if safetensors_files:
            model_file = safetensors_files[0]
            file_type = "safetensors"
    if model_file is None:
        if (args.llm_dir / "pytorch_model.bin").exists():
            model_file = args.llm_dir / "pytorch_model.bin"
            file_type = "pytorch"
        else:
            pytorch_files = list(args.llm_dir.glob("*.bin"))
            if pytorch_files:
                model_file = pytorch_files[0]
                file_type = "pytorch"

    if model_file is None:
        print(f"No model files (.safetensors or .bin) found in {args.llm_dir}")
        return

    print(f"Using {file_type} file: {model_file}")

    if file_type == "safetensors":
        weights_dict = load_file(str(model_file))
    else:
        weights_dict = torch.load(str(model_file), map_location="cpu")

    all_weights_list = []
    for key, tensor in weights_dict.items():
        if (
            any(sub in key for sub in ["proj", "fc", "w1", "w2", "gate", "linear"])
            and "norm" not in key
        ):
            all_weights_list.append(tensor.flatten())

    if not all_weights_list:
        print("No weights found matching the filter criteria. Exiting.")
        return

    all_weights = torch.cat(all_weights_list).float().cpu().numpy()
    print(f"Global weights collected. Total parameters: {all_weights.size}")

    global_min, global_max = all_weights.min(), all_weights.max()
    num_bins = 400
    global_bins = np.linspace(global_min, global_max, num_bins + 1)

    plot_histogram(
        all_weights,
        "Global Weight Distribution",
        save_dir / "global_histogram.png",
        global_bins,
    )

    global_hist, _ = np.histogram(all_weights, bins=global_bins, density=True)
    global_hist += 1e-10

    analysis_results = []
    for layer_idx in range(1):  # Assuming 32 layers, adjust if necessary
        for weight_name in args.view_weights:
            matching_keys = [
                key
                for key in weights_dict.keys()
                if weight_name in key and f"layers.{layer_idx}." in key
            ]

            for key in matching_keys:
                weight_tensor = weights_dict[key]
                if weight_tensor.dim() != 2:
                    print(
                        f"Skipping {key} (shape: {weight_tensor.shape}) as it is not a 2D tensor."
                    )
                    continue

                print(f"Analyzing {key}...")
                weight_np = weight_tensor.float().cpu().numpy()

                chunk_size = args.chunk_size
                if chunk_size == -1:
                    chunk_size = weight_np.shape[1]

                if chunk_size == 0:
                    continue

                num_chunks_per_row = weight_np.shape[1] // chunk_size

                if num_chunks_per_row == 0:
                    continue

                num_rows_to_analyze = weight_np.shape[0]
                if args.num_rows != -1:
                    num_rows_to_analyze = min(args.num_rows, weight_np.shape[0])

                for i in range(num_rows_to_analyze):
                    row = weight_np[i, :]
                    for j in range(num_chunks_per_row):
                        chunk = row[j * chunk_size : (j + 1) * chunk_size]
                        chunk_hist, _ = np.histogram(
                            chunk, bins=global_bins, density=True
                        )
                        chunk_hist += 1e-10

                        js_div = jensenshannon(global_hist, chunk_hist) ** 2

                        analysis_results.append(
                            {
                                "key": key,
                                "row": i,
                                "chunk_idx": j,
                                "divergence": js_div,
                                "data": chunk,
                            }
                        )

    sorted_results = sorted(
        analysis_results, key=lambda x: x["divergence"], reverse=True
    )

    # Save all divergence scores to a log file
    log_file_path = save_dir / "js_divergence_log.txt"
    with open(log_file_path, "w") as f:
        f.write("Weight Key, Row, Chunk Index, JS Divergence\n")
        for res in sorted_results:
            f.write(
                f"{res['key']}, {res['row']}, {res['chunk_idx']}, {res['divergence']:.8f}\n"
            )
    print(f"Saved all JS divergence scores to {log_file_path}")

    # Plot JS divergence distribution
    all_divergences = [res["divergence"] for res in sorted_results]
    plot_histogram(
        np.array(all_divergences),
        "Distribution of JS Divergence Scores",
        save_dir / "js_divergence_distribution.png",
        bins=400,
    )

    print(f"\nPlotting {args.k_samples} randomly sampled chunks...")
    if len(analysis_results) < args.k_samples:
        sampled_results = analysis_results
    else:
        sampled_results = random.sample(analysis_results, args.k_samples)

    for res in sampled_results:
        safe_key = res["key"].replace(".", "_").replace("/", "_")
        title = f"Hist for {res['key']} (row {res['row']}, chunk {res['chunk_idx']})<br>JS-Div: {res['divergence']:.6f}"
        filename = f"rand_{safe_key}_row{res['row']}_chunk{res['chunk_idx']}_hist.png"
        save_path = save_dir / filename

        plot_histogram(res["data"], title, save_path, global_bins)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
