from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


def view_hist(num_tokens, ds_path):
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(num_tokens, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Tokens in Mixture-of-Thoughts Dataset")
    plt.grid(True, alpha=0.3)

    # Add some statistics to the plot
    mean_tokens = np.mean(num_tokens)
    median_tokens = np.median(num_tokens)
    plt.axvline(
        mean_tokens,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_tokens:.1f}",
    )
    plt.axvline(
        median_tokens,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Median: {median_tokens:.1f}",
    )
    plt.legend()

    # Save the plot to ds_path directory
    output_path = ds_path / "num_tokens_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"image saved to {output_path}")


if __name__ == "__main__":
    ds_path = Path("data/Mixture-of-Thoughts")
    seq_len = 8192
    max_n_samples = 10240

    ds = load_from_disk(ds_path)
    print(f"length of the original dataset: {len(ds)}")
    num_tokens = ds["num_tokens"]

    view_hist(num_tokens, ds_path)
    filtered_ds = ds.filter(lambda x: x["num_tokens"] >= seq_len)
    print(f"filtered dataset length: {len(ds)}, remaining: {len(filtered_ds)}")
    if len(filtered_ds) > max_n_samples:
        filtered_ds = filtered_ds.select(range(max_n_samples))
        print(f"truncated dataset length: {len(filtered_ds)}")

    save_ds_path = ds_path.with_suffix(f".seq{seq_len}")
    filtered_ds.save_to_disk(save_ds_path)
