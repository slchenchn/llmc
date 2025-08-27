from pathlib import Path
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def l2_loss(x, y):
    """Calculate L2 loss between two tensors."""
    return (x - y).pow(2).sum(-1)


def load_data(root, methods, key0, layer, key1s):
    """Load data from specified paths."""
    datas = defaultdict(dict)
    print("=" * 100)
    print("Loading data...")
    
    for method in methods:
        for key1 in key1s:
            pt_path = root / method / key0 / f"layer{layer}_{key1}.pt"
            cur_data = torch.load(pt_path)
            datas[method][key1] = cur_data
            
            print("-" * 100)
            print(f"{method} {key1}")
            print(f"min: {cur_data.min()}, max: {cur_data.max()}")
            print(f"mean: {cur_data.mean()}, std: {cur_data.std()}")
            print(f"shape: {cur_data.shape}")
    
    return datas


def create_visualization(data, title, xlabel, ylabel, save_path, method_name=""):
    """Create heatmap and histogram visualization for given data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    data_reshaped = (
        data.reshape(-1, data.shape[-1])
        if len(data.shape) > 2
        else data.unsqueeze(0)
    )
    im = ax1.imshow(
        data_reshaped.detach().numpy().squeeze(),
        cmap="viridis",
        aspect="auto",
        norm=mcolors.LogNorm(),
    )
    ax1.set_title(f"{method_name} - {title} Scale")
    ax1.set_xlabel("Feature Dimension")
    ax1.set_ylabel("Sample Index")
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.colorbar(im, ax=ax1, label=f"{title} (log scale)")
    
    # Histogram with symlog scale
    data_np = data.detach().numpy().flatten()
    
    ax2.set_xscale('symlog')
    ax2.set_yscale('log')
    ax2.hist(
        data_np,
        bins=1000,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax2.set_title(f"{method_name} - {title} Distribution")
    ax2.set_xlabel(f"{title} (symlog scale)")
    
    ax2.set_ylabel("Frequency (log scale)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {title} visualization to: {save_path}")
    plt.show()


def visualize_data(datas, root, methods, layer, key1s):
    """Create visualizations for all loaded data."""
    print("Creating data visualizations...")
    
    for method in methods:
        for key1 in key1s:
            cur_data = datas[method][key1]
            save_path = root / method / f"{key1}_analysis_layer{layer}.png"
            create_visualization(cur_data, key1, "Feature Dimension", "Sample Index", save_path, method)


def analyze_loss(datas, methods, root, layer):
    """Analyze L2 loss between target and quant_out for each method."""
    print("=" * 100)
    print("Loss Analysis")
    
    for method in methods:
        print("-" * 100)
        print(f"{method}")
        
        loss = l2_loss(datas[method]["target"], datas[method]["quant_out"])
        print(f"{method} loss mean: {loss.mean()}, shape: {loss.shape}")
        print(f"loss min: {loss.min()}, max: {loss.max()}")
        print(f"loss mean: {loss.mean()}, std: {loss.std()}")
        
        # Create loss visualization
        save_path = root / method / f"loss_analysis_layer{layer}.png"
        create_visualization(loss, "Loss", "Feature Dimension", "Sample Index", save_path, method)


def main():
    """Main function to run the analysis."""
    # Configuration
    root = Path("cache/tesseraq/")
    methods = ["wiki2_tesseraq", "wiki2_quarot_tesseraq"]
    key0 = "before_train"
    layer = 2
    key1s = ["target", "quant_out"]
    
    # Load data
    datas = load_data(root, methods, key0, layer, key1s)
    
    # Create visualizations for raw data
    visualize_data(datas, root, methods, layer, key1s)
    
    # Analyze loss
    analyze_loss(datas, methods, root, layer)
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()
