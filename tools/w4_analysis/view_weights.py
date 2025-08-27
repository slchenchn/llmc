from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os


def create_3d_weight_plot(weight_tensor, title, save_path):
    """Create 3D surface plot of weight tensor"""
    # Convert to numpy and handle different tensor shapes
    # Convert to float32 first to handle BFloat16 and other unsupported types
    if weight_tensor.dtype in [torch.bfloat16, torch.float16]:
        weight_tensor = weight_tensor.float()
    weight_np = weight_tensor.detach().cpu().numpy()

    if len(weight_np.shape) == 1:
        # 1D tensor - create a simple line plot in 3D
        x = np.arange(len(weight_np))
        y = np.zeros_like(x)
        z = weight_np

        fig = go.Figure(
            data=go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+lines",
                marker=dict(size=3, color=z, colorscale="Viridis"),
                line=dict(color="darkblue", width=2),
            )
        )

    elif len(weight_np.shape) == 2:
        # 2D tensor - create surface plot
        fig = go.Figure(
            data=go.Surface(z=weight_np, colorscale="Viridis", showscale=True)
        )

    else:
        # Higher dimensional tensors - flatten to 2D for visualization
        if len(weight_np.shape) > 2:
            # Take a 2D slice or reshape
            if (
                weight_np.shape[0] * weight_np.shape[1] <= 10000
            ):  # Reasonable size limit
                weight_np = weight_np.reshape(weight_np.shape[0], -1)
            else:
                # Take first two dimensions
                weight_np = weight_np[
                    : min(100, weight_np.shape[0]), : min(100, weight_np.shape[1])
                ]

        fig = go.Figure(
            data=go.Surface(z=weight_np, colorscale="Viridis", showscale=True)
        )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Weight Value",
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6)),
        ),
        width=800,
        height=600,
    )

    # Save as PNG
    fig.write_image(save_path)
    print(f"Saved 3D plot: {save_path}")


def create_histogram_plot(weight_tensor, title, save_path):
    """Create histogram plot of weight tensor"""
    # Convert to numpy and handle different tensor shapes
    # Convert to float32 first to handle BFloat16 and other unsupported types
    if weight_tensor.dtype in [torch.bfloat16, torch.float16]:
        weight_tensor = weight_tensor.float()
    weight_np = weight_tensor.detach().cpu().numpy().flatten()

    # Create histogram
    fig = go.Figure(
        data=go.Histogram(
            x=weight_np,
            nbinsx=200,
            name="Weight Distribution",
            marker_color="rgba(55, 128, 191, 0.7)",
            marker_line=dict(color="rgba(55, 128, 191, 1.0)", width=1),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Weight Value",
        yaxis_title="Frequency (log scale)",
        yaxis_type="log",
        showlegend=False,
        width=800,
        height=600,
        template="plotly_white",
    )

    # Add statistics text
    mean_val = np.mean(weight_np)
    std_val = np.std(weight_np)
    min_val = np.min(weight_np)
    max_val = np.max(weight_np)

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Mean: {mean_val:.4f}<br>Std: {std_val:.4f}<br>Min: {min_val:.4f}<br>Max: {max_val:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
    )

    # Save as PNG
    fig.write_image(save_path)
    print(f"Saved histogram plot: {save_path}")


def create_heatmap_plot(weight_tensor, title, save_path):
    """Create heatmap plot of the first row of weight tensor, reshaped to 2D with 64 columns"""
    # Convert to numpy and handle different tensor shapes
    # Convert to float32 first to handle BFloat16 and other unsupported types
    if weight_tensor.dtype in [torch.bfloat16, torch.float16]:
        weight_tensor = weight_tensor.float()
    weight_np = weight_tensor.detach().cpu().numpy()

    # Get the first row
    if len(weight_np.shape) == 1:
        first_row = weight_np
    else:
        first_row = weight_np[0, :]  # Take first row

    # Reshape into 2D matrix with 64 columns per row
    cols = 64
    if len(first_row) < cols:
        # If less than 64 elements, pad with zeros
        padded_length = ((len(first_row) + cols - 1) // cols) * cols
        first_row_padded = np.pad(
            first_row, (0, padded_length - len(first_row)), "constant"
        )
        reshaped = first_row_padded.reshape(-1, cols)
    else:
        # Truncate to make it divisible by 64, then reshape
        truncated_length = (len(first_row) // cols) * cols
        first_row_truncated = first_row[:truncated_length]
        reshaped = first_row_truncated.reshape(-1, cols)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=reshaped,
            colorscale="RdBu_r",  # Red-Blue colorscale, reversed
            showscale=True,
            colorbar=dict(title="Weight Value"),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Column Index (64 per row)",
        yaxis_title="Row Index",
        width=800,
        height=600,
        template="plotly_white",
    )

    # Add statistics text
    mean_val = np.mean(first_row)
    std_val = np.std(first_row)
    min_val = np.min(first_row)
    max_val = np.max(first_row)

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"First Row Stats:<br>Mean: {mean_val:.4f}<br>Std: {std_val:.4f}<br>Min: {min_val:.4f}<br>Max: {max_val:.4f}<br>Shape: {reshaped.shape}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
    )

    # Save as PNG
    fig.write_image(save_path)
    print(f"Saved heatmap plot: {save_path}")


if __name__ == "__main__":
    # llm_dir = Path('/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B')
    llm_dir = Path(
        "checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_sym_dynamic/vllm_quant_model"
    )
    save_dir = Path("figs/view_weights/qwen3-1.7B/gptq/w4a8_sym_dynamic")
    view_weights = [
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]  # weights to view
    layers = [0]  # layers to view

    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Configure plotly to use kaleido for PNG export
    pio.kaleido.scope.default_format = "png"

    # Iterate through layers and weights
    for layer_idx in layers:
        # Try to find model files in order of preference
        model_file = None
        file_type = None

        # First try safetensors
        safetensors_file = llm_dir / "model.safetensors"
        if safetensors_file.exists():
            model_file = safetensors_file
            file_type = "safetensors"
        else:
            # Try to find any safetensors files
            safetensors_files = list(llm_dir.glob("*.safetensors"))
            if safetensors_files:
                model_file = safetensors_files[0]
                file_type = "safetensors"

        # If no safetensors, try pytorch .bin files
        if model_file is None:
            pytorch_file = llm_dir / "pytorch_model.bin"
            if pytorch_file.exists():
                model_file = pytorch_file
                file_type = "pytorch"
            else:
                # Try to find any .bin files
                pytorch_files = list(llm_dir.glob("*.bin"))
                if pytorch_files:
                    model_file = pytorch_files[0]
                    file_type = "pytorch"

        if model_file is None:
            print(f"No model files (.safetensors or .bin) found in {llm_dir}")
            continue

        print(f"Using {file_type} file: {model_file}")

        try:
            # Load the model file
            if file_type == "safetensors":
                weights_dict = load_file(str(model_file))
            else:  # pytorch
                weights_dict = torch.load(str(model_file), map_location="cpu")
            print(
                f"Available keys in {file_type} file: {list(weights_dict.keys())[:10]}..."
            )  # Show first 10 keys

            # Find weights matching the specified patterns
            for weight_name in view_weights:
                matching_keys = [
                    key
                    for key in weights_dict.keys()
                    if weight_name in key and f"layers.{layer_idx}." in key
                ]

                if not matching_keys:
                    print(
                        f"No weights found matching '{weight_name}' for layer {layer_idx}"
                    )
                    continue

                for key in matching_keys:
                    weight_tensor = weights_dict[key]
                    print(f"Processing weight: {key}, shape: {weight_tensor.shape}")

                    # Create safe filename base
                    safe_key = key.replace(".", "_").replace("/", "_")

                    # # Create 3D plot
                    # filename_3d = f"layer_{layer_idx}_{safe_key}_3d.png"
                    # save_path_3d = save_dir / filename_3d
                    # title_3d = f"3D Weight Visualization: {key}"
                    # create_3d_weight_plot(weight_tensor, title_3d, str(save_path_3d))

                    # Create histogram plot
                    filename_hist = f"layer_{layer_idx}_{safe_key}_histogram.png"
                    save_path_hist = save_dir / filename_hist
                    title_hist = f"Weight Distribution Histogram: {key}"
                    create_histogram_plot(
                        weight_tensor, title_hist, str(save_path_hist)
                    )

                    # Create heatmap plot
                    filename_heatmap = f"layer_{layer_idx}_{safe_key}_heatmap.png"
                    save_path_heatmap = save_dir / filename_heatmap
                    title_heatmap = f"First Row Heatmap (64 cols): {key}"
                    create_heatmap_plot(
                        weight_tensor, title_heatmap, str(save_path_heatmap)
                    )

        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")

    print("Weight visualization complete!")
