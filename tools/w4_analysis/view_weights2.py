from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os


def calculate_row_statistics(tensor, name):
    """Calculate min, max, mean, var for each row of a tensor"""
    if len(tensor.shape) < 2:
        print(f"Skipping {name}: tensor shape {tensor.shape} is not 2D or higher")
        return None
    
    # Reshape to 2D if needed (flatten all dimensions except the first)
    if len(tensor.shape) > 2:
        tensor_2d = tensor.view(tensor.shape[0], -1)
    else:
        tensor_2d = tensor
    
    # Calculate statistics for each row
    row_min = torch.min(tensor_2d, dim=1)[0]
    row_max = torch.max(tensor_2d, dim=1)[0]
    row_mean = torch.mean(tensor_2d, dim=1)
    row_var = torch.var(tensor_2d, dim=1)
    
    stats = {
        'name': name,
        'shape': tensor.shape,
        'num_rows': tensor_2d.shape[0],
        'row_min': row_min,
        'row_max': row_max,
        'row_mean': row_mean,
        'row_var': row_var
    }
    
    return stats


def analyze_weights(llm_dir):
    """Analyze all weight files in the model directory"""
    all_stats = []
    
    # Find all safetensors files
    weight_files = list(llm_dir.glob("*.safetensors"))
    if not weight_files:
        print(f"No safetensors files found in {llm_dir}")
        return
    
    print(f"Found {len(weight_files)} weight files")
    
    for weight_file in weight_files:
        print(f"\nAnalyzing {weight_file.name}...")
        
        try:
            # Load weights from safetensors file
            weights = load_file(weight_file)
            
            for param_name, tensor in weights.items():
                print(f"  Processing {param_name}: shape {tensor.shape}")
                
                stats = calculate_row_statistics(tensor, f"{weight_file.name}::{param_name}")
                if stats is not None:
                    all_stats.append(stats)
                    
                    # Print summary statistics
                    print(f"    Rows: {stats['num_rows']}")
                    print(f"    Row min range: [{stats['row_min'].min():.6f}, {stats['row_min'].max():.6f}]")
                    print(f"    Row max range: [{stats['row_max'].min():.6f}, {stats['row_max'].max():.6f}]")
                    print(f"    Row mean range: [{stats['row_mean'].min():.6f}, {stats['row_mean'].max():.6f}]")
                    print(f"    Row var range: [{stats['row_var'].min():.6f}, {stats['row_var'].max():.6f}]")
                    
        except Exception as e:
            print(f"Error processing {weight_file.name}: {e}")
    
    return all_stats


def save_detailed_stats(stats_list, output_dir):
    """Save detailed statistics to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for stats in stats_list:
        filename = stats['name'].replace('::', '_').replace('/', '_') + '_stats.txt'
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"Weight: {stats['name']}\n")
            f.write(f"Shape: {stats['shape']}\n")
            f.write(f"Number of rows: {stats['num_rows']}\n\n")
            
            f.write("Row-wise statistics:\n")
            f.write("Row_Index\tMin\tMax\tMean\tVar\n")
            
            for i in range(stats['num_rows']):
                f.write(f"{i}\t{stats['row_min'][i]:.6f}\t{stats['row_max'][i]:.6f}\t"
                       f"{stats['row_mean'][i]:.6f}\t{stats['row_var'][i]:.6f}\n")
        
        print(f"Saved detailed stats to {filepath}")


if __name__ == "__main__":
    llm_dir = Path("/data/chenshuailin/checkpoints/Qwen/Qwen3-1.7B")
    
    if not llm_dir.exists():
        print(f"Model directory {llm_dir} does not exist!")
        exit(1)
    
    print(f"Analyzing weights in: {llm_dir}")
    
    # Analyze all weights
    stats_list = analyze_weights(llm_dir)
    
    if stats_list:
        # Save detailed statistics to files
        output_dir = llm_dir / "weight_analysis"
        save_detailed_stats(stats_list, output_dir)
        
        print(f"\nAnalysis complete! Processed {len(stats_list)} weight tensors.")
        print(f"Detailed statistics saved to: {output_dir}")
    else:
        print("No weight tensors found to analyze!")
