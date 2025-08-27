from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from collections import defaultdict
import seaborn as sns


def load_model_weights(model_dir):
    """加载模型权重"""
    model_dir = Path(model_dir)
    
    # 方法1: 尝试读取权重索引文件
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        print("Found index file, loading weights from safetensors with index...")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # 按文件分组权重
        file_weights = defaultdict(list)
        for weight_name, file_name in index_data["weight_map"].items():
            file_weights[file_name].append(weight_name)
        
        # 加载所有权重
        all_weights = {}
        for file_name, weight_names in file_weights.items():
            file_path = model_dir / file_name
            weights = load_file(file_path)
            for weight_name in weight_names:
                if weight_name in weights:
                    all_weights[weight_name] = weights[weight_name]
        
        return all_weights
    
    # 方法2: 如果没有索引文件，尝试读取所有safetensors文件
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if safetensors_files:
        print(f"No index file found, loading {len(safetensors_files)} safetensors files...")
        all_weights = {}
        for file_path in safetensors_files:
            print(f"Loading {file_path.name}...")
            weights = load_file(file_path)
            all_weights.update(weights)
        
        return all_weights
    
    # 方法3: 如果没有safetensors文件，尝试读取.bin文件
    bin_files = list(model_dir.glob("*.bin"))
    if bin_files:
        print(f"No safetensors files found, loading {len(bin_files)} .bin files...")
        all_weights = {}
        for file_path in bin_files:
            print(f"Loading {file_path.name}...")
            weights = torch.load(file_path, map_location='cpu')
            # .bin文件可能直接包含权重字典，或者包含在'state_dict'键中
            if isinstance(weights, dict):
                if 'state_dict' in weights:
                    all_weights.update(weights['state_dict'])
                else:
                    all_weights.update(weights)
            else:
                print(f"Warning: {file_path.name} contains unexpected data structure")
        
        return all_weights
    
    # 如果都没有找到，抛出错误
    raise FileNotFoundError(f"No model weights found in {model_dir}. "
                          f"Expected .safetensors files (with or without index.json) or .bin files.")


def filter_linear_weights(weights):
    """过滤出线性层的二维权重矩阵"""
    linear_weights = {}
    
    for name, weight in weights.items():
        # 排除 embeddings, lm_head, norm 权重
        if any(skip in name for skip in ['embed', 'lm_head', 'norm']):
            continue
        
        # 只保留二维权重矩阵
        if len(weight.squeeze().shape) == 2:
            linear_weights[name] = weight
    
    return linear_weights


def compute_singular_values(weights):
    """计算每个权重矩阵的奇异值"""
    singular_values = {}
    
    for name, weight in weights.items():
        # 转换为numpy并计算SVD (先转换为float32处理BFloat16)
        weight_np = weight.float().cpu().numpy().astype(np.float32)
        s = np.linalg.svd(weight_np, compute_uv=False)
        
        # 按从高到低排序
        s_sorted = np.sort(s)[::-1]
        singular_values[name] = s_sorted
        
        print(f"{name}: shape={weight.shape}, rank={len(s)}, max_sv={s_sorted[0]:.4f}, min_sv={s_sorted[-1]:.4f}")
    
    return singular_values


def plot_singular_value_distributions(singular_values, save_dir):
    """为每个权重矩阵绘制单独的奇异值分布图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    plt.style.use('default')
    
    print(f"\nGenerating individual plots for {len(singular_values)} weight matrices...")
    
    # 为每个权重矩阵生成单独的图
    for weight_name, sv in singular_values.items():
        plt.figure(figsize=(10, 6))
        
        # 绘制奇异值分布
        plt.plot(sv, 'b-', linewidth=2, alpha=0.8)
        plt.fill_between(range(len(sv)), sv, alpha=0.3)
        
        # 添加关键统计信息到图中
        effective_rank = np.sum(sv > 0.01 * sv[0])
        condition_number = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')
        
        plt.xlabel('Singular Value Index', fontsize=12)
        plt.ylabel('Singular Value', fontsize=12)
        plt.title(f'Singular Value Distribution\n{weight_name}', fontsize=14, pad=20)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 添加文本信息
        info_text = f'Shape: {len(sv)}\nEffective Rank: {effective_rank}\nCondition Number: {condition_number:.2e}\nSV Range: [{sv[-1]:.6f}, {sv[0]:.6f}]'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        # 生成安全的文件名
        safe_name = weight_name.replace('/', '_').replace('.', '_')
        plt.savefig(save_dir / f'{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {safe_name}.png")
    
    # 生成汇总图：按层类型分组
    print("\nGenerating summary plots...")
    
    # 按权重类型分组
    weight_groups = {
        'attention_q': [name for name in singular_values.keys() if 'q_proj' in name],
        'attention_k': [name for name in singular_values.keys() if 'k_proj' in name],
        'attention_v': [name for name in singular_values.keys() if 'v_proj' in name],
        'attention_o': [name for name in singular_values.keys() if 'o_proj' in name],
        'mlp_gate': [name for name in singular_values.keys() if 'gate_proj' in name],
        'mlp_up': [name for name in singular_values.keys() if 'up_proj' in name],
        'mlp_down': [name for name in singular_values.keys() if 'down_proj' in name],
    }
    
    for group_name, weight_names in weight_groups.items():
        if not weight_names:
            continue
            
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(weight_names)))
        
        for i, weight_name in enumerate(sorted(weight_names)):
            sv = singular_values[weight_name]
            layer_idx = weight_name.split('.')[2] if 'layers.' in weight_name else 'N/A'
            plt.plot(sv, color=colors[i], alpha=0.7, linewidth=1.5, 
                    label=f'Layer {layer_idx}' if i % 3 == 0 else '')
        
        plt.xlabel('Singular Value Index', fontsize=12)
        plt.ylabel('Singular Value', fontsize=12)
        plt.title(f'Singular Value Distributions - {group_name.replace("_", " ").title()}', fontsize=14)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / f'summary_{group_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: summary_{group_name}.png")
    
    # 统计分析
    print("\n=== Singular Value Statistics ===")
    for name, sv in singular_values.items():
        effective_rank = np.sum(sv > 0.01 * sv[0])
        condition_number = sv[0] / sv[-1] if sv[-1] > 0 else float('inf')
        
        print(f"{name}:")
        print(f"  Full rank: {len(sv)}")
        print(f"  Effective rank (>1% max): {effective_rank}")
        print(f"  Condition number: {condition_number:.2e}")
        print(f"  Singular value range: [{sv[-1]:.6f}, {sv[0]:.6f}]")
        print()


if __name__ == "__main__":
    model_dir = Path("checkpoints/Qwen3-1.7B/quarot/sym_w8_a8-dynamic/vllm_quant_model")
    save_dir = Path("figs/weight_analysis/Qwen3-1.7B-quarot-w8a8/per-weight")
    
    print("Loading model weights...")
    all_weights = load_model_weights(model_dir)
    
    print("Filtering linear layer weights...")
    linear_weights = filter_linear_weights(all_weights)
    print(f"Found {len(linear_weights)} linear layer weights")
    
    print("Computing singular values...")
    singular_values = compute_singular_values(linear_weights)
    
    print("Plotting distributions...")
    plot_singular_value_distributions(singular_values, save_dir)
    
    print(f"Analysis complete! Results saved to {save_dir}")
    
