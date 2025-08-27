#!/usr/bin/env python3
"""
Model Weight Comparison Tool

This script compares weights between dequantized and fake-quantized models,
providing detailed analysis and visualizations of the differences.
Now supports quantizing original models and comparing with quantized versions.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import sys

# Add the project root to the Python path to import LLMC modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class QuantizerConfig:
    """Configuration for quantizer."""

    bit: int = 4
    symmetric: bool = True
    granularity: str = "per_group"  # per_group, per_channel, per_tensor
    group_size: int = 64
    calib_algo: str = "minmax"  # minmax, hqq
    round_mode: str = "round"


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""

    model1_dir: Path
    model2_dir: Path
    model1_name: str = "Model 1"
    model2_name: str = "Model 2"
    output_prefix: str = "model_comparison"
    max_diffs_per_layer: int = 10
    max_diff_samples: int = 1000
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    save_dir: Path = Path
    # 新增量化相关配置
    quantize_before_compare: bool = False  # 是否在比较前对两个模型进行量化
    quantizer_config: Optional[QuantizerConfig] = None  # 量化器配置
    ref_model: str = (
        "model2"  # 'model1' or 'model2', a_q = f(b_q, a_fp) or b_q = f(a_q, b_fp)
    )


@dataclass
class LayerStats:
    """Statistics for a single layer."""

    abs_max: List[float] = field(default_factory=list)
    abs_mean: List[float] = field(default_factory=list)
    rel_mean: List[float] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    diffs: List[np.ndarray] = field(default_factory=list)


class SimpleQuantizer:
    """Simple quantizer for per-group quantization."""

    def __init__(self, config: QuantizerConfig):
        self.config = config
        self.bit = config.bit
        self.symmetric = config.symmetric
        self.granularity = config.granularity
        self.group_size = config.group_size

        # Calculate quantization ranges
        if self.symmetric:
            self.qmin = -(2 ** (self.bit - 1))
            self.qmax = 2 ** (self.bit - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**self.bit - 1

    def reshape_tensor_for_group_quant(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for per-group quantization."""
        if self.granularity != "per_group":
            return tensor

        # Reshape for group-wise quantization
        # tensor shape: [out_features, in_features]
        out_features, in_features = tensor.shape

        # Pad if necessary
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
            in_features = tensor.shape[1]

        # Reshape to [out_features, num_groups, group_size]
        num_groups = in_features // self.group_size
        tensor = tensor.view(out_features, num_groups, self.group_size)

        return tensor

    def restore_tensor_shape(
        self, tensor: torch.Tensor, original_shape: torch.Size
    ) -> torch.Tensor:
        """Restore tensor to original shape after quantization."""
        if self.granularity != "per_group":
            return tensor

        # Flatten back to 2D
        out_features = tensor.shape[0]
        tensor = tensor.view(out_features, -1)

        # Trim to original size if padding was added
        original_in_features = original_shape[1]
        if tensor.shape[1] > original_in_features:
            tensor = tensor[:, :original_in_features]

        return tensor

    def get_scales_and_zeros(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantization scales and zero points."""
        if self.granularity == "per_group":
            # tensor shape: [out_features, num_groups, group_size]
            min_vals = tensor.min(dim=-1, keepdim=True)[
                0
            ]  # [out_features, num_groups, 1]
            max_vals = tensor.max(dim=-1, keepdim=True)[
                0
            ]  # [out_features, num_groups, 1]
        elif self.granularity == "per_channel":
            # Per output channel
            min_vals = tensor.min(dim=-1, keepdim=True)[0]  # [out_features, 1]
            max_vals = tensor.max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        else:  # per_tensor
            min_vals = tensor.min()
            max_vals = tensor.max()

        if self.symmetric:
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            scales = abs_max / (2 ** (self.bit - 1) - 1)
            zeros = torch.zeros_like(scales)
        else:
            scales = (max_vals - min_vals) / (2**self.bit - 1)
            zeros = self.qmin - min_vals / scales

        # Avoid division by zero
        scales = torch.clamp(scales, min=1e-8)

        return scales, zeros

    def dequantize_tensor(
        self, q_tensor: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize tensor."""
        original_shape = q_tensor.shape
        reshaped_q_tensor = self.reshape_tensor_for_group_quant(q_tensor)

        if self.symmetric:
            dq_tensor_reshaped = reshaped_q_tensor.float() * scales
        else:
            dq_tensor_reshaped = (reshaped_q_tensor.float() - zeros) * scales

        dq_tensor = self.restore_tensor_shape(dq_tensor_reshaped, original_shape)
        return dq_tensor

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor.
        If scales and zeros are provided, use them. Otherwise, compute them.
        Returns quantized tensor, scales, and zeros.
        """
        original_shape = tensor.shape

        reshaped_tensor = self.reshape_tensor_for_group_quant(tensor)

        # Compute scales and zeros only if not provided
        if scales is None or zeros is None:
            scales, zeros = self.get_scales_and_zeros(reshaped_tensor)

        if self.symmetric:
            q_tensor_reshaped = torch.clamp(
                torch.round(reshaped_tensor / scales), self.qmin, self.qmax
            )
        else:
            q_tensor_reshaped = torch.clamp(
                torch.round(reshaped_tensor / scales + zeros), self.qmin, self.qmax
            )

        q_tensor = self.restore_tensor_shape(q_tensor_reshaped, original_shape)

        return q_tensor, scales, zeros


class ModelComparator:
    """Compare weights between two models."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.layer_patterns = [
            r"layers\.([0-9]+)\.",
            r"layer\.([0-9]+)\.",
            r"h\.([0-9]+)\.",
            r"transformer\.h\.([0-9]+)\.",
            r"blocks\.([0-9]+)\.",
        ]

        # Initialize quantizer if needed
        self.quantizer = None
        if self.config.quantize_before_compare and self.config.quantizer_config:
            self.quantizer = SimpleQuantizer(self.config.quantizer_config)

    def extract_layer_id(self, param_name: str) -> int:
        """Extract layer ID from parameter name."""
        for pattern in self.layer_patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        return -1

    def load_model_weights(
        self, model_dir: Path, model_name: str
    ) -> Dict[str, torch.Tensor]:
        """Load model weights from safetensors files."""
        safetensors_files = list(model_dir.glob("*.safetensors"))

        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {model_dir}")

        print(f"Loading {model_name} from {len(safetensors_files)} files...")

        model_dict = {}
        filtered_count = 0
        total_count = 0

        for file in safetensors_files:
            cur_dict = load_file(file)
            for k, v in cur_dict.items():
                total_count += 1

                # Filter out buffers
                if "buf" in k:
                    filtered_count += 1
                    continue

                # Filter out embed_tokens and lm_head
                if "embed_tokens" in k or "lm_head" in k:
                    filtered_count += 1
                    continue

                # Filter out non-2D tensors
                if len(v.shape) != 2:
                    filtered_count += 1
                    continue

                model_dict[k] = v.double()

        print(
            f"✓ {model_name} loaded: {len(model_dict)} parameters (filtered out {filtered_count}/{total_count})"
        )
        return model_dict

    def quantize_model_weights(
        self,
        model_dict: Dict[str, torch.Tensor],
        external_qparams: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple]]:
        """
        Apply quantization to model weights.
        If external_qparams is provided, use them for quantization.
        """
        if not self.quantizer:
            return model_dict

        print("Applying quantization to model weights...")
        quantized_dict = {}

        # Patterns for weights that should NOT be quantized (normalization layers)
        non_quantizable_patterns = [
            r".*norm.*",  # Normalization layers
            r".*ln.*",  # Layer norm
            r".*layernorm.*",  # Layer norm variations
        ]

        quantized_count = 0
        for key, tensor in model_dict.items():
            should_quantize = True

            # Check if this should be excluded from quantization
            for pattern in non_quantizable_patterns:
                if re.search(pattern, key, re.IGNORECASE):
                    should_quantize = False
                    break

            # All tensors in model_dict are already filtered to be 2D weights
            if should_quantize:
                scales, zeros = None, None
                if external_qparams and key in external_qparams:
                    scales, zeros = external_qparams[key]
                    print(f"  Quantizing {key} using external qparams: {tensor.shape}")
                else:
                    print(
                        f"  Quantizing {key} by computing new qparams: {tensor.shape}"
                    )

                quantized_dict[key] = self.quantizer.quantize_tensor(
                    tensor, scales, zeros
                )
                quantized_count += 1
            else:
                quantized_dict[key] = tensor

        print(
            f"✓ Quantization applied to {quantized_count}/{len(model_dict)} parameters"
        )
        return quantized_dict

    def dequantize_model_dict(
        self, q_model_dict: Dict[str, Union[torch.Tensor, Tuple]]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize an entire model dictionary."""
        dequantized_dict = {}
        for key, value in q_model_dict.items():
            if isinstance(value, tuple):  # It's a quantized weight tuple
                q_tensor, scales, zeros = value
                dequantized_dict[key] = self.quantizer.dequantize_tensor(
                    q_tensor, scales, zeros
                )
            else:  # It's a regular tensor
                dequantized_dict[key] = value
        return dequantized_dict

    def compute_parameter_differences(
        self, param1: torch.Tensor, param2: torch.Tensor, is_q_level: bool = False
    ) -> Tuple[float, float, float]:
        """
        Compute absolute and relative differences between two parameters.
        If is_q_level is True, computes integer differences for q_tensors.
        """
        diff = param1.float() - param2.float()
        # if diff.abs.unique() >

        abs_max = diff.abs().max().item()
        abs_mean = diff.abs().mean().item()

        # DEBUG:
        if abs_max > 2:
            print(f"abs_max: {abs_max}, abs_mean: {abs_mean}")

        # Set epsilon based on comparison type
        epsilon = 1.0 if is_q_level else 1e-8

        # Compute relative difference with epsilon to avoid division by zero
        rel_diff = diff.abs() / (param2.float().abs() + epsilon)
        rel_mean = torch.mean(rel_diff.flatten().float()).item()

        return abs_max, abs_mean, rel_mean

    def analyze_differences(
        self, model1_dict: Dict, model2_dict: Dict
    ) -> Dict[int, LayerStats]:
        """Analyze differences between two models."""
        print(
            f"\nAnalyzing {self.config.model1_name} vs {self.config.model2_name} differences..."
        )

        model1_keys = set(model1_dict.keys())
        model2_keys = set(model2_dict.keys())

        if model1_keys != model2_keys:
            missing_in_model2 = model1_keys - model2_keys
            missing_in_model1 = model2_keys - model1_keys

            error_msg = "Models do not have exactly the same parameters"
            if missing_in_model2:
                error_msg += f"\nParameters in {self.config.model1_name} but not in {self.config.model2_name}: {sorted(missing_in_model2)}"
            if missing_in_model1:
                error_msg += f"\nParameters in {self.config.model2_name} but not in {self.config.model1_name}: {sorted(missing_in_model1)}"

            raise ValueError(error_msg)

        common_keys = model1_keys
        print(f"Total parameters (both models): {len(common_keys)}")

        layer_stats = defaultdict(LayerStats)

        for key in sorted(common_keys):
            val1 = model1_dict[key]
            val2 = model2_dict[key]

            is_q_level_comparison = isinstance(val1, tuple) and isinstance(val2, tuple)

            param1 = val1[0] if is_q_level_comparison else val1
            param2 = val2[0] if is_q_level_comparison else val2

            try:
                # Check shape compatibility
                if param1.shape != param2.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}"
                    )

                # Compute differences
                abs_max, abs_mean, rel_mean = self.compute_parameter_differences(
                    param1, param2, is_q_level=is_q_level_comparison
                )
                layer_id = self.extract_layer_id(key)

                # Store statistics
                layer_stats[layer_id].abs_max.append(abs_max)
                layer_stats[layer_id].abs_mean.append(abs_mean)
                layer_stats[layer_id].rel_mean.append(rel_mean)
                layer_stats[layer_id].param_names.append(key)

                # Store sample differences for histogram
                if len(layer_stats[layer_id].diffs) < self.config.max_diffs_per_layer:
                    diff = param1.float() - param2.float()
                    diff_sample = (
                        diff.flatten()[: self.config.max_diff_samples].cpu().numpy()
                    )
                    layer_stats[layer_id].diffs.append(diff_sample)

                # Log parameter-level statistics
                unit = "levels" if is_q_level_comparison else ""
                if layer_id >= 0:
                    print(
                        f"  Layer {layer_id:2d} - {key}: max={abs_max:.6f} {unit}, mean={abs_mean:.6f} {unit}, rel_mean={rel_mean:.6f}"
                    )
                else:
                    print(
                        f"  Other - {key}: max={abs_max:.6f} {unit}, mean={abs_mean:.6f} {unit}, rel_mean={rel_mean:.6f}"
                    )

            except Exception as e:
                print(f"Error processing {key}: {e}")

        print(f"  Completed: {len(common_keys)} parameters processed")
        return dict(layer_stats)

    def prepare_plot_data(
        self, stats: Dict[int, LayerStats]
    ) -> Tuple[List[float], List[float], List[float], List[str]]:
        """Prepare data for plotting."""
        all_layer_ids = sorted(stats.keys())
        layer_ids = [lid for lid in all_layer_ids if lid >= 0]

        abs_max_vals = []
        abs_mean_vals = []
        rel_mean_vals = []
        layer_labels = []

        # Process layer-wise statistics
        for lid in layer_ids:
            abs_max = np.max(stats[lid].abs_max) if stats[lid].abs_max else 0
            abs_mean = np.mean(stats[lid].abs_mean) if stats[lid].abs_mean else 0
            rel_mean = np.mean(stats[lid].rel_mean) if stats[lid].rel_mean else 0

            abs_max_vals.append(abs_max)
            abs_mean_vals.append(abs_mean)
            rel_mean_vals.append(rel_mean)
            layer_labels.append(f"L{lid}")

        # Add non-layer parameters
        if -1 in stats:
            abs_max = np.mean(stats[-1].abs_max) if stats[-1].abs_max else 0
            abs_mean = np.mean(stats[-1].abs_mean) if stats[-1].abs_mean else 0
            rel_mean = np.mean(stats[-1].rel_mean) if stats[-1].rel_mean else 0

            abs_max_vals.append(abs_max)
            abs_mean_vals.append(abs_mean)
            rel_mean_vals.append(rel_mean)
            layer_labels.append("Other")

        return abs_max_vals, abs_mean_vals, rel_mean_vals, layer_labels

    def create_plots(self, stats: Dict[int, LayerStats]) -> str:
        """Create comparison plots."""
        abs_max_vals, abs_mean_vals, rel_mean_vals, layer_labels = (
            self.prepare_plot_data(stats)
        )

        if not layer_labels:
            raise ValueError("No data available for plotting")

        print("\nCreating plots...")

        # Check if we are in q-level comparison mode
        is_q_level_comparison = self.config.quantize_before_compare
        y_label_suffix = " (Quant Levels)" if is_q_level_comparison else ""
        hist_xlabel = (
            "Difference (Quant Levels)" if is_q_level_comparison else "Difference Value"
        )

        title_ref_info = ""
        if is_q_level_comparison:
            ref_model_name = (
                self.config.model1_name
                if self.config.ref_model == "model1"
                else self.config.model2_name
            )
            title_ref_info = f" (Ref: {ref_model_name})"

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle(
            f"{self.config.model1_name} vs {self.config.model2_name} Comparison{title_ref_info}",
            fontsize=16,
        )

        x_positions = range(len(layer_labels))

        # Plot 1: Maximum absolute differences
        axes[0, 0].bar(x_positions, abs_max_vals, color="skyblue", alpha=0.7)
        axes[0, 0].set_xlabel("Layer")
        axes[0, 0].set_ylabel(f"Max Absolute Difference{y_label_suffix}")
        axes[0, 0].set_title(f"Maximum Absolute Differences by Layer")
        axes[0, 0].set_xticks(x_positions)
        # axes[0, 0].set_xticklabels(layer_labels, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Mean absolute differences
        axes[0, 1].bar(x_positions, abs_mean_vals, color="lightcoral", alpha=0.7)
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel(f"Mean Absolute Difference{y_label_suffix}")
        axes[0, 1].set_title(f"Mean Absolute Differences by Layer")
        axes[0, 1].set_xticks(x_positions)
        # axes[0, 1].set_xticklabels(layer_labels, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Relative mean differences
        axes[1, 0].bar(x_positions, rel_mean_vals, color="lightgreen", alpha=0.7)
        axes[1, 0].set_xlabel("Layer")
        axes[1, 0].set_ylabel("Mean Relative Difference")
        axes[1, 0].set_title("Mean Relative Differences by Layer")
        axes[1, 0].set_xticks(x_positions)
        # axes[1, 0].set_xticklabels(layer_labels, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Combined histogram
        all_diffs = []
        for layer_stats in stats.values():
            for diff_array in layer_stats.diffs:
                all_diffs.extend(diff_array)

        if all_diffs:
            axes[1, 1].hist(
                all_diffs, bins=100, alpha=0.7, color="purple", density=False
            )
            axes[1, 1].set_xlabel(hist_xlabel)
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_yscale("log")
            axes[1, 1].set_title("Combined Difference Histogram")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            self.config.save_dir
            / f"{self.config.output_prefix}_ref_{self.config.ref_model}.png"
        )
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        print(f"✓ Plot saved as '{output_path}'")

        return output_path

    def run_comparison(self) -> str:
        """Run the complete comparison analysis."""
        print("=" * 50)
        print(
            f"{self.config.model1_name} vs {self.config.model2_name} Comparison Analysis"
        )
        print("=" * 50)

        # Step 1: Load original models
        model1_orig_dict = self.load_model_weights(
            self.config.model1_dir, self.config.model1_name
        )
        model2_orig_dict = self.load_model_weights(
            self.config.model2_dir, self.config.model2_name
        )

        model1_to_compare = model1_orig_dict
        model2_to_compare = model2_orig_dict

        # Step 2: Apply quantization if configured
        if self.config.quantize_before_compare:
            print("\n🔢 Applying quantization and comparing q_tensors directly...")
            print(
                f"Quantizer Config: {self.config.quantizer_config.bit}-bit, "
                f"{self.config.quantizer_config.granularity}, "
                f"group_size={self.config.quantizer_config.group_size}"
            )

            # Determine which model is the reference
            if self.config.ref_model == "model1":
                ref_model_dict = model1_orig_dict
                ref_model_name = self.config.model1_name
                other_model_dict = model2_orig_dict
                other_model_name = self.config.model2_name
            else:  # model2 is the reference
                ref_model_dict = model2_orig_dict
                ref_model_name = self.config.model2_name
                other_model_dict = model1_orig_dict
                other_model_name = self.config.model1_name

            print(f"Reference model for qparams: {ref_model_name}")

            # 2a: Quantize the reference model and get its qparams
            print(f"\n--- Step 2a: Quantizing '{ref_model_name}' to get qparams ---")
            q_ref_model_dict = self.quantize_model_weights(ref_model_dict)

            # Extract scales and zeros from the quantized reference model
            ref_qparams = {
                k: (s, z)
                for k, (q, s, z) in q_ref_model_dict.items()
                if isinstance(q_ref_model_dict.get(k), tuple)
            }

            # 2b: Quantize the other model using the reference model's qparams
            print(
                f"\n--- Step 2b: Quantizing '{other_model_name}' using extracted qparams ---"
            )
            q_other_model_dict = self.quantize_model_weights(
                other_model_dict, external_qparams=ref_qparams
            )

            # Re-assign to model1 and model2 based on the original roles
            if self.config.ref_model == "model1":
                q_model1_dict = q_ref_model_dict
                q_model2_dict = q_other_model_dict
            else:
                q_model1_dict = q_other_model_dict
                q_model2_dict = q_ref_model_dict

            # Set models to compare to the quantized dictionaries (containing tuples)
            model1_to_compare = q_model1_dict
            model2_to_compare = q_model2_dict
            # model1_to_compare = {k: q for k, (q, s, z) in q_model1_dict.items()}
            # model2_to_compare = {k: q for k, (q, s, z) in q_model2_dict.items()}

        # Step 3: Analyze differences
        stats = self.analyze_differences(model1_to_compare, model2_to_compare)

        # Step 4: Create plots
        plot_path = self.create_plots(stats)

        print("\nAnalysis completed successfully!")
        return plot_path


def get_config() -> ComparisonConfig:
    """Create configuration for model comparison."""
    # --- Main Configuration ---
    QUANTIZE_BEFORE_COMPARE = (
        True  # Set to True to quantize both models before comparing
    )
    REFERENCE_MODEL = "model1"  # 'model1' or 'model2'. Determines which model's qparams are used as reference.

    # Model directories (used for all modes)
    # MODEL1_DIR = Path(
    #     "/ms/FM/chenshuailin/code/llmc/checkpoints/Qwen3-1.7B/tesseraqv2/quarot_tesseraqv2_w4a16g64_sym_s10_c4_noclip_seq4k_kl_top_mse_it100/dequant_model"
    # )
    # MODEL2_DIR = Path(
    #     "/ms/FM/chenshuailin/code/llmc/checkpoints/Qwen3-1.7B/tesseraqv2/quarot_tesseraqv2_w4a16g64_sym_s10_c4_noclip_seq4k_kl_top_mse_it100/fake_quant_model"
    # )
    MODEL1_DIR = Path(
        "checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/dequant_model"
    )
    MODEL2_DIR = Path(
        "checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_sym_dynamic/fake_quant_model"
    )

    # Quantizer settings (only used if QUANTIZE_BEFORE_COMPARE is True)
    QUANTIZER_CONFIG = QuantizerConfig(
        bit=4,
        symmetric=True,
        granularity="per_group",
        group_size=64,
        calib_algo="minmax",
        round_mode="round",
    )

    # --- Create Config Object ---
    return ComparisonConfig(
        model1_dir=MODEL1_DIR,
        model2_dir=MODEL2_DIR,
        model1_name="Dequant",
        model2_name="Fake Quant",
        output_prefix="quantized_comparison",
        save_dir=Path("figs/tesseraq/compare_fake_dequant_gptq"),
        # Quantization settings
        quantize_before_compare=QUANTIZE_BEFORE_COMPARE,
        quantizer_config=QUANTIZER_CONFIG if QUANTIZE_BEFORE_COMPARE else None,
        ref_model=REFERENCE_MODEL,
    )


def main():
    """Main function to run the comparison."""
    config = get_config()

    if config.quantize_before_compare:
        print("▶️ Running in Quantized Comparison Mode")
    else:
        print("▶️ Running in Standard Comparison Mode")

    comparator = ModelComparator(config)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
