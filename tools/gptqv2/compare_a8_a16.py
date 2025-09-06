import re
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class QuantizationMetrics:
    """Container for quantization metrics"""

    H_norm: List[float]
    dXXT_norm: List[float]
    ratio: List[float]
    P_norm: List[float]
    module: List[str]


class LogAnalyzer:
    """Analyzes GPTQv2 log files for quantization metrics"""

    PATTERNS = {
        "module": r"\[GPTQv2\]\[(.*?)\]",
        "H_norm": r"\|\|H\|\|_F=([\d.e+-]+)",
        "dXXT_norm": r"\|\|dXXT\|\|_F=([\d.e+-]+)",
        "ratio": r"ratio=([\d.]+)%",
        "P_norm": r"\|\|P\|\|_F=([\d.e+-]+)",
    }

    @staticmethod
    def extract_metrics(log_path: str) -> QuantizationMetrics:
        """Extract quantization metrics from log file"""
        metrics = QuantizationMetrics(
            H_norm=[], dXXT_norm=[], ratio=[], P_norm=[], module=[]
        )

        with open(log_path) as f:
            for line in f:
                if not ("[GPTQv2]" in line and ("pre-P:" in line or "post-P:" in line)):
                    continue

                module = re.search(LogAnalyzer.PATTERNS["module"], line)
                metrics.module.append(module.group(1) if module else "unknown")

                for metric in ["H_norm", "dXXT_norm", "ratio"]:
                    match = re.search(LogAnalyzer.PATTERNS[metric], line)
                    if match:
                        getattr(metrics, metric).append(float(match.group(1)))

                p_match = re.search(LogAnalyzer.PATTERNS["P_norm"], line)
                if p_match:
                    metrics.P_norm.append(float(p_match.group(1)))

        return metrics


class Visualizer:
    """Handles visualization of quantization metrics"""

    @staticmethod
    def plot_comparison(
        a16_metrics: QuantizationMetrics,
        a8_metrics: QuantizationMetrics,
        metric_name: str,
        title: str,
    ) -> None:
        """Generate comparison plots for given metric"""
        plt.figure(figsize=(12, 6))

        # Get data for both configurations
        a16_data = getattr(a16_metrics, metric_name)
        a8_data = getattr(a8_metrics, metric_name)

        # Calculate bins using logspace for uniform bin width in log scale
        a16_data = [x for x in a16_data if x != 0]
        a8_data = [x for x in a8_data if x != 0]
        all_data = a16_data + a8_data
        assert min(all_data) > 0, f'Data contains non-positive values: {all_data}'
        
        min_val = max(min(all_data), 1e-10)  # Avoid log(0)
        max_val = max(all_data)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)

        plt.subplot(1, 2, 1)
        plt.hist(a16_data, bins=bins, alpha=0.7)
        plt.title(f"w4a16 {title}")
        plt.xlabel(metric_name)
        plt.ylabel("Frequency")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)

        # Add statistical lines
        if a16_data:
            mean_val = np.mean(a16_data)
            median_val = np.median(a16_data)
            percentile_5 = np.percentile(a16_data, 5)
            percentile_95 = np.percentile(a16_data, 95)

            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
            plt.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2e}')
            plt.axvline(percentile_5, color='blue', linestyle=':', linewidth=2, label=f'5%: {percentile_5:.2e}')
            plt.axvline(percentile_95, color='orange', linestyle='-', linewidth=2, label=f'95%: {percentile_95:.2e}')
            plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(a8_data, bins=bins, alpha=0.7)
        plt.title(f"w4a8 {title}")
        plt.xlabel(metric_name)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)

        # Add statistical lines
        if a8_data:
            mean_val = np.mean(a8_data)
            median_val = np.median(a8_data)
            percentile_5 = np.percentile(a8_data, 5)
            percentile_95 = np.percentile(a8_data, 95)

            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')
            plt.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2e}')
            plt.axvline(percentile_5, color='blue', linestyle=':', linewidth=2, label=f'5%: {percentile_5:.2e}')
            plt.axvline(percentile_95, color='orange', linestyle='-', linewidth=2, label=f'95%: {percentile_95:.2e}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"comparison_{metric_name}.png")
        plt.close()


if __name__ == "__main__":
    # Process logs
    a16 = "logs/quarot.gptqv2.qwen3.0.6B.w4a16.sym_20250905_155546.log"
    a8 = "logs/quarot.olrotate.gptqv2.qwen3.0.6B.w4a8.sym_20250905_155620.log"

    analyzer = LogAnalyzer()
    a16_metrics = analyzer.extract_metrics(a16)
    a8_metrics = analyzer.extract_metrics(a8)

    # Generate visualizations
    visualizer = Visualizer()
    for metric in ["H_norm", "dXXT_norm", "ratio", "P_norm"]:
        visualizer.plot_comparison(
            a16_metrics, a8_metrics, metric, f"{metric} comparison"
        )

    print("Comparison plots saved as:")
    print("- comparison_H_norm.png")
    print("- comparison_dXXT_norm.png")
    print("- comparison_ratio.png")
    print("- comparison_P_norm.png")
