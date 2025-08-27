from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def format_number(value):
    """Format large numbers in abbreviated form with one decimal place"""
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.1f}k"
    else:
        return f"{value:.1f}"


def read_log(log_path, pats):
    results = defaultdict(list)
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            for pat_name, pat_val in pats.items():
                if re.search(pat_val, line):
                    results[pat_name].append(float(re.search(pat_val, line).group(1)))
                    break
    return results


def plot_and_save(results, save_fig_dir, title, annotation_interval=5):
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    # Create subplots for before and after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set the main title for both subplots
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Plot before measurements
    for log_name, log_results in results.items():
        if "before" in log_results:
            values = log_results["before"]
            # Plot all points
            x_plot = list(range(len(values)))
            y_plot = values

            line = ax1.plot(x_plot, y_plot, label=log_name, marker="o")

            # Add text annotations for y-values every n points
            for i, (x, y) in enumerate(zip(x_plot, y_plot)):
                if (
                    i % annotation_interval == 0 or i == len(values) - 1
                ):  # Annotate every n points plus the last point
                    ax1.annotate(
                        format_number(y),
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                    )

    ax1.set_xlabel("layer")
    ax1.set_ylabel("reconstruction loss")
    ax1.grid(True)
    # ax1.set_yscale("log")
    ax1.set_title("Before TesseraQ")
    ax1.legend()

    # Plot after measurements
    for log_name, log_results in results.items():
        if "after" in log_results:
            values = log_results["after"]
            # Plot all points
            x_plot = list(range(len(values)))
            y_plot = values

            line = ax2.plot(x_plot, y_plot, label=log_name, marker="s")

            # Add text annotations for y-values every n points
            for i, (x, y) in enumerate(zip(x_plot, y_plot)):
                if (
                    i % annotation_interval == 0 or i == len(values) - 1
                ):  # Annotate every n points plus the last point
                    ax2.annotate(
                        format_number(y),
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                    )

    ax2.set_xlabel("layer")
    ax2.set_ylabel("reconstruction loss")
    ax2.grid(True)
    # ax2.set_yscale("log")
    ax2.set_title("After TesseraQ")
    ax2.legend()

    plt.tight_layout()
    save_path = save_fig_dir / ("--".join(log_paths.keys()) + ".png")
    plt.savefig(save_path)  # 保存为一张图
    print(f"Saved figure to {save_path}")
    plt.close()


if __name__ == "__main__":
    configs = {
        # "group_size_128_sym": {
        #     "title": "group_size=128, sym",
        #     "w_awq": "logs/2.tesseraq.qwen3-1.7B.w4a16g128.sym_20250719_011441.log",
        #     "wo_awq": "logs/step2_tesseraq_qwen3-1.7B_sym_s10_w4a16_g128_20250718_064042.log",
        # },
        # "awq_tesseraq_group128": {
        #     "title": "AWQ + TesseraQ, group_size=128",
        #     "asym": "logs/2.tesseraq.qwen3-1.7B.w4a16g128_20250718_083038.log",
        #     "sym": "logs/2.tesseraq.qwen3-1.7B.w4a16g128.sym_20250719_011441.log",
        # },
        # "awq_tesseraq_sym": {
        #     "title": "AWQ + TesseraQ, sym",
        #     "g128": "logs/2.tesseraq.qwen3-1.7B.w4a16g128.sym_20250719_011441.log",
        #     "g64": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_024006.log",
        # },
        # "group_size_64_sym": {
        #     "title": "group_size=64, sym",
        #     "vanilla": "logs/tesseraq_qwen3-1.7B_sym_s10_w4a16_g64_wiki2_20250721_073206.log",
        #     # "awq": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_024006.log",
        #     "quarot": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_032336.log",
        #     "quarot_fp32": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.fp32_20250721_080230.log",
        #     # "quarot-awq": "logs/3.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_060147.log",
        # },
        # "awq_group_size_64_sym": {
        #     "title": "AWQ group_size=64, sym",
        #     "seq2k": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_024006.log",
        #     "seq4k": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.seq4k_20250720_065503.log",
        #     "seq8k": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.seq8k_20250720_070807.log",
        # },
        # "group_size_64_sym": {
        #     "title": "group_size=64, sym, c4, seq8k",
        #     # "vanilla-clipv1-wiki2": "logs/tesseraq_qwen3-1.7B_sym_s10_w4a16_g64_wiki2_20250721_073206.log",
        #     # "vanilla-noclip-c4": "logs/tesseraq_qwen3-1.7B_sym_s10_w4a16_g64_c4_noclip_20250721_110638.log",
        #     # "awq-clipv1": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq8k_20250720_115019.log",
        #     # "awq-noclip": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq8k.noclip_20250721_111937.log",
        #     "quarot-clipv1-wiki2-seq2k": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_032336.log",
        #     "quarot-noclip-c4-seq8k": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq8k.noclip_20250721_111216.log",
        # },
        # "group_size_64_sym_c4_seq8k": {
        #     "title": "group_size=64, sym, c4, seq8k",
        #     # "vanilla": "logs/tesseraq_qwen3-1.7B_sym_s10_w4a16_g64_wiki2_20250721_073206.log",
        #     "awq": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq8k.noclip_20250721_111937.log",
        #     "quarot_awq": "logs/3.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_060147.log",
        #     # "quarot_fp32": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.fp32_20250721_080230.log",
        #     # "quarot-awq": "logs/3.tesseraq.qwen3-1.7B.w4a16g64.sym_20250720_060147.log",
        # },
        # "group_size_64_sym_c4_seq4k_kl": {
        #     "title": "group_size=64, sym, c4, seq4k, kl",
        #     "s10_it25": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it25_20250723_030158.log",
        #     "s10_it50": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it50_20250723_013929.log",
        #     # "s10_it100": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100_20250722_084304.log",
        #     "s10_it250": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl_20250722_082505.log",
        #     "s20_it250": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.s20.c4.seq4k.noclip.kl_20250722_091015.log",
        # },
        # "group_size_64_sym_c4_seq4k_kl_s10_it100": {
        #     "title": "group_size=64, sym, c4, seq4k, kl, s10, it100",
        #     "awq": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100_20250722_084304.log",
        #     "quarot": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100_20250723_013911.log",
        # },
        "group_size_64_sym_c4_seq4k_kl_s10_it100": {
            "title": "group_size=64, sym, c4, seq4k, kl, s10, it100",
            # "v1-vanilla": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100_20250722_084304.log",
            # "v1-lr10x": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.lr10x.it100_20250724_071251.log",
            "v2-vanilla": "logs/2.tesseraqv2.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100_20250724_065108.log",
            "v2-10xrd2": "logs/2.tesseraqv2.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100.10xrd2_20250724_071455.log",
            "v2-5xrd2": "logs/2.tesseraqv2.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it100.5xrd2_20250724_081005.log",
        },
        # "group_size_64_sym_c4_seq4k_kl": {
        #     "title": "group_size=64, sym, c4, seq4k, kl",
        #     "s10_it25": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it25_20250723_030158.log",
        #     "s10_it50": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.c4.seq4k.noclip.kl.it50_20250723_013929.log",
        #     "s20_it25": "logs/2.tesseraq.qwen3-1.7B.w4a16g64.sym.s20.c4.seq4k.noclip.kl.it25_20250723_063453.log",
        # },
    }

    pats = {
        "before": "Before \w+, the \w+ loss: (\d+\.\d+)",
        "after": "After \w+, the \w+ loss: (\d+\.\d+)",
    }
    save_fig_dir = Path("figs/tesseraq/training_dynamics")

    # Process each configuration
    for config_name, config in configs.items():
        title = config["title"]
        log_paths = {k: v for k, v in config.items() if k != "title"}

        results = {}
        for log_name, log_path in log_paths.items():
            log_path = Path(log_path)
            results[log_name] = read_log(log_path, pats)

        plot_and_save(results, save_fig_dir, title)
