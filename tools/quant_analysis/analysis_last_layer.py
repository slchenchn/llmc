import os
import pickle
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def compare_before_and_after_normed():
    def _compare_act(res_path):
        with open(res_path, "rb") as f:
            res = pickle.load(f)
        input_act = res["block_61.input_layernorm"].abs()
        normed_act = res["block_61.lm_head"].abs()
        print("input_act:")
        print(f"shape: {input_act.shape}")
        print(f"min: {input_act.min():,}")
        print(f"max: {input_act.max():,}")
        print(f"mean: {input_act.mean():,}")
        print(f"std: {input_act.std():,}")

        print("-" * 80)
        print("normed_act:")
        print(f"shape: {normed_act.shape}")
        print(f"min: {normed_act.min():,}")
        print(f"max: {normed_act.max():,}")
        print(f"mean: {normed_act.mean():,}")
        print(f"std: {normed_act.std():,}")

    pickled_res_paths = [
        "/analysis_model/DeepSeek-R1/bf16/pickled/block61.res.pkl",
        "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled.bk/block61.res.pkl",
    ]
    for res_path in pickled_res_paths:
        print("=" * 100)
        _compare_act(Path(res_path))


def analysis_massive_outlier():
    def _analysis_massive_outlier(res_path, y_axis_type="log"):
        with open(res_path, "rb") as f:
            res = pickle.load(f)
        input_act = res["block_61.input_layernorm"]
        print("input_act:")
        print(f"shape: {input_act.shape}")
        print(f"abs max: {input_act.abs().max():,}, abs min: {input_act.abs().min():,}, abs mean: {input_act.abs().mean():,}")

        per_channel_max = input_act.abs().max(dim=0)[0].max(dim=0)[0].float()
        massive_index = torch.where(per_channel_max > 65_000)[0]
        print(f"massive_index (abs > 65,000) (len: {len(massive_index)}): {massive_index}")

        # Create figure with log scale y-axis
        fig = make_subplots()
        fig.add_trace(
            # go.Bar(
            go.Scatter(
                x=list(range(len(per_channel_max))),
                y=per_channel_max.cpu().numpy(),
                name="Channel Max Values",
                # marker_color='black',
            )
        )

        fig.update_layout(
            title=f"Per-Channel Maximum Absolute Values - {Path(res_path).stem}",
            xaxis_title="Channel Index",
            yaxis_title="Max Absolute Value",
            yaxis_type=y_axis_type,
        )

        # Save figure as png
        output_file = res_path.parent / f"{Path(res_path).stem}_channel_max.png"
        fig.write_image(str(output_file))
        print(f"Plot saved to {output_file}")

    pickled_res_paths = [
        "analysis_model/DeepSeek-R1/bf16/pickled/20250418_104623/block61.res.pkl",
        "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled/20250418_104623/block61.res.pkl"

    ]
    for res_path in pickled_res_paths:
        print("=" * 100)
        _analysis_massive_outlier(Path(res_path))


if __name__ == "__main__":
    # compare_before_and_after_normed()
    analysis_massive_outlier()

    print("done")
