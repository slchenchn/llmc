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

        input_act = [res["block_60.mlp.shared_experts.down_proj"]]
        for i in range(256):
            try:
                cur_down_proj = res[f"block_60.mlp.experts.{i}.down_proj"]
                if cur_down_proj.ndim == 2:
                    cur_down_proj = cur_down_proj.unsqueeze(0)
            except KeyError:
                continue
            input_act.append(cur_down_proj)
        input_act = torch.cat(input_act, dim=1)
        print("input_act:")
        print(f"shape: {input_act.shape}")
        print(f"abs max: {input_act.abs().max():,}")

        per_channel_max = input_act.abs().max(dim=1)[0].max(dim=0)[0].float()

        # Create figure with log scale y-axis
        fig = make_subplots()
        fig.add_trace(
            # go.Bar(
            go.Scatter(
                x=list(range(len(per_channel_max))),
                y=per_channel_max.cpu().numpy(),
                name="Channel Max Values",
                # marker_color="royalblue",
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
        # "analysis_model/DeepSeek-R1/bf16/pickled/20250418_104623/block61.res.pkl",
        "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled/20250418_104623/block60.res.pkl"
    ]
    for res_path in pickled_res_paths:
        print("=" * 100)
        _analysis_massive_outlier(Path(res_path))


if __name__ == "__main__":
    # compare_before_and_after_normed()
    analysis_massive_outlier()

    print("done")
