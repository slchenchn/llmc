import os
import pickle
import re
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def plot_per_channel_max(
    ori_act,
    prefixed_act,
    save_dir,
    y_axis_type="log",
):
    ori_per_channel_max = ori_act.abs().max(dim=0)[0].float()
    prefixed_per_channel_max = prefixed_act.abs().max(dim=0)[0].float()
    # massive_index = torch.where(per_channel_max > 65_000)[0]
    # print(
    #     f"massive_index (abs > 65,000) (len: {len(massive_index)}): {massive_index}"
    # )

    # Create figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original Activations", "Prefixed Activations"),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Original activations in the first subplot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(ori_per_channel_max))),
            y=ori_per_channel_max.cpu().numpy(),
            name="Original",
            mode="lines",
            line=dict(color="blue", width=2),
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    # Prefixed activations in the second subplot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prefixed_per_channel_max))),
            y=prefixed_per_channel_max.cpu().numpy(),
            name="Prefixed",
            mode="lines",
            line=dict(color="red", width=2),
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Per-Channel Maximum Absolute Values Comparison",
        xaxis2_title="Channel Index",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Set y-axis type for both subplots
    fig.update_yaxes(title_text="Max Absolute Value", type=y_axis_type, row=1, col=1)
    fig.update_yaxes(title_text="Max Absolute Value", type=y_axis_type, row=2, col=1)

    # Save figure as png
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / "channel_max_comparison.png"
    fig.write_image(str(output_file))
    print(f"Plot saved to {output_file}")


def plot_per_token_max(
    ori_act,
    prefixed_act,
    save_dir,
    y_axis_type="log",
):
    ori_per_token_max = ori_act.abs().max(dim=1)[0].float()
    prefixed_per_token_max = prefixed_act.abs().max(dim=1)[0].float()

    # Create figure with two subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original Activations", "Prefixed Activations"),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Original activations in the first subplot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(ori_per_token_max))),
            y=ori_per_token_max.cpu().numpy(),
            name="Original",
            mode="lines",
            line=dict(color="blue", width=2),
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    # Prefixed activations in the second subplot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prefixed_per_token_max))),
            y=prefixed_per_token_max.cpu().numpy(),
            name="Prefixed",
            mode="lines",
            line=dict(color="red", width=2),
            opacity=0.5,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Per-Token Maximum Absolute Values Comparison",
        xaxis2_title="Token Index",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Set y-axis type for both subplots
    fig.update_yaxes(title_text="Max Absolute Value", type=y_axis_type, row=1, col=1)
    fig.update_yaxes(title_text="Max Absolute Value", type=y_axis_type, row=2, col=1)

    # Save figure as png
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / "token_max_comparison.png"
    fig.write_image(str(output_file))
    print(f"Plot saved to {output_file}")


def extract_act_by_pattern(state_dict, pattern):
    acts = []
    for key, value in state_dict.items():
        if re.search(pattern, key):
            if value.ndim == 3:
                value = value.squeeze()
                assert value.ndim == 2
            acts.append(value)
    return torch.cat(acts, dim=0)


def analysis_massive_outlier(
    ori_res_path, prefixed_res_path, y_axis_type="log", save_dir=None
):
    inspect_key = "down_proj"
    with open(ori_res_path, "rb") as f:
        ori_res = pickle.load(f)
    ori_act = extract_act_by_pattern(ori_res, inspect_key)
    del ori_res

    with open(prefixed_res_path, "rb") as f:
        prefixed_res = pickle.load(f)
    prefixed_act = extract_act_by_pattern(prefixed_res, inspect_key)
    del prefixed_res
    
    plot_per_token_max(ori_act, prefixed_act, save_dir, y_axis_type)

    # exclude the prefix token
    ori_act = ori_act[:-1]
    prefixed_act = prefixed_act[1:]
    plot_per_channel_max(ori_act, prefixed_act, save_dir, y_axis_type)


if __name__ == "__main__":
    pickled_res_paths = [
        {
            "ori_res": "analysis_model/DeepSeek-R1/bf16/pickled/20250423_041048/block60.res.pkl",
            "prefixed_res": "analysis_model/DeepSeek-R1/bf16/pickled/20250423_023357/block60.res.pkl",
            "save_dir": "analysis_model/DeepSeek-R1/bf16/PrefixQuant/block60",
        },
        {
            "ori_res": "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled/20250423_041048/block60.res.pkl",
            "prefixed_res": "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/pickled/20250423_023357/block60.res.pkl",
            "save_dir": "analysis_model/DeepSeek-R1/quarot/sym_w8a8_dynamic2/PrefixQuant/block60",
        },
    ]
    for res_path in pickled_res_paths:
        print("=" * 100)
        analysis_massive_outlier(
            Path(res_path["ori_res"]),
            Path(res_path["prefixed_res"]),
            save_dir=Path(res_path["save_dir"]),
        )

    print("done")
