from pathlib import Path
from loguru import logger

import torch
from safetensors.torch import load_file, save_file
from natsort import natsorted
import math
import os
import json
from collections import defaultdict
import sys
import scipy

sys.path.append(str(Path(__file__).parent.parent.parent))
from llmc.compression.quantization.quant import IntegerQuantizer

from quant_mtp_block import (
    plot_histogram,
    get_hadamard_matrix,
    quant_mtp_block,
    save_mtp_block_v2,
    get_param_by_pat,
    fuse_ln_layers,
    rotate_pre_layers,
    rotate_post_layers,
)


def create_random_hadamard_matrix(ori_had_mat):
    return ori_had_mat
    n = ori_had_mat.shape[0]
    while True:
        # find a different random hadamard matrix
        # random_had_mat = scipy.linalg.hadamard(n)
        random_had_mat = ori_had_mat.T.clone()
        if torch.norm(random_had_mat - ori_had_mat) > 1e-2:
            break
    return random_had_mat


def rotate_mtp_embeddings(state_dict, mtp_pat, rotated_emb, ori_had_mat, new_had_mat):
    emb_dim = rotated_emb.shape[1]

    # re-rotate for input activation
    eh_proj_name = mtp_pat + "eh_proj.weight"
    eh_proj = state_dict[eh_proj_name]
    ori_dtype = eh_proj.dtype
    eh_proj = eh_proj.view(-1, 2, emb_dim).double()
    # eh_proj[:, 1, :] = eh_proj[:, 1, :] @ ori_had_mat
    eh_proj = eh_proj @ ori_had_mat
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj after first rotation",
        f"logs/mtp_hists_v4/{mtp_pat}eh_proj_after_first_rotation.png",
        split_by_second_dim=True,
    )

    # rotate for output activation
    eh_proj = eh_proj.view(-1, 2 * emb_dim).contiguous()
    eh_proj = new_had_mat.T @ eh_proj
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj after second rotation",
        f"logs/mtp_hists_v4/{mtp_pat}eh_proj_after_second_rotation.png",
        split_by_second_dim=True,
    )
    state_dict[eh_proj_name] = eh_proj.to(ori_dtype)
    logger.info(f"rotate eh_proj with had_mat")

    mtp_emb_name = mtp_pat + "embed_tokens.weight"
    mtp_emb = state_dict[mtp_emb_name].clone()
    ori_dtype = mtp_emb.dtype
    mtp_emb = mtp_emb.double() @ ori_had_mat
    state_dict[mtp_emb_name] = mtp_emb.to(ori_dtype)
    logger.info(f"rotate MTP embeddings with ori_had_mat")


def rotate_mtp_block(
    model_dir: Path, mtp_pat: str, had_mat: torch.Tensor, num_experts: int, emb_pat: str
):
    logger.info(f"---------------- rotate_mtp_block ----------------")
    state_dict = get_param_by_pat(model_dir, mtp_pat)
    fuse_ln_layers(state_dict, mtp_pat, num_experts)

    rotated_emb = get_param_by_pat(model_dir, emb_pat)
    new_had_mat = create_random_hadamard_matrix(had_mat)
    rotate_mtp_embeddings(state_dict, mtp_pat, rotated_emb, had_mat, new_had_mat)
    del had_mat
    rotate_pre_layers(state_dict, mtp_pat, new_had_mat, num_experts)
    rotate_post_layers(state_dict, mtp_pat, new_had_mat, num_experts)
    return state_dict


if __name__ == "__main__":
    ori_model_dir = Path("/ms/FM/checkpoints/deepseek-ai/DeepSeek-R1-bf16")
    rotated_model_root = Path("checkpoints/DeepSeek-R1/quarot/sym_w8a8_dynamic2")
    save_model_dir = rotated_model_root / "vllm_quant_model_mtp_v4"
    n_bits = 8
    mtp_pat = "model.layers.61."
    emb_pat = "model.embed_tokens.weight"
    num_experts = 256
    quant_granularity = "per_channel"

    log_file = "logs/quant_mtp_block_v4.log"
    logger.add(log_file)
    logger.info(f"Logging to {log_file}")

    had_mat = get_hadamard_matrix(ori_model_dir, rotated_model_root, emb_pat)
    state_dict = rotate_mtp_block(ori_model_dir, mtp_pat, had_mat, num_experts, emb_pat)
    state_dict = quant_mtp_block(state_dict, quant_granularity, n_bits)

    save_mtp_block_v2(
        state_dict, ori_model_dir, save_model_dir, rotated_model_root, mtp_pat
    )
