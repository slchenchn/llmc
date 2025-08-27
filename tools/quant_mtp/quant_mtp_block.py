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

sys.path.append(str(Path(__file__).parent.parent.parent))
from llmc.compression.quantization.quant import IntegerQuantizer


def plot_histogram(tensor, title, save_path, split_by_second_dim=False):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    if split_by_second_dim and tensor.ndim >= 2:
        # Split by second dimension into two parts
        mid_point = tensor.shape[1] // 2
        part1 = tensor[:, :mid_point]
        part2 = tensor[:, mid_point:]

        # Plot histograms with different colors
        plt.hist(
            part1.cpu().float().numpy().flatten(),
            bins=200,
            alpha=0.7,
            label="embed",
            color="blue",
            log=True,
        )
        plt.hist(
            part2.cpu().float().numpy().flatten(),
            bins=200,
            alpha=0.7,
            label="hidden_states",
            color="red",
            log=True,
        )
        plt.legend()
    else:
        # Original single histogram
        plt.hist(tensor.cpu().float().numpy().flatten(), bins=200, log=True)

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency (log scale)")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved histogram to {save_path}")


def get_model_index_path(model_dir: Path):
    safetensors_index_path = model_dir / "model.safetensors.index.json"
    bin_index_path = model_dir / "pytorch_model.bin.index.json"
    if safetensors_index_path.exists():
        index_path = safetensors_index_path
    elif bin_index_path.exists():
        index_path = bin_index_path
    else:
        raise FileNotFoundError(f"No index file found in {model_dir}")
    return index_path


def load_shard(shard_path: Path):
    if shard_path.suffix == ".safetensors":
        shard_ckpt = load_file(shard_path)
    else:
        shard_ckpt = torch.load(str(shard_path), map_location="cpu")
    return shard_ckpt


def get_param_by_pat(model_dir: Path, param_pat: str):
    index_path = get_model_index_path(model_dir)

    model_state = {}
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files_to_load = set()
    for k, v in weight_map.items():
        if param_pat in k:
            shard_files_to_load.add(v)

    for shard_file in shard_files_to_load:
        shard_path = model_dir / shard_file
        model_state.update(load_shard(shard_path))

    shard_params = {}
    for k, v in model_state.items():
        if param_pat in k:
            shard_params[k] = v.cuda()

    if len(shard_params) == 1:
        shard_params = list(shard_params.values())[0]

    return shard_params


def get_hadamard_matrix(ori_model_dir: Path, rotated_model_root: Path, emb_pat: str):
    logger.info(f"---------------- get_hadamard_matrix ----------------")
    ori_emb = get_param_by_pat(ori_model_dir, emb_pat)
    rotated_model_dir = rotated_model_root / "transformed_model"
    rotated_emb = get_param_by_pat(rotated_model_dir, emb_pat)
    assert ori_emb.shape == rotated_emb.shape
    logger.info(f"emb.shape: {ori_emb.shape}")

    # B=A^T C(C^T C)^-1，通过伪逆求解
    # had_mat = ori_emb.T @ torch.linalg.pinv(rotated_emb)
    had_mat = torch.linalg.lstsq(ori_emb.double(), rotated_emb.double()).solution

    # round to -1 and +1
    scale = math.sqrt(ori_emb.shape[1])
    had_mat = had_mat * scale
    had_mat_q = torch.sign(had_mat).double()

    # evaluate error
    n = had_mat_q.shape[0]
    identity = torch.eye(n, device=had_mat_q.device)
    ortho_error = torch.norm(had_mat_q @ had_mat_q.T - n * identity)
    logger.info(f"Orthogonality error of rounded hadamard matrix: {ortho_error}")

    reconstructed_emb = ori_emb.double() @ had_mat_q / scale

    mse = torch.nn.functional.mse_loss(reconstructed_emb, rotated_emb.float())
    logger.info(f"Reconstruction MSE: {mse}")

    return had_mat_q / scale


def fuse_ln_layers(state_dict, mtp_pat, num_experts):
    enorm = state_dict[mtp_pat + "enorm.weight"]
    hnorm = state_dict[mtp_pat + "hnorm.weight"]
    ehnorm = torch.cat([enorm, hnorm], dim=0).unsqueeze(0)
    eh_proj = state_dict[mtp_pat + "eh_proj.weight"]
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj before scaling",
        f"logs/mtp_hists/{mtp_pat}eh_proj_before_scaling.png",
        split_by_second_dim=True,
    )
    eh_proj = (eh_proj.double() * ehnorm.double()).to(eh_proj.dtype)
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj after scaling",
        f"logs/mtp_hists/{mtp_pat}eh_proj_after_scaling.png",
        split_by_second_dim=True,
    )
    state_dict[mtp_pat + "eh_proj.weight"] = eh_proj
    enorm.fill_(1.0)
    hnorm.fill_(1.0)
    state_dict[mtp_pat + "enorm.weight"] = enorm
    state_dict[mtp_pat + "hnorm.weight"] = hnorm
    logger.info(f"fuse eh_proj with enorm and hnorm")

    layers = []
    # attention
    layers.append(
        {
            "layers": ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
            "prev_op": "input_layernorm",
        }
    )
    # mlp
    layers.append(
        {
            "layers": [f"mlp.experts.{i}.gate_proj" for i in range(num_experts)]
            + [f"mlp.experts.{i}.up_proj" for i in range(num_experts)]
            + [
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.gate",
            ],
            "prev_op": "post_attention_layernorm",
        }
    )
    # head
    layers.append(
        {
            "layers": ["shared_head.head"],
            "prev_op": "shared_head.norm",
        }
    )
    for layer in layers:
        ln_name = mtp_pat + layer["prev_op"]
        assert (ln_name + ".bias") not in state_dict, "ln bias is not supported"
        ln_weight = state_dict[ln_name + ".weight"]
        for fc in layer["layers"]:
            fc_name = mtp_pat + fc
            assert (fc_name + ".bias") not in state_dict, "fc bias is not supported"
            fc_weight = state_dict[fc_name + ".weight"]
            fc_weight = (fc_weight.double() * ln_weight.double()).to(fc_weight.dtype)
            state_dict[fc_name + ".weight"] = fc_weight
            logger.info(f"fuse {fc_name} with {ln_name}")
        ln_weight.fill_(1.0)
        state_dict[ln_name + ".weight"] = ln_weight
    return layers


def rotate_mtp_embeddings(state_dict, mtp_pat, rotated_emb, had_mat):
    emb_dim = rotated_emb.shape[1]

    # re-rotate for input activation
    eh_proj = state_dict[mtp_pat + "eh_proj.weight"]
    ori_dtype = eh_proj.dtype
    eh_proj = eh_proj.view(-1, 2, emb_dim).double()
    eh_proj = eh_proj @ had_mat
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj after first rotation",
        f"logs/mtp_hists/{mtp_pat}eh_proj_after_first_rotation.png",
        split_by_second_dim=True,
    )

    # rotate for output activation
    eh_proj = eh_proj.view(-1, 2 * emb_dim).contiguous()
    eh_proj = had_mat.T @ eh_proj
    plot_histogram(
        eh_proj,
        f"{mtp_pat}eh_proj after second rotation",
        f"logs/mtp_hists/{mtp_pat}eh_proj_after_second_rotation.png",
        split_by_second_dim=True,
    )
    state_dict[mtp_pat + "eh_proj.weight"] = eh_proj.to(ori_dtype)
    logger.info(f"rotate eh_proj with had_mat")

    state_dict[mtp_pat + "embed_tokens.weight"] = rotated_emb
    logger.info(f"replace MPT block's embed_tokens with rotated_emb")


def rotate_pre_layers(state_dict, mtp_pat, had_mat, num_experts):
    layers = []
    # attention
    layers.append(
        {
            "layers": ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
            "prev_op": "input_layernorm",
        }
    )
    # mlp
    layers.append(
        {
            "layers": [f"mlp.experts.{i}.gate_proj" for i in range(num_experts)]
            + [f"mlp.experts.{i}.up_proj" for i in range(num_experts)]
            + [
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.gate",
            ],
            "prev_op": "post_attention_layernorm",
        }
    )
    # head
    layers.append(
        {
            "layers": ["shared_head.head"],
            "prev_op": "shared_head.norm",
        }
    )
    for layer in layers:
        for fc in layer["layers"]:
            fc_name = mtp_pat + fc
            assert (fc_name + ".bias") not in state_dict, "fc bias is not supported"
            fc_weight = state_dict[fc_name + ".weight"]
            ori_dtype = fc_weight.dtype
            fc_weight = fc_weight.double() @ had_mat
            fc_weight = fc_weight.to(ori_dtype)
            state_dict[fc_name + ".weight"] = fc_weight
            logger.info(f"rotate {fc_name} with pre-had_mat")


def rotate_post_layers(state_dict, mtp_pat, had_mat, num_experts):
    layers = []
    # routed experts
    layers.append(
        {
            "layers": [f"mlp.experts.{i}.down_proj" for i in range(num_experts)],
            "prev_op": [f"mlp.experts.{i}.up_proj" for i in range(num_experts)],
        }
    )
    # shared experts
    layers.append(
        {
            "layers": ["mlp.shared_experts.down_proj"],
            "prev_op": ["mlp.shared_experts.up_proj"],
        }
    )
    # o_proj
    layers.append(
        {
            "layers": ["self_attn.o_proj"],
            "prev_op": None,
        }
    )
    for layer in layers:
        for fc in layer["layers"]:
            fc_name = mtp_pat + fc
            assert (fc_name + ".bias") not in state_dict, "fc bias is not supported"
            fc_weight = state_dict[fc_name + ".weight"]
            ori_dtype = fc_weight.dtype
            fc_weight = had_mat.T @ fc_weight.double()
            fc_weight = fc_weight.to(ori_dtype)
            state_dict[fc_name + ".weight"] = fc_weight
            logger.info(f"rotate {fc_name} with post-had_mat")


def rotate_mtp_block(
    model_dir: Path, mtp_pat: str, had_mat: torch.Tensor, num_experts: int, emb_pat: str
):
    logger.info(f"---------------- rotate_mtp_block ----------------")
    state_dict = get_param_by_pat(model_dir, mtp_pat)
    fuse_ln_layers(state_dict, mtp_pat, num_experts)

    rotated_emb = get_param_by_pat(model_dir, emb_pat)
    rotate_mtp_embeddings(state_dict, mtp_pat, rotated_emb, had_mat)

    rotate_pre_layers(state_dict, mtp_pat, had_mat, num_experts)
    rotate_post_layers(state_dict, mtp_pat, had_mat, num_experts)
    return state_dict


def quant_mtp_block(state_dict, quant_granularity, n_bits):
    logger.info(f"---------------- quant_mtp_block ----------------")
    quantizer = IntegerQuantizer(
        bit=n_bits, symmetric=True, granularity=quant_granularity
    )
    names = sorted(list(state_dict.keys()))
    for name in names:
        weight = state_dict[name]
        if weight.ndim != 2:
            continue

        if (
            "embed_tokens" in name
            or "head" in name
            or "eh_proj" in name
            or ".gate." in name
        ):
            logger.info(f"skip {name} as not quantized")
            continue

        quant_weight, scales, zeros = quantizer.real_quant_weight_dynamic(weight)
        assert zeros is None, "zeros is not supported"
        state_dict[name] = quant_weight
        state_dict[name + "_scale"] = scales
        logger.info(f"quant {name}")
    return state_dict


def save_mtp_block_v2(
    state_dict, ori_model_dir, save_model_dir, rotated_model_root, mtp_pat
):
    logger.info(f"---------------- save_mtp_block ----------------")
    save_model_dir.mkdir(parents=True, exist_ok=True)
    rotated_model_dir = rotated_model_root / "vllm_quant_model"
    index_path = get_model_index_path(ori_model_dir)
    with open(index_path) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    mtp_weights = {}
    for name, shard_file in weight_map.items():
        if mtp_pat in name:
            mtp_weights[name] = state_dict[name]

    index_path = get_model_index_path(rotated_model_dir)
    with open(index_path) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    safetensor = natsorted(rotated_model_dir.glob("*.safetensors"))[-1]
    rotated_model_ckpt = load_file(safetensor)
    for name, weight in mtp_weights.items():
        assert name not in rotated_model_ckpt, f"name {name} already exists"
        rotated_model_ckpt[name] = weight
        weight_map[name] = safetensor.name
        if name + "_scale" in state_dict:
            assert weight.dtype == torch.int8
            rotated_model_ckpt[name + "_scale"] = state_dict[name + "_scale"]
            weight_map[name + "_scale"] = safetensor.name
        logger.info(f"Updated {name}")
    dst_path = save_model_dir / safetensor.name
    try:
        os.remove(dst_path)
    except FileNotFoundError:
        pass
    save_file(rotated_model_ckpt, dst_path)

    logger.info(f"Saved new model to {save_model_dir}")
    model_index["weight_map"] = weight_map
    try:
        os.remove(save_model_dir / "model.safetensors.index.json")
    except FileNotFoundError:
        pass
    with open(save_model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(model_index, f, indent=4)


if __name__ == "__main__":
    ori_model_dir = Path("/ms/FM/checkpoints/deepseek-ai/DeepSeek-R1-bf16")
    rotated_model_root = Path("checkpoints/DeepSeek-R1/quarot/sym_w8a8_dynamic2")
    save_model_dir = rotated_model_root / "vllm_quant_model_mtp"
    n_bits = 8
    mtp_pat = "model.layers.61."
    emb_pat = "model.embed_tokens.weight"
    num_experts = 256
    quant_granularity = "per_channel"

    log_file = "logs/quant_mtp_block.log"
    logger.add(log_file)
    logger.info(f"Logging to {log_file}")

    had_mat = get_hadamard_matrix(ori_model_dir, rotated_model_root, emb_pat)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(had_mat, save_model_dir / "had_mat.pt")

    state_dict = rotate_mtp_block(ori_model_dir, mtp_pat, had_mat, num_experts, emb_pat)
    state_dict = quant_mtp_block(state_dict, quant_granularity, n_bits)

    save_mtp_block_v2(
        state_dict, ori_model_dir, save_model_dir, rotated_model_root, mtp_pat
    )
