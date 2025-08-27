import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import torch
from loguru import logger

from llmc.data import BaseDataset
from llmc.eval.eval_ppl import PerplexityEval
from llmc.models import *
from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY
from llmc.compression.quantization.base_blockwise_quantization import (
    BaseBlockwiseQuantization,
)
from easydict import EasyDict


def setup_environment(output_dir, seed, log_dir_name, extra_log_str=None):
    """Setup environment for the analysis.

    This includes:
    1. Configuring the logger to output to both terminal and file
    2. Creating logs directory if it doesn't exist
    3. Setting random seeds for reproducibility

    Args:
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    logs_dir = output_path / log_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if extra_log_str is None:
        log_name = f"eval_ppl_{timestamp}.log"
    else:
        log_name = f"{extra_log_str}_{timestamp}.log"
    logger.add(logs_dir / log_name, level="INFO")

    seed_all(seed)

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    return timestamp


def setup_online_rotate(model, online_rotate):
    if not online_rotate:
        return

    model.find_blocks()
    blockwise_opt = BaseBlockwiseQuantization(
        model,
        quant_config={
            "weight": {
                "tp": 1,
                "bit": 16,
                "symmetric": True,
                "granularity": "per_channel",
            },
            "special": {"online_rotate": True, "fp32_had": True},
        },
        input=None,
        padding_mask=None,
        config={},
    )
    for block in model.blocks:
        blockwise_opt.replace_rotate_linears(block)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikitext2")
    parser.add_argument("--data_path", type=str, default="data/wikitext2")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="logs/ppl")
    parser.add_argument("--model_type", type=str, default="Qwen2")
    parser.add_argument("--online_rotate", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/Qwen3-1.7B/gptq/quarot_gptq_w4a8_g64_dq_sym_dynamic/fake_quant_model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-1.7B_quarot_gptq_w4a8_g64_dg_sym_dynamic",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="torch.bfloat16",
    )
    return parser.parse_args()


@torch.inference_mode()
def main(args=None):
    if args is None:
        args = get_args()

    log_str = f"{args.model_name}"
    log_dir_name = f"n{args.n_samples}_seq{args.seq_len}"
    setup_environment(args.output_dir, args.seed, log_dir_name, log_str)
    logger.info(f"args : {args}")

    model_cfg = EasyDict(
        {
            "model": {
                "path": args.model_path,
                "torch_dtype": args.torch_dtype,
                "type": args.model_type,
            }
        }
    )
    logger.info(f"model_type: {args.model_type}, model_cfg: {model_cfg}")
    model = MODEL_REGISTRY[args.model_type](model_cfg)
    setup_online_rotate(model, args.online_rotate)

    tokenizer = model.get_tokenizer()
    logger.info(model)

    eval_cfg = EasyDict(
        {
            "eval": {
                "type": "ppl",
                "name": args.dataset_name,
                "path": args.data_path,
                "num_samples": args.n_samples,
                "bs": args.bs,
                "seq_len": args.seq_len,
                "download": False,
                "seed": args.seed,
            }
        }
    )
    eval_cfg.update(model_cfg)
    evaluator = PerplexityEval(model, eval_cfg)
    ppl = evaluator.eval(model)

    logger.info(
        f"Perplexity for model '{args.model_name}' on '{args.dataset_name}': {ppl}"
    )
    return ppl


if __name__ == "__main__":
    main()
