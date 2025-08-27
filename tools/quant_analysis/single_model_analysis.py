import argparse
import functools
import gc
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import pickle

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm

# Import from local utils
from utils import (
    block_forward,
    calculate_absmax,
    calculate_absmax_channel,
    calculate_kurtosis,
    calculate_kurtosis_channel,
    calculate_outlier_degree,
    calculate_outlier_degree_channel,
    draw,
    get_calib_config,
    get_model_config,
    setup_output_dir,
)

from llmc.compression.quantization.hadamard_utils import get_hadK
from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_,
    _TRANSFORMERS_LINEAR_TYPES_,
    RotateLinear,
    Rotater,
)
from llmc.data import BaseDataset
from llmc.models import *
from llmc.utils import mkdirs, seed_all
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY

metric_mapper = {
    "kurtosis": {
        "per_channel": calculate_kurtosis_channel,
        "per_group": calculate_kurtosis_channel,
        "per_tensor": calculate_kurtosis,
    },
    "absmax": {
        "per_channel": calculate_absmax_channel,
        "per_group": calculate_absmax_channel,
        "per_tensor": calculate_absmax,
    },
    "outlier_degree": {
        "per_channel": calculate_outlier_degree_channel,
        "per_group": calculate_outlier_degree_channel,
        "per_tensor": calculate_outlier_degree,
    },
}


def analysis_block_outlier(res, org_w, args, metric, prefix_len=0):
    """
    Analyze the outlier of the original model using sequential processing.

    Args:
        res (dict): Dictionary of original model activations
        org_w (dict): Dictionary of original model weights
        args: Command line arguments
    """
    # Ensure the outlier directory exists if draw is enabled
    if not metric:
        return

    if args.draw:
        save_outlier_path = os.path.join(args.output_dir, "outlier")
        os.makedirs(save_outlier_path, exist_ok=True)

    # Use appropriate kurtosis function
    metric_funcs = {
        metric: metric_mapper[metric][args.prof_gra] for metric in args.vis_metric
    }

    # Process each layer sequentially
    for name in res:
        logger.info(name)

        # Calculate weight kurtosis and absmax
        weight = org_w[name]
        weight_metric_values = {}
        for metric, metric_func in metric_funcs.items():
            # outlier_degree is not supported for weight
            if metric == "outlier_degree":
                continue
            weight_metric_values[metric] = metric_func(weight)
            logger.info(f"The {metric} of weight is: {weight_metric_values[metric]}")

        # Process activation tensor
        tensor = res[name]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.float().mean(dim=0)[prefix_len:]

        # Calculate activation kurtosis and absmax
        tensor_metric_values = {}
        for metric, metric_func in metric_funcs.items():
            cur_metric = metric_func(tensor)
            if isinstance(cur_metric, dict):
                for k, v in cur_metric.items():
                    tensor_metric_values[f"{metric}.{k}"] = v
            else:
                tensor_metric_values[metric] = cur_metric

        for metric, value in tensor_metric_values.items():
            logger.info(f"The {metric} of act is: {value}")

        # Draw if requested
        if args.draw:
            save_outlier_path = os.path.join(args.output_dir, "outlier")

            min_val = tensor.amin(dim=0).detach().cpu().numpy()
            max_val = tensor.amax(dim=0).detach().cpu().numpy()

            try:
                draw(
                    save_dir=save_outlier_path,
                    save_name=name,
                    X=range(tensor.shape[-1]),
                    Y1=min_val,
                    Y2=max_val,
                )
            except Exception as e:
                logger.info(f"error of {name}, {tensor.shape=}")
                raise e


def register_hook(block, idx, args):
    hooks = []
    for name, m in block.named_modules():
        if (
            isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_))
            or "layernorm" in name
        ):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(
                        stat_input_hook,
                        w=m.weight.data,
                        name=name,
                        idx=idx,
                        args=args,
                    )
                )
            )

    return hooks


def stat_input_hook(m, x, y, w, name, idx, args):
    if isinstance(x, tuple):
        x = x[0]

    layer_name = f"block_{idx}.{name}"

    if args.online_rotate:
        if "down_proj" in layer_name:
            global down_rotater
            x = down_rotater.rotate(x)
        # elif "o_proj" in layer_name:
        #     global o_rotater
        #     x = o_rotater.rotate(x)

    res[layer_name] = x
    org_w[layer_name] = w


def setup_environment(output_dir, seed, extra_log_str=None):
    """Setup environment for the analysis.

    This includes:
    1. Configuring the logger to output to both terminal and file
    2. Creating logs directory if it doesn't exist
    3. Setting random seeds for reproducibility

    Args:
        seed: Random seed for reproducibility
    """
    # Ensure logs directory exists
    logs_dir = Path(output_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Configure loguru logger to output to both file and terminal
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Add terminal output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if extra_log_str is None:
        log_name = f"model_analysis_{timestamp}.log"
    else:
        log_name = f"{extra_log_str}_{timestamp}.log"
    logger.add(logs_dir / log_name, level="INFO")  # Add file output with timestamp

    # Set random seeds
    seed_all(seed)

    # Set environment variables for distributed training
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    return timestamp


class LastBlockWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        # add the last norm and lm_head into a single block
        self.input_layernorm = model.get_pre_head_layernorm_layers()[0]
        self.lm_head = model.get_head_layers()[0]

    def forward(self, x, **kwargs):
        x = self.input_layernorm(x)
        x = self.lm_head(x)
        return x


def prepare_model(model, calib_data, padding_mask):
    model.find_blocks()
    model.find_embed_layers()
    model.collect_first_block_input(calib_data, padding_mask)
    fp_inps = model.get_first_block_input()
    model.blocks.append(LastBlockWrapper(model))
    return fp_inps


def block_forward_lifecycle(
    model, i, args, fp_inps, metric, add_hook=True, prefix_len=0
):
    block = model.blocks[i]
    block.cuda()

    if add_hook:
        hooks = register_hook(block, i, args)
    fp_inps["data"] = block_forward(block, fp_inps["data"], fp_inps["kwargs"])

    block.cpu()

    if add_hook:
        for h in hooks:
            h.remove()
        analysis_block_outlier(res, org_w, args, metric, prefix_len)
    model.blocks[i] = None
    del block


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pileval")
    parser.add_argument("--data_path", type=str, default="data/pileval")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preproc", type=str, default="txt_general_preproc")
    parser.add_argument("--output_dir", type=str, default="analysis_model")
    parser.add_argument("--model_type", type=str, default="DeepseekV3")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-bf16",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="torch.bfloat16",
    )
    parser.add_argument("--tokenizer_mode", type=str, default="fast")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--prof_gra", type=str, default="per_tensor")
    parser.add_argument("--online_rotate", action="store_true")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/csl/quarot_gptq_ol-rotate/quarot_r1_sym_w8a8_dynamic_ol-rotate.yml",
    )
    parser.add_argument(
        "--vis_metric",
        type=str,
        nargs="+",
        # default=[],
        default=["outlier_degree", "kurtosis", "absmax"],
        # default=["outlier_degree"],
        # default=["kurtosis", "absmax"],
        # default=["absmax"],
    )
    parser.add_argument("--prefix_token_ids", nargs="+", type=int, default=[])
    return parser.parse_args()


@torch.inference_mode()
def main(args=None):
    """Main function for model analysis.

    Args:
        args: Command line arguments (optional, parsed if None)

    Returns:
        int: Exit code
    """
    if args is None:
        args = get_args()

    metric_str = "_".join(args.vis_metric)
    timestamp = setup_environment(args.output_dir, args.seed, metric_str)
    logger.info(f"args : {args}")

    calib_cfg = get_calib_config(args)
    model_cfg = get_model_config(args, args.model_path)
    logger.info(f"model_type: {args.model_type}, model_cfg: {model_cfg}")
    model = MODEL_REGISTRY[args.model_type](model_cfg)
    tokenizer = model.get_tokenizer()
    logger.info(model)

    dataset = BaseDataset(tokenizer, calib_cfg)
    calib_data, padding_mask = dataset.get_calib_dataset(
        prefix_token_ids=args.prefix_token_ids
    )

    fp_inps = prepare_model(model, calib_data, padding_mask)
    # if args.online_rotate:
    #     with open(args.config_path, "r") as file:
    #         config = yaml.safe_load(file)
    #     config = EasyDict(config)
    #     config.quant.modality = "language"

    #     logger.info(f"quant method: {config.quant.method}")
    #     blockwise_opt = ALGO_REGISTRY[config.quant.method](
    #         model, config.quant, fp_inps, None, config, do_preprocess=False
    #     )

    #     # blockwise_opt.run_block_loop()
    #     # logger.info(f"start replace rotate linears")
    #     # for i in range(len(model.blocks)):
    #     #     blockwise_opt.replace_rotate_linears(model.blocks[i], exclude_keys=['o_proj'])

    #     model = blockwise_opt.model

    #     # for n, m in model.model.named_modules():
    #     #     if isinstance(m, RotateLinear):
    #     #         logger.info(m)
    #     #         if "down_proj" in n:
    #     #             global down_rotater
    #     #             down_rotater = m.rotater
    #     #         # else:
    #     #         #     global o_rotater
    #     #         #     o_rotater = m.rotater

    global res, org_w
    res = {}
    org_w = {}
    pickle_dir = Path(args.output_dir) / "pickled" / timestamp
    pickle_dir.mkdir(exist_ok=True, parents=True)
    prefix_len = len(args.prefix_token_ids)
    for i in tqdm(range(len(model.blocks))):
        logger.info(f"processing the {i}th block")

        # logger.info(f"replace rotate linears for the {i}th block")
        # blockwise_opt.replace_rotate_linears(model.blocks[i], exclude_keys=['o_proj'])
        # for n, m in model.blocks[i].named_modules():
        #     if isinstance(m, RotateLinear):
        #         logger.info(m)
        #         if "down_proj" in n:
        #             global down_rotater
        #             down_rotater = m.rotater
        #         # else:
        #         #     global o_rotater
        #         #     o_rotater = m.rotater
        if args.online_rotate:
            if args.model_type == "DeepseekV3":
                if i < 3:
                    # hidden_dim = model.intermediate_size
                    hidden_dim = 18432
                else:
                    # hidden_dim = model.moe_intermediate_size
                    hidden_dim = 2048
            elif args.model_type == "Qwen2":
                hidden_dim = 25600
            else:
                raise NotImplementedError

            had_K, K = get_hadK(hidden_dim)
            params_dict = {
                "had_K": had_K,
                "K": K,
                "online_full_had": True,
                "online_partial_had": False,
                "had_dim": None,
                "fp32_had": True,
            }
            global down_rotater
            down_rotater = Rotater(**params_dict)

        if i >= len(model.blocks) - 2:
            block_forward_lifecycle(
                model,
                i,
                args,
                fp_inps,
                metric=args.vis_metric,
                add_hook=True,
                prefix_len=prefix_len,
            )
            with open(pickle_dir / f"block{i}.res.pkl", "wb") as f:
                pickle.dump(res, f)
        else:
            block_forward_lifecycle(
                model,
                i,
                args,
                fp_inps,
                metric=args.vis_metric,
                add_hook=True,
                prefix_len=prefix_len,
            )

        res.clear()
        org_w.clear()

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
