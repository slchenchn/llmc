import argparse
import functools
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# Import from local utils
from utils import (
    block_forward,
    calculate_kurtosis,
    calculate_kurtosis_channel,
    cosine_similarity,
    draw,
    get_calib_config,
    get_model_config,
    setup_output_dir,
)

from llmc.compression.quantization import FakeQuantLinear
from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_,
    _TRANSFORMERS_LINEAR_TYPES_,
    RotateLinear,
)
from llmc.compression.quantization.quant import BaseQuantizer
from llmc.data import BaseDataset
from llmc.models import *
from llmc.utils import mkdirs, seed_all
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY


def analysis_block_cosine(res, t_res, args):
    """
    Analyze the cosine similarity between the original and quantized model.

    Args:
        res (dict): Dictionary of original model activations
        t_res (dict): Dictionary of quantized model activations
        args (dict): Analysis arguments
    """
    for name in res:
        oups = res[name]
        t_oups = t_res[name]

        layer_cosine_dict = {}
        for j in range(oups.shape[0]):
            cos = cosine_similarity(oups[j], t_oups[j])

            if name not in layer_cosine_dict:
                layer_cosine_dict[name] = []

            layer_cosine_dict[name].append(cos.item())

        for name in layer_cosine_dict:
            cos_values = layer_cosine_dict[name]
            min_cos = min(cos_values)
            avg_cos = sum(cos_values) / len(cos_values)
            logger.info(name)
            logger.info(f"min_cos : {min_cos}")
            logger.info(f"avg_cos : {avg_cos}")


def analysis_block_outlier(res, t_res, org_w, trans_w, args):
    """
    Analyze the outlier of the original and quantized model.

    Args:
        res (dict): Dictionary of original model activations
        t_res (dict): Dictionary of quantized model activations
        org_w (dict): Dictionary of original model weights
    """
    if args.prof_gra in ["per_channel", "per_group"]:
        kurt_func = calculate_kurtosis_channel
    else:
        kurt_func = calculate_kurtosis

    for name in res:
        logger.info(name)

        weight = org_w[name]
        t_weight = trans_w[name]

        if args.prof_gra == "per_group":
            weight = wquanter.reshape_tensor(weight)
            t_weight = wquanter.reshape_tensor(t_weight)

        k_w = kurt_func(weight)
        k_t_w = kurt_func(t_weight)

        logger.info(f"The kurtosis of org weight is :{k_w}")
        logger.info(f"The kurtosis of trans weight is :{k_t_w}")

        tensor = res[name].mean(dim=0)
        tensor = tensor.float()

        t_tensor = t_res[name].mean(dim=0)
        t_tensor = t_tensor.float()

        k_a = kurt_func(tensor)
        k_t_a = kurt_func(t_tensor)

        logger.info(f"The kurtosis of org act is :{k_a}")
        logger.info(f"The kurtosis of trans act is :{k_t_a}")

        if args.draw:
            save_outlier_path = os.path.join(args.output_dir, "outlier")
            save_t_outlier_path = os.path.join(args.output_dir, "t_outlier")

            t_min_val = t_tensor.amin(dim=0).detach().cpu().numpy()
            t_max_val = t_tensor.amax(dim=0).detach().cpu().numpy()

            min_val = tensor.amin(dim=0).detach().cpu().numpy()
            max_val = tensor.amax(dim=0).detach().cpu().numpy()

            draw(
                save_dir=save_outlier_path,
                save_name=name,
                X=range(tensor.shape[-1]),
                Y1=min_val,
                Y2=max_val,
            )

            draw(
                save_dir=save_t_outlier_path,
                save_name=name,
                X=range(t_tensor.shape[-1]),
                Y1=t_min_val,
                Y2=t_max_val,
            )


def register_hook(block, idx, args):
    hooks = []
    for name, m in block.named_modules():
        if not args.cosine:
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
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
        else:
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            stat_output_hook, name=name, idx=idx, args=args
                        )
                    )
                )

    return hooks


def stat_input_hook(m, x, y, w, name, idx, args):
    if isinstance(x, tuple):
        x = x[0]

    layer_name = f"block_{idx}.{name}"

    if args.online_rotate and is_quantized_model:
        if "down_proj" in layer_name:
            x = down_rotater.rotate(x)
        elif "o_proj" in layer_name:
            x = o_rotater.rotate(x)

    if is_quantized_model:
        t_res[layer_name] = x
        trans_w[layer_name] = w
    else:
        res[layer_name] = x
        org_w[layer_name] = w


def stat_output_hook(m, x, y, name, idx, args):
    if isinstance(y, tuple):
        y = y[0]
    layer_name = f"block_{idx}.{name}"
    if is_quantized_model:
        t_res[layer_name] = y
    else:
        res[layer_name] = y


class analysis_quanter(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)

    def fake_quant_weight_dynamic(self, module, args={}):
        weight = module.weight
        if "int_indices" in args:
            if self.granularity == "per_group":
                assert len(args["int_indices"]) % self.group_size == 0
            q_weight = weight[:, args["int_indices"]]
            fp_weight = weight[:, args["fp_indices"]]

        elif "dim" in args and "ic" in args["dim"]:
            q_weight = weight.T
        else:
            q_weight = weight

        if "current_bit" in args:
            org_bit = self.bit
            self.bit = args["current_bit"]

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        q_weight, scales, zeros, max_int, min_int = self.get_tensor_qparams(
            q_weight, args
        )

        q_weight = self.quant_dequant(q_weight, scales, zeros, max_int, min_int)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if "current_bit" in args:
            self.bit = org_bit

        if "int_indices" in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args["int_indices"]] = q_weight
            mix_weight[:, args["fp_indices"]] = fp_weight
            return mix_weight

        elif "dim" in args and "ic" in args["dim"]:
            q_weight = q_weight.T

        return q_weight


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pileval")
    parser.add_argument("--data_path", type=str, default="data/pileval")
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preproc", type=str, default="txt_general_preproc")
    parser.add_argument("--output_dir", type=str, default="analysis_quant")
    parser.add_argument("--model_type", type=str, default="DeepseekV3")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1-bf16",
        # default="/data/chenshuailin/checkpoints/deepseek-ai/DeepSeek-R1",
    )
    parser.add_argument(
        "--t_model_path",
        type=str,
        default="/data/chenshuailin/checkpoints/llmc/DeepSeek-R1/quarot/sym_w8a8-dynamic2/fake_quant_model",
        # default="/ms/FM/chenshuailin/code/llmc/checkpoints/DeepSeek-R1/quarot/fp8_sym_w8a8_dynamic/fake_quant_model",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        # default="auto",
        default="torch.bfloat16",
    )

    parser.add_argument("--tokenizer_mode", type=str, default="fast")

    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--cosine", action="store_true")

    parser.add_argument("--w_only", action="store_true")
    parser.add_argument("--wbit", type=int, default=6)
    parser.add_argument("--wsym", action="store_true")
    parser.add_argument("--wgra", type=str, default="per_channel")
    parser.add_argument("--group_size", type=int, default=-1)

    parser.add_argument("--abit", type=int, default=6)
    parser.add_argument("--asym", action="store_true")
    parser.add_argument("--agra", type=str, default="per_token")

    parser.add_argument("--prof_gra", type=str, default="per_tensor")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--online_rotate", action="store_true")

    parser.add_argument("--debugpy", action="store_true")
    args = parser.parse_args()
    return args


def maybe_prepare_online_rotate(model, t_model, tokenizer, args):
    if args.online_rotate:

        with open(args.config_path, "r") as file:
            config = yaml.safe_load(file)
        config = EasyDict(config)

        dataset = BaseDataset(tokenizer.get_tokenizer(), config.calib)
        calib_data = dataset.get_calib_dataset()
        t_model.collect_first_block_input(calib_data)
        del calib_data
        gc.collect()
        torch.cuda.empty_cache()

        blockwise_opt = ALGO_REGISTRY[config.quant.method](
            t_model, config.quant, t_model.get_first_block_input(), None, config
        )
        blockwise_opt.run_block_loop()
        t_model = blockwise_opt.model

        global down_rotater, o_rotater
        for n, m in t_model.model.named_modules():
            if isinstance(m, RotateLinear):
                logger.info(m)
                if "down_proj" in n:
                    down_rotater = m.rotater
                else:
                    o_rotater = m.rotater


def setup_environment(output_dir, seed):
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
    logs_dir.mkdir(exist_ok=True)

    # Configure loguru logger to output to both file and terminal
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Add terminal output
    logger.add(
        logs_dir / f"quant_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log", level="INFO"
    )  # Add file output with timestamp

    # Set random seeds
    seed_all(seed)

    # Set environment variables for distributed training
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


def prepare_model(model, calib_data, padding_mask):
    model.find_blocks()
    model.find_embed_layers()
    model.collect_first_block_input(calib_data, padding_mask)
    fp_inps = model.get_first_block_input()
    return fp_inps


def block_forward_lifecycle(model, i, args, fp_inps):
    block = model.blocks[i]
    block.cuda()

    hooks = register_hook(block, i, args)
    fp_inps["data"] = block_forward(block, fp_inps["data"], fp_inps["kwargs"])

    block.cpu()

    for h in hooks:
        h.remove()

    if args.cosine:
        analysis_block_cosine(res, t_res, args)
    else:
        analysis_block_outlier(res, t_res, org_w, trans_w, args)


@torch.inference_mode()
def main(args=None):
    """Main function for quantized model analysis.

    Args:
        args: Command line arguments (optional, parsed if None)

    Returns:
        int: Exit code
    """
    if args is None:
        args = get_args()

    if args.debugpy:
        import debugpy

        print("Waiting for debugpy connection...", flush=True)
        debugpy.listen(12345)
        debugpy.wait_for_client()

    setup_environment(args.output_dir, args.seed)

    logger.info(f"args : {args}")

    calib_cfg = get_calib_config(args)
    model_cfg = get_model_config(args, args.model_path)
    t_model_cfg = get_model_config(args, args.t_model_path)
    model = MODEL_REGISTRY[args.model_type](model_cfg)
    t_model = MODEL_REGISTRY[args.model_type](t_model_cfg)
    tokenizer = model.get_tokenizer()
    maybe_prepare_online_rotate(model, t_model, tokenizer, args)
    logger.info(t_model)
    logger.info(model)

    dataset = BaseDataset(tokenizer, calib_cfg)
    calib_data, padding_mask = dataset.get_calib_dataset()

    fp_inps = prepare_model(model, calib_data, padding_mask)
    t_fp_inps = prepare_model(t_model, calib_data, padding_mask)

    global res, t_res, org_w, trans_w, is_quantized_model, wquanter
    res = {}
    t_res = {}
    org_w = {}
    trans_w = {}

    if args.cosine:
        params_dict = {}
        params_dict["w_qdq"] = wquanter.fake_quant_weight_dynamic
        params_dict["a_qdq"] = None if args.w_only else a_qdq
        t_model.replace_language_module_all(FakeQuantLinear, params_dict)

    for i in tqdm(range(len(model.blocks))):
        is_quantized_model = True
        block_forward_lifecycle(t_model, i, args, t_fp_inps)
        is_quantized_model = False
        block_forward_lifecycle(model, i, args, fp_inps)

        res.clear()
        t_res.clear()
        org_w.clear()
        trans_w.clear()

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
