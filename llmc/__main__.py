import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Type, Union

import psutil
import torch
import torch.distributed as dist
import yaml
from accelerate import infer_auto_device_map, init_empty_weights
from easydict import EasyDict
from loguru import logger
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from llmc.compression.quantization import *
from llmc.compression.sparsification import *
from llmc.compression.token_reduction import *
from llmc.data import BaseDataset
from llmc.eval.utils import eval_model, get_eval_list
from llmc.models import *
from llmc.utils import (
    check_config,
    deploy_all_modality,
    get_modality,
    mkdirs,
    print_important_package_version,
    seed_all,
    update_autoawq_quant_config,
    update_vllm_quant_config,
)
from llmc.utils.loggings import setup_wandb
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY


def calculate_per_gpu_quantization_memory(
    model: torch.nn.Module,
    device_map: Dict[str, Union[int, str]],
) -> Dict[int, int]:
    """Calculate the quantization memory requirements for each GPU based on the
    device map.

    :param model: The model to calculate memory requirements for
    :param device_map: The device mapping for model layers
    :return: Dictionary mapping GPU indices to their required quantization memory in bytes
    """
    reserved_memory_per_gpu = defaultdict(int)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            device = "cpu"
            for k, v in device_map.items():
                if name.startswith(k):
                    device = v
            if isinstance(device, int):  # weights
                # Calculate memory for this specific module
                max_quant_shape = 0
                for param in module.parameters():
                    param_quant_shape = param.shape[0] // 128
                    if len(param.size()) > 1:  # weights
                        param_quant_shape *= param.shape[1]
                    max_quant_shape += param_quant_shape * 4

                bytes_ratio = 32 // 16  # assuming float16
                module_memory = max_quant_shape * bytes_ratio
                reserved_memory_per_gpu[device] += module_memory

    return reserved_memory_per_gpu


def calculate_offload_device_map(
    model: nn.Module,
    num_gpus: int = 8,
    torch_dtype: torch.dtype = torch.bfloat16,
    model_cls: Type = AutoModelForCausalLM,
    safe_margin=1,
    first_gpu_extra_margin: float = 0.9,
    **model_kwargs,
) -> Dict[Union[int, str], Union[int, str]]:
    """Calculates the optimal gpu mappings for model_stub stored as
    torch_dtype. Takes into account extra memory required for quantization and
    (optionally) GPTQ hessians.

    :param model_stub: local path or HF stub to calculate mapping for
    :param num_gpus: number of gpus to utilize
    :param torch_dtype: dtype to use for model weights
    :param model_cls: model class to use when initializing model structure, default is
        AutoModelForCausalLM
    :param safe_margin: safety margin for GPU memory (0-1)
    :param exclude_first_gpu: whether to exclude GPU 0 from model placement
    :param model_kwargs: keyword arguments to pass to model initializer
    :return: memory mapping for layers of model_stub to be passed to from_pretrained()
    """
    max_cpu_memory = psutil.virtual_memory().available
    max_gpu_memory = torch.cuda.mem_get_info(0)[0]
    available_gpus = torch.cuda.device_count()

    if available_gpus < num_gpus:
        raise ValueError(
            f"Requested {num_gpus} GPUs but only {available_gpus} are available."
        )

    max_gpu_memory *= safe_margin
    max_gpu_memory = [max_gpu_memory] * num_gpus
    print(f"{max_gpu_memory=}")

    device_map = {}
    with init_empty_weights():
        # First get the device map without reserving memory
        memory_limits = {
            idx: max_memory for idx, max_memory in enumerate(max_gpu_memory)
        }

        memory_limits[0] *= first_gpu_extra_margin

        memory_limits["cpu"] = max_cpu_memory
        print(f"{memory_limits=}")

        initial_device_map = infer_auto_device_map(
            model,
            max_memory=memory_limits,
            no_split_module_classes=model._no_split_modules,
        )
        device_map = initial_device_map

    print(f"{device_map=}")
    return device_map


def dispatch_model(model, device_map):
    """Dispatches a model according to a given device map.

    Args:
        model (`torch.nn.Module`): The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names to the device they should go to.
            Keys serve as prefixes, and any module whose name starts with a key
            will be moved to the corresponding device.
    """
    # Convert device_map values to proper devices
    for key, device in device_map.items():
        if isinstance(device, int):
            if device >= 0:
                device_map[key] = f"cuda:{device}"  # noqa: E231
            else:
                device_map[key] = "cpu"

    # Dispatch model components based on longest prefix matching
    for i, block in tqdm(
        enumerate(model.model.model.layers), desc="Dispatching model by prefix"
    ):
        layer_name = f"model.layers.{i}"
        # Find the longest matching prefix
        best_match = None
        best_match_length = -1

        for prefix, device in device_map.items():
            if layer_name.startswith(prefix):
                if len(prefix) > best_match_length:
                    best_match = device
                    best_match_length = len(prefix)

        if best_match is None:
            raise ValueError(f"No device found for module {layer_name}")

        try:
            block.to(best_match)
        except Exception as e:
            logger.error("Error dispatching model: ")
            logger.error(f"Target device: {best_match}")
            logger.error(f"Layer name: {layer_name}")
            raise e

    return model


def auto_dispatch_model(model, config):
    try:
        dispatch_config = config.dispatch
    except BaseException:
        dispatch_config = {}

    safe_margin = dispatch_config.get("safe_margin", 1)
    first_gpu_extra_margin = dispatch_config.get("first_gpu_extra_margin", 1)
    num_gpus = dispatch_config.get("num_gpus", 8)
    device_map = calculate_offload_device_map(
        model.model,
        safe_margin=safe_margin,
        num_gpus=num_gpus,
        first_gpu_extra_margin=first_gpu_extra_margin,
    )
    dispatch_model(model, device_map)


def main(config):
    model = MODEL_REGISTRY[config.model.type](config)
    auto_dispatch_model(model, config)

    logger.info(f"model: {model}")
    logger.info(f"tokenizer: {model.get_tokenizer()}")

    blockwise_opts = []
    modalities, modality_configs = get_modality(config)
    for modality, modality_config in zip(modalities, modality_configs):
        model.set_modality(modality)
        eval_list = get_eval_list(model, config)
        eval_model(model, None, eval_list, eval_pos="pretrain")
        if not config.get("calib", False):
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                input=None,
                padding_mask=None,
                config=config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()
        else:
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()

    print(">> end of blockwise transform")
    eval_model(model, blockwise_opts, eval_list, eval_pos="transformed")
    if int(os.environ["RANK"]) == 0:
        if "save" in config and config.save.get("save_trans", False):
            print(">> save transformed model")
            blockwise_opt.save_model(save_trans_path)

        if "save" in config and config.save.get("save_trtllm", False):
            blockwise_opt.save_model(save_trtllm_trans_path)
            from llmc.utils.export_trtllm import cvt_trtllm_engine

            cvt_trtllm_engine(
                save_trtllm_trans_path,
                save_trtllm_engine_path,
                config.save.get("trtllm_cfg"),
            )

        eval_model(model, blockwise_opts, eval_list, eval_pos="fake_quant")
        eval_model(model, blockwise_opts, eval_list, eval_pos="fake_quant_wo_kv")

        if "save" in config and config.save.get("save_fake", False):
            deploy_all_modality(blockwise_opts, "fake_quant")
            blockwise_opt.save_model(save_fake_path)

        if "save" in config:
            if (
                config.save.get("save_vllm", False)
                or config.save.get("save_sgl", False)
                or config.save.get("save_lightllm", False)
                or config.save.get("save_vllm_nvfp4", False)
            ):
                for modality_config in modality_configs:
                    w, a = modality_config.weight, modality_config.get("act")

                    if isinstance(w.bit, str):
                        assert w.symmetric, "Only symmetric quant is supported."
                        assert w.bit in ["e4m3", "e3m4"], "Supported quant: w8a16."
                        if a:
                            assert w.symmetric and a.symmetric, (
                                "Only symmetric quant is supported."
                            )
                            assert (
                                w.bit == a.bit
                                and w.bit in ["e4m3", "e5m2"]
                                and a.bit in ["e4m3", "e5m2"]
                            ), "Only WA FP8 quant is supported"
                    else:
                        assert w.symmetric, "Only symmetric quant is supported."
                        assert w.bit in [4, 8], "Supported quant: w4a16, w8a16, w8a8."
                        if a:
                            assert a.symmetric, "Only symmetric quant is supported."
                            assert a.bit in [4, 8], (
                                "Supported quant: w4a4, w4a16, w8a16, w8a8."
                            )

                if config.save.get("save_vllm", False):
                    deploy_all_modality(blockwise_opts, "vllm_quant")
                if config.save.get("save_vllm_nvfp4", False):
                    deploy_all_modality(blockwise_opts, "vllm_nvfp4_quant")
                if config.save.get("save_lightllm", False):
                    deploy_all_modality(blockwise_opts, "lightllm_quant")
                if config.save.get("save_sgl", False):
                    deploy_all_modality(blockwise_opts, "sgl_quant")

                blockwise_opt.save_model(save_quant_path)
                update_vllm_quant_config(blockwise_opt.model, config, save_quant_path)

        if "save" in config and config.save.get("save_autoawq", False):
            for modality_config in modality_configs:
                assert (
                    modality_config.weight.bit in [4] and "act" not in modality_config
                ), "AutoAWQ supports only 4-bit weight-only quantization."
                assert not modality_config.weight.symmetric, (
                    "Only asymmetric quant is supported."
                )

            deploy_all_modality(blockwise_opts, "autoawq_quant")
            blockwise_opt.save_model(save_quant_path)
            update_autoawq_quant_config(config, save_quant_path)

        if "save" in config and config.save.get("save_mlcllm", False):
            for modality_config in modality_configs:
                assert (
                    modality_config.weight.bit in [4] and "act" not in modality_config
                ), "MlcLLM supports only 4-bit weight-only quantization."
                assert not modality_config.weight.symmetric, (
                    "Only asymmetric quant is supported."
                )

            deploy_all_modality(blockwise_opts, "mlcllm_quant")
            blockwise_opt.save_model(save_quant_path)
            update_autoawq_quant_config(config, save_quant_path)

        if "opencompass" in config:
            assert config.save.get("save_trans", False)
            cfg_path = config["opencompass"]["cfg_path"]
            output_path = config["opencompass"]["output_path"]
            eval_model_path = os.path.abspath(save_trans_path)
            opencompass_cmd = (
                f"opencompass {cfg_path} -w {output_path} "
                f"--llmc_cfg {args.config} "
                f"--llmc_eval_mode quant "
                f"--llmc_model_path {eval_model_path}"
            )
            logger.info(f"opencompass_cmd: {opencompass_cmd}")
            os.system(opencompass_cmd)
    dist.barrier()


if __name__ == "__main__":
    logger.add(sys.stdout, level="INFO")
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--debugpy", action="store_true")
    args = parser.parse_args()
    if args.debugpy:
        print("waiting for debugpy connection...")
        import debugpy

        debugpy.listen(12345)
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if int(os.environ["RANK"]) != 0:
        logger.remove()

    check_config(config)
    setup_wandb(args, config)

    logger.info(f"args: {args}")
    logger.info(f"config: \n{json.dumps(config, ensure_ascii=False, indent=4)}")

    print_important_package_version()

    logger.info(f"WORLD_SIZE: {int(os.environ['WORLD_SIZE'])}")

    seed_all(config.base.seed + int(os.environ["RANK"]))

    # Ensure only the main process creates directories
    if int(os.environ["RANK"]) == 0:
        if "save" in config:
            if config.save.get("save_trans", False):
                save_trans_path = os.path.join(
                    config.save.save_path, "transformed_model"
                )
                mkdirs(save_trans_path)
            if config.save.get("save_trtllm", False):
                save_trtllm_trans_path = os.path.join(
                    config.save.save_path, "trtllm_transformed_model"
                )
                mkdirs(save_trtllm_trans_path)
                save_trtllm_engine_path = os.path.join(
                    config.save.save_path, "trtllm_engine"
                )
                mkdirs(save_trtllm_engine_path)
            if config.save.get("save_vllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "vllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_vllm_nvfp4", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "vllm_nvfp4_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_lightllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "lightllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_sgl", False):
                save_quant_path = os.path.join(config.save.save_path, "sgl_quant_model")
                mkdirs(save_quant_path)
            if config.save.get("save_autoawq", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "autoawq_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_mlcllm", False):
                save_quant_path = os.path.join(
                    config.save.save_path, "mlcllm_quant_model"
                )
                mkdirs(save_quant_path)
            if config.save.get("save_fake", False):
                save_fake_path = os.path.join(config.save.save_path, "fake_quant_model")
                mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config)

    destroy_process_group()

    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f"llmc_duration_time: {llmc_duration_time} s")
    logger.info("--- llmc finished ---")
