import copy
import functools
import gc
import math
import os
import pdb
import random
from contextlib import nullcontext
from math import inf
from random import sample
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from llmc.utils.registry_factory import ALGO_REGISTRY
from llmc.utils import warning_once

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear, RectifiedSigmoid
from .train_utils import AvgMeter, LossFunction, NativeScalerWithGradNormCount


@ALGO_REGISTRY
class TesseraQV2(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.add_quant_config()

        self.attention_mask = self.input["kwargs"][0].get("attention_mask")
        model_type = self.config["model"]["type"]
        self.position_ids = (
            self.input["kwargs"][0].get("position_ids")
            if model_type in ["Llama", "Mistral", "Qwen2"]
            else None
        )

        if self.deactive_amp:
            self.batch_mask = self._repeat_attention_mask()
        else:
            self.batch_mask = (
                self._repeat_attention_mask().float()
                if self.attention_mask is not None
                else None
            )

        self.dev = torch.device("cuda")
        self.model_dtype = next(self.model.model.parameters()).dtype
        logger.info("self model dtype: {}".format(self.model_dtype))

        self.sigmoid = RectifiedSigmoid(-0.1, 1.1)

    def _repeat_attention_mask(self):
        if self.attention_mask is not None:
            return self.attention_mask.repeat(
                self.input["data"][0].shape[0], 1, 1, 1
            ).cuda()
        return None

    def w_q(self, module, wquantizer):
        args = {}
        if self.optimize_scale:
            args["output_scale_factor"] = 2 * self.sigmoid(
                module.buf_output_scale_factor
            )
        if hasattr(module, "buf_upbound_factor"):
            args["upbound_factor"] = module.buf_upbound_factor
            args["lowbound_factor"] = None
        if hasattr(module, "buf_lowbound_factor"):
            args["lowbound_factor"] = module.buf_lowbound_factor

        return wquantizer.real_quant_weight_dynamic(module.weight.data, args)

    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        self.loss_func = LossFunction(method="l2")
        special_config = self.quant_config.get("special", {})

        self.deactive_amp = special_config.get("deactive_amp", False)
        self.wd = special_config.get("wd", None)
        self.lr = special_config.get("lr", None)
        self.iterations = special_config.get("iterations", 0)
        self.batch_size = special_config.get("batch_size", 1)
        self.optimize_scale = special_config.get("optimize_scale", False)
        self.thresholds = special_config.get("thresholds", [])
        self.load_transform = special_config.get("load_transform", False)
        self.reduce_memory = special_config.get("reduce_memory", False)

        if self.load_transform:
            assert "scale_path" in special_config, (
                "scale_path must be specified when load_transform is True"
            )
            self.scale_path = special_config["scale_path"]
            self.act_scales = torch.load(
                os.path.join(self.scale_path, "scales.pth"), map_location="cpu"
            )
            for k in self.act_scales:
                self.act_scales[k] = self.act_scales[k].to(torch.float32)

        self.scale_lr = special_config.get("scale_lr", None)

        if self.deactive_amp:
            self.dtype = torch.float
            self.traincast = nullcontext
        else:
            self.dtype = torch.bfloat16
            self.traincast = torch.cuda.amp.autocast

        self.aug_loss = special_config.get("aug_loss", None)

        self.loss_type = special_config.get("loss_type", "mse")
        logger.info(f"Using {self.loss_type} loss.")

        # K3 estimator specific configuration
        self.k3_sample_size = special_config.get(
            "k3_sample_size", 1000
        )  # Number of samples for k3 estimation
        self.k3_temperature = special_config.get(
            "k3_temperature", 1.0
        )  # Temperature scaling for k3

        # KL_top specific configuration
        self.kl_top_k = special_config.get(
            "kl_top_k", 1000
        )  # Number of top tokens for kl_top loss

        if self.loss_type in ["kl", "k3", "kl_top", "kl_top_mse"] or "kl_top" in self.loss_type:
            self.lm_head = self.model.get_head_layers()
            assert len(self.lm_head) == 1
            self.lm_head = self.lm_head[0]
            self.norm = self.model.get_pre_head_layernorm_layers()
            assert len(self.norm) == 1
            self.norm = self.norm[0]
            for p in self.lm_head.parameters():
                p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False

        if self.weight_clip and self.clip_version == "v2":
            self.wquantizer.calib_algo = "learnable"
            self.clip_path = special_config.get("clip_path", None)
            if self.clip_path:
                self.weight_clips = torch.load(
                    os.path.join(self.clip_path, "clips.pth"), map_location="cpu"
                )

        self.change_ratio = {}
        self.save_quant_act_path = special_config.get("save_quant_act_path", None)
        if self.save_quant_act_path:
            self.save_quant_act_path = Path(self.save_quant_act_path)
            self.save_quant_act_path.mkdir(parents=True, exist_ok=True)
        self.eval_sample_num = special_config.get("eval_sample_num", 4)
        self.round2_lr_scale = special_config.get("round2_lr_scale", 1)

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input["data"]

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if (
                "attention_mask" in self.input["kwargs"][i]
                and self.input["kwargs"][i]["attention_mask"] is not None
            ):
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input["kwargs"][i])[0]
                    output.append(out)
        return output

    def get_original_out(self, block):
        if self.block_idx == 0:
            self.ori_out = self.block_forward(block)
            if self.aug_loss:
                self.ori_out2 = self.ori_out
        else:
            self.ori_out = self.block_forward(block, self.ori_out)
            if self.aug_loss:
                self.ori_out2 = self.block_forward(block)

    @torch.no_grad()
    def collect_block_qparams(self, block, input_feat):
        """
        Collects quantization parameters (scales, zeros, min/max integers) for linear layers within a block.
        This method registers these parameters as buffers for later use during quantization.

        Args:
            block (nn.Module): The current model block being processed.
            input_feat (dict): A dictionary containing input features for different layers.
        """
        named_linears = self.model.get_block_linears(block)
        for n, m in named_linears.items():
            args = {}
            # Check for existing low and upper bound factors (from clipping)
            if hasattr(m, "buf_lowbound_factor"):
                args["lowbound_factor"] = m.buf_lowbound_factor
            if hasattr(m, "buf_upbound_factor"):
                args["upbound_factor"] = m.buf_upbound_factor

            # Get quantization parameters for the weight tensor
            (
                tensor,
                scales,
                zeros,
                max_int,
                min_int,
            ) = self.wquantizer.get_tensor_qparams(m.weight.data, args=args)

            # Register quantization parameters as buffers in the module
            m.register_buffer("buf_scales", scales)
            m.register_buffer("buf_zeros", zeros)
            m.register_buffer("buf_qmax", torch.tensor(max_int).to(self.dev))
            m.register_buffer("buf_qmin", torch.tensor(min_int).to(self.dev))

            # If static activation quantization is enabled, collect activation quantization parameters
            if self.act_static:
                subsets = self.model.get_subsets_in_block(block)
                for index, subset in enumerate(subsets):
                    layers_dict = subset["layers"]
                    input_name = subset["input"][0]
                    input_tensors = copy.deepcopy(input_feat[input_name])
                    self.register_act_qparams(layers_dict, input_tensors)
                    del input_tensors

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f"Start transform the {self.block_idx + 1}-th block")

        with torch.no_grad():
            block.float()

        if self.online_rotate:
            self.replace_rotate_linears(block)

        for i in range(len(self.input["data"])):
            self.input["data"][i] = self.input["data"][i].to(self.dtype)
        self.get_original_out(block)  # collect block output

        if self.load_transform:
            self.tesseraq_load_transform(block, input_feat)
        if self.weight_clip:
            self.tesseraq_weight_clip(block, input_feat)

        self.collect_block_qparams(
            block, input_feat
        )  # collect quant range after transformation
        self.register_tesseraq_parameters(block)

        self.tesseraq_train(block)
        self.merge_tesseraq_parameters_and_clear_tmp(block)
        self.set_rounding_opt_mode(block, on=False)

        # convert it back to original dtype
        if self.reduce_memory:
            block.to(self.model_dtype)

        logger.info(f"End transform the {self.block_idx + 1}-th block")

    def tesseraq_train(self, block):
        self.set_dynamic_tmp_quant(block, on=True)
        for n, p in block.named_parameters():
            p.requires_grad = False

        thresholds = self.thresholds
        self.input["data"] = torch.cat(self.input["data"], dim=0)
        self.ori_out = torch.cat(self.ori_out, dim=0)

        # evaluate loss before reconstruction
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss_prev = self.get_tesseraq_loss(
                    block,
                    self.input["data"][: self.eval_sample_num],
                    self.ori_out[: self.eval_sample_num],
                    save_prefix="before_train",
                )
                logger.info(
                    f"Before TesseraQV2, the {self.loss_type} loss: {loss_prev.item()}"
                )

        for i in range(len(thresholds)):
            self.set_rounding_opt_mode(block, on=True)
            self.update_mask(block, quantile_threshold=thresholds[i])

            params_r, params_r2, params_s = self.get_rounding_parameters(block)
            if self.optimize_scale:
                optimizer = torch.optim.Adam(
                    [
                        {"params": params_r, "lr": self.lr},
                        {"params": params_r2, "lr": self.lr * self.round2_lr_scale},
                        {
                            "params": params_s,
                            "lr": self.scale_lr or self.lr,
                            "weight_decay": 1e-4,
                        },
                    ],
                    lr=self.lr,
                )
            else:
                optimizer = torch.optim.Adam(params_r + params_r2, self.lr)

            loss_scaler = NativeScalerWithGradNormCount()

            with torch.enable_grad():
                for p in params_r + params_r2 + params_s:
                    p.requires_grad = True

                for iters in range(self.iterations):
                    # DEBUG:
                    if iters == self.iterations - 1:
                        print()

                    indices = torch.randperm(self.config["calib"]["n_samples"])[
                        : self.batch_size
                    ]

                    with self.traincast():
                        target2 = self.ori_out2[indices] if self.aug_loss else None
                        loss = self.get_tesseraq_loss(
                            block,
                            self.input["data"][indices],
                            self.ori_out[indices],
                            target2,
                        )

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    optimizer.zero_grad()

                    norm = loss_scaler(
                        loss, optimizer, parameters=params_r + params_r2 + params_s
                    )

                logger.info(
                    f"block {self.block_idx} iter {i + 1} loss:{loss.item():5f} \
                    norm:{norm.item():4f} HR progress:{(1 - thresholds[i]) * 100:1f}% "
                )
                for p in params_r + params_r2 + params_s:
                    p.requires_grad = False

            del optimizer

        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                # set to hard masking
                m.buf_rounding = 100 * m.buf_rounding.sign()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                loss_now = self.get_tesseraq_loss(
                    block,
                    self.input["data"][: self.eval_sample_num],
                    self.ori_out[: self.eval_sample_num],
                    save_prefix="after_train",
                )
                self.low_now = loss_now.item()
                logger.info(
                    f"After TesseraQV2, the {self.loss_type} loss: {loss_now.item()}"
                )

        self.input["data"] = list(
            torch.split(self.input["data"], split_size_or_sections=1, dim=0)
        )
        self.ori_out = list(torch.split(self.ori_out, split_size_or_sections=1, dim=0))

    @torch.no_grad()
    def tesseraq_load_transform(self, block, input_feat):
        logger.info("loading scales...")
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            prev_op = subset["prev_op"]
            layers_dict = subset["layers"]
            layers = list(layers_dict.values())

            if (
                isinstance(prev_op[0], (nn.Linear, FakeQuantLinear))
                and prev_op[0].out_features != layers[0].in_features * 3
                and prev_op[0].out_features != layers[0].in_features
            ):
                logger.info("Cannot apply scale. Do not transform this subset.")
                continue

            for n in layers_dict:
                layer_name = f"{self.model.block_name_prefix}.{self.block_idx}.{n}"
            scale = self.act_scales[layer_name].cuda()
            self.apply_scale(scale, prev_op, layers)
            self.update_input_feat(scale, input_feat, layers_dict)

    @torch.no_grad()
    def update_input_feat(self, scale, input_feat, layers_dict):
        for layer_name in layers_dict:
            for i in range(len(input_feat[layer_name])):
                inp = input_feat[layer_name][i]
                inp.div_(scale.view(1, -1).to(inp.device))

    def tesseraq_weight_clip(self, block, input_feat):
        if self.clip_version in ("v1", "v3", "v4"):
            self.auto_clipper.run(block, self.block_idx, input_feat, n_sample_token=512)
        elif self.clip_version == "v2":
            logger.info("loading clips...")
            for n, m in block.named_modules():
                if isinstance(m, nn.Linear):
                    if any([_ in n for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                        m.register_buffer("buf_upbound_factor", None)
                        m.register_buffer("buf_lowbound_factor", None)
                        continue
                    layer_name = f"{n}.weight_quantizer."
                    upbound_factor = self.weight_clips[self.block_idx][
                        layer_name + "upbound_factor"
                    ]
                    lowbound_factor = self.weight_clips[self.block_idx][
                        layer_name + "lowbound_factor"
                    ]
                    m.register_buffer(
                        "buf_upbound_factor",
                        upbound_factor.cuda().float(),
                    )
                    m.register_buffer(
                        "buf_lowbound_factor",
                        lowbound_factor.cuda().float()
                        if lowbound_factor is not None
                        else None,
                    )

    def get_tesseraq_loss(self, block, x, target, target2=None, save_prefix=None):
        if self.position_ids is not None:
            pos_emb = self.rotary_emb(x, self.position_ids)
            quant_out = block(
                x,
                attention_mask=self.batch_mask,
                position_ids=self.position_ids,
                position_embeddings=pos_emb,
            )[0]
        else:
            quant_out = block(x, attention_mask=self.batch_mask)[0]

        if self.save_quant_act_path and save_prefix:
            cur_save_dir = self.save_quant_act_path / save_prefix
            cur_save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                quant_out.detach().cpu(),
                cur_save_dir / f"layer{self.block_idx}_quant_out.pt",
            )
            torch.save(
                target.detach().cpu(), cur_save_dir / f"layer{self.block_idx}_target.pt"
            )

        if self.loss_type == "mse":
            loss = self.loss_func(target, quant_out)
        elif self.loss_type == "kl":
            # import time
            loss = self.compute_kl_divergence(target, quant_out)
        elif self.loss_type == "kl_residual":
            raise NotImplementedError
            loss = self.compute_kl_divergence(target, quant_out)
        elif self.loss_type == "k3":
            raise NotImplementedError
            loss = self.compute_k3_divergence(target, quant_out)
        elif self.loss_type == "kl_top_mse":
            kl_loss = self.compute_kl_top_divergence(target, quant_out)
            mse_loss = self.loss_func(target, quant_out)
            loss = kl_loss + mse_loss
        elif self.loss_type == "kl_top" or "kl_top" in self.loss_type:
            loss = self.compute_kl_top_divergence(target, quant_out)

        if target2 is not None:
            if self.loss_type == "mse":
                loss = (loss + self.loss_func(target2, quant_out)) / 2
            elif self.loss_type == "kl":
                raise NotImplementedError("Not implemented")
            elif self.loss_type == "k3":
                loss2 = self.compute_k3_divergence(target2, quant_out)
                loss = (loss + loss2) / 2
            elif self.loss_type == "kl_top" or "kl_top" in self.loss_type:
                loss2 = self.compute_kl_top_divergence(target2, quant_out)
                loss = (loss + loss2) / 2
            elif self.loss_type == "kl_top_mse":
                kl_loss2 = self.compute_kl_top_divergence(target2, quant_out)
                mse_loss2 = self.loss_func(target2, quant_out)
                loss2 = kl_loss2 + mse_loss2
                loss = (loss + loss2) / 2
        return loss

    def register_tesseraq_parameters(self, block):
        module = FakeQuantLinear
        self.model.replace_module_block(
            module,
            block,
            self.block_idx,
            self.get_replacement_params(
                mode="fake_quant", w_only=self.w_only, name=None
            ),
        )
        self.register_rounding_parameters(block)

    def register_rounding_parameters(self, block):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                rounding = m.weight.data.clone()
                scales = m.buf_scales
                rounding = self.wquantizer.reshape_tensor(rounding).div(scales)
                rounding = rounding - torch.floor(rounding)
                rounding = self.sigmoid.inverse(rounding)

                m.register_buffer("buf_rounding", rounding)

                # Register second rounding variable initialized to 0 (which maps to 0.5 after sigmoid)
                rounding2 = torch.zeros_like(rounding)
                m.register_buffer("buf_rounding2", rounding2)

                if self.optimize_scale:
                    output_scale_factor = torch.zeros_like(scales)
                    m.register_buffer("buf_output_scale_factor", output_scale_factor)

    @torch.no_grad()
    def update_mask(self, block, quantile_threshold):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                # Compute combined rounding value
                # First rounding: 0 to 1
                rounding1 = self.sigmoid(m.buf_rounding)
                # Second rounding: map to -1 to 1 range
                rounding2 = 2 * self.sigmoid(m.buf_rounding2) - 1
                # Combined value ranges from -1 to 2
                combined = rounding1 + rounding2

                # Compute distance to nearest integer
                # The nearest integers are -1, 0, 1, 2
                rounded_combined = torch.round(combined)
                sim_to_int = 0.5 - (combined - rounded_combined).abs()
                # distance_to_int = (combined - rounded_combined).abs()

                # Find threshold value based on quantile
                # torch do not support large tensor quantile, so we use numpy
                # mask_thres = torch.quantile(sim_to_int, q=quantile_threshold)
                mask_thres = np.quantile(sim_to_int.cpu().numpy(), q=quantile_threshold)
                logger.info(
                    f"layer: {n}, quantile_threshold: {quantile_threshold}, mask_thres: {mask_thres}"
                )

                # Create mask for values close to integers
                mask = sim_to_int > mask_thres

                # Apply mask to both rounding parameters
                # Set to inf/-inf to fix them during optimization
                # For values close to integers, we want to fix them
                # Determine which integer each masked value is closest to

                # Set buf_rounding based on target combination
                # target = -1: rounding1 = 0, rounding2 = -1
                # target = 0:  rounding1 = 1, rounding2 = -1
                # target = 1:  rounding1 = 0, rounding2 = 1
                # target = 2:  rounding1 = 1, rounding2 = 1

                # For buf_rounding (maps to 0 or 1)
                m.buf_rounding[mask] = torch.where(
                    (rounded_combined[mask] == 0) | (rounded_combined[mask] == 2),
                    float("inf"),  # sigmoid(inf) = 1
                    -float("inf"),  # sigmoid(-inf) = 0
                )

                # For buf_rounding2 (maps to -1 or 1)
                # rounding2 = 2*sigmoid(buf_rounding2) - 1
                # sigmoid(inf) = 1 => rounding2 = 1
                # sigmoid(-inf) = 0 => rounding2 = -1
                m.buf_rounding2[mask] = torch.where(
                    (rounded_combined[mask] == 1) | (rounded_combined[mask] == 2),
                    float("inf"),  # For rounding2 = 1
                    -float("inf"),  # For rounding2 = -1
                )

                del sim_to_int

    def set_rounding_opt_mode(self, block, on=True):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                if not hasattr(m, "buf_rounding_opt"):
                    m.register_buffer("buf_rounding_opt", torch.tensor(on))
                else:
                    m.buf_rounding_opt = torch.tensor(on)

    def set_dynamic_tmp_quant(self, block, on=True):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                m.dynamic_quant_tmp_weight = on

    def get_rounding_parameters(self, block):
        params_r = []
        params_r2 = []
        params_s = []
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                params_r += [m.buf_rounding]
                params_r2 += [m.buf_rounding2]
                if self.optimize_scale:
                    params_s += [m.buf_output_scale_factor]
        return params_r, params_r2, params_s

    def merge_tesseraq_parameters_and_clear_tmp(self, block):
        for n, m in block.named_modules():
            if isinstance(m, FakeQuantLinear):
                r1 = (m.buf_rounding > 0).float()
                r2 = 2 * ((m.buf_rounding2 > 0).float() - 0.5)
                combined_rounding = r1 + r2  # {-1, 0, 1, 2}

                w_shape = m.weight.shape
                W = self.wquantizer.reshape_tensor(m.weight.data) / m.buf_scales

                # Standard rounding (0 or 1)
                standard_rounding = (W - torch.floor(W) > 0.5).float()

                # Adjustment relative to standard rounding
                # combined_rounding ranges from {-1, 0, 1, 2}
                # standard_rounding is {0, 1}
                # So the adjustment is combined_rounding - standard_rounding
                m.buf_rounding = combined_rounding - standard_rounding

                # Change ratio: count elements where adjustment is non-zero
                cr = torch.count_nonzero(m.buf_rounding) / m.buf_rounding.numel()

                if n not in self.change_ratio:
                    self.change_ratio[n] = 0
                self.change_ratio[n] = self.change_ratio[n] + cr
                logger.info(
                    "layer {}, change ratio: {}%".format(
                        n, self.change_ratio[n] / (self.block_idx + 1) * 100
                    )
                )
                m.buf_rounding *= 0.5 * m.buf_scales
                m.buf_rounding = self.wquantizer.restore_tensor(m.buf_rounding, w_shape)
                m.weight.data.add_(m.buf_rounding.to(self.model_dtype))

                delattr(m, "buf_rounding")
                delattr(m, "buf_rounding2")
                delattr(m, "tmp_weight")
                delattr(m, "tmp_bias")
                m.dynamic_quant_weight = False
                m.dynamic_quant_tmp_weight = False

                gc.collect()
                torch.cuda.empty_cache()

    def cache_input_hook(self, m, x, y, name, feat_dict):
        super().cache_input_hook(m, x, y, name, feat_dict)
        if len(feat_dict[name]) > 128:
            del feat_dict[name][-1]

    def w_qdq(self, module, wquantizer):
        weight = module.weight

        args = {}
        args["scales"] = module.buf_scales
        if hasattr(module, "buf_zeros"):
            args["zeros"] = module.buf_zeros
        else:
            args["zeros"] = None
        args["qmax"] = module.buf_qmax
        args["qmin"] = module.buf_qmin

        if hasattr(module, "buf_rounding_opt") and module.buf_rounding_opt:
            # Combine both roundings for quantization
            # First rounding: sigmoid output directly (0 to 1)
            rounding1 = self.sigmoid(module.buf_rounding)
            # Second rounding: map sigmoid output to {-1, 1} range
            # sigmoid(buf_rounding2) gives 0-1, we want to map this to -1 or 1
            # Use 2 * sigmoid - 1 to get continuous range, then round
            rounding2 = 2 * self.sigmoid(module.buf_rounding2) - 1
            if (rounding1 + rounding2).max() >= 2 or (
                rounding1 + rounding2
            ).min() <= -1:
                print()
            # Combined rounding for fake quantization
            args["rounding"] = rounding1 + rounding2

        if self.optimize_scale:
            args["output_scale_factor"] = 2 * self.sigmoid(
                module.buf_output_scale_factor
            )

        weight = wquantizer.fake_quant_weight_static(weight, args)

        return weight

    def deploy(self, quant_format):
        super().deploy(quant_format)
        # self.model.convert_dtype(self.model_dtype)

    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)

    def compute_kl_divergence(self, target, quant_out):
        quant_out = quant_out.to(torch.bfloat16)
        with torch.no_grad():
            target = target.to(torch.bfloat16)
            self.lm_head.to(target.device, dtype=torch.bfloat16)
            self.norm.to(target.device, dtype=torch.bfloat16)
            self.norm.to(target.device, dtype=torch.bfloat16)

            # # 步骤5: 计算target概率分布
            # step5_start = time.time()
            target_probs = F.softmax(self.lm_head(self.norm(target)), dim=-1)
            # step5_time = time.time() - step5_start
            # print(f"Step 5 - Compute target_probs: {step5_time:.6f}s")

        # 步骤6: 计算量化后的log概率分布
        # step6_start = time.time()
        quant_log_probs = F.log_softmax(self.lm_head(self.norm(quant_out)), dim=-1)
        # step6_time = time.time() - step6_start
        # print(f"Step 6 - Compute quant_log_probs: {step6_time:.6f}s")

        # 步骤7: 计算KL散度损失
        # step7_start = time.time()
        loss = F.kl_div(
            quant_log_probs,
            target_probs,
            reduction="batchmean",
        )
        # step7_time = time.time() - step7_start
        # print(f"Step 7 - Compute KL divergence loss: {step7_time:.6f}s")
        return loss

    def compute_k3_divergence(self, target, quant_out):
        """
        Compute KL divergence using k3 estimator: k3 = (r - 1) - log(r)
        where r = p(x)/q(x) is the probability ratio.

        This method is more memory-efficient than standard KL divergence computation
        as it only processes sampled tokens rather than the full vocabulary.
        """
        batch_size, seq_len, hidden_dim = target.shape

        # Sample a subset of positions to compute the divergence
        if batch_size * seq_len > self.k3_sample_size:
            # Randomly sample positions
            total_positions = batch_size * seq_len
            sample_indices = torch.randperm(total_positions)[: self.k3_sample_size]

            # Reshape to (batch_size * seq_len, hidden_dim)
            target_flat = target.reshape(-1, hidden_dim)
            quant_out_flat = quant_out.reshape(-1, hidden_dim)

            # Sample the positions
            target_sampled = target_flat[sample_indices]
            quant_out_sampled = quant_out_flat[sample_indices]
        else:
            target_sampled = target.reshape(-1, hidden_dim)
            quant_out_sampled = quant_out.reshape(-1, hidden_dim)

        # Move norm to the same device as target
        self.norm.to(target.device)

        with torch.no_grad():
            # Compute normalized target representations
            target_norm = self.norm(target_sampled)

        # Compute normalized quantized representations
        quant_norm = self.norm(quant_out_sampled)

        # Use vocabulary sampling for k3 estimation
        # Get a subset of the vocabulary to reduce memory usage
        vocab_subset_size = min(1000, self.lm_head.out_features)  # Limit vocabulary

        # Create a temporary smaller head for computation
        with torch.no_grad():
            # Sample random vocabulary indices (in practice, you might want top-k frequent tokens)
            vocab_indices = torch.randperm(self.lm_head.out_features)[
                :vocab_subset_size
            ].to(target.device)

            # Get subset of weight matrix
            weight_subset = self.lm_head.weight[vocab_indices].to(target.device)
            bias_subset = None
            if self.lm_head.bias is not None:
                bias_subset = self.lm_head.bias[vocab_indices].to(target.device)

            # Compute logits for subset
            target_logits_subset = F.linear(target_norm, weight_subset, bias_subset)
            target_probs_subset = F.softmax(
                target_logits_subset / self.k3_temperature, dim=-1
            )

        quant_logits_subset = F.linear(quant_norm, weight_subset, bias_subset)
        quant_probs_subset = F.softmax(
            quant_logits_subset / self.k3_temperature, dim=-1
        )

        # Compute probability ratios: r = p(target) / p(quant)
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        r = (target_probs_subset + eps) / (quant_probs_subset + eps)

        # Compute k3 estimator: k3 = (r - 1) - log(r)
        # This is unbiased, low-variance, and always positive
        k3_values = (r - 1.0) - torch.log(r.clamp(min=eps))

        # Average across positions and vocabulary
        loss = k3_values.mean()

        return loss

    def compute_kl_top_divergence(self, target, quant_out):
        """
        Compute KL divergence using only top-k logits for efficiency.
        Based on the implementation from OSTQuant trainer.py.
        
        Args:
            target: Original model hidden states
            quant_out: Quantized model hidden states
            
        Returns:
            KL divergence loss computed on top-k logits
        """
        # Determine k value
        if self.loss_type == "kl_top":
            k = self.kl_top_k
        elif self.loss_type == "kl_top_mse":
            k = self.kl_top_k
        else:
            # Extract k from loss_type like "kl_top_500"
            try:
                k = int(self.loss_type.split("_")[-1])
            except (ValueError, IndexError):
                warning_once(f"Cannot parse k from loss_type {self.loss_type}, using default k={self.kl_top_k}")
                k = self.kl_top_k
        
        quant_out = quant_out.to(torch.bfloat16)
        
        with torch.no_grad():
            target = target.to(torch.bfloat16)
            self.lm_head.to(target.device, dtype=torch.bfloat16)
            self.norm.to(target.device, dtype=torch.bfloat16)
            
            # Compute original logits
            ori_logits = self.lm_head(self.norm(target))
            
        # Compute quantized logits
        logits = self.lm_head(self.norm(quant_out))
        
        # Get top-k indices from original logits
        top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
        
        # Extract corresponding logits from quantized output using the same indices
        top_logits = logits.gather(-1, indices)
        
        # Compute KL divergence on top-k logits
        # KL(P||Q) where P is target (original) and Q is quantized
        loss = F.kl_div(
            F.log_softmax(top_logits, dim=-1).flatten(0, -2),
            F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
            reduction="batchmean",
        )
        
        return loss
