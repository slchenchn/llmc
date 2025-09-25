import copy
import math
import os
# No abstract base classes used here

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from loguru import logger
from collections import defaultdict
import functools
from llmc.utils.utils import get_decoder_layer_ori_device
from tqdm import trange

from llmc.utils.loggings import safe_wandb_log
from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (
    _LLMC_LINEAR_TYPES_,
    _TRANSFORMERS_LINEAR_TYPES_,
    FakeQuantLinear,
    RotateLinear,
)


@ALGO_REGISTRY
class GPTQv2(BaseBlockwiseQuantization):
    def __init__(
        self, model, quant_config, input, padding_mask, config, modality="language"
    ):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device("cuda")
        self.model_dtype = next(self.model.model.parameters()).dtype
        self.add_quant_config()
        self.layers_cache = {}
        self.collect_model_qparams()
        self.original_blocks = [b.cpu() for b in copy.deepcopy(self.blocks)]
        self.fp_input_data = self.input["data"]

    @torch.no_grad()
    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix

        # calib_bs = self.quant_config["calib"]["bs"]
        # assert calib_bs == 1, f"GPTQv2 only supports bs=1, but got {calib_bs}"

        special_config = self.quant_config["special"]

        self.true_sequential = special_config["true_sequential"]
        self.static_groups = special_config["static_groups"]
        self.actorder = special_config["actorder"]
        self.percdamp = special_config["percdamp"]
        self.blocksize = special_config["blocksize"]

        self.owq = special_config.get("owq", False)
        self.chunk_num = special_config.get("chunk_num", 1)
        self.v2_alpha = special_config.get("v2_alpha", 0.25)

        # GPTQv2/GPTAQ: asymmetric calibration hooks (inputs from partially quantized model,
        # targets from full-precision). We enable data collection by default.
        self.asymmetric_calib = True

        if self.owq:
            self.n_outs = special_config["n_outs"]
            self.static_groups = False
            self.actorder = False

        self.need_perm = (
            self.wquantizer.granularity == "per_group"
            and not self.static_groups
            and self.actorder
        ) or self.owq

        # self.keep_sample_ratio = special_config.get("keep_sample_ratio", 1.0)
        # self.keep_sample_part = special_config.get("keep_sample_part", "bottom")
        self.online_rotate_exclude = special_config.get("online_rotate_exclude", [])
        self.eager_transform_dtype = special_config.get("eager_transform_dtype", False)
        self.force_scale_dtype = special_config.get("force_scale_dtype", False)
        logger.info(f"eager_transform_dtype: {self.eager_transform_dtype}")
        logger.info(f"force_scale_dtype: {self.force_scale_dtype}")

    def hessian_sorting(self, name):
        H = self.layers_cache[name]["H"]

        if not self.owq:
            if self.actorder:
                self.perm = torch.argsort(torch.diag(H), descending=True)
            return

        temp_mask = torch.full([self.columns], True, device=self.dev)
        H_diag = torch.diag(H)
        descending_ids = torch.argsort(H_diag, descending=True)
        temp_mask[descending_ids[: self.n_out]] = False

        if self.actorder:
            perm = torch.cat(
                [descending_ids[self.n_out :], descending_ids[: self.n_out]]
            )
        else:
            perm = torch.cat(
                [
                    torch.arange(self.columns, device=self.dev)[temp_mask],
                    descending_ids[: self.n_out],
                ]
            )

        self.perm = perm

    @torch.no_grad()
    def block_transform(self, block, input_feat, block_kwargs):
        if self.eager_transform_dtype:
            ori_dtypes = {}
            for name, module in block.named_modules():
                if type(module) in _LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_:
                    ori_dtypes[name] = module.weight.dtype
        if self.online_rotate:
            self.replace_rotate_linears(block, self.online_rotate_exclude)
        if self.owq and not hasattr(self, "n_out_dict"):
            named_linears = self.model.get_block_linears(block)
            self.n_out_dict = {}
            for i, name in enumerate(named_linears.keys()):
                self.n_out_dict[name] = self.n_outs[i]
        super().block_transform(block, input_feat, block_kwargs)
        if self.eager_transform_dtype:
            for name, module in block.named_modules():
                if type(module) in _LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_:
                    module.weight.data = module.weight.data.to(ori_dtypes[name])

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset["layers"]
        for name, layer in layers_dict.items():
            if not isinstance(
                layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
            ):
                continue
            self.layer_transform(layer, name)
            self.free(name)

    @torch.no_grad()
    def layer_transform(self, layer, name):
        self.initialize_qparams_and_prepare_weights(layer, name)
        W, Hinv = self.process_hessian_and_weights(layer, name)
        self.update_layer_with_transformed_weights(layer, W, Hinv, name)

    def initialize_qparams_and_prepare_weights(self, layer, name):
        self.qparams = {}
        self.columns = self.layers_cache[name]["columns"]
        self.n_out = self.n_out_dict[name] if self.owq else 0
        self.n_nonout = self.columns - self.n_out

        if self.actorder or self.owq:
            self.hessian_sorting(name)

    def process_hessian_and_weights(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(layer, transformers.Conv1D):
            W = W.t()

        W = W.float()
        H = self.layers_cache[name].pop("H")
        dXXT = self.layers_cache[name].pop("dXXT")

        dead = torch.diag(H) == 0
        if dead.sum() > 0:
            logger.info(f"{name} has {dead.sum()} dead channels")
            H[dead, dead] = 1
            W[:, dead] = 0
            dXXT[:, dead] = 0

        if not self.ready():
            if self.wquantizer.granularity == "per_group":
                self.groups = []
                self.search_group_qparams(layer)
            else:
                self.search_layer_qparams(layer)

        if self.actorder or self.owq:
            W = W[:, self.perm]
            H = H[self.perm][:, self.perm]
            dXXT = dXXT[self.perm][:, self.perm]
            self.invperm = torch.argsort(self.perm)

            layer.register_buffer("buf_perm", self.perm)
            layer.register_buffer("buf_invperm", self.invperm)

            if self.owq:
                layer.register_buffer("buf_n_nonout", torch.tensor(self.n_nonout))
                if self.wquantizer.granularity == "per_channel":
                    _, layer.buf_scales, layer.buf_zeros, _, _ = (
                        self.wquantizer.get_tensor_qparams(W[:, : self.n_nonout])
                    )
                    self.qparams["scale"], self.qparams["zero"] = (
                        layer.buf_scales,
                        layer.buf_zeros,
                    )

        # Diagnostics before constructing P
        try:
            H_fro = torch.linalg.norm(H, ord="fro").item()
        except Exception:
            H_fro = torch.norm(H, p="fro").item()
        try:
            dXXT_fro = torch.linalg.norm(dXXT, ord="fro").item()
        except Exception:
            dXXT_fro = torch.norm(dXXT, p="fro").item()
        ratio = dXXT_fro / (H_fro + 1e-12)
        logger.info(
            f"[GPTQv2][{name}] pre-P: ||H||_F={H_fro:.4e}, ||dXXT||_F={dXXT_fro:.4e}, ratio={ratio:.2%}"
        )
        safe_wandb_log(
            {
                f"{name}/preP_H_fro": H_fro,
                f"{name}/preP_dXXT_fro": dXXT_fro,
                f"{name}/preP_H_dXXT_ratio": ratio,
            },
            step=self.block_idx,
        )

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # H condition number
        try:
            cond_num = torch.linalg.cond(H).item()
            logger.info(f'[GPTQv2][{name}] H cond num={cond_num:.4e}')
            cond_val = cond_num
        except Exception:
            logger.info(f'[GPTQv2][{name}] H cond num=inf')
            cond_val = float("inf")
        safe_wandb_log({f"{name}/H_cond_num": cond_val}, step=self.block_idx)

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        P = self.v2_alpha * ((dXXT @ Hinv.T).triu(diagonal=1)) @ Hinv
        self.P = P

        # Diagnostics after constructing P
        try:
            P_fro = torch.linalg.norm(P, ord="fro").item()
        except Exception:
            P_fro = torch.norm(P, p="fro").item()
        logger.info(f"[GPTQv2][{name}] post-P: ||P||_F={P_fro:.4e}")
        safe_wandb_log({f"{name}/postP_P_fro": P_fro}, step=self.block_idx)

        return W, Hinv

    def update_layer_with_transformed_weights(self, layer, W, Hinv, name):
        Losses = torch.zeros_like(W)
        tmp = torch.zeros_like(W)

        self.weight_transform(W, Hinv, Losses, tmp)
        torch.cuda.synchronize()
        total_error = torch.sum(Losses).item()
        logger.info(f"error {total_error}")
        safe_wandb_log(
            {
                f"{name}/gptq error": total_error,
            },
            step=self.block_idx,
        )

        if self.actorder or self.owq:
            tmp[:, self.n_nonout :] = W[:, self.n_nonout :]
            tmp = tmp[:, self.invperm]

        if isinstance(layer, transformers.Conv1D):
            tmp = tmp.t()

        layer.weight.data = tmp.reshape(layer.weight.shape)

        if self.wquantizer.granularity == "per_group" and not self.static_groups:
            self.update_model_qparams(layer)

    @torch.no_grad()
    def weight_transform(self, W, Hinv, Losses, tmp):
        # Same column-wise Cholesky backsubstitution as GPTQ. Hooks for asymmetric
        # calibration are collected in cache but not used here yet.
        P = self.P
        for i1 in range(0, self.n_nonout, self.blocksize):
            i2 = min(i1 + self.blocksize, self.n_nonout)
            count = i2 - i1
            W1, Hinv1, P1 = (
                W[:, i1:i2].clone(),
                Hinv[i1:i2, i1:i2],
                self.P[i1:i2, i1:i2],
            )
            tmp1, Err1, Losses1 = (
                torch.zeros_like(W1),
                torch.zeros_like(W1),
                torch.zeros_like(W1),
            )

            for i in range(count):
                w, d = W1[:, i], Hinv1[i, i]
                if self.wquantizer.granularity == "per_group":
                    idx = i1 + i
                    if not self.static_groups:
                        if idx % self.wquantizer.group_size == 0:
                            column_tensors = W[
                                :,
                                idx : min(
                                    (idx + self.wquantizer.group_size),
                                    (self.columns - self.n_out),
                                ),
                            ]
                            self.search_column_qparams(column_tensors, idx)
                    else:
                        if self.actorder:
                            idx = self.perm[idx]
                        self.qparams = self.groups[idx // self.wquantizer.group_size]

                if self.is_nvfp4:
                    q = self.wquantizer.quant_dequant(
                        w.unsqueeze(1),
                        self.qparams["global_scale"],
                        self.qparams["local_scales"],
                        self.qparams["qmax"],
                        self.qparams["qmin"],
                    )
                else:
                    q = self.wquantizer.quant_dequant(
                        w.unsqueeze(1),
                        self.qparams["scale"],
                        self.qparams["zero"],
                        self.qparams["qmax"],
                        self.qparams["qmin"],
                    )
                q = q.squeeze(1)

                tmp1[:, i] = q
                Losses1[:, i] = ((w - q) ** 2) / (2 * d**2)
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(
                    Hinv1[i, i:].unsqueeze(0)
                ) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

                # DEBUG:
                if Losses1.isnan().any() or Losses1.isinf().any():
                    raise ValueError("Losses is nan or inf")

            tmp[:, i1:i2], Losses[:, i1:i2] = tmp1, Losses1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

    @torch.no_grad()
    def cache_fp_input_hook(self, m, inp, out, name, feat_dict):
        if isinstance(inp, tuple):
            inp = inp[0]
        feat_dict[name].append(inp.cpu())

    @torch.no_grad()
    def block_opt(self, block):
        block_idx = self.block_idx
        logger.info(f"Collecting fp inputs for block {block_idx}")

        if self.online_rotate:
            self.replace_rotate_linears(block, self.online_rotate_exclude)

        block_ori_device = get_decoder_layer_ori_device(block)
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)
        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }

        self.fp_input_feat = defaultdict(list)
        fp_handles = []
        for name, module in input_feat_modules.items():
            handle = module.register_forward_hook(
                functools.partial(
                    self.cache_fp_input_hook,
                    name=name,
                    feat_dict=self.fp_input_feat,
                )
            )
            fp_handles.append(handle)

        next_fp_input_data = self.block_forward(
            block, [d.cuda() for d in self.fp_input_data]
        )
        for h in fp_handles:
            h.remove()

        self.fp_input_data = [d.cpu() for d in next_fp_input_data]
        block.to(block_ori_device)
        torch.cuda.empty_cache()
        logger.info(f"Finished collecting fp inputs for block {block_idx}")

        super().block_opt(block)

    @torch.no_grad()
    def cache_input_hook(self, m, inp, out, name, feat_dict):
        quant_inp = inp[0].data

        if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
            fp_inp = self.fp_input_feat[name][len(feat_dict[name])].to(quant_inp.device)
            self.add_batch(self.named_layers[name], name, quant_inp, out.data, fp_inp)

        feat_dict[name].append(quant_inp.cpu())

        # if self.act_static:
        #     super().cache_input_hook(m, inp, out, name, feat_dict)

    @torch.no_grad()
    def add_batch(self, layer, name, inp, out, fp_inp):
        world_size = int(os.environ["WORLD_SIZE"])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            fp_inp = fp_inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(
            layer, (FakeQuantLinear, nn.Linear, transformers.Conv1D, RotateLinear)
        ):
            if isinstance(layer, RotateLinear):
                fp_inp = layer.rotater.rotate(fp_inp)
                inp = layer.rotater.rotate(inp)
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                fp_inp = fp_inp.reshape((-1, fp_inp.shape[-1]))

            # # activation quantization
            # if "act" in self.quant_config:
            #     inp = self.a_qdq(inp, layer, self.aquantizer)
            inp = inp.t()
            fp_inp = fp_inp.t()

        if isinstance(layer, nn.Conv2d):
            unfold = nn.Unfold(
                layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

            fp_inp = unfold(fp_inp)
            fp_inp = fp_inp.permute([1, 0, 2])
            fp_inp = fp_inp.flatten(1)

        # if self.keep_sample_ratio < 1.0:
        #     keep_sample_num = int(inp.shape[1] * self.keep_sample_ratio)
        #     keep_sample_num = keep_sample_num // self.chunk_num * self.chunk_num
        #     tmp = tmp * self.keep_sample_ratio
        #     inp_l2_norm = torch.norm(inp, p=2, dim=0)
        #     sorted_ids = torch.argsort(inp_l2_norm, descending=True)
        #     if self.keep_sample_part == "top":
        #         keep_sample_ids = sorted_ids[:keep_sample_num]
        #     else:
        #         keep_sample_ids = sorted_ids[-keep_sample_num:]
        #     inp = inp[:, keep_sample_ids]
        #     fp_inp = fp_inp[:, keep_sample_ids]

        assert inp.shape[1] % self.chunk_num == 0, (
            f"Error: inp.shape[1] ({inp.shape[1]}) cannot be evenly divided by chunk_num."
        )
        chunks = torch.chunk(inp, self.chunk_num, dim=1)
        fp_chunks = torch.chunk(fp_inp, self.chunk_num, dim=1)

        prev_bs = self.layers_cache[name]["nsamples"]
        cur_bs = prev_bs + tmp
        self.layers_cache[name]["H"] *= prev_bs / cur_bs
        self.layers_cache[name]["dXXT"] *= prev_bs / cur_bs
        self.layers_cache[name]["nsamples"] = cur_bs

        for idx, chunk in enumerate(chunks):
            chunk = math.sqrt(2 / self.layers_cache[name]["nsamples"]) * chunk.float()
            self.layers_cache[name]["H"] += chunk.matmul(chunk.t())

            fp_chunk = fp_chunks[idx]
            fp_chunk = (
                math.sqrt(2 / self.layers_cache[name]["nsamples"]) * fp_chunk.float()
            )

            if (
                "act" not in self.quant_config
                and self.block_idx == 0
                and name
                in (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                )
            ):
                # # DEBUG:
                # print('debug')
                assert torch.allclose(fp_chunk, chunk)

            dX = fp_chunk - chunk
            # logger.info(f'layer.{self.block_idx}.{name}: dX.norm: {dX.norm()}, dX.max: {dX.abs().max()}')
            self.layers_cache[name]["dXXT"] += dX.matmul(chunk.t())

        # dist.all_reduce(self.layers_cache[name]["H"], op=dist.ReduceOp.SUM)
        # dist.all_reduce(
        #     torch.tensor(self.layers_cache[name]["nsamples"]).cuda(),
        #     op=dist.ReduceOp.SUM,
        # )
        # self.layers_cache[name]["H"] /= world_size

        # if "dXXT" in self.layers_cache[name]:
        #     dist.all_reduce(self.layers_cache[name]["dXXT"], op=dist.ReduceOp.SUM)
        #     self.layers_cache[name]["dXXT"] /= world_size

    @torch.no_grad()
    def layer_init(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.layers_cache[name]["H"] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev
        )
        self.layers_cache[name]["nsamples"] = 0
        self.layers_cache[name]["columns"] = W.shape[1]
        # Optional cross-covariance accumulator
        self.layers_cache[name]["dXXT"] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev
        )

    @torch.no_grad()
    def subset_init(self, subset):
        self.named_layers = subset["layers"]
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def block_init(self, block):
        self.named_layers = self.model.get_block_linears(block)
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def collect_model_qparams(self):
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            block = block.cuda()
            self.collect_block_qparams(block)
            block = block.cpu()

    @torch.no_grad()
    def split_qparams(self, qparams):
        group_qparams = []
        group_num = math.ceil(self.columns / self.wquantizer.group_size)
        qparams = qparams.reshape(math.ceil(qparams.shape[0] / group_num), -1)
        qparams = qparams.t()
        group_qparams = list(torch.split(qparams, 1, dim=0))
        for i in range(len(group_qparams)):
            group_qparams[i] = group_qparams[i].reshape(-1, 1)
        return group_qparams

    @torch.no_grad()
    def merge_qparams(self, qparams):
        if isinstance(qparams, int):
            return qparams
        if self.wquantizer.granularity == "per_head":
            head_size = self.rows // self.head_num
            qparams = qparams.t()
            qparams = qparams.repeat(head_size, 1)
            qparams = qparams.t()
            qparams = qparams.reshape(-1, 1)
        elif self.wquantizer.granularity == "per_group":
            qparams = torch.stack(qparams, dim=1)
            qparams = qparams.reshape(-1, 1)
        return qparams

    @torch.no_grad()
    def search_column_qparams(self, c_tensor, idx):
        _, scale, zero, qmax, qmin = self.wquantizer.get_tensor_qparams(c_tensor)
        self.qparams["scale"] = (
            scale.to(self.model_dtype) if self.force_scale_dtype else scale
        )
        self.qparams["zero"] = zero
        self.qparams["qmax"] = qmax
        self.qparams["qmin"] = qmin
        qparams = copy.deepcopy(self.qparams)
        self.groups[idx // self.wquantizer.group_size] = qparams

    @torch.no_grad()
    def search_layer_qparams(self, layer):
        scales = layer.buf_scales
        zeros = layer.buf_zeros
        scales = self.merge_qparams(scales)
        if not self.wquantizer.sym:
            zeros = self.merge_qparams(zeros)
        self.qparams["scale"] = (
            scales.to(self.model_dtype) if self.force_scale_dtype else scales
        )
        self.qparams["zero"] = zeros
        self.qparams["qmax"] = layer.buf_qmax
        self.qparams["qmin"] = layer.buf_qmin

    @torch.no_grad()
    def search_group_qparams(self, layer):
        if self.is_nvfp4:
            self.global_scale = layer.buf_global_scale
            local_scales = layer.buf_local_scales
            local_scales = self.split_qparams(local_scales)
            for i in range(len(local_scales)):
                qparams = {}
                qparams["global_scale"] = self.global_scale
                qparams["local_scales"] = local_scales[i]
                qparams["qmax"] = layer.buf_qmax
                qparams["qmin"] = layer.buf_qmin
                self.groups.append(qparams)
        else:
            scales = layer.buf_scales
            zeros = layer.buf_zeros
            self.group_scales = self.split_qparams(scales)
            if not self.wquantizer.sym:
                self.group_zeros = self.split_qparams(zeros)
            for i in range(len(self.group_scales)):
                qparams = {}
                qparams["scale"] = self.group_scales[i]
                if not self.wquantizer.sym:
                    qparams["zero"] = self.group_zeros[i]
                else:
                    qparams["zero"] = torch.tensor(0.0)
                qparams["qmax"] = layer.buf_qmax
                qparams["qmin"] = layer.buf_qmin
                self.groups.append(qparams)

    @torch.no_grad()
    def update_model_qparams(self, layer):
        _scales = []
        _zeros = []
        for g in self.groups:
            _scales.append(g["scale"])
            _zeros.append(g["zero"])
        scales = self.merge_qparams(_scales)
        layer.buf_scales = copy.deepcopy(scales)

        if not self.wquantizer.sym:
            zeros = self.merge_qparams(_zeros)
            layer.buf_zeros = copy.deepcopy(zeros)

    @torch.no_grad()
    def w_q(self, module, wquantizer):
        weight = module.weight.data
        args = {}
        args["qmax"] = module.buf_qmax
        args["qmin"] = module.buf_qmin
        if self.is_nvfp4:
            args["global_scale"] = module.buf_global_scale
            args["local_scales"] = module.buf_local_scales
        else:
            args["scales"] = module.buf_scales
            args["zeros"] = module.buf_zeros
            # args["scales"] = args["scales"]
            args["scales"] = args["scales"].to(self.model_dtype)

        return wquantizer.real_quant_weight_static(weight, args)

    @torch.no_grad()
    def w_qdq(self, module, wquantizer):
        weight = module.weight
        if self.need_perm:
            perm = module.buf_perm
            weight = module.weight[:, perm]

        args = {}
        if self.is_nvfp4:
            args["global_scale"] = module.buf_global_scale
            args["local_scales"] = module.buf_local_scales
        else:
            args["scales"] = module.buf_scales

        if hasattr(module, "buf_zeros"):
            args["zeros"] = module.buf_zeros
        else:
            args["zeros"] = None
        args["qmax"] = module.buf_qmax
        args["qmin"] = module.buf_qmin

        if self.owq:
            fp_weight = weight[:, module.buf_n_nonout :]

        weight = wquantizer.fake_quant_weight_static(weight, args).to(self.model_dtype)

        if self.owq:
            weight[:, module.buf_n_nonout :] = fp_weight.to(self.model_dtype)

        if self.need_perm:
            invperm = module.buf_invperm
            weight = weight[:, invperm]

        return weight

    @torch.no_grad()
    def deploy(self, quant_format):
        if quant_format not in ["fake_quant", "origin_float"]:
            assert not self.need_perm
        super().deploy(quant_format)

        if not self.is_nvfp4:
            self.model.convert_dtype(self.model_dtype)

    @torch.no_grad()
    def save_model(self, path):
        if not self.is_nvfp4:
            self.model.convert_dtype(self.model_dtype)
        super().save_model(path)

    @torch.no_grad()
    def free(self, name):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.P = None
        del self.layers_cache[name]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def ready(self):
        if self.is_nvfp4:
            if "global_scale" not in self.qparams or "local_scales" not in self.qparams:
                return False
            return torch.all(self.qparams["global_scale"] > 100) and torch.all(
                self.qparams["local_scales"] > 0
            )
        else:
            if "scale" not in self.qparams:
                return False
            return torch.all(self.qparams["scale"] != 0)
