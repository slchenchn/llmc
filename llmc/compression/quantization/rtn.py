import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization


@ALGO_REGISTRY
class RTN(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device("cuda")

    @torch.no_grad()
    def get_shared_global_absmax(self, block):
        q_abs_max = block.self_attn.q_proj.weight.abs().max()
        k_abs_max = block.self_attn.k_proj.weight.abs().max()
        v_abs_max = block.self_attn.v_proj.weight.abs().max()
        up_abs_max = block.mlp.up_proj.weight.abs().max()
        gate_abs_max = block.mlp.gate_proj.weight.abs().max()

        qkv_abs_max = max(q_abs_max, k_abs_max, v_abs_max)
        upgate_abs_max = max(gate_abs_max, up_abs_max)

        qkv_global_scale = self.wquantizer.get_global_scale(qkv_abs_max)
        upgate_global_scale = self.wquantizer.get_global_scale(upgate_abs_max)

        block.self_attn.q_proj.register_buffer("buf_global_scale", qkv_global_scale)
        block.self_attn.k_proj.register_buffer("buf_global_scale", qkv_global_scale)
        block.self_attn.v_proj.register_buffer("buf_global_scale", qkv_global_scale)
        block.mlp.gate_proj.register_buffer("buf_global_scale", upgate_global_scale)
        block.mlp.up_proj.register_buffer("buf_global_scale", upgate_global_scale)

        if self.act_static:
            _, _, q_local_scales, _, _ = self.wquantizer.get_tensor_qparams(
                block.self_attn.q_proj.weight.clone(),
                {"global_scale": qkv_global_scale},
            )
            block.self_attn.q_proj.register_buffer("buf_local_scales", q_local_scales)
            _, _, k_local_scales, _, _ = self.wquantizer.get_tensor_qparams(
                block.self_attn.k_proj.weight.clone(),
                {"global_scale": qkv_global_scale},
            )
            block.self_attn.k_proj.register_buffer("buf_local_scales", k_local_scales)
            _, _, v_local_scales, _, _ = self.wquantizer.get_tensor_qparams(
                block.self_attn.v_proj.weight.clone(),
                {"global_scale": qkv_global_scale},
            )
            block.self_attn.v_proj.register_buffer("buf_local_scales", v_local_scales)

            _, _, gate_local_scales, _, _ = self.wquantizer.get_tensor_qparams(
                block.mlp.gate_proj.weight.clone(),
                {"global_scale": upgate_global_scale},
            )
            block.mlp.gate_proj.register_buffer("buf_local_scales", gate_local_scales)
            _, _, up_local_scales, _, _ = self.wquantizer.get_tensor_qparams(
                block.mlp.up_proj.weight.clone(),
                {"global_scale": upgate_global_scale},
            )
            block.mlp.up_proj.register_buffer("buf_local_scales", up_local_scales)

    @torch.no_grad()
    def block_opt(self, block, *opt_kwargs):
        if self.quant_kvcache:
            self.register_kv_cache(block)

        weight_cfg = self.quant_config.weight
        if getattr(weight_cfg, "quant_type", None) == "int-quant" and getattr(weight_cfg, "share_global_scale", False):
            # self.get_shared_global_absmax(block)
            self.collect_block_qparams(block)

        if self.act_static:
            super().block_opt(block, *opt_kwargs)

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        pass

    @torch.no_grad()
    def w_q(self, module, wquantizer):
        weight_cfg = self.quant_config.weight
        if getattr(weight_cfg, "quant_type", None) == "int-quant" and getattr(weight_cfg, "share_global_scale", False):
            args = {"global_scale": getattr(module, "buf_global_scale", None)}
        else:
            args = {}

        return wquantizer.real_quant_weight_dynamic(module.weight.data, args)
