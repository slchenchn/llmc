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
    def block_opt(self, block, *opt_kwargs):
        if self.quant_kvcache:
            self.register_kv_cache(block)

        weight_cfg = self.quant_config.weight
        if getattr(weight_cfg, "quant_type", None) == "nvfp4" and getattr(
            weight_cfg, "share_global_scale", False
        ):
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
        if getattr(weight_cfg, "quant_type", None) == "nvfp4" and getattr(
            weight_cfg, "share_global_scale", False
        ):
            args = {"global_scale": getattr(module, "buf_global_scale", None)}
            if hasattr(module, "buf_local_scales"):
                args["local_scales"] = module.buf_local_scales
            if hasattr(module, "buf_qmax"):
                args["qmax"] = module.buf_qmax
            if hasattr(module, "buf_qmin"):
                args["qmin"] = module.buf_qmin
                
            if len(args) == 4:
                return wquantizer.real_quant_weight_static(module.weight.data, args)
        else:
            args = {}

        return wquantizer.real_quant_weight_dynamic(module.weight.data, args)
