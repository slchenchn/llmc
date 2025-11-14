import torch

from .quant_nvfp4 import FP4_E2M1_DATA, FP8_E4M3_DATA, NVFP4Quantizer


class NVFP4AblationQuantizer(NVFP4Quantizer):
    """NVFP4 quantizer variants for isolating rounding error sources.

    This class optionally disables the FP4 rounding of the quantized tensor and/or
    the FP8 rounding of the local scales so that the contribution of each step can
    be measured independently.
    """

    def __init__(
        self,
        bit: int,
        symmetric: bool,
        granularity: str,
        *,
        quantize_fp4: bool = True,
        quantize_local_scales: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.quantize_fp4 = quantize_fp4
        self.quantize_local_scales = quantize_local_scales

    def get_local_scales(self, global_scale, absmax):
        """Return (optionally quantized) local scales."""
        local_scales = global_scale * (absmax / FP4_E2M1_DATA.max)

        if self.quantize_local_scales:
            local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)
        else:
            local_scales = local_scales.float()

        return local_scales

    def quant(self, tensor, global_scale, local_scales, qmax, qmin):
        """Quantize tensor with optional FP4 rounding."""
        if self.quantize_local_scales:
            if local_scales.dtype != torch.float8_e4m3fn:
                local_scales = FP8_E4M3_DATA.cast_to_positive_fp8(local_scales)
        else:
            assert local_scales.dtype == torch.float32, "local_scales must be float32"
            local_scales = local_scales.clamp(FP8_E4M3_DATA.min, FP8_E4M3_DATA.max)

        x_scaled = (tensor / local_scales.float()) * global_scale

        if self.quantize_fp4:
            x_quant = FP4_E2M1_DATA.cast_to_fp4(x_scaled)
        else:
            x_quant = x_scaled.clamp(qmin, qmax)

        return x_quant

    def real_quant_weight_static(self, weight, args):
        raise NotImplementedError(
            "NVFP4AblationQuantizer only supports fake quantization flows."
        )

    def real_quant_weight_dynamic(self, weight, args=None):
        raise NotImplementedError(
            "NVFP4AblationQuantizer only supports fake quantization flows."
        )

    def real_quant_act_dynamic(self, act, args=None):
        raise NotImplementedError(
            "NVFP4AblationQuantizer only supports fake quantization flows."
        )


def build_nvfp4_ablation_quantizers(group_size: int = 16):
    """Factory returning the NVFP4 quantizer variants used in activation ablations."""
    base_kwargs = dict(
        bit=4, symmetric=True, granularity="per_group", group_size=group_size
    )

    quantizers = {
        "NVFP4": NVFP4Quantizer(**base_kwargs),
        "NVFP4_no_fp4": NVFP4AblationQuantizer(
            **base_kwargs, quantize_fp4=False, quantize_local_scales=True
        ),
        "NVFP4_no_fp8_scale": NVFP4AblationQuantizer(
            **base_kwargs, quantize_fp4=True, quantize_local_scales=False
        ),
    }
    return quantizers
