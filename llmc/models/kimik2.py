import math
from typing import Tuple

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY

from .deepseekv3 import DeepseekV3


@MODEL_REGISTRY
class KimiK2Thinking(DeepseekV3):
    """
    Model wrapper that understands the packed-int4 MoE experts shipped with
    Kimi-K2-Thinking checkpoints and converts them back to floating-point
    weights before running LLMC quantization pipelines.
    """

    def build_model(self):
        super().build_model()
        converted = self._convert_packed_int4_weights()
        logger.info(f"Converted {converted} packed int4 expert weights for Kimi-K2.")

    def _convert_packed_int4_weights(self) -> int:
        converted = 0
        for module in self.model.modules():
            if hasattr(module, "weight_packed"):
                self._convert_single_module(module)
                converted += 1
        return converted

    def _convert_single_module(self, module: nn.Module) -> None:
        weight_packed = self._pop_tensor(module, "weight_packed")
        weight_scale = self._pop_tensor(module, "weight_scale")
        weight_shape = self._pop_tensor(module, "weight_shape")

        if weight_packed is None or weight_scale is None or weight_shape is None:
            raise ValueError(
                "Packed int4 module is missing required tensors: "
                f"weight_packed={weight_packed is not None}, "
                f"weight_scale={weight_scale is not None}, "
                f"weight_shape={weight_shape is not None}"
            )

        out_features, in_features = self._parse_shape(weight_shape)
        unpacked = self._unpack_int4(weight_packed, out_features, in_features)

        scales = weight_scale.to(torch.float32)
        groups = scales.shape[-1]
        group_size = math.ceil(in_features / groups)
        scales = scales.repeat_interleave(group_size, dim=1)[:, :in_features]

        weight = unpacked.to(torch.float32) * scales
        weight = weight.to(self.torch_dtype)

        parameter = nn.Parameter(weight, requires_grad=False)
        if "weight" in module._parameters:
            module._parameters["weight"] = parameter
        else:
            module.register_parameter("weight", parameter)

    @staticmethod
    def _parse_shape(weight_shape: torch.Tensor) -> Tuple[int, int]:
        if weight_shape.ndim != 1 or weight_shape.numel() != 2:
            raise ValueError(f"Unexpected weight_shape tensor: {weight_shape}")
        out_features = int(weight_shape[0].item())
        in_features = int(weight_shape[1].item())
        return out_features, in_features

    @staticmethod
    def _unpack_int4(
        packed: torch.Tensor, out_features: int, in_features: int
    ) -> torch.Tensor:
        packed = packed.to(torch.int32).view(out_features, -1)
        pack_factor = 32 // 4
        mask = (1 << 4) - 1
        chunks = []
        for idx in range(pack_factor):
            chunk = (packed >> (idx * 4)) & mask
            chunks.append(chunk)
        unpacked = torch.stack(chunks, dim=-1).reshape(out_features, -1)
        unpacked = unpacked[:, :in_features]
        unpacked = unpacked.to(torch.int8) - 8
        return unpacked.to(torch.float32)

    @staticmethod
    def _pop_tensor(module: nn.Module, name: str) -> torch.Tensor:
        if name in module._parameters and module._parameters[name] is not None:
            tensor = module._parameters.pop(name).detach()
        elif name in module._buffers and module._buffers[name] is not None:
            tensor = module._buffers.pop(name).detach()
        elif hasattr(module, name):
            tensor = getattr(module, name).detach()
            delattr(module, name)
        else:
            tensor = None

        if tensor is not None and tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return tensor


