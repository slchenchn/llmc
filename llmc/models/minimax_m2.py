import torch
from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class MiniMaxM2(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        self.calibrate_all_experts = config.model.get("calibrate_all_experts", False)
        logger.info(f"calibrate_all_experts: {self.calibrate_all_experts}")

        self.expert_down_name_template = "block_sparse_moe.experts.{}.w2"
        self.expert_gate_name_template = "block_sparse_moe.experts.{}.w1"
        self.expert_up_name_template = "block_sparse_moe.experts.{}.w3"

        self.num_experts_per_tok = self.model.num_experts_per_tok
        self.is_moe = True

        self.convert_weights_to_model_dtype()

    def convert_weights_to_model_dtype(self):
        def _may_need_convert(name, param):
            if "scale" in name or 'bias' in name:
                return

            if 'inv_freq' in name:
                return 
                
            if param.dtype in convert_dtypes and param.dtype != target_dtype:
                logger.info(f"Converted {name} from {param.dtype} to {target_dtype}")
                param.data = param.data.to(target_dtype)

        target_dtype = self.model.dtype
        convert_dtypes = (torch.bfloat16, torch.float16, torch.float32)
        for name, param in self.model.named_parameters():
            _may_need_convert(name, param)
        for name, buffer in self.model.named_buffers():
            _may_need_convert(name, buffer)

    def find_blocks(self):
        self.blocks = self.model.model.layers

        # add attr:no_quant to moe.gate
        for block in self.blocks:
            # assert not hasattr(block.block_sparse_moe.gate, "no_quant"), "moe.gate already has no_quant"
            setattr(block.block_sparse_moe.gate, "no_quant", True)

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens

    def find_block_name(self):
        self.block_name_prefix = "model.layers"

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def get_extra_modules(self, block):
        return {"block_sparse_moe": block.block_sparse_moe}

    def skip_layer_name(self):
        return ["lm_head"]

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            "input_layernorm": block.input_layernorm,
            "post_attention_layernorm": block.post_attention_layernorm,
        }

    def get_attn_in_block(self, block):
        return {"self_attn": block.self_attn}

    def get_matmul_in_block(self, block):
        raise NotImplementedError("MiniMaxM2 does not support matmul in block")
        return {
            "self_attn.matmul_1": block.self_attn.matmul_1,
            "self_attn.matmul_2": block.self_attn.matmul_2,
        }

    def get_softmax_in_block(self, block):
        raise NotImplementedError("MiniMaxM2 does not support softmax in block")
        return {"self_attn.softmax": block.self_attn.softmax}

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_moe_gate(self, block):
        if hasattr(block.block_sparse_moe, "gate"):
            return {"block_sparse_moe.gate": block.block_sparse_moe.gate}
        else:
            return None

    def get_subsets_in_block(self, block):
        self.num_experts = len(block.block_sparse_moe.experts)
        if self.calibrate_all_experts:
            return self._get_subsets_in_block_with_all_experts(block)
        else:
            return self._get_subsets_in_block_without_all_experts(block)

    def _get_attn_subsets_in_block(self, block):
        layers = []
        layers.append(
            {
                "layers": {
                    "self_attn.q_proj": block.self_attn.q_proj,
                    "self_attn.k_proj": block.self_attn.k_proj,
                    "self_attn.v_proj": block.self_attn.v_proj,
                },
                "prev_op": [block.input_layernorm],
                "input": ["self_attn.q_proj"],
                "inspect": block.self_attn.q_proj,
                "has_kwargs": True,
            }
        )

        layers.append(
            {
                "layers": {"self_attn.o_proj": block.self_attn.o_proj},
                "prev_op": [block.self_attn.v_proj],
                "input": ["self_attn.o_proj"],
                "inspect": block.self_attn.o_proj,
                "has_kwargs": False,
            }
        )
        return layers

    def _get_down_proj_in_block(self, block):
        layers = []
        for i in range(self.num_experts):
            layers.append(
                {
                    "layers": {
                        f"block_sparse_moe.experts.{i}.w2": block.block_sparse_moe.experts[
                            i
                        ].w2
                    },
                    "prev_op": [block.block_sparse_moe.experts[i].w3],
                    "input": [f"block_sparse_moe.experts.{i}.w2"],
                    "inspect": block.block_sparse_moe.experts[i].w2,
                    "has_kwargs": False,
                    "is_mlp": True,
                    "true_sequential": False,
                }
            )

        return layers

    def _get_subsets_in_block_with_all_experts(self, block):
        layers = self._get_attn_subsets_in_block(block)

        layers.append(
            {
                "layers": {
                    **{
                        f"block_sparse_moe.experts.{i}.w1": block.block_sparse_moe.experts[
                            i
                        ].w1
                        for i in range(self.num_experts)
                    },
                    **{
                        f"block_sparse_moe.experts.{i}.w3": block.block_sparse_moe.experts[
                            i
                        ].w3
                        for i in range(self.num_experts)
                    },
                    "block_sparse_moe.gate": block.block_sparse_moe.gate,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["block_sparse_moe.gate"],
                "inspect": block.block_sparse_moe.gate,
                "has_kwargs": False,
                "is_mlp": True,
            }
        )
        layers.extend(self._get_down_proj_in_block(block))
        return layers

    def _get_subsets_in_block_without_all_experts(self, block):
        layers = self._get_attn_subsets_in_block(block)

        # mlp
        layers.append(
            {
                "layers": {
                    "block_sparse_moe.gate": block.block_sparse_moe.gate,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["block_sparse_moe.gate"],
                "inspect": block.block_sparse_moe.gate,
                "has_kwargs": False,
                "is_mlp": True,
            }
        )
        for i in range(self.num_experts):
            layers.append(
                {
                    "layers": {
                        f"block_sparse_moe.experts.{i}.w1": block.block_sparse_moe.experts[
                            i
                        ].w1,
                        f"block_sparse_moe.experts.{i}.w3": block.block_sparse_moe.experts[
                            i
                        ].w3,
                    },
                    "prev_op": [block.post_attention_layernorm],
                    "input": [f"block_sparse_moe.experts.{i}.w1"],
                    "inspect": block.block_sparse_moe.experts[i].w1,
                    "has_kwargs": False,
                    "is_mlp": True,
                    "true_sequential": True if i == self.num_experts - 1 else False,
                }
            )

        layers.extend(self._get_down_proj_in_block(block))
        return layers
