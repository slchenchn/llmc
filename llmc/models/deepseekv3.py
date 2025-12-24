from loguru import logger

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class DeepseekV3(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        self.calibrate_all_experts = config.model.get("calibrate_all_experts", False)
        logger.info(f"calibrate_all_experts: {self.calibrate_all_experts}")

        self.expert_down_name_template = "mlp.experts.{}.down_proj"
        self.expert_gate_name_template = "mlp.experts.{}.gate_proj"
        self.expert_up_name_template = "mlp.experts.{}.up_proj"

        self.num_experts_per_tok = self.model_config.num_experts_per_tok
        self.is_moe = True

    def find_blocks(self):
        self.blocks = self.model.model.layers

        for block in self.blocks:
            if hasattr(block.mlp, "gate"):
                setattr(block.mlp.gate, "no_quant", True)

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens

    def find_block_name(self):
        self.block_name_prefix = "model.layers"

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def get_extra_modules(self, block):
        return {"mlp": block.mlp}

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
        return {
            "self_attn.matmul_1": block.self_attn.matmul_1,
            "self_attn.matmul_2": block.self_attn.matmul_2,
        }

    def get_softmax_in_block(self, block):
        return {"self_attn.softmax": block.self_attn.softmax}

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_moe_gate(self, block):
        if hasattr(block.mlp, "gate"):
            return {"mlp.gate": block.mlp.gate}
        else:
            return None

    def get_subsets_in_block(self, block):
        """
        NOTE: when true_sequential=True, the order of layers matters
        """
        if hasattr(block.mlp, "gate"):
            self.num_experts = len(block.mlp.experts)
            if self.calibrate_all_experts:
                return self._get_subsets_in_block_with_all_experts(block)
            else:
                return self._get_subsets_in_block_without_all_experts(block)
        else:
            layers = self._get_attn_subsets_in_block(block)
            layers.append(
                {
                    "layers": {
                        "mlp.gate_proj": block.mlp.gate_proj,
                        "mlp.up_proj": block.mlp.up_proj,
                    },
                    "prev_op": [block.post_attention_layernorm],
                    "input": ["mlp.gate_proj"],
                    "inspect": block.mlp,
                    "has_kwargs": False,
                    "is_mlp": True,
                }
            )

            layers.append(
                {
                    "layers": {"mlp.down_proj": block.mlp.down_proj},
                    "prev_op": [block.mlp.up_proj],
                    "input": ["mlp.down_proj"],
                    "inspect": block.mlp.down_proj,
                    "has_kwargs": False,
                    "is_mlp": True,
                }
            )
            return layers

    def _get_attn_subsets_in_block(self, block):
        layers = []

        layers.append(
            {
                "layers": {
                    "self_attn.q_a_proj": block.self_attn.q_a_proj,
                    "self_attn.kv_a_proj_with_mqa": block.self_attn.kv_a_proj_with_mqa,  # noqa
                },
                "prev_op": [block.input_layernorm],
                "input": ["self_attn.q_a_proj"],
                "inspect": block.self_attn,
                "has_kwargs": True,
            }
        )
        layers.append(
            {
                "layers": {"self_attn.q_b_proj": block.self_attn.q_b_proj},
                "prev_op": [block.self_attn.q_a_layernorm],
                "input": ["self_attn.q_b_proj"],
                "inspect": block.self_attn.q_b_proj,
                "has_kwargs": False,
                "true_sequential": False,
                "skip_rotate": True,
            }
        )

        layers.append(
            {
                "layers": {"self_attn.kv_b_proj": block.self_attn.kv_b_proj},
                "prev_op": [block.self_attn.kv_a_layernorm],
                "input": ["self_attn.kv_b_proj"],
                "inspect": block.self_attn.kv_b_proj,
                "has_kwargs": False,
                "skip_rotate": True,
            }
        )
        layers.append(
            {
                "layers": {"self_attn.o_proj": block.self_attn.o_proj},
                "prev_op": [None],
                "input": ["self_attn.o_proj"],
                "inspect": block.self_attn.o_proj,
                "has_kwargs": False,
            },
        )
        return layers

    def _get_down_proj_in_block(self, block):
        layers = []
        for i in range(self.num_experts):
            layers.append(
                {
                    "layers": {
                        f"mlp.experts.{i}.down_proj": block.mlp.experts[i].down_proj
                    },  # noqa
                    "prev_op": [block.mlp.experts[i].up_proj],
                    "input": [f"mlp.experts.{i}.down_proj"],
                    "inspect": block.mlp.experts[i].down_proj,
                    "has_kwargs": False,
                    "is_mlp": True,
                    "true_sequential": False,
                }
            )

        layers.append(
            {
                "layers": {
                    "mlp.shared_experts.down_proj": block.mlp.shared_experts.down_proj
                },  # noqa
                "prev_op": [block.mlp.shared_experts.up_proj],
                "input": ["mlp.shared_experts.down_proj"],
                "inspect": block.mlp.shared_experts.down_proj,
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
                        f"mlp.experts.{i}.gate_proj": block.mlp.experts[i].gate_proj
                        for i in range(self.num_experts)
                    },
                    **{
                        f"mlp.experts.{i}.up_proj": block.mlp.experts[i].up_proj
                        for i in range(self.num_experts)
                    },
                    "mlp.shared_experts.gate_proj": block.mlp.shared_experts.gate_proj,  # noqa
                    "mlp.shared_experts.up_proj": block.mlp.shared_experts.up_proj,
                    "mlp.gate": block.mlp.gate,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["mlp.gate"],
                "inspect": block.mlp.gate,
                "has_kwargs": False,
                "is_mlp": True,
            }
        )
        layers.extend(self._get_down_proj_in_block(block))
        return layers

    def _get_subsets_in_block_without_all_experts(self, block):
        layers = self._get_attn_subsets_in_block(block)

        layers.append(
            {
                "layers": {
                    "mlp.gate": block.mlp.gate,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["mlp.gate"],
                "inspect": block.mlp.gate,
                "has_kwargs": False,
                "is_mlp": True,
            }
        )

        for i in range(self.num_experts):
            layers.append(
                {
                    "layers": {
                        f"mlp.experts.{i}.gate_proj": block.mlp.experts[i].gate_proj,
                        f"mlp.experts.{i}.up_proj": block.mlp.experts[i].up_proj,
                    },
                    "prev_op": [block.post_attention_layernorm],
                    "input": [f"mlp.experts.{i}.gate_proj"],
                    "inspect": block.mlp.experts[i].gate_proj,
                    "has_kwargs": False,
                    "is_mlp": True,
                    "true_sequential": False,
                }
            )

        layers.append(
            {
                "layers": {
                    "mlp.shared_experts.gate_proj": block.mlp.shared_experts.gate_proj,
                    "mlp.shared_experts.up_proj": block.mlp.shared_experts.up_proj,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["mlp.shared_experts.gate_proj"],
                "inspect": block.mlp.shared_experts.gate_proj,
                "has_kwargs": False,
                "is_mlp": True,
                "true_sequential": True,
            }
        )

        layers.extend(self._get_down_proj_in_block(block))
        return layers
