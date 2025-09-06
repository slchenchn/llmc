from importlib.metadata import version

import packaging

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Qwen3MOE(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        self.rotary_emb = self.model.model.rotary_emb

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens
        if packaging.version.parse(version("transformers")) >= packaging.version.parse(
            "4.45.0"
        ):
            self.rotary_emb = self.model.model.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = "model.layers"
        self.pairs = {"q_proj": "qkv", "o_proj": "out", "up_proj": "fc1"}

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_attn_in_block(self, block):
        return {"self_attn": block.self_attn}

    def get_attention_rotary_layers(self):
        if packaging.version.parse(version("transformers")) >= packaging.version.parse(
            "4.45.0"
        ):
            return [self.rotary_emb]
        else:
            return []

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version("transformers")) >= packaging.version.parse(
            "4.45.0"
        ):
            return [
                self.embed_tokens,
                self.rotary_emb,
                self.model.model.norm,
                self.model.lm_head,
            ]  # noqa
        else:
            return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def skip_layer_name(self):
        return ["lm_head"]

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            "input_layernorm": block.input_layernorm,
            "post_attention_layernorm": block.post_attention_layernorm,
        }

    # flake8: noqa
    def apply_chat_template(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text

    def get_subsets_in_block(self, block):
        layers = []
        # attn
        layers.append(
            {
                "layers": {
                    "self_attn.q_proj": block.self_attn.q_proj,
                    "self_attn.k_proj": block.self_attn.k_proj,
                    "self_attn.v_proj": block.self_attn.v_proj,
                },
                "prev_op": [block.input_layernorm],
                "input": ["self_attn.q_proj"],
                "inspect": block.self_attn,
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

        # mlp
        layers.append(
            {
                "layers": {
                    **{
                        f"mlp.experts.{i}.gate_proj": block.mlp.experts[i].gate_proj
                        for i in range(len(block.mlp.experts))
                    },
                    **{
                        f"mlp.experts.{i}.up_proj": block.mlp.experts[i].up_proj
                        for i in range(len(block.mlp.experts))
                    },
                    "mlp.gate": block.mlp.gate,
                },
                "prev_op": [block.post_attention_layernorm],
                "input": ["mlp"],
                "inspect": block.mlp,
                "has_kwargs": False,
                "is_mlp": True,
            }
        )
        for i in range(len(block.mlp.experts)):
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
                }
            )
        return layers
