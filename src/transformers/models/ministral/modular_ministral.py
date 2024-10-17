# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Ministral model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    logging,
)
from .configuration_ministral import MinistralConfig
from ..mistral.configuration_mistral import MistralConfig
from ..mistral.modeling_mistral import (
    MistralRMSNorm,
    MistralRotaryEmbedding,
    MistralMLP,
    MistralDecoderLayer,
    MistralModel,
    MistralPreTrainedModel,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralForTokenClassification,

)
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2SdpaAttention,
    Gemma2FlashAttention2,
)


logger = logging.get_logger(__name__)


class MinistralConfig(MistralConfig):
    def __init__(**super_kwargs):
        super().__init__()


class MinistralRMSNorm(MistralRMSNorm):
    pass


class MinistralRotaryEmbedding(MistralRotaryEmbedding):
    pass



class MinistralMLP(MistralMLP):
    pass


class MinistralAttention(Gemma2Attention):
    def __init__(self, config: MinistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MinistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = 1. / math.sqrt(self.head_dim)


class MinistralFlashAttention2(MinistralAttention, Gemma2FlashAttention2):
    pass


class MinistralSdpaAttention(Gemma2SdpaAttention):
    pass


class MinistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MinistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MinistralMLP(config)
        self.input_layernorm = MinistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MinistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = not bool(layer_idx % 2)
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class MinistralPreTrainedModel(MistralPreTrainedModel):
    pass


class MinistralModel(MinistralPreTrainedModel, MistralModel):
    def __init__(self, config: MinistralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MinistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class MinistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MinistralModel(config)



class MinistralForSequenceClassification(MistralForSequenceClassification):
    pass


class MinistralForTokenClassification(MistralForTokenClassification):
    pass


class MinistralForQuestionAnswering(MistralPreTrainedModel):
    pass


__all__ = [
    "MinistralConfig",
    "MinistralPreTrainedModel",
    "MinistralModel",
    "MinistralForCausalLM",
    "MinistralForQuestionAnswering",
    "MinistralForSequenceClassification",
    "MinistralForTokenClassification",
]