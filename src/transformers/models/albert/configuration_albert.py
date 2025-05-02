# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""ALBERT model configuration"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

from huggingface_hub.dataclasses import strict, validated_field

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...validators import activation_fn_key, interval, probability, token


@strict(accept_kwargs=True)
@dataclass
class AlbertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AlbertModel`] or a [`TFAlbertModel`]. It is used
    to instantiate an ALBERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the ALBERT
    [albert/albert-xxlarge-v2](https://huggingface.co/albert/albert-xxlarge-v2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30000):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AlbertModel`] or [`TFAlbertModel`].
        embedding_size (`int`, *optional*, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_hidden_groups (`int`, *optional*, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 16384):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        inner_group_num (`int`, *optional*, defaults to 1):
            The number of inner repetition of attention and ffn.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`AlbertModel`] or [`TFAlbertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for attached classifiers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.

    Examples:

    ```python
    >>> from transformers import AlbertConfig, AlbertModel

    >>> # Initializing an ALBERT-xxlarge style configuration
    >>> albert_xxlarge_configuration = AlbertConfig()

    >>> # Initializing an ALBERT-base style configuration
    >>> albert_base_configuration = AlbertConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072,
    ... )

    >>> # Initializing a model (with random weights) from the ALBERT-base style configuration
    >>> model = AlbertModel(albert_xxlarge_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = validated_field(interval(min=1), default=30000)
    embedding_size: int = validated_field(interval(min=1), default=128)
    hidden_size: int = validated_field(interval(min=1), default=4096)
    num_hidden_layers: int = validated_field(interval(min=1), default=12)
    num_hidden_groups: int = validated_field(interval(min=1), default=1)
    num_attention_heads: int = validated_field(interval(min=0), default=64)
    intermediate_size: int = validated_field(interval(min=1), default=16384)
    inner_group_num: int = validated_field(interval(min=0), default=1)
    hidden_act: str = validated_field(activation_fn_key, default="gelu_new")
    hidden_dropout_prob: float = validated_field(probability, default=0.0)
    attention_probs_dropout_prob: float = validated_field(probability, default=0.0)
    max_position_embeddings: int = validated_field(interval(min=0), default=512)
    type_vocab_size: int = validated_field(interval(min=1), default=2)
    initializer_range: float = validated_field(interval(min=0.0), default=0.02)
    layer_norm_eps: float = validated_field(interval(min=0.0), default=1e-12)
    classifier_dropout_prob: float = validated_field(probability, default=0.1)
    position_embedding_type: Literal["absolute", "relative_key", "relative_key_query"] = "absolute"
    pad_token_id: Optional[int] = validated_field(token, default=0)
    bos_token_id: Optional[int] = validated_field(token, default=2)
    eos_token_id: Optional[int] = validated_field(token, default=3)

    # Not part of __init__
    model_type = "albert"

    def __post_init__(self):
        """Called after `__init__`: validates the instance."""
        self.validate()

    def validate(self):
        """Ensures the configuration is valid by assessing combinations of arguments."""
        # Architecture validation
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads}"
            )

        # Token validation
        for token_name in ["pad_token_id", "bos_token_id", "eos_token_id"]:
            token_id = getattr(self, token_name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"{token_name} must be in the vocabulary with size {self.vocab_size}, i.e. between 0 and "
                    f"{self.vocab_size - 1}, got {token_id}."
                )


# Copied from transformers.models.bert.configuration_bert.BertOnnxConfig with Roberta->Albert
class AlbertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


__all__ = ["AlbertConfig", "AlbertOnnxConfig"]
