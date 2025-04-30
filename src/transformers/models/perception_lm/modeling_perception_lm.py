# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PerceptionLM model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from timm.models.pe import PE
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_perception_lm import PerceptionLMConfig, PerceptionEncoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PerceptionLMConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/Perception-LM-1B"


class PerceptionEncoder(PE):
    def __init__(self, config: PerceptionEncoderConfig):
        assert config.pool_type == "none"
        self.use_cls_token = config.use_cls_token
        # Converting configs to timm PE args
        super().__init__(
            img_size=config.image_size,
            patch_size=config.patch_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            use_cls_token=config.use_cls_token,
            use_abs_posemb=config.use_abs_posemb,
            use_ln_post=config.use_ln_post,
            ls_init_value=config.ls_init_value,
            output_dim=config.width,
            use_attn_pool=False,
            use_proj=False,
        )
    
    def forward(self, x):
        x = super().forward(x)
        if self.use_cls_token:
            return x[:, 1:, :]
        else:
            return x

@dataclass
# Copied from transformers.models.llava.modeling_llava.LlavaCausalLMOutputWithPast with Llava->PerceptionLM
class PerceptionLMCausalLMOutputWithPast(ModelOutput):
    """
    Base class for PerceptionLM causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


# # Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->PerceptionLM
# class PerceptionLMMultiModalProjector(nn.Module):
#     def __init__(self, config: PerceptionLMConfig):
#         super().__init__()
#         # We have hidden_size * the number of vision feature layers
#         num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
#         self.linear_1 = nn.Linear(
#             config.vision_config.hidden_size * num_feature_layers,
#             config.text_config.hidden_size,
#             bias=config.multimodal_projector_bias,
#         )
#         self.act = ACT2FN[config.projector_hidden_act]
#         self.linear_2 = nn.Linear(
#             config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
#         )

#     def forward(self, image_features):
#         hidden_states = self.linear_1(image_features)
#         hidden_states = self.act(hidden_states)
#         hidden_states = self.linear_2(hidden_states)
#         return hidden_states


class AdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super(AdaptiveAvgPooling, self).__init__()
        self.pooling_ratio = pooling_ratio

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens

        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = F.adaptive_avg_pool2d(x, shape)
        x = x.flatten(2).transpose(1, 2)

        return x


class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__()
        input_size = config.vision_config.width
        output_size = config.text_config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=output_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=output_size,
                out_features=output_size,
                bias=True,
            ),
        )
        self.pooling = (
            AdaptiveAvgPooling(config.projector_pooling_ratio)
            if config.projector_pooling_ratio > 1
            else nn.Identity()
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.projector(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.pooling(x)
        return x


PERCEPTION_LM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PerceptionLMConfig`] or [`PerceptionLMVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    PERCEPTION_LM_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->PerceptionLM,llava->perception_lm
class PerceptionLMPreTrainedModel(PreTrainedModel):
    config_class = PerceptionLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PerceptionLMVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        # important: this ported version of PerceptionLM isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/perception_lm should serve for that purpose
        std = getattr(
            self.config,
            "initializer_range",
            self.config.get_text_config().initializer_range,
        )

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


PERCEPTION_LM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`PerceptionLMProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`Union[int, List[int]], *optional*, defaults to -2`):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The PERCEPTION_LM model which consists of a vision backbone and a language model.""",
    PERCEPTION_LM_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration with LLAVA->PERCEPTION_LM,Llava->PerceptionLM,llava-hf/llava-1.5-7b-hf->facebook/Perception-LM-1B
class PerceptionLMForConditionalGeneration(
    PerceptionLMPreTrainedModel, GenerationMixin
):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__(config)
        self.vision_model = PerceptionEncoder(config.vision_config)

        self.multi_modal_projector = PerceptionLMMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            )

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        print("pixel_values shape: ", pixel_values.shape)
        image_outputs = self.vision_model(pixel_values[0])
        # image_features = torch.load(
        #     "/checkpoint/vision_encoder/smhu/debug/0/h_img_dump_0.pt"
        # ).to(pixel_values.device).to(torch.bfloat16)
        print("image_outputs shape: ", image_outputs.shape)
        image_features = self.multi_modal_projector(image_outputs)
        return image_features

    @add_start_docstrings_to_model_forward(PERCEPTION_LM_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PerceptionLMCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, PerceptionLMCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

        >>> model = PerceptionLMForConditionalGeneration.from_pretrained("facebook/Perception-LM-1B")
        >>> processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values.to(inputs_embeds),
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            # if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            #     n_image_tokens = (input_ids == self.config.image_token_id).sum()
            #     n_image_features = image_features.shape[0] * image_features.shape[1]
            #     raise ValueError(
            #         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            #     )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )
            # print(inputs_embeds.shape)
            # occhi_embeds = torch.load(
            #     "/checkpoint/vision_encoder/smhu/debug/0/h_post_stitching_dump_0.pt"
            # )
            # print(occhi_embeds.shape)
            # inputs_embeds = occhi_embeds

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PerceptionLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["PerceptionLMForConditionalGeneration", "PerceptionLMPreTrainedModel"]
