# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...image_processing_utils_fast import (
    BatchFeature,
    get_size_dict,
)
from ...image_transforms import convert_to_rgb, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...processing_utils import Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    LossKwargs,
    TensorType,
    filter_out_non_signature_kwargs,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..deepseek_vl.configuration_deepseek_vl import DeepseekVLConfig
from ..deepseek_vl.image_processing_deepseek_vl import DeepseekVLImageProcessor
from ..deepseek_vl.modeling_deepseek_vl import (
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    DeepseekVLPreTrainedModel,
)
from ..deepseek_vl.processing_deepseek_vl import DeepseekVLProcessor, DeepseekVLProcessorKwargs
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..sam.modeling_sam import SamLayerNorm, SamVisionNeck


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekVLHybridConfig"
_CHECKPOINT_FOR_DOC = "deepseek-ai/deepseek-vl-7b-chat-hf"
_EXPECTED_OUTPUT_SHAPE = [1, 628, 4096]


class DeepseekVLHybridConfig(DeepseekVLConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekVLHybridModel`]. It is used to instantiate a
    DeepseekVLHybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekVLHybrid
    [deepseek-ai/deepseek-vl-7b-chat-hf](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        high_res_vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SamVisionConfig`):
            The config object or dictionary of the high resolution vision backbone.
        image_token_id (`int`, *optional*, defaults to 100015):
            The index representing image tokens in the model's token vocabulary.

    Example:

    ```python
    >>> from transformers import DeepseekVLHybridConfig, DeepseekVLHybridModel

    >>> # Initializing a DeepseekVLHybrid deepseek-ai/deepseek-vl-7b-chat-hf style configuration
    >>> configuration = DeepseekVLHybridConfig()

    >>> # Initializing a model (with random weights) from the deepseek-ai/deepseek-vl-7b-chat-hf style configuration
    >>> model = DeepseekVLHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl_hybrid"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "high_res_vision_config": AutoConfig}

    def __init__(
        self,
        text_config: AutoConfig = None,
        vision_config: AutoConfig = None,
        high_res_vision_config: AutoConfig = None,
        image_token_id: int = 100015,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            **kwargs,
        )

        if high_res_vision_config is None:
            high_res_vision_config = {}
            logger.info("`high_res_vision_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        if isinstance(high_res_vision_config, dict):
            high_res_vision_config["model_type"] = high_res_vision_config.get("model_type", "sam_vision_model")
            high_res_vision_config = CONFIG_MAPPING[high_res_vision_config["model_type"]](**high_res_vision_config)

        self.high_res_vision_config = high_res_vision_config


@dataclass
class DeepseekVLHybridBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


@dataclass
class DeepseekVLHybridCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class DeepseekVLHybridLayerNorm(SamLayerNorm):
    pass


class DeepseekVLSamVisionNeck(SamVisionNeck):
    def __init__(self, config):
        super().__init__(config)


class DeepseekVLSamVisionProj(nn.Module):
    def __init__(self, config, output_size: int = 24):
        super().__init__()
        self.config = config
        self.output_size = output_size

        self.conv1 = nn.Conv2d(
            config.output_channels, config.output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            config.output_channels * 2, config.output_channels * 4, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # interpolate Sam encodings to match Siglip encodings
        features = torch.nn.functional.interpolate(
            features,
            size=(4 * self.output_size, 4 * self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.conv1(features)
        features = self.conv2(features)
        return features


class DeepseekVLHybridAligner(nn.Module):
    def __init__(self, config: DeepseekVLHybridConfig):
        super().__init__()

        in_channels = config.vision_config.hidden_size
        high_res_in_channels = config.high_res_vision_config.output_channels * 4
        out_channels = config.text_config.hidden_size

        self.vision_proj = nn.Linear(in_channels, out_channels // 2)
        self.high_res_vision_proj = nn.Linear(high_res_in_channels, out_channels // 2)

        self.act = nn.GELU()
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(
        self,
        vision_encodings: torch.Tensor,
        high_res_vision_encodings: torch.Tensor,
    ) -> torch.Tensor:
        vision_encodings = self.vision_proj(vision_encodings)
        high_res_vision_encodings = self.high_res_vision_proj(high_res_vision_encodings)

        encodings = torch.concat([high_res_vision_encodings, vision_encodings], dim=-1)
        encodings = self.act(encodings)
        encodings = self.proj(encodings)

        return encodings


class DeepseekVLHybridPreTrainedModel(DeepseekVLPreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekVLHybridLayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, DeepseekVLHybridModel):
            module.high_res_vision_alpha.data.zero_()


DEEPSEEK_VL_HYBRID_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`].
        high_res_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`].
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
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class DeepseekVLHybridModel(DeepseekVLModel):
    def __init__(self, config):
        self.output_size = config.vision_config.image_size // config.vision_config.patch_size
        self.global_attn_index = config.high_res_vision_config.global_attn_indexes[0]

        self.high_res_vision_model = AutoModel.from_config(config.high_res_vision_config)
        self.high_res_vision_neck = DeepseekVLSamVisionNeck(config.high_res_vision_config)
        self.high_res_vision_proj = DeepseekVLSamVisionProj(
            config.high_res_vision_config, output_size=self.output_size
        )
        self.high_res_vision_alpha = nn.Parameter(torch.zeros(1))

        super().__init__(config)

    def get_low_res_image_features(self, pixel_values):
        output = self.vision_model(pixel_values)
        output = output[0]
        return output

    def get_high_res_image_features(self, pixel_values):
        output = self.high_res_vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        last_hidden_state = output[0]
        last_hidden_state = self.high_res_vision_proj(last_hidden_state)

        hidden_states = output[1]
        global_hidden_state = hidden_states[self.global_attn_index + 1]  # +1 for embedding layer
        global_hidden_state = self.high_res_vision_neck(global_hidden_state)
        global_hidden_state = self.high_res_vision_proj(global_hidden_state)

        output = last_hidden_state + global_hidden_state * self.high_res_vision_alpha

        # batch_size, hidden_size, height, width -> batch_size, seq_len, hidden_size
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.shape[0], -1, output.shape[-1])

        return output

    def get_image_features(self, pixel_values, high_res_pixel_values):
        vision_encodings = self.get_low_res_image_features(pixel_values)
        high_res_vision_encodings = self.get_high_res_image_features(high_res_pixel_values)
        images_embeds = self.aligner(vision_encodings, high_res_vision_encodings)
        return images_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        high_res_pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and high_res_pixel_values is None:
            raise ValueError("Both pixel_values and high_res_pixel_values should be specified at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, high_res_pixel_values)
            image_attention_mask = input_ids == self.config.image_token_id

            embed_dim = inputs_embeds.shape[-1]
            image_features = image_features.reshape(-1, embed_dim)
            image_attention_mask = image_attention_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            flash_attn_kwargs=flash_attn_kwargs,
        )

        output = DeepseekVLHybridBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

        return output


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class DeepseekVLHybridForConditionalGeneration(DeepseekVLForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        high_res_pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        >>> import torch
        >>> from transformers import DeepseekVLHybridForConditionalGeneration, DeepseekVLHybridProcessor

        >>> model_id = "deepseek-ai/deepseek-vl-7b-chat-hf"

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {'type':'image', 'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
        ...             {'type':"text", "text":"What do you see in this image?."}
        ...         ]
        ...     },
        ... ]

        >>> processor = DeepseekVLHybridProcessor.from_pretrained(model_id)
        >>> model = DeepseekVLHybridForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     add_generation_prompt=True,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device, dtype=torch.bfloat16)

        >>> output = model.generate(**inputs, max_new_tokens=40, do_sample=True)
        >>> text = processor.decode(output[0], skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            high_res_pixel_values=high_res_pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        output = DeepseekVLHybridCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        high_res_pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
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
            model_inputs["high_res_pixel_values"] = high_res_pixel_values

        return model_inputs


class DeepseekVLHybridImageProcessor(DeepseekVLImageProcessor):
    r"""
    Constructs a DeepseekVLHybrid image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        high_res_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the high resolution output image after resizing. Can be overridden by the `high_res_size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        high_res_resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        high_res_image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the high resolution image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `high_res_image_mean` parameter in the `preprocess` method.
        high_res_image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            Standard deviation to use if normalizing the high resolution image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `high_res_image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        high_res_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        high_res_resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        high_res_image_mean: Optional[Union[float, List[float]]] = None,
        high_res_image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        **kwargs,
    ) -> None:
        high_res_size = high_res_size if high_res_size is not None else {"height": 1024, "width": 1024}
        high_res_size = get_size_dict(high_res_size, default_to_square=True)

        self.high_res_size = high_res_size
        self.high_res_image_mean = high_res_image_mean if high_res_image_mean is not None else OPENAI_CLIP_MEAN
        self.high_res_image_std = high_res_image_std if high_res_image_std is not None else OPENAI_CLIP_STD

        self.resample = resample
        self.high_res_resample = high_res_resample

        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )

        self.background_color = tuple([int(x * 255) for x in self.high_res_image_mean])

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        high_res_size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        high_res_resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        high_res_image_mean: Optional[Union[float, List[float]]] = None,
        high_res_image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            high_res_size (`Dict[str, int]`, *optional*, defaults to `self.high_res_size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the high resolution output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            high_res_resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BICUBIC`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            high_res_image_mean (`float` or `List[float]`, *optional*, defaults to `self.high_res_image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            high_res_image_std (`float` or `List[float]`, *optional*, defaults to `self.high_res_image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        high_res_resample = high_res_resample if high_res_resample is not None else self.high_res_resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        high_res_image_mean = high_res_image_mean if high_res_image_mean is not None else self.high_res_image_mean
        high_res_image_std = high_res_image_std if high_res_image_std is not None else self.high_res_image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)
        high_res_size = high_res_size if high_res_size is not None else self.high_res_size
        high_res_size_dict = get_size_dict(high_res_size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_images = []
        all_high_res_images = []
        for image in images:
            # high_res_image: resize (high) -> rescale -> normalize (high)
            # low_res_image:  resize (high) -> rescale -> resize (low) -> normalize (low)
            high_res_image = image

            if do_resize:
                high_res_image = self.resize(
                    image=high_res_image,
                    size=high_res_size_dict,
                    resample=high_res_resample,
                    input_data_format=input_data_format,
                )
                image = self.resize(
                    image=high_res_image, size=size_dict, resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                high_res_image = self.rescale(
                    image=high_res_image, scale=rescale_factor, input_data_format=input_data_format
                )

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )
                high_res_image = self.normalize(
                    image=high_res_image,
                    mean=high_res_image_mean,
                    std=high_res_image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            high_res_image = to_channel_dimension_format(
                high_res_image, data_format, input_channel_dim=input_data_format
            )

            all_images.append(image)
            all_high_res_images.append(high_res_image)

        data = {"pixel_values": all_images, "high_res_pixel_values": all_high_res_images}
        return BatchFeature(data=data, tensor_type=return_tensors)


class DeepseekVLHybridProcessorKwargs(DeepseekVLProcessorKwargs):
    pass


class DeepseekVLHybridProcessor(DeepseekVLProcessor):
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        **kwargs: Unpack[DeepseekVLHybridProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        DeepseekVLHybridImageProcessor's [`~DeepseekVLHybridImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            DeepseekVLHybridProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = []
        one_img_tokens = self.image_token * self.num_image_tokens
        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # process images if pixel_values are provided
        if images is not None:
            images = make_flat_list_of_images(images)
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            data["pixel_values"] = inputs["pixel_values"]
            data["high_res_pixel_values"] = inputs["high_res_pixel_values"]

        return BatchFeature(data=data)


__all__ = [
    "DeepseekVLHybridConfig",
    "DeepseekVLHybridPreTrainedModel",
    "DeepseekVLHybridModel",
    "DeepseekVLHybridForConditionalGeneration",
    "DeepseekVLHybridImageProcessor",
    "DeepseekVLHybridProcessor",
]
