import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoder, SiglipEncoderLayer

from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging
from ..auto import AutoModelForCausalLM
from .configuration_ovis2 import Ovis2Config, Ovis2VisionConfig


logger = logging.get_logger(__name__)


def hard_softmax(logits: torch.Tensor, dim: int):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
        torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
    )
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Ovis2RMSNorm(LlamaRMSNorm):
    pass


class Ovis2VisionMLP(LlamaMLP):
    pass


class Ovis2VisionEmbeddings(nn.Module):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="xy")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_h = grid_h.flatten()[..., None] @ omega[None, :]
        out_w = grid_w.flatten()[..., None] @ omega[None, :]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, _, height, width = pixel_values.size()
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = self.rms_norm(hidden_states)

        if self.config.image_size != height or self.config.image_size != width:
            pos_embed = self.build_2d_sincos_position_embedding(
                height // self.patch_size, width // self.patch_size, embed_dim=self.config.hidden_size
            )
        else:
            pos_embed = self.position_embeddings(self.position_ids)

        hidden_states = hidden_states + pos_embed
        return hidden_states


class Ovis2VisionAttention(SiglipAttention):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)


class Ovis2EncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__(config)
        self.layer_norm1 = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Ovis2VisionMLP(config)
        self.layer_norm2 = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Ovis2VisionAttention(config)


class Ovis2Encoder(SiglipEncoder):
    pass


class Ovis2VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Ovis2VisionEmbeddings(config)
        self.encoder = Ovis2Encoder(config)
        self.rms_norm = Ovis2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.rms_norm(last_hidden_state)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Ovis2VisualEmbeddingTable(nn.Embedding):
    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0.0, std=1.0) -> None:
        nn.init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


# class Ovis2VisionPreTrainedModel(PreTrainedModel):
#     config_class = Ovis2VisionConfig
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["Ovis2EncoderLayer", "Ovis2VisionEmbeddings"]
#     _supports_sdpa = True
#     _supports_flash_attn_2 = True


class Ovis2VisionModel(nn.Module):
    def __init__(self, config: Ovis2VisionConfig):
        super().__init__()
        self.config = config
        self.transformer = Ovis2VisionTransformer(config)
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        self.vocab_size = config.vocab_size
        self.head_linear = nn.Linear(
            config.hidden_size * config.hidden_stride * config.hidden_stride,
            self.vocab_size - self.num_visual_indicator_tokens,
            bias=False,
        )
        self.head_norm = nn.LayerNorm(self.vocab_size - self.num_visual_indicator_tokens)

    def get_prob_token(self, logits):
        if self.config.tokenize_function == "gumbel_argmax":
            prob_token = gumbel_softmax(logits, dim=-1, hard=True)
        else:
            if self.config.tokenize_function == "st_argmax":
                prob_token = hard_softmax(logits, dim=-1)
            else:
                prob_token = nn.functional.softmax(logits, dim=-1)
        return prob_token

    def forward(self, pixel_values: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(pixel_values)
        last_hidden_state = outputs.last_hidden_state

        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = last_hidden_state[:, 1:, :]
        elif self.config.vision_feature_select_strategy == "full":
            selected_image_feature = last_hidden_state

        if self.config.hidden_stride > 1:
            n, seq_len, d = selected_image_feature.shape
            hs = self.config.hidden_stride

            sqrt_l = int(math.sqrt(seq_len))
            assert sqrt_l * sqrt_l == seq_len, "Token sequence length must be a perfect square"
            pad_size = (hs - (sqrt_l % hs)) % hs
            selected_image_feature = nn.functional.pad(
                selected_image_feature, (0, 0, 0, pad_size, 0, pad_size), "constant", 0
            )
            sqrt_l += pad_size

            selected_image_feature = selected_image_feature.reshape(n, sqrt_l // hs, hs, sqrt_l // hs, hs, d)
            selected_image_feature = selected_image_feature.permute(0, 1, 3, 2, 4, 5)
            selected_image_feature = selected_image_feature.reshape(n, -1, hs * hs * d)  # (n, (sqrt_l//hs)^2, hs^2*d)

        logits = self.head_linear(selected_image_feature)
        logits = self.head_norm(logits)
        prob_token = self.get_prob_token(logits)

        return prob_token


class Ovis2CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


class Ovis2PreTrainedModel(PreTrainedModel):
    config_class = Ovis2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ovis2VisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (Ovis2RMSNorm, nn.LayerNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Ovis2VisualEmbeddingTable):
            module.reset_parameters()


OVIS2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Ovis2ImageProcessor.__call__`] for details. [`Ovis2Processor`] uses
            [`Ovis2ImageProcessor`] for processing images.
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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class Ovis2ForConditionalGeneration(Ovis2PreTrainedModel, GenerationMixin):
    def __init__(self, config: Ovis2Config):
        super().__init__(config)
        self.vision_tower = Ovis2VisionModel(config.vision_config)
        self.visual_table = Ovis2VisualEmbeddingTable(config.vision_config.vocab_size, config.hidden_size)

        self.visual_vocab_size = config.vision_config.vocab_size
        self.vocab_size = config.vocab_size

        self.visual_indicator_token_ids = config.visual_indicator_token_ids

        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

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
    ) -> torch.FloatTensor:
        image_features = self.vision_tower(pixel_values)
        b, l, _ = image_features.shape
        padding_tensor = torch.zeros(
            (b, l, self.vision_tower.num_visual_indicator_tokens),
            dtype=image_features.dtype,
            device=image_features.device,
            requires_grad=False,
            layout=image_features.layout,
        )
        image_features = torch.cat([image_features, padding_tensor], dim=2)
        image_features = self.visual_table(image_features)

        visual_indicator = torch.arange(
            self.visual_vocab_size - self.vision_tower.num_visual_indicator_tokens,
            self.visual_vocab_size,
            dtype=torch.long,
        ).to(image_features.device)
        visual_indicator_features = self.visual_table(visual_indicator)

        return image_features, visual_indicator_features

    @add_start_docstrings_to_model_forward(OVIS2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        grids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Ovis2CausalLMOutputWithPast]:
        r"""
        Args:
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
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "", torch_dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        ...             },
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
        ...             },
        ...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        The images depict the Statue of Liberty and the Golden Gate Bridge.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features, visual_indicator_features = self.get_image_features(pixel_values=pixel_values)

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_features = image_features.reshape(-1, image_features.shape[-1])
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            for i, visual_indicator_id in enumerate(self.visual_indicator_token_ids):
                mask = (input_ids == visual_indicator_id).to(inputs_embeds.device)
                inputs_embeds[mask] = (
                    visual_indicator_features[i]
                    .expand_as(inputs_embeds[mask])
                    .to(inputs_embeds.device, inputs_embeds.dtype)
                )

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
            **kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Ovis2CausalLMOutputWithPast(
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
        grids=None,
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
            model_inputs["grids"] = grids

        return model_inputs


__all__ = ["Ovis2PreTrainedModel", "Ovis2ForConditionalGeneration"]
