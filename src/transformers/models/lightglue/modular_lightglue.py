from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..auto import CONFIG_MAPPING
from ..auto.modeling_auto import AutoModelForKeypointDetection
from ..llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward
from .configuration_lightglue import LightGlueConfig


if TYPE_CHECKING:
    from ..superpoint import SuperPointConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "LightGlueConfig"
_CHECKPOINT_FOR_DOC_ = "stevenbucaille/lightglue_superpoint"


class LightGlueConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightGlueForKeypointMatching`]. It is used to
    instantiate a LightGlue model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LightGlue
    [stevenbucaille/lightglue_superpoint](https://huggingface.co/stevenbucaille/lightglue_superpoint) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        keypoint_detector_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SuperPointConfig`):
            The config object or dictionary of the keypoint detector.
        descriptor_dim (`int`, *optional*, defaults to 256):
            The dimension of the descriptors.
        num_hidden_layers (`int`, *optional*, defaults to 9):
            The number of self and cross attention layers.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of heads in the multi-head attention.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        depth_confidence (`float`, *optional*, defaults to 0.95):
            The confidence threshold used to perform early stopping
        width_confidence (`float`, *optional*, defaults to 0.99):
            The confidence threshold used to prune points
        filter_threshold (`float`, *optional*, defaults to 0.1):
            The confidence threshold used to filter matches
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function to be used in the hidden layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.

    Examples:
        ```python
        >>> from transformers import LightGlueConfig, LightGlueForKeypointMatching

        >>> # Initializing a LightGlue style configuration
        >>> configuration = LightGlueConfig()

        >>> # Initializing a model from the LightGlue style configuration
        >>> model = LightGlueForKeypointMatching(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "lightglue"

    def __init__(
        self,
        keypoint_detector_config: "SuperPointConfig" = None,
        descriptor_dim: int = 256,
        num_hidden_layers: int = 9,
        num_attention_heads: int = 4,
        num_key_value_heads=None,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        initializer_range: float = 0.02,
        hidden_act: str = "gelu",
        attention_dropout=0.0,
        attention_bias=True,
        **kwargs,
    ):
        if descriptor_dim % num_attention_heads != 0:
            raise ValueError("descriptor_dim % num_heads is different from zero")

        self.descriptor_dim = descriptor_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.filter_threshold = filter_threshold
        self.initializer_range = initializer_range

        if isinstance(keypoint_detector_config, dict):
            keypoint_detector_config["model_type"] = (
                keypoint_detector_config["model_type"] if "model_type" in keypoint_detector_config else "superpoint"
            )
            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config
            )
        if keypoint_detector_config is None:
            keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        self.keypoint_detector_config = keypoint_detector_config

        self.hidden_size = descriptor_dim
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        super().__init__(**kwargs)


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Normalize keypoints locations based on image image_shape

    Args:
        keypoints (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`):
            Keypoints locations in (x, y) format.
        height (`int`):
            Image height.
        width (`int`):
            Image width.

    Returns:
        Normalized keypoints locations of shape (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`).
    """
    size = torch.tensor([width, height], device=keypoints.device, dtype=keypoints.dtype)[None]
    shift = size / 2
    scale = size.max(-1).values / 2
    keypoints = (keypoints - shift[..., None, :]) / scale[..., None, None]
    return keypoints


def get_matches_from_scores(scores: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """obtain matches from a score matrix [Bx M+1 x N+1]"""
    batch_size, _, _ = scores.shape
    # For each keypoint, get the best match
    max0 = scores[:, :-1, :-1].max(2)
    max1 = scores[:, :-1, :-1].max(1)
    matches0 = max0.indices
    matches1 = max1.indices

    # Mutual check for matches
    indices0 = torch.arange(matches0.shape[1], device=matches0.device)[None]
    indices1 = torch.arange(matches1.shape[1], device=matches1.device)[None]
    mutual0 = indices0 == matches1.gather(1, matches0)
    mutual1 = indices1 == matches0.gather(1, matches1)

    # Get matching scores and filter based on mutual check and thresholding
    max0 = max0.values.exp()
    zero = max0.new_tensor(0)
    matching_scores0 = torch.where(mutual0, max0, zero)
    matching_scores1 = torch.where(mutual1, matching_scores0.gather(1, matches1), zero)
    valid0 = mutual0 & (matching_scores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, matches1)

    # Filter matches based on mutual check and thresholding of scores
    matches0 = torch.where(valid0, matches0, -1)
    matches1 = torch.where(valid1, matches1, -1)
    matches = torch.stack([matches0, matches1]).transpose(0, 1).reshape(batch_size * 2, -1)
    matching_scores = torch.stack([matching_scores0, matching_scores1]).transpose(0, 1).reshape(batch_size * 2, -1)

    return matches, matching_scores


@dataclass
class LightGlueKeypointMatchingOutput(ModelOutput):
    """
    Base class for outputs of LightGlue keypoint matching models. Due to the nature of keypoint detection and matching,
    the number of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the
    batch of images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask
    tensor is used to indicate which values in the keypoints, matches, matching_scores and prune tensors are keypoint
    matching information.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Absolute (x, y) coordinates of predicted keypoints in a given image.
        prune (`torch.IntTensor` of shape `(batch_size, num_keypoints)`):
            Pruning mask indicating which keypoints are removed and at which layer.
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches, matching_scores, keypoints and prune are keypoint matching
            information.
        hidden_states (`Tuple[torch.FloatTensor, ...]`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)` returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`
        attentions (`Tuple[torch.FloatTensor, ...]`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)` returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`
    """

    loss: Optional[torch.FloatTensor] = None
    matches: Optional[torch.FloatTensor] = None
    matching_scores: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.FloatTensor] = None
    prune: Optional[torch.IntTensor] = None
    mask: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LightGluePositionalEncoder(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()
        self.projector = nn.Linear(2, config.descriptor_dim // config.num_attention_heads // 2, bias=False)

    def forward(
        self, keypoints: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        projected_keypoints = self.projector(keypoints)
        embeddings = projected_keypoints.repeat_interleave(2, dim=-1)
        cosines = torch.cos(embeddings)
        sines = torch.sin(embeddings)
        embeddings = (cosines, sines)
        output = (embeddings, projected_keypoints) if output_hidden_states else (embeddings,)
        return output


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class LightGlueAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        current_attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        key_states = self.k_proj(current_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(current_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            current_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LightGlueMLP(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.layer_norm = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
        self.activation = ACT2FN[config.hidden_act]
        self.output = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states


class LightGlueAttentionBlock(nn.Module):
    def __init__(self, config: LightGlueConfig, layer_idx: int):
        super().__init__()
        self.attention = LightGlueAttention(config, layer_idx)
        self.mlp = LightGlueMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        attention_outputs = self.attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        message = attention_outputs[0]
        intermediate_states = torch.cat([hidden_states, message], dim=-1)
        output_states = self.mlp(intermediate_states)
        hidden_states = hidden_states + output_states

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (intermediate_states, output_states)
        attention_probs = attention_outputs[1:] if output_attentions else None

        outputs = (hidden_states, all_hidden_states, attention_probs)
        return outputs


class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig, layer_idx: int):
        super().__init__()
        self.self_attention_block = LightGlueAttentionBlock(config, layer_idx)
        self.cross_attention_block = LightGlueAttentionBlock(config, layer_idx)

    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (descriptors,)

        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        self_attention_output = self.self_attention_block(
            descriptors,
            position_embeddings=keypoints,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = self_attention_output[0]

        # Reshape hidden_states to group by image_pairs :
        #   (batch_size, num_keypoints, descriptor_dim) -> (batch_size, 2, num_keypoints, descriptor_dim)
        # Flip dimension 1 to perform cross attention :
        #   (image0, image1) -> (image1, image0)
        # Reshape back to original shape :
        #   (batch_size, 2, num_keypoints, descriptor_dim) -> (batch_size, num_keypoints, descriptor_dim)
        encoder_hidden_states = (
            descriptors.reshape(-1, 2, num_keypoints, descriptor_dim)
            .flip(1)
            .reshape(batch_size, num_keypoints, descriptor_dim)
        )
        # Same for mask
        encoder_attention_mask = (
            attention_mask.reshape(-1, 2, 1, 1, num_keypoints).flip(1).reshape(batch_size, 1, 1, num_keypoints)
            if attention_mask is not None
            else None
        )

        cross_attention_output = self.cross_attention_block(
            descriptors,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = cross_attention_output[0]
        if output_hidden_states:
            all_hidden_states = (
                all_hidden_states
                + (self_attention_output[0].reshape(batch_size, num_keypoints, descriptor_dim),)
                + self_attention_output[1]
                + (cross_attention_output[0].reshape(batch_size, num_keypoints, descriptor_dim),)
                + cross_attention_output[1]
            )

        if output_attentions:
            all_attentions = all_attentions + self_attention_output[2] + cross_attention_output[2]

        return descriptors, all_hidden_states, all_attentions


def sigmoid_log_double_softmax(
    similarity: torch.Tensor, matchability0: torch.Tensor, matchability1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    batch_size, num_keypoints_0, num_keypoints_1 = similarity.shape
    certainties = nn.functional.logsigmoid(matchability0) + nn.functional.logsigmoid(matchability1).transpose(1, 2)
    scores0 = nn.functional.log_softmax(similarity, 2)
    scores1 = nn.functional.log_softmax(similarity.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = similarity.new_full((batch_size, num_keypoints_0 + 1, num_keypoints_1 + 1), 0)
    scores[:, :num_keypoints_0, :num_keypoints_1] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = nn.functional.logsigmoid(-matchability0.squeeze(-1))
    scores[:, -1, :-1] = nn.functional.logsigmoid(-matchability1.squeeze(-1))
    return scores


class LightGlueMatchAssignmentLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.descriptor_dim = config.descriptor_dim
        self.final_projection = nn.Linear(self.descriptor_dim, self.descriptor_dim, bias=True)
        self.matchability = nn.Linear(self.descriptor_dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        # Final projection and similarity computation
        m_descriptors = self.final_projection(descriptors)
        m_descriptors = m_descriptors / torch.tensor(self.descriptor_dim) ** 0.25
        m_descriptors = m_descriptors.reshape(batch_size // 2, 2, num_keypoints, descriptor_dim)
        m_descriptors0 = m_descriptors[:, 0]
        m_descriptors1 = m_descriptors[:, 1]
        similarity = m_descriptors0 @ m_descriptors1.transpose(-1, -2)
        if mask is not None:
            mask = mask.reshape(batch_size // 2, 2, num_keypoints)
            mask0 = mask[:, 0].unsqueeze(-1)
            mask1 = mask[:, 1].unsqueeze(-1).transpose(-1, -2)
            mask = mask0 * mask1
            similarity = similarity.masked_fill(mask == 0, torch.finfo(similarity.dtype).min)

        # Compute matchability of descriptors
        matchability = self.matchability(descriptors)
        matchability = matchability.reshape(batch_size // 2, 2, num_keypoints, 1)
        matchability_0 = matchability[:, 0]
        matchability_1 = matchability[:, 1]

        # Compute scores from similarity and matchability
        scores = sigmoid_log_double_softmax(similarity, matchability_0, matchability_1)
        return scores

    def get_matchability(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Get matchability of descriptors as a probability"""
        matchability = self.matchability(descriptors)
        matchability = nn.functional.sigmoid(matchability).squeeze(-1)
        return matchability


class LightGlueTokenConfidenceLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.token = nn.Linear(config.descriptor_dim, 1)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        token = self.token(descriptors.detach())
        token = nn.functional.sigmoid(token).squeeze(-1)
        return token


class LightGluePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LightGlueConfig
    base_model_prefix = "lightglue"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LIGHTGLUE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LightGlueConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

LIGHTGLUE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`LightPointImageProcessor`]. See
            [`LightPointImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "LightGlue model taking images as inputs and outputting the matching of them.",
    LIGHTGLUE_START_DOCSTRING,
)
class LightGlueForKeypointMatching(LightGluePreTrainedModel):
    """
    LightGlue is a model matching keypoints in images by leveraging detections from a keypoint detector such as
    SuperPoint. It is based on the SuperGlue architecture and is designed to be lightweight and efficient.
    It consists of :
        1. Keypoint Encoder
        2. A Graph Neural Network with self and cross attention layers
        3. Matching Assignment layers

    The correspondence ids use -1 to indicate non-matching points.

    Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys. LightGlue: Local Feature Matching at Light Speed.
    In ICCV 2023. https://arxiv.org/pdf/2306.13643.pdf
    """

    def __init__(self, config: LightGlueConfig):
        super().__init__(config)

        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)

        self.descriptor_dim = config.descriptor_dim
        self.num_layers = config.num_hidden_layers
        self.filter_threshold = config.filter_threshold
        self.depth_confidence = config.depth_confidence
        self.width_confidence = config.width_confidence

        if self.descriptor_dim != config.keypoint_detector_config.descriptor_decoder_dim:
            self.input_projection = nn.Linear(
                config.keypoint_detector_config.descriptor_decoder_dim, self.descriptor_dim, bias=True
            )
        else:
            self.input_projection = nn.Identity()

        self.positional_encoder = LightGluePositionalEncoder(config)

        self.transformer_layers = nn.ModuleList(
            [LightGlueTransformerLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.match_assignment_layers = nn.ModuleList(
            [LightGlueMatchAssignmentLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.token_confidence = nn.ModuleList(
            [LightGlueTokenConfidenceLayer(config) for _ in range(config.num_hidden_layers - 1)]
        )

        self.post_init()

    def _get_confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold for a given layer"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.num_layers)
        return np.clip(threshold, 0, 1)

    def _keypoint_processing(
        self, descriptors: torch.Tensor, keypoints: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        descriptors = descriptors.detach().contiguous()
        projected_descriptors = self.input_projection(descriptors)
        keypoint_encoding_output = self.positional_encoder(keypoints, output_hidden_states=output_hidden_states)
        return projected_descriptors, keypoint_encoding_output

    def _get_early_stopped_image_pairs(
        self, keypoint_confidences: torch.Tensor, layer_index: int, mask: torch.Tensor, num_points: torch.Tensor
    ) -> torch.Tensor:
        """evaluate whether we should stop inference based on the confidence of the keypoints"""
        batch_size, _ = mask.shape
        if layer_index < self.num_layers - 1:
            # If the current layer is not the last layer, we compute the confidence of the keypoints and check
            # if we should stop the forward pass through the transformer layers for each pair of images.
            keypoint_confidences = keypoint_confidences.masked_fill(mask == 0, 1)
            keypoint_confidences = keypoint_confidences.reshape(batch_size // 2, -1)
            threshold = self._get_confidence_threshold(layer_index)
            ratio_confident = 1.0 - (keypoint_confidences < threshold).float().sum(dim=1) / num_points
            early_stopped_pairs = ratio_confident > self.depth_confidence
        else:
            # If the current layer is the last layer, we stop the forward pass through the transformer layers for
            # all pairs of images.
            early_stopped_pairs = torch.ones(batch_size, dtype=torch.bool)
        return early_stopped_pairs

    def _get_keypoint_matching(self, descriptors, mask, layer_index, early_stops=None):
        if early_stops is not None:
            descriptors = descriptors[early_stops]
            mask = mask[early_stops]
        scores = self.match_assignment_layers[layer_index](descriptors, mask)
        matches, matching_scores = get_matches_from_scores(scores, self.filter_threshold)
        return matches, matching_scores

    def _get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self._get_confidence_threshold(layer_index)
        return keep

    def _do_layer_keypoint_pruning(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        indices: torch.Tensor,
        prune_output: torch.Tensor,
        keypoint_confidences: torch.Tensor,
        layer_index: int,
    ):
        """
        For a given layer, prune keypoints based on the confidence of the keypoints and the matchability of the
        descriptors.
        """
        batch_size, _, _ = descriptors.shape
        descriptors_matchability = self.match_assignment_layers[layer_index].get_matchability(descriptors)
        pruned_keypoints_mask = self._get_pruning_mask(keypoint_confidences, descriptors_matchability, layer_index)
        pruned_keypoints_mask = pruned_keypoints_mask.masked_fill(mask == 0, torch.tensor(False))

        # For each image, we extract the pruned indices and the corresponding descriptors and keypoints.
        pruned_descriptors, pruned_keypoints_0, pruned_keypoints_1, pruned_mask, pruned_indices = (
            [t[mask] for t, mask in zip(tensor, pruned_keypoints_mask)]
            for tensor in [descriptors, keypoints[0], keypoints[1], pruned_keypoints_mask, indices]
        )
        for i in range(batch_size):
            prune_output[i, pruned_indices[i]] += 1

        # Pad the pruned descriptors, keypoints, indices and mask to have the same shape across the batch.
        pruned_descriptors, pruned_keypoints_0, pruned_keypoints_1, pruned_mask = (
            pad_sequence(pruned_tensor, batch_first=True)
            for pruned_tensor in [pruned_descriptors, pruned_keypoints_0, pruned_keypoints_1, pruned_mask]
        )
        pruned_keypoints = (pruned_keypoints_0, pruned_keypoints_1)
        pruned_indices = pad_sequence(pruned_indices, batch_first=True, padding_value=-1)

        return pruned_descriptors, pruned_keypoints, pruned_indices, pruned_mask, prune_output

    def _concat_early_stopped_outputs(
        self,
        early_stops_indices,
        final_pruned_keypoints_indices,
        final_pruned_keypoints_iterations,
        matches,
        matching_scores,
    ):
        early_stops_indices = torch.stack(early_stops_indices)
        matches, final_pruned_keypoints_indices = (
            pad_sequence(tensor, batch_first=True, padding_value=-1)
            for tensor in [matches, final_pruned_keypoints_indices]
        )
        matching_scores, final_pruned_keypoints_iterations = (
            pad_sequence(tensor, batch_first=True, padding_value=0)
            for tensor in [matching_scores, final_pruned_keypoints_iterations]
        )
        matches, matching_scores, final_pruned_keypoints_indices, final_pruned_keypoints_iterations = (
            tensor[early_stops_indices]
            for tensor in [
                matches,
                matching_scores,
                final_pruned_keypoints_indices,
                final_pruned_keypoints_iterations,
            ]
        )
        return final_pruned_keypoints_indices, final_pruned_keypoints_iterations, matches, matching_scores

    def _do_final_keypoint_pruning(
        self,
        indices: torch.Tensor,
        matches: torch.Tensor,
        matching_scores: torch.Tensor,
        num_keypoints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, num_keypoints) -> (batch_size // 2, 2, num_keypoints) -> 2 * (batch_size // 2, num_keypoints) to
        # have tensors from
        batch_size, _ = indices.shape
        indices, matches, matching_scores = (
            tensor.reshape(batch_size // 2, 2, -1) for tensor in [indices, matches, matching_scores]
        )
        indices0 = indices[:, 0]
        indices1 = indices[:, 1]
        matches0 = matches[:, 0]
        matches1 = matches[:, 1]
        matching_scores0 = matching_scores[:, 0]
        matching_scores1 = matching_scores[:, 1]

        # Prepare final matches and matching scores
        _matches = torch.full((batch_size // 2, 2, num_keypoints), -1, device=indices.device, dtype=matches.dtype)
        _matching_scores = torch.zeros(
            (batch_size // 2, 2, num_keypoints), device=indices.device, dtype=matching_scores.dtype
        )
        # Fill the matches and matching scores for each image pair
        for i in range(batch_size // 2):
            _matches[i, 0, indices0[i]] = torch.where(
                matches0[i] == -1, -1, indices1[i].gather(0, matches0[i].clamp(min=0))
            )
            _matches[i, 1, indices1[i]] = torch.where(
                matches1[i] == -1, -1, indices0[i].gather(0, matches1[i].clamp(min=0))
            )
            _matching_scores[i, 0, indices0[i]] = matching_scores0[i]
            _matching_scores[i, 1, indices1[i]] = matching_scores1[i]
        return _matches, _matching_scores

    def _match_image_pair(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        height: int,
        width: int,
        mask: torch.Tensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if keypoints.shape[2] == 0:  # no keypoints
            shape = keypoints.shape[:-1]
            return (
                keypoints.new_full(shape, -1, dtype=torch.int),
                keypoints.new_zeros(shape),
                keypoints.new_zeros(shape),
                all_hidden_states,
                all_attentions,
            )

        device = keypoints.device
        batch_size, _, initial_num_keypoints, _ = keypoints.shape
        num_points_per_pair = torch.sum(mask.reshape(batch_size, -1), dim=1)
        # (batch_size, 2, num_keypoints, 2) -> (batch_size * 2, num_keypoints, 2)
        keypoints = keypoints.reshape(batch_size * 2, initial_num_keypoints, 2)
        mask = mask.reshape(batch_size * 2, initial_num_keypoints) if mask is not None else None
        descriptors = descriptors.reshape(batch_size * 2, initial_num_keypoints, self.descriptor_dim)
        image_indices = torch.arange(batch_size * 2, device=device)
        # Keypoint normalization
        keypoints = normalize_keypoints(keypoints, height, width)

        descriptors, keypoint_encoding_output = self._keypoint_processing(
            descriptors, keypoints, output_hidden_states=output_hidden_states
        )

        keypoints = keypoint_encoding_output[0]

        # Early stop consists of stopping the forward pass through the transformer layers when the confidence of the
        # keypoints is above a certain threshold.
        do_early_stop = self.depth_confidence > 0
        # Keypoint pruning consists of removing keypoints from the input of the transformer layers when the confidence of
        # the keypoints is below a certain threshold.
        do_keypoint_pruning = self.width_confidence > 0

        early_stops_indices = []
        matches = []
        matching_scores = []
        final_pruned_keypoints_indices = []
        final_pruned_keypoints_iterations = []

        pruned_keypoints_indices = torch.arange(0, initial_num_keypoints, device=device).expand(batch_size * 2, -1)
        pruned_keypoints_iterations = torch.ones_like(pruned_keypoints_indices)

        for layer_index in range(self.num_layers):
            input_shape = descriptors.size()
            if mask is not None:
                extended_attention_mask = self.get_extended_attention_mask(mask, input_shape)
            else:
                extended_attention_mask = torch.ones((batch_size, input_shape[-2]), device=keypoints.device)
            layer_output = self.transformer_layers[layer_index](
                descriptors,
                keypoints,
                attention_mask=extended_attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            descriptors, hidden_states, attention = layer_output
            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states
            if output_attentions:
                all_attentions = all_attentions + attention

            if do_early_stop:
                if layer_index < self.num_layers - 1:
                    # Get the confidence of the keypoints for the current layer
                    keypoint_confidences = self.token_confidence[layer_index](descriptors)

                    # Determine which pairs of images should be early stopped based on the confidence of the keypoints for
                    # the current layer.
                    early_stopped_pairs = self._get_early_stopped_image_pairs(
                        keypoint_confidences, layer_index, mask, num_points=num_points_per_pair
                    )
                else:
                    # Early stopping always occurs at the last layer
                    early_stopped_pairs = torch.ones(batch_size, dtype=torch.bool)

                if torch.any(early_stopped_pairs):
                    # If a pair of images is considered early stopped, we compute the matches for the remaining
                    # keypoints and stop the forward pass through the transformer layers for this pair of images.
                    early_stops = early_stopped_pairs.repeat_interleave(2)
                    early_stopped_image_indices = image_indices[early_stops]
                    early_stopped_matches, early_stopped_matching_scores = self._get_keypoint_matching(
                        descriptors, mask, layer_index, early_stops=early_stops
                    )
                    early_stops_indices.extend(list(early_stopped_image_indices))
                    matches.extend(list(early_stopped_matches))
                    matching_scores.extend(list(early_stopped_matching_scores))
                    if do_keypoint_pruning:
                        final_pruned_keypoints_indices.extend(list(pruned_keypoints_indices[early_stops]))
                        final_pruned_keypoints_iterations.extend(list(pruned_keypoints_iterations[early_stops]))

                    # Remove image pairs that have been early stopped from the forward pass
                    num_points_per_pair = num_points_per_pair[~early_stopped_pairs]
                    descriptors, keypoints_0, keypoint_1, mask, image_indices = tuple(
                        (
                            tensor[~early_stops]
                            for tensor in [descriptors, keypoints[0], keypoints[1], mask, image_indices]
                        )
                    )
                    keypoints = (keypoints_0, keypoint_1)
                    if do_keypoint_pruning:
                        pruned_keypoints_indices, pruned_keypoints_iterations, keypoint_confidences = tuple(
                            (
                                tensor[~early_stops]
                                for tensor in [
                                    pruned_keypoints_indices,
                                    pruned_keypoints_iterations,
                                    keypoint_confidences,
                                ]
                            )
                        )
                # If all pairs of images are early stopped, we stop the forward pass through the transformer
                # layers for all pairs of images.
                if torch.all(early_stopped_pairs):
                    break

            if do_keypoint_pruning:
                # Prune keypoints from the input of the transformer layers for the next iterations if the confidence of
                # the keypoints is below a certain threshold.
                descriptors, keypoints, pruned_keypoints_indices, mask, pruned_keypoints_iterations = (
                    self._do_layer_keypoint_pruning(
                        descriptors,
                        keypoints,
                        mask,
                        pruned_keypoints_indices,
                        pruned_keypoints_iterations,
                        keypoint_confidences,
                        layer_index,
                    )
                )

        if do_early_stop and do_keypoint_pruning:
            # Concatenate early stopped outputs together and perform final keypoint pruning
            final_pruned_keypoints_indices, final_pruned_keypoints_iterations, matches, matching_scores = (
                self._concat_early_stopped_outputs(
                    early_stops_indices,
                    final_pruned_keypoints_indices,
                    final_pruned_keypoints_iterations,
                    matches,
                    matching_scores,
                )
            )
            matches, matching_scores = self._do_final_keypoint_pruning(
                final_pruned_keypoints_indices,
                matches,
                matching_scores,
                initial_num_keypoints,
            )
        else:
            matches, matching_scores = self._get_keypoint_matching(descriptors, mask, self.num_layers - 1)
            final_pruned_keypoints_iterations = torch.ones_like(matching_scores) * self.num_layers

        final_pruned_keypoints_iterations = final_pruned_keypoints_iterations.reshape(
            batch_size, 2, initial_num_keypoints
        )

        return (
            matches,
            matching_scores,
            final_pruned_keypoints_iterations,
            all_hidden_states,
            all_attentions,
        )

    @add_start_docstrings_to_model_forward(LIGHTGLUE_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LightGlueKeypointMatchingOutput]:
        loss = None
        if labels is not None:
            raise ValueError("LightGlue is not trainable, no labels should be provided.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        batch_size, _, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * 2, channels, height, width)
        keypoint_detections = self.keypoint_detector(pixel_values)

        keypoints, _, descriptors, mask = keypoint_detections[:4]
        keypoints = keypoints.reshape(batch_size, 2, -1, 2).to(pixel_values)
        descriptors = descriptors.reshape(batch_size, 2, -1, self.descriptor_dim).to(pixel_values)
        mask = mask.reshape(batch_size, 2, -1)

        absolute_keypoints = keypoints.clone()
        absolute_keypoints[:, :, :, 0] = absolute_keypoints[:, :, :, 0] * width
        absolute_keypoints[:, :, :, 1] = absolute_keypoints[:, :, :, 1] * height

        matches, matching_scores, prune, hidden_states, attentions = self._match_image_pair(
            absolute_keypoints,
            descriptors,
            height,
            width,
            mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if not return_dict:
            return tuple(
                v
                for v in [matches, matching_scores, keypoints, prune, mask, hidden_states, attentions]
                if v is not None
            )

        return LightGlueKeypointMatchingOutput(
            loss=loss,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            prune=prune,
            mask=mask,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["LightGluePreTrainedModel", "LightGlueForKeypointMatching", "LightGlueConfig"]
