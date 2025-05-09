# coding=utf-8
# Copyright 2024 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch SAM 2 model."""

import collections
import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam2 import Sam2Config, Sam2ImageEncoderConfig, Sam2MaskDecoderConfig, Sam2PromptEncoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Sam2Config"
# TODO: update checkpoint
_CHECKPOINT_FOR_DOC = "hkhedr93/sam2_hiera_base_plus"

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0
CUDA_KERNELS = None


def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    global CUDA_KERNELS

    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "sam2"
    src_files = [root / "connected_components.cu"]

    CUDA_KERNELS = load(
        "CUDA_KERNELS",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).
    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.
    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """

    return CUDA_KERNELS.get_connected_components(mask.to(torch.uint8).contiguous())


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()


@dataclass
class Sam2ImageEncoderOutput(ModelOutput):
    """
    Base class for sam2 vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    fpn_hidden_states: Optional[torch.FloatTensor] = None
    fpn_position_encoding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Sam2ImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    image_embeddings: Tuple[torch.FloatTensor, ...] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Sam2PatchEmbeddings(nn.Module):
    r"""
    Turns pixel values into patch embeddings for transformer consumption.

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Sam2ImageProcessor.__call__`] for details.

    Returns:
        embeddings (`torch.FloatTensor`):
            Patch embeddings depend on image_size, patch_kernel_size, patch_stride and patch_padding
    """

    def __init__(self, config: Sam2ImageEncoderConfig):
        super().__init__()
        image_size, patch_kernel_size, patch_stride, patch_padding = (
            config.image_size,
            config.patch_kernel_size,
            config.patch_stride,
            config.patch_padding,
        )
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_kernel_size = (
            patch_kernel_size
            if isinstance(patch_kernel_size, collections.abc.Iterable)
            else (patch_kernel_size, patch_kernel_size)
        )
        patch_stride = (
            patch_stride if isinstance(patch_stride, collections.abc.Iterable) else (patch_stride, patch_stride)
        )
        patch_padding = (
            patch_padding if isinstance(patch_padding, collections.abc.Iterable) else (patch_padding, patch_padding)
        )
        self.image_size = image_size
        self.num_channels = num_channels

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_kernel_size, stride=patch_stride, padding=patch_padding
        )

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class Sam2VisionNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.position_encoding = Sam2PositionEmbeddingSine(
            num_pos_feats=config.fpn_hidden_size, normalize=True, temperature=10000
        )
        self.convs = nn.ModuleList()
        for in_channels in config.backbone_channel_list:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config.fpn_hidden_size,
                    kernel_size=config.fpn_kernel_size,
                    stride=config.fpn_stride,
                    padding=config.fpn_padding,
                ),
            )

        self.fpn_interpolation_mode = config.fpn_interpolation_mode
        self.fuse_type = config.fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if config.fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            config.fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(config.fpn_top_down_levels)

    def forward(self, hidden_states):
        fpn_hidden_states = ()
        fpn_position_encoding = ()

        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            lateral_features = hidden_states[i].permute(0, 3, 1, 2)
            lateral_features = self.convs[n - i](lateral_features)
            if i not in self.fpn_top_down_levels or i == n:
                prev_features = lateral_features
            else:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interpolation_mode,
                    align_corners=(None if self.fpn_interpolation_mode == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "average":
                    prev_features /= 2

            prev_position_encoding = self.position_encoding(prev_features).to(prev_features.dtype)

            fpn_hidden_states += (prev_features,)
            fpn_position_encoding += (prev_position_encoding,)

        return fpn_hidden_states, fpn_position_encoding


class Sam2ImageEncoder(nn.Module):
    def __init__(self, config: Sam2ImageEncoderConfig):
        super().__init__()
        self.config = config

        # Patch embdding
        self.patch_embed = Sam2PatchEmbeddings(config)
        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.hidden_size, *config.window_positional_embedding_background_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, config.hidden_size, config.window_spec[0], config.window_spec[0])
        )

        self.stage_ends = [sum(config.stages[:i]) - 1 for i in range(1, len(config.stages) + 1)]
        self.global_attention_blocks = config.global_attention_blocks

        self.blocks = nn.ModuleList()
        embed_dim = config.hidden_size
        num_heads = config.num_heads
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.stages))
        ]  # stochastic depth decay rule
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][: config.q_pool]
        cur_stage = 1
        for i in range(sum(config.stages)):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = config.window_spec[cur_stage - 1]

            if self.global_attention_blocks is not None:
                window_size = 0 if i in self.global_attention_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * config.dim_mul)
                num_heads = int(num_heads * config.head_mul)
                cur_stage += 1

            block = Sam2MultiScaleBlock(
                config=config,
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=config.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.neck = Sam2VisionNeck(config)
        self.num_feature_levels = None

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Sam2ImageEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self._get_pos_embed(hidden_states.shape[1:3])

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        intermediate_hidden_states = ()
        for i, block_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            block_outputs = block_module(hidden_states, output_attentions=output_attentions)
            hidden_states = block_outputs[0]

            if (i == self.stage_ends[-1]) or (i in self.stage_ends):
                intermediate_hidden_states = intermediate_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (block_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Forward through backbone
        fpn_hidden_states, fpn_position_encoding = self.neck(intermediate_hidden_states)
        fpn_hidden_states, fpn_position_encoding = (
            fpn_hidden_states[-self.num_feature_levels :][::-1],
            fpn_position_encoding[-self.num_feature_levels :][::-1],
        )

        if not return_dict:
            outputs = (hidden_states, fpn_hidden_states, fpn_position_encoding)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return Sam2ImageEncoderOutput(
            last_hidden_state=hidden_states,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Sam2PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.scale
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.hidden_size // 2)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]
        coordinates.to(torch.float32)

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


# Copied from transformers.models.sam.modeling_sam.SamMaskEmbedding with Sam->Sam2
class Sam2MaskEmbedding(nn.Module):
    def __init__(self, config: Sam2PromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = Sam2LayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = Sam2LayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class Sam2PromptEncoder(nn.Module):
    def __init__(self, config: Sam2PromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = Sam2MaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    # Ignore copy
    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitely
        # specificed as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 2)[:, :, :, None],
            point_embedding + self.point_embed[2].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 3)[:, :, :, None],
            point_embedding + self.point_embed[3].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class Sam2TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        config,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Sam2Attention(config.two_way_transformer_embedding_dim, config.two_way_transformer_num_heads)
        self.layer_norm1 = nn.LayerNorm(config.two_way_transformer_embedding_dim)

        self.cross_attn_token_to_image = Sam2Attention(
            config.two_way_transformer_embedding_dim,
            config.two_way_transformer_num_heads,
            downsample_rate=config.two_way_transformer_attention_downsample_rate,
        )
        self.layer_norm2 = nn.LayerNorm(config.two_way_transformer_embedding_dim)

        self.mlp = Sam2FeedForward(
            config.two_way_transformer_embedding_dim,
            config.two_way_transformer_mlp_dim,
            config.two_way_transformer_embedding_dim,
            num_layers=2,
            activation=config.two_way_transformer_activation,
        )
        self.layer_norm3 = nn.LayerNorm(config.two_way_transformer_embedding_dim)

        self.layer_norm4 = nn.LayerNorm(config.two_way_transformer_embedding_dim)
        self.cross_attn_image_to_token = Sam2Attention(
            config.two_way_transformer_embedding_dim,
            config.two_way_transformer_num_heads,
            downsample_rate=config.two_way_transformer_attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_point_embedding: Tensor, key_point_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)

        return queries, keys


class Sam2TwoWayTransformer(nn.Module):
    def __init__(
        self,
        config: Sam2MaskDecoderConfig,
    ):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()

        for i in range(config.two_way_transformer_depth):
            self.layers.append(
                Sam2TwoWayAttentionBlock(
                    config,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Sam2Attention(
            config.two_way_transformer_embedding_dim,
            config.two_way_transformer_num_heads,
            downsample_rate=config.two_way_transformer_attention_downsample_rate,
        )
        self.layer_norm_final_attn = nn.LayerNorm(config.two_way_transformer_embedding_dim)

    def forward(
        self,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        point_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        # batchxHxW -> BxHWxC == B x N_image_tokens x C
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
            )

        # Apply the final attention layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        return queries, keys


class Sam2MaskDecoder(nn.Module):
    def __init__(self, config: Sam2MaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, config.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, config.hidden_size)

        self.transformer = Sam2TwoWayTransformer(config)

        self.pred_obj_scores = config.pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, config.hidden_size)
        self.use_multimask_token_for_object_pointer = config.use_multimask_token_for_object_pointer

        self.upscale_conv1 = nn.ConvTranspose2d(config.hidden_size, config.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(
            config.hidden_size // 4, config.hidden_size // 8, kernel_size=2, stride=2
        )
        self.upscale_layer_norm = Sam2LayerNorm(config.hidden_size // 4, data_format="channels_first")
        self.activation = ACT2FN[config.hidden_act]

        self.use_high_resolution_features = config.use_high_resolution_features
        if self.use_high_resolution_features:
            self.conv_s0 = nn.Conv2d(config.hidden_size, config.hidden_size // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(config.hidden_size, config.hidden_size // 4, kernel_size=1, stride=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                Sam2FeedForward(
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_size // 8,
                    3,
                    activation=config.feed_forward_hidden_act,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = Sam2FeedForward(
            config.hidden_size,
            config.iou_head_hidden_dim,
            self.num_mask_tokens,
            config.iou_head_depth,
            activation=config.feed_forward_hidden_act,
            sigmoid_output=config.iou_prediction_use_sigmoid,
        )
        if config.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(config.hidden_size, 1)
            if config.pred_obj_scores_mlp:
                self.pred_obj_score_head = Sam2FeedForward(
                    config.hidden_size, config.hidden_size, 1, 3, activation="relu"
                )

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = config.dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = config.dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = config.dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        high_resolution_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_positional_embeddings (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer
        hs, image_embeddings = self.transformer(image_embeddings, image_positional_embeddings, tokens)
        iou_token_out = hs[:, :, s, :]
        mask_tokens_out = hs[:, :, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        if not self.use_high_resolution_features:
            upscaled_embedding = self.upscale_conv1(image_embeddings)
            upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
            upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))
        else:
            feat_s0, feat_s1 = high_resolution_features
            upscaled_embedding = self.upscale_conv1(image_embeddings) + feat_s1
            upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
            upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, :, 1:, :, :]
            iou_pred = iou_pred[:, :, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, :, 0:1, :, :]
            iou_pred = iou_pred[:, :, 0:1]

        if multimask_output and self.use_multimask_token_for_object_pointer:
            sam_tokens_out = mask_tokens_out[:, :, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, :, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out


class Sam2PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Sam2FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "gelu",
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = ACT2FN[activation]
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam2
class Sam2LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class Sam2MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(hidden_states).reshape(batch_size, height * width, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        query, key, value = torch.unbind(qkv, 2)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            query = do_pool(query.reshape(batch_size, height, width, -1), self.q_pool)
            height, width = query.shape[1:3]  # downsampled shape
            query = query.reshape(batch_size, height * width, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        attn_output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        # Transpose back
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Sam2
class Sam2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Sam2MultiScaleBlock(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        q_stride: Tuple[int, int] = None,
        window_size: int = 0,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.layer_norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = Sam2MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = Sam2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.layer_norm2 = nn.LayerNorm(dim_out, eps=config.layer_norm_eps)
        self.mlp = Sam2FeedForward(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=config.hidden_act,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states  # batch_size, height, width, channel

        hidden_states = self.layer_norm1(hidden_states)

        # Skip connection
        if self.dim != self.dim_out:
            residual = do_pool(self.proj(hidden_states), self.pool)

        # Window partition
        window_size = self.window_size
        if self.window_size > 0:
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = self.window_partition(hidden_states, window_size)

        # Window Attention + Q Pooling (if stage change)
        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = residual.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, window_size, pad_hw, (H, W))

        hidden_states = residual + self.drop_path(hidden_states)
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.drop_path(self.mlp(layernorm_output))

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class Sam2Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_heads)
        key = self._separate_heads(key, self.num_heads)
        value = self._separate_heads(value, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=USE_FLASH_ATTN,
            # if Flash attention kernel is off, then math kernel needs to be enabled
            enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
            enable_mem_efficient=OLD_GPU,
        ):
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)

        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


class Sam2RoPEAttention(Sam2Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=USE_FLASH_ATTN,
            # if Flash attention kernel is off, then math kernel needs to be enabled
            enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
            enable_mem_efficient=OLD_GPU,
        ):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class Sam2MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.dim_feedforward = config.dim_feedforward
        self.self_attn = Sam2RoPEAttention(
            rope_theta=config.rope_theta,
            feat_sizes=config.rope_feat_sizes,
            embedding_dim=config.rope_embedding_dim,
            num_heads=config.rope_num_heads,
            downsample_rate=config.rope_downsample_rate,
            dropout=config.rope_dropout,
        )
        self.cross_attn_image = Sam2RoPEAttention(
            rope_theta=config.rope_theta,
            feat_sizes=config.rope_feat_sizes,
            embedding_dim=config.rope_embedding_dim,
            num_heads=config.rope_num_heads,
            downsample_rate=config.rope_downsample_rate,
            dropout=config.rope_dropout,
            rope_k_repeat=True,
            kv_in_dim=64,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(config.hidden_size, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_size)

        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = ACT2FN[config.hidden_act]

        # Where to add pos enc
        self.apply_pe_at_self_attn = config.apply_pe_at_self_attn
        self.apply_pe_at_cross_attn_queries = config.apply_pe_at_cross_attn_queries
        self.apply_pe_at_cross_attn_keys = config.apply_pe_at_cross_attn_keys

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Optional[Tensor] = None,
        key_point_embedding: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        # Self-Attention
        query = self.layer_norm1(queries)
        if self.apply_pe_at_self_attn:
            query = self.self_attn(query + query_point_embedding, query + query_point_embedding, v=query)
        else:
            query = self.self_attn(query, query, v=query)
        queries = queries + self.dropout1(query)

        # Cross-Attention
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, Sam2RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        query = self.layer_norm2(queries)
        query = self.cross_attn_image(
            q=query + query_point_embedding if self.apply_pe_at_cross_attn_queries else query,
            k=keys + key_point_embedding if self.apply_pe_at_cross_attn_keys else keys,
            v=keys,
            **kwds,
        )
        queries = queries + self.dropout2(query)

        # MLP
        query = self.layer_norm3(queries)
        query = self.linear2(self.dropout(self.activation(self.linear1(query))))
        queries = queries + self.dropout3(query)
        return queries


class Sam2MemoryAttention(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        layer = Sam2MemoryAttentionLayer(config)
        self.layers = get_clones(layer, config.num_layers)

        self.hidden_size = config.hidden_size
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.apply_pe_at_input = config.apply_pe_at_input
        self.batch_first = config.batch_first

    def forward(
        self,
        current_vision_features: torch.Tensor,
        memory: torch.Tensor,
        current_vision_poisition_embeddings: Optional[Tensor] = None,
        memory_posision_embeddings: Optional[Tensor] = None,
        num_object_pointer_tokens: int = 0,
    ):
        """
        Args:
            current_vision_features (`torch.FloatTensor`):
                The current vision features used for self-attention.
            memory (`torch.FloatTensor`):
                The memory features used for cross-attention.
            current_vision_poisition_embeddings (`torch.FloatTensor`, *optional*):
                The position embeddings for the current vision features.
            memory_posision_embeddings (`torch.FloatTensor`, *optional*):
                The position embeddings for the memory features.
            num_object_pointer_tokens (`int`, *optional*):
                The number of object pointer tokens.
        """
        if isinstance(current_vision_features, list):
            assert isinstance(current_vision_poisition_embeddings, list)
            assert len(current_vision_features) == len(current_vision_poisition_embeddings) == 1
            current_vision_features, current_vision_poisition_embeddings = (
                current_vision_features[0],
                current_vision_poisition_embeddings[0],
            )

        assert current_vision_features.shape[1] == memory.shape[1], "Batch size must be the same for curr and memory"

        output = current_vision_features
        if self.apply_pe_at_input and current_vision_poisition_embeddings is not None:
            output = output + 0.1 * current_vision_poisition_embeddings

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            current_vision_poisition_embeddings = current_vision_poisition_embeddings.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_posision_embeddings = memory_posision_embeddings.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, Sam2RoPEAttention):
                kwds = {"num_k_exclude_rope": num_object_pointer_tokens}

            output = layer(
                queries=output,
                keys=memory,
                query_point_embedding=current_vision_poisition_embeddings,
                key_point_embedding=memory_posision_embeddings,
                **kwds,
            )

        normed_output = self.layer_norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            current_vision_poisition_embeddings = current_vision_poisition_embeddings.transpose(0, 1)

        return normed_output


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class Sam2MemoryFuserCXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        config,
        drop_path=0.0,
    ):
        super().__init__()
        memory_fuser_embed_dim = config.memory_fuser_embed_dim
        memory_fuser_layer_scale_init_value = config.memory_fuser_layer_scale_init_value
        self.depthwise_conv = nn.Conv2d(
            memory_fuser_embed_dim,
            memory_fuser_embed_dim,
            kernel_size=config.memory_fuser_kernel_size,
            padding=config.memory_fuser_padding,
            groups=memory_fuser_embed_dim if config.memory_fuser_use_depthwise_conv else 1,
        )  # depthwise conv
        self.layer_norm = Sam2LayerNorm(memory_fuser_embed_dim, eps=1e-6)
        self.activation = ACT2FN[config.memory_fuser_hidden_act]
        self.pointwise_conv1 = nn.Linear(
            memory_fuser_embed_dim, 4 * memory_fuser_embed_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pointwise_conv2 = nn.Linear(4 * memory_fuser_embed_dim, memory_fuser_embed_dim)
        self.scale = (
            nn.Parameter(
                memory_fuser_layer_scale_init_value * torch.ones((memory_fuser_embed_dim)), requires_grad=True
            )
            if memory_fuser_layer_scale_init_value > 0
            else None
        )
        self.drop_path = Sam2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        input = hidden_states
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)
        if self.scale is not None:
            hidden_states = self.scale * hidden_states
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        hidden_states = input + self.drop_path(hidden_states)
        return hidden_states


class Sam2MemoryFuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_projection = nn.Identity()
        layer = Sam2MemoryFuserCXBlock(config)
        self.layers = get_clones(layer, config.memory_fuser_num_layers)
        if config.memory_fuser_input_projection:
            assert config.memory_fuser_embed_dim is not None
            embed_dim = config.memory_fuser_embed_dim
            self.input_projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, hidden_states):
        # normally hidden_states: (N, C, H, W)
        hidden_states = self.input_projection(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Sam2MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        num_layers = int(math.log2(config.mask_downsampler_total_stride) // math.log2(config.mask_downsampler_stride))

        self.encoder = nn.Sequential()
        self.activation = ACT2FN[config.mask_downsampler_hidden_act]
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (config.mask_downsampler_stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=config.mask_downsampler_kernel_size,
                    stride=config.mask_downsampler_stride,
                    padding=config.mask_downsampler_padding,
                )
            )
            self.encoder.append(Sam2LayerNorm(mask_out_chans))
            self.encoder.append(self.activation)
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, config.mask_downsampler_embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


class Sam2MemoryEncoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        hidden_size = config.hidden_size
        output_channels = config.output_channels
        self.mask_downsampler = Sam2MaskDownSampler(config)
        self.feature_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.memory_fuser = Sam2MemoryFuser(config)
        self.position_encoding = Sam2PositionEmbeddingSine(num_pos_feats=output_channels)
        self.projection = nn.Identity()
        if output_channels != hidden_size:
            self.projection = nn.Conv2d(hidden_size, output_channels, kernel_size=1)

    def forward(
        self,
        vision_features: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pixel_features and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        vision_features = vision_features.to(masks.device)

        vision_features = self.feature_projection(vision_features)
        vision_features = vision_features + masks
        vision_features = self.memory_fuser(vision_features)
        vision_features = self.projection(vision_features)

        vision_pos_enc = self.position_encoding(vision_features).to(vision_features.dtype)

        return {"vision_features": vision_features, "vision_pos_enc": [vision_pos_enc]}


class Sam2PreTrainedModel(PreTrainedModel):
    config_class = Sam2Config
    base_model_prefix = "sam2"
    # main_input_name = "pixel_values"
    # _no_split_modules = ["SamVisionAttention"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


SAM2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Sam2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# TODO: update docstring
SAM2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and botton right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# TODO: update docstring
@add_start_docstrings(
    "Segment Anything Model 2 (SAM 2) for generating segmentation masks in images",
    SAM2_START_DOCSTRING,
)
class Sam2Model(Sam2PreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = Sam2PositionalEmbedding(config.prompt_encoder_config)

        # For single image inference
        self.image_encoder = Sam2ImageEncoder(config.image_encoder_config)
        self.prompt_encoder = Sam2PromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = Sam2MaskDecoder(config.mask_decoder_config)
        # For video sequence inference
        self.memory_attention = Sam2MemoryAttention(config.memory_attention_config)
        self.memory_encoder = Sam2MemoryEncoder(config.memory_encoder_config)

        self.use_high_resolution_features = config.mask_decoder_config.use_high_resolution_features
        self.num_feature_levels = 3 if self.use_high_resolution_features else 1
        # hacky_solution for giving image_encoder self.num_feature_levels
        self.image_encoder.num_feature_levels = self.num_feature_levels

        # memory encoder related part
        # a single token to indicate no memory embedding from previous frames
        self.no_memory_embedding = torch.nn.Parameter(torch.zeros(1, 1, config.image_encoder_config.fpn_hidden_size))
        self.no_memory_positional_encoding = torch.nn.Parameter(
            torch.zeros(1, 1, config.image_encoder_config.fpn_hidden_size)
        )
        self.directly_add_no_memory_embedding = config.directly_add_no_memory_embedding

        self.hidden_dim = config.image_encoder_config.fpn_hidden_size

        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "projection") and hasattr(self.memory_encoder.projection, "weight"):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.projection.weight.shape[0]
        self.num_maskmem = config.num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.memory_temporal_positional_encoding = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )

        # prompt encoder part
        self.use_mlp_for_object_pointer_proj = config.use_mlp_for_object_pointer_proj
        self.use_object_pointers_in_encoder = config.use_object_pointers_in_encoder
        self.proj_tpos_enc_in_object_pointers = config.proj_tpos_enc_in_object_pointers

        if config.pred_obj_scores and config.use_object_pointers_in_encoder:
            self.no_object_pointer = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        if self.use_object_pointers_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
            # a linear projection on SAM output tokens to turn them into object pointers
            self.object_pointer_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_object_pointer_proj:
                self.object_pointer_proj = Sam2FeedForward(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.object_pointer_proj = torch.nn.Identity()

        if self.proj_tpos_enc_in_object_pointers:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.object_pointer_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.object_pointer_tpos_proj = torch.nn.Identity()

        self.no_obj_embed_spatial = None
        if config.no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))

        if torch.cuda.is_available():
            try:
                logger.info("Building CUDA kernel, this might take some time...")
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load custom CUDA kernels for postprocessing: {e}")
        # Model compilation
        if config.compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print("Image encoder compilation is enabled. First forward pass will be slow.")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

        self.post_init()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @add_start_docstrings_to_model_forward(SAM2_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("danelcsb/sam2.1_hiera_tiny")
        >>> processor = AutoProcessor.from_pretrained("danelcsb/sam2.1_hiera_tiny")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.image_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            feature_maps = vision_outputs[1]
            vision_embeddings = vision_outputs[2]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[-2]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

            if self.use_high_resolution_features:
                # precompute projected level 0 and level 1 features in SAM decoder
                # to avoid running it again on every SAM click
                feature_maps = list(feature_maps)
                feature_maps[0] = self.mask_decoder.conv_s0(feature_maps[0])
                feature_maps[1] = self.mask_decoder.conv_s1(feature_maps[1])

            feature_maps = feature_maps[-self.num_feature_levels :]
            vision_embeddings = vision_embeddings[-self.num_feature_levels :]

            # flatten NxCxHxW to HWxNxC
            feature_maps = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
            vision_embeddings = [
                vision_embedding.flatten(2).permute(2, 0, 1) for vision_embedding in vision_embeddings
            ]

            if self.directly_add_no_memory_embedding:
                feature_maps[-1] = feature_maps[-1] + self.no_memory_embedding

            image_embeddings = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(feature_maps, self.config._bb_feat_sizes)
            ]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and vision_embeddings[-1].shape[1] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(vision_embeddings[-1].shape[1], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions, _ = self.mask_decoder(
            image_embeddings=image_embeddings[-1],
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            high_resolution_features=image_embeddings[:-1],
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks, image_embeddings)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return Sam2ImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            image_embeddings=image_embeddings,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )


__all__ = ["Sam2Model", "Sam2PreTrainedModel"]
