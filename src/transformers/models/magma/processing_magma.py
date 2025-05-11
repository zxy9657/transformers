# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for Magma.
"""

from typing import List, Optional, Union

import torch
import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin, ProcessingKwargs, Unpack, ImagesKwargs
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from transformers.utils import TensorType


class MagmaImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class MagmaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MagmaImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class MagmaProcessor(ProcessorMixin):
    r"""
    Constructs a Magma processor which wraps a Magma image processor and a LLaMa tokenizer into a single processor.

    [`MagmaProcessor`] offers all the functionalities of [`MagmaImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MagmaProcessor.__call__`] and [`~MagmaProcessor.decode`] for more information.

    Args:
        image_processor ([`MagmaImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer        
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput]] = None,
        text: Union[TextInput, List[TextInput]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs: Unpack[MagmaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MagmaImageProcessor's [`~MagmaImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.        
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True` will pad the images in the batch to the largest image in the batch
                and create a pixel mask. Padding will be applied to the bottom and right of the image with zeros.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
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
        if images is not None:
            image_inputs = self.image_processor(images, do_pad=do_pad, return_tensors=return_tensors)
        else:
            image_inputs = {}
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        if len(image_inputs) > 0:
            # expand the image_token_id to the number of image tokens
            image_token_id = self.image_token_id
            input_ids = text_inputs["input_ids"]
            # replace the image_token_id with the number of image tokens that is equal to the number of image tokens in the image
            image_sizes = image_inputs["image_sizes"]  # (batch_size, num_images, 2)
            assert input_ids.shape[0] == image_sizes.shape[0]
            input_ids_list = input_ids.tolist()
            assert len(input_ids_list) == image_sizes.shape[0]
            
            input_ids_list_filled = []
            for i in range(input_ids.shape[0]):
                input_ids_orig = input_ids[i].tolist()
                input_ids_list_filled.append([])
                img_idx = 0
                for id in input_ids_orig:
                    if id == image_token_id:
                        # replace the image_token_id with the number of image tokens that is equal to the number of image tokens in the image
                        assert image_sizes[i][img_idx][0] != 0 and image_sizes[i][img_idx][1] != 0, "some mismatch happens, please double check your prompt processor"
                        num_image_tokens = image_sizes[i][img_idx][0] * image_sizes[i][img_idx][1] * 256 + image_sizes[i][img_idx][0] * 16
                        input_ids_list_filled[i].extend([image_token_id] * num_image_tokens)                        
                        img_idx += 1
                    else:
                        input_ids_list_filled[i].append(id)
            
            text_inputs["input_ids"] = torch.tensor(input_ids_list_filled)
            text_inputs["attention_mask"] = torch.ones_like(text_inputs["input_ids"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

__all__ = ["MagmaProcessor"]