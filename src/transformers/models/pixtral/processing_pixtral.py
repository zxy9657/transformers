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
Processor class for Pixtral.
"""

from typing import List, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from .image_processing_pixtral import get_resize_output_image_size


logger = logging.get_logger(__name__)


class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {},
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


class PixtralProcessor(ProcessorMixin):
    r"""
    Constructs a Pixtral processor which wraps a Pixtral image processor and a Pixtral tokenizer into a single processor.

    [`PixtralProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~PixtralProcessor.__call__`] and [`~PixtralProcessor.decode`] for more information.

    Args:
        image_processor ([`PixtralImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*, defaults to 16):
            Patch size from the vision tower.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The downsampling factor for the spatial merge operation.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"[IMG]"`):
            Special token used to denote image location.
        image_break_token (`str`, *optional*, defaults to `"[IMG_BREAK]"`):
            Special token used to denote the end of a line of pixels in an image.
        image_end_token (`str`, *optional*, defaults to `"[IMG_END]"`):
            Special token used to denote the end of an image input.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "spatial_merge_size",
        "image_token",
        "image_break_token",
        "image_end_token",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(self.image_break_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[PixtralProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PixtralProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        patch_size = self.patch_size * self.spatial_merge_size

        if images is not None:
            if is_image_or_image_url(images):
                images = [images]
            elif isinstance(images, (list, tuple)) and is_image_or_image_url(images[0]):
                pass
            elif (
                isinstance(images, (list, tuple))
                and isinstance(images[0], (list, tuple))
                and is_image_or_image_url(images[0][0])
            ):
                images = [image for sublist in images for image in sublist]
            else:
                raise ValueError(
                    "Invalid input images. Please provide a single image, a list of images, or a list of lists of images."
                )
            images = [load_image(im) if isinstance(im, str) else im for im in images]
            image_inputs = self.image_processor(images, patch_size=patch_size, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            # Replace the image token with the expanded image token sequence
            image_sizes = iter(image_inputs["image_sizes"])
            prompt_strings = []
            replace_strings = []

            for sample in text:
                while self.image_token in sample:
                    height, width = next(image_sizes)
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [
                        [self.image_token] * num_width_tokens + [self.image_break_token]
                    ] * num_height_tokens
                    # Flatten list
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]
                    replace_tokens[-1] = self.image_end_token
                    replace_str = "".join(replace_tokens)
                    replace_strings.append(replace_str)
                    sample = sample.replace(self.image_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    replace_str = replace_strings.pop(0)
                    sample = sample.replace("<placeholder>", replace_str, 1)
                prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.image_end_token_id] = 1
            mm_token_type_ids[array_ids == self.image_break_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_mm_tokens_from_sizes(
        self, image_sizes=None, video_sizes=None, audio_lengths=None, **mm_processor_kwargs
    ):
        """
        Computes the number of placeholder tokens needed for each multimodal input type
        (image, video, and audio) with the given input sizes.
        Args:
            image_sizes (List[List[str]], *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (List[List[str]], *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
            audio_lengths (List[int], *optional*):
                The input length formatted as per each audio.
        Returns:
            Dict[str, List[int]]: A dictionary mapping each modality ("image", "video", "audio")
            to a list containing the number of placeholder tokens required. If the model doesn't accept
            a certain modality or no input sizes are provided, the dict value is set to an empty list.
        """
        size = mm_processor_kwargs.get("size", None) or self.image_processor.size
        patch_size = mm_processor_kwargs.get("patch_size", None) or self.image_processor.patch_size
        patch_height, patch_width = patch_size["height"], patch_size["width"]

        batch_num_image_tokens = []
        for height, width in image_sizes:
            resized_height, resized_width = get_resize_output_image_size(
                height,
                width,
                size=(size["longest_edge"], size["longest_edge"]),
                patch_size=(patch_height, patch_width),
            )
            num_height_tokens = resized_height // patch_height
            num_width_tokens = resized_width // patch_width
            num_image_tokens = (num_width_tokens + 1) * num_height_tokens
            batch_num_image_tokens.append(num_image_tokens)

        return {"image": batch_num_image_tokens, "video": [], "audio": []}

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


__all__ = ["PixtralProcessor"]
