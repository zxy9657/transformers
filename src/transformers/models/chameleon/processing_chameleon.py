# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Chameleon.
"""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack, _validate_images_text_input_order
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class ChameleonTextKwargs(TextKwargs, total=False):
    return_for_text_completion: bool


class ChameleonProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: ChameleonTextKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_for_text_completion": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class ChameleonProcessor(ProcessorMixin):
    r"""
    Constructs a Chameleon processor which wraps a Chameleon image processor and a Chameleon tokenizer into a single
    processor.

    [`ChameleonProcessor`] offers all the functionalities of [`ChameleonImageProcessor`] and [`LlamaTokenizerFast`].
    See the [`~ChameleonProcessor.__call__`] and [`~ChameleonProcessor.decode`] for more information.

    Args:
        image_processor ([`ChameleonImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    valid_kwargs = ["image_seq_length", "image_token"]
    image_processor_class = "ChameleonImageProcessor"

    def __init__(self, image_processor, tokenizer, image_seq_length: int = 1024, image_token: str = "<image>"):
        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token = (
            tokenizer.boi_token if hasattr(tokenizer, "boi_token") else "<racm3:break>"
        )  # fixed tokens for start and end, so can hardcode
        self.image_end_token = tokenizer.eoi_token if hasattr(tokenizer, "eoi_token") else "<eoss>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[ChameleonProcessorKwargs],
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
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")
        if text is None and images is None:
            raise ValueError("You must provide either text or images")

        output_kwargs = self._merge_kwargs(
            ChameleonProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_for_text_completion = output_kwargs["text_kwargs"].pop("return_for_text_completion", False)

        # Replace the image token with the expanded image token sequence
        prompt_strings = []
        one_img_tokens = self.image_start_token + (self.image_token * self.image_seq_length) + self.image_end_token
        for sample in text:
            sample = sample.replace(self.image_token, one_img_tokens)
            if not return_for_text_completion:
                sample += self.tokenizer.sep_token  # special Chameleon treatment to add sep for chat mode
            prompt_strings.append(sample)

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.image_start_token_id] = 1
            mm_token_type_ids[array_ids == self.image_end_token_id] = 1
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
        return {"image": [self.image_seq_length + 2] * len(image_sizes), "video": 0, "audio": 0}

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


__all__ = ["ChameleonProcessor"]
