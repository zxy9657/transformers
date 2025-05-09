# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import to_pil_image
from ...image_utils import ImageInput, make_list_of_images
from ...utils import TensorType, logging, requires_backends
from ...utils.import_utils import is_timm_available, is_torch_available, requires


if is_timm_available():
    import timm

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

_DATA_ARG_KEYS = (
    "input_size",
    "img_size",
    "interpolation",
    "in_chans",
    "mean",
    "std",
    "use_test_size",
    "use_train_size",
)


@requires(backends=("torch", "timm", "torchvision"))
class TimmWrapperImageProcessor(BaseImageProcessor):
    """
    Wrapper class for timm models to be used within transformers.

    Args:
        pretrained_cfg (`Dict[str, Any]`):
            The configuration of the pretrained model used to resolve evaluation and
            training transforms.
        architecture (`Optional[str]`, *optional*):
            Name of the architecture of the model.
    """

    main_input_name = "pixel_values"

    def __init__(
        self,
        pretrained_cfg: Dict[str, Any],
        architecture: Optional[str] = None,
        **kwargs,
    ):
        requires_backends(self, "timm")
        super().__init__(architecture=architecture)

        data_arg_overrides = {}
        for k in _DATA_ARG_KEYS:
            if k in kwargs:
                data_arg_overrides[k] = kwargs.pop(k)

        # Many convnets in timm have a different, higher test time input size default to leverage the
        # train-test resolution discrepancy. This isn't expected in most transformers use cases so,
        # we default to using train resolution unless `use_test_size = True` or `use_train_size = False` is
        # passed as a kwarg.
        # Both flags are checked because the defaults are opposite between here and timm useage.
        use_test_size = data_arg_overrides.get("use_test_size", data_arg_overrides.get("use_train_size", True))
        self.data_config = timm.data.resolve_data_config(
            args=data_arg_overrides,  # will override values in pretrained_cfg
            pretrained_cfg=pretrained_cfg,
            model=None,
            use_test_size=use_test_size,
            verbose=False,
        )

        self.val_transforms = timm.data.create_transform(**self.data_config, is_training=False, **kwargs)

        # useful for training, see examples/pytorch/image-classification/run_image_classification.py
        self.train_transforms = timm.data.create_transform(**self.data_config, is_training=True, **kwargs)

        # If `ToTensor` is in the transforms, then the input should be numpy array or PIL image.
        # Otherwise, the input can be a tensor. In later timm versions, `MaybeToTensor` is used
        # which can handle both numpy arrays / PIL images and tensors.
        self._not_supports_tensor_input = any(
            transform.__class__.__name__ == "ToTensor" for transform in self.val_transforms.transforms
        )

    def get_size_dict(self):
        input_size = self.data_config["input_size"][1:]  # remove leading channel dim
        # timm can do more advanced crop (aspect preserving shortest edge w/ non-square target)
        # This logic should provide least surprises to downstream users attempting to match.
        if self.data_config["crop_mode"] == "center" and input_size[0] == input_size[1]:
            # Aspect preserving, resize shortest edge to target & crop
            # NOTE in timm the resize & centre crop is aspect preserving even if height != width
            # but most other code that leverages this size is not able to match so we must prioritize
            # getting the correct final height & width, we fallback to squash crop below.
            return {"shortest_edge": input_size[0]}
        elif self.data_config["crop_mode"] == "border" and input_size[0] == input_size[1]:
            # Aspect preserving, resize longest edge to target & pad
            # NOTE same concerns as "center" above
            return {"longest_edge": input_size[0]}
        else:
            # Non aspect preserving, resize dims to target aka 'squash' in timm terms
            return {"height": input_size[0], "width": input_size[1]}

    @property
    def size(self) -> Dict:
        return self.get_size_dict()

    @property
    def crop_size(self) -> Dict:
        input_size = self.data_config["input_size"]
        return {"height": input_size[-2], "width": input_size[-1]}

    @property
    def crop_pct(self) -> float:
        return self.data_config["crop_pct"]

    @property
    def image_mean(self) -> List[int]:
        return self.data_config["mean"]

    @property
    def image_std(self) -> List[int]:
        return self.data_config["std"]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        """
        output = super().to_dict()
        output.pop("train_transforms", None)
        output.pop("val_transforms", None)
        output.pop("_not_supports_tensor_input", None)
        return output

    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the image processor dict for the model.
        """
        requires_backends(cls, "timm")

        image_processor_filename = kwargs.pop("image_processor_filename", "config.json")
        image_processor_dict, kwargs = super().get_image_processor_dict(
            pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
        )

        # Only pass through architecture and pretrained_cfg from config.json
        image_processor_dict = {
            "architecture": image_processor_dict["architecture"],
            "pretrained_cfg": image_processor_dict["pretrained_cfg"],
        }

        # Merge kwargs that should be passed through to timm transform factory into image_processor_dict
        create_transform_keys = tuple(inspect.signature(timm.data.create_transform).parameters.keys())
        all_timm_keys = set(_DATA_ARG_KEYS + create_transform_keys)
        for k in all_timm_keys:
            if k in kwargs:
                image_processor_dict[k] = kwargs.pop(k)

        return image_processor_dict, kwargs

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.
        """
        if return_tensors != "pt":
            raise ValueError(f"return_tensors for TimmWrapperImageProcessor must be 'pt', but got {return_tensors}")

        if self._not_supports_tensor_input and isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        # If the input is a torch tensor, then no conversion is needed
        # Otherwise, we need to pass in a list of PIL images
        if isinstance(images, torch.Tensor):
            images = self.val_transforms(images)
            # Add batch dimension if a single image
            images = images.unsqueeze(0) if images.ndim == 3 else images
        else:
            images = make_list_of_images(images)
            images = [to_pil_image(image) for image in images]
            images = torch.stack([self.val_transforms(image) for image in images])

        return BatchFeature({"pixel_values": images}, tensor_type=return_tensors)

    def save_pretrained(self, *args, **kwargs):
        # disable it to make checkpoint the same as in `timm` library.
        logger.warning_once(
            "The `save_pretrained` method is disabled for TimmWrapperImageProcessor. "
            "The image processor configuration is saved directly in `config.json` when "
            "`save_pretrained` is called for saving the model."
        )


__all__ = ["TimmWrapperImageProcessor"]
