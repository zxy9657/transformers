# coding=utf-8
# Copyright 2025-present the HuggingFace Inc. team.
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
Validators to be used with `huggingface_hub.dataclasses.validated_field`. We recommend using the validator(s) that best
describe the constraints of your dataclass fields, for the best user experience (e.g. better error messages).
"""

from typing import Callable, Optional

from .activations import ACT2CLS


# Numerical validators


def interval(
    min: Optional[int | float] = None,
    max: Optional[int | float] = None,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> Callable:
    """
    Parameterized validator that ensures that `value` is within the defined interval. Optionally, the interval can be
    open on either side. Expected usage: `validated_field(interval(min=0), default=8)`

    Args:
        min (`int` or `float`, *optional*):
            Minimum value of the interval.
        max (`int` or `float`, *optional*):
            Maximum value of the interval.
        exclude_min (`bool`, *optional*, defaults to `False`):
            If True, the minimum value is excluded from the interval.
        exclude_max (`bool`, *optional*, defaults to `False`):
            If True, the maximum value is excluded from the interval.
    """
    error_message = "Value must be"
    if min is not None:
        if exclude_min:
            error_message += f" greater than {min}"
        else:
            error_message += f" greater or equal to {min}"
    if min is not None and max is not None:
        error_message += " and"
    if max is not None:
        if exclude_max:
            error_message += f" smaller than {max}"
        else:
            error_message += f" smaller or equal to {max}"
    error_message += ", got {value}."

    min = min or float("-inf")
    max = max or float("inf")

    def _inner(value: int | float):
        min_valid = min <= value if not exclude_min else min < value
        max_valid = value <= max if not exclude_max else value < max
        if not (min_valid and max_valid):
            raise ValueError(error_message.format(value=value))

    return _inner


def probability(value: float):
    """Ensures that `value` is a valid probability number, i.e. [0,1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"Value must be a probability between 0.0 and 1.0, got {value}.")


def token(value: Optional[int]):
    """Ensures that `value` is a potential token. A token, when set, must be a non-negative integer."""
    if value is not None and value < 0:
        raise ValueError(f"A token, when set, must be a non-negative integer, got {value}.")


# String validators


def activation_fn_key(value: str):
    """Ensures that `value` is a string corresponding to an activation function."""
    # TODO (joao): in python 3.11+, we can build a Literal type from the keys of ACT2CLS
    if value not in ACT2CLS:
        raise ValueError(
            f"Value must be one of {list(ACT2CLS.keys())}, got {value}. "
            "Make sure to use a string that corresponds to an activation function."
        )


__all__ = ["interval", "probability", "token", "activation_fn_key"]
