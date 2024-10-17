import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import AutoTokenizer


# fmt: off
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"^output.weight":                            r"lm_head.weight",

    # Model keys
    r"^norm.weight":                              r"model.norm.weight",
    r"^tok_embeddings.weight":                    r"model.embed_tokens.weight",

    # Layers keys
    r"^layers.(\d+).attention_norm.weight":       r"model.layers.\1.input_layernorm.weight",
    r"^layers.(\d+).ffn_norm.weight":             r"model.layers.\1.post_attention_layernorm.weight",

    # Attention keys
    r"^layers.(\d+).attention.w(q|k|v|o).weight": r"model.layers.\1.self_attn.\2_proj.weight",


    # MLP keys
    r"^layers.(\d+).feed_forward.w1.weight":      r"model.layers.\1.mlp.gate_proj.weight",
    r"^layers.(\d+).feed_forward.w2.weight":      r"model.layers.\1.mlp.down_proj.weight",
    r"^layers.(\d+).feed_forward.w3.weight":      r"model.layers.\1.mlp.up_proj.weight",
}
# fmt: on


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def permute_for_rope(value, n_heads, dim1, dim2):
    return value.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_state_dict(original_state_dict: dict, config):
    new_dict = {}

    n_heads = config.num_attention_heads
    dim = config.hidden_size
    dims_per_head = dim // n_heads
    num_key_value_heads = config.num_key_value_heads
    key_value_dim = dims_per_head * num_key_value_heads


    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        if "q_proj" in new_key:
            value = permute_for_rope(value.view(n_heads, dims_per_head, dim).reshape(dim, dim), n_heads, dim, dim)
        elif "k_proj" in new_key:
            value = permute_for_rope(value.view(num_key_value_heads, dims_per_head, dim).reshape(key_value_dim, dim), num_key_value_heads, key_value_dim, dim)
        elif "v_proj" in new_key:
            value = value.view(num_key_value_heads, dims_per_head, dim).reshape(key_value_dim, dim)

        new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict, config_class):
    key_mapping = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "intermediate_size": "hidden_dim",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_seq_len",
    }
    similar_keys_to_keep = [
        "head_dim",
        "vocab_size",
        "rope_theta",
    ]

    new_config_kwargs = {k: original_config[v] for k, v in key_mapping.items()}
    new_config_kwargs.update({k: v for k, v in original_config.items() if k in similar_keys_to_keep})
    # IMPORTANT:
    # This if fine for now as they limited the max_seq_len to the sliding window anyway! But improve later!
    new_config_kwargs["sliding_window"] = min(x for x in original_config["_sliding_window"] if x is not None)
    # new_config_kwargs["cache_implementation"] = "hybrid"

    new_config = config_class(**new_config_kwargs)
    return new_config


def convert_ministral_tokenizer(input_dir):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    template = """
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- '[INST]' + system_message + '\n\n' + message['content'] + '[/INST]' }}
        {%- else %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}""".lstrip()
    
    tokenizer.chat_template = template

    return tokenizer


def convert_ministral_model(input_dir, output_dir, class_name: str):

    if class_name == "Mistral":
        from transformers import MistralConfig, MistralForCausalLM
        model_class = MistralForCausalLM
        config_class = MistralConfig
    elif class_name == "Ministral":
        from transformers import MinistralConfig, MinistralForCausalLM
        model_class = MinistralForCausalLM
        config_class = MinistralConfig

    # Load and convert config
    with open(os.path.join(input_dir, "params.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config, config_class)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = load_file(os.path.join(input_dir, "consolidated.safetensors"))
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = model_class(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    # Load and convert tokenizer
    tokenizer = convert_ministral_tokenizer(input_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model class. Mistral or Ministral (We use Mistral while waiting for new model definition).",
        choices=["Mistral", "Ministral"]
    )

    args = parser.parse_args()
    convert_ministral_model(args.input_dir, args.output_dir, args.model)
