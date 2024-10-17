import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file
from tokenizers import processors

from transformers import MinistralConfig, MinistralForCausalLM, PreTrainedTokenizerFast


# fmt: off
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"output.weight":                            r"lm_head.weight",

    # Model keys
    r"norm.weight":                              r"model.norm.weight",
    r"tok_embeddings.weight":                    r"model.embed_tokens.weight",

    # Layers keys
    r"layers.(\d+).attention_norm.weight":       r"model.layers.\1.input_layernorm.weight",
    r"layers.(\d+).ffn_norm.weight":             r"model.layers.\1.post_attention_layernorm.weight",

    # Attention keys
    r"layers.(\d+).attention.w(q|k|v|o).weight": r"model.layers.\1.self_attn.\2_proj.weight",


    # MLP keys
    r"layers.(\d+).feed_forward.w1.weight":      r"model.layers.\1.mlp.gate_proj.weight",
    r"layers.(\d+).feed_forward.w2.weight":      r"model.layers.\1.mlp.up_proj.weight",
    r"layers.(\d+).feed_forward.w3.weight":      r"model.layers.\1.mlp.down_proj.weight",
}
# fmt: on


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def convert_state_dict(original_state_dict: dict, config: MinistralConfig):
    new_dict = {}

    # head_dim = config.hidden_size // config.num_attention_heads
    # query_size = config.num_attention_heads * head_dim
    # kv_size = config.num_key_value_heads * head_dim

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)

        # if "qkv_proj." in new_key:
        #     q_proj, k_proj, v_proj = (
        #         value[:query_size, ...],
        #         value[query_size : query_size + kv_size, ...],
        #         value[query_size + kv_size :, ...],
        #     )
        #     new_dict[new_key.replace("qkv_proj.", "q_proj.")] = q_proj
        #     new_dict[new_key.replace("qkv_proj.", "k_proj.")] = k_proj
        #     new_dict[new_key.replace("qkv_proj.", "v_proj.")] = v_proj
        # else:
        #     new_dict[new_key] = value
        new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict):
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

    new_config = MinistralConfig(**new_config_kwargs)
    return new_config


def convert_ministral_tokenizer(input_dir):
    fast_tok = PreTrainedTokenizerFast.from_pretrained(input_dir, model_input_names=["input_ids", "attention_mask"])
    # Add the two tokens automatically with post processor
    fast_tok._tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single="[gMASK]:0 <sop>:0 $A:0",
                pair="[gMASK]:0 <sop>:0 $A:0 $B:1",
                special_tokens=[("[gMASK]", 151331), ("<sop>", 151333)],
            ),
        ],
    )

    return fast_tok


def convert_ministral_model(input_dir, output_dir):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = load_file(os.path.join(input_dir, "consolidated.safetensors"))
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = MinistralForCausalLM(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    # Load and convert tokenizer
    # tokenizer = convert_ministral_tokenizer(input_dir)
    # tokenizer.save_pretrained(output_dir)


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

    args = parser.parse_args()
    convert_ministral_model(args.input_dir, args.output_dir)
