# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniMax model."""

import unittest

import pytest

from transformers import MiniMaxConfig, is_torch_available
from transformers.cache_utils import Cache
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MiniMaxForCausalLM,
        MiniMaxForQuestionAnswering,
        MiniMaxForSequenceClassification,
        MiniMaxForTokenClassification,
        MiniMaxModel,
    )


class MiniMaxModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
        router_jitter_noise=0.1,
        attn_type_list=[1, 0],
        block_size=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.router_jitter_noise = router_jitter_noise
        self.attn_type_list = attn_type_list
        self.block_size = block_size

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return MiniMaxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            num_experts_per_tok=2,
            num_local_experts=2,
            router_jitter_noise=self.router_jitter_noise,
            attn_type_list=self.attn_type_list,
            block_size=self.block_size,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->MiniMax
    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MiniMaxModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->MiniMax
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
# Copied from tests.models.mixtral.test_modeling_mixtral.MixtralModelTest with Mixtral->MiniMax
class MiniMaxModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MiniMaxModel,
            MiniMaxForCausalLM,
            MiniMaxForSequenceClassification,
            MiniMaxForTokenClassification,
            MiniMaxForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MiniMaxModel,
            "text-classification": MiniMaxForSequenceClassification,
            "token-classification": MiniMaxForTokenClassification,
            "text-generation": MiniMaxForCausalLM,
            "zero-shot": MiniMaxForSequenceClassification,
            "question-answering": MiniMaxForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def setUp(self):
        self.model_tester = MiniMaxModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniMaxConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_torch_fx_output_loss(self):
        super().test_torch_fx_output_loss()

    def test_MiniMax_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniMaxForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_MiniMax_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniMaxForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_MiniMax_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = MiniMaxForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_token_classification_model with Llama->MiniMax,llama->MiniMax
    def test_MiniMax_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = MiniMaxForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="MiniMax flash attention does not support right padding")

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_local_experts = 8
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MiniMaxForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_local_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of includding padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())

    # Ignore copy
    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))
        use_cache = decoder_past_key_values is not None

        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            expected_shape = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                prompt_length + generated_length,
            )
            for layer_idx, layer_attention in enumerate(iter_attentions):
                if config.attn_type_list[layer_idx] == 1:
                    self.assertEqual(layer_attention.shape, expected_shape)

    # Ignore copy
    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))

        # (batch, head, seq_length, head_features)
        key_value_cache_expected_shape = (
            batch_size,
            config.num_key_value_heads,
            cache_length,
            config.hidden_size // config.num_attention_heads,
        )
        # (batch, head, head_features, head_features)
        linear_cache_expected_shape = (
            batch_size,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )

        for layer_idx in range(config.num_hidden_layers):
            if config.attn_type_list[layer_idx] == 0:
                self.assertEqual(decoder_past_key_values[layer_idx][0].shape, linear_cache_expected_shape)
            else:
                self.assertEqual(decoder_past_key_values[layer_idx][0].shape, key_value_cache_expected_shape)
                self.assertEqual(decoder_past_key_values[layer_idx][1].shape, key_value_cache_expected_shape)

    # Ignore copy
    @pytest.mark.generate
    def test_past_key_values_format(self, custom_all_cache_shapes=None):
        """
        Test that the KV cache is formatted correctly.
        """
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config).to(torch_device)
            model = model.eval()
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            past_kv = outputs["past_key_values"]

            batch_size, seq_length = inputs["input_ids"].shape
            self._check_past_key_values_for_generate(batch_size, past_kv, seq_length, config)

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_contrastive_generate_low_memory(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_assisted_decoding_sample(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_assisted_decoding_with_logits_to_keep(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    # Ignore copy
    @unittest.skip(reason="MiniMaxCache doesnot support `crop()` method")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass


@require_torch
@require_torch_accelerator
@slow
class MiniMaxIntegrationTest(unittest.TestCase):
    def test_small_model_logits(self):
        model_id = "geetu040/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(
            torch_device
        )
        expected_slice = (
            torch.Tensor([[-0.3320, -0.5703, -0.3633], [0.6133, 0.9609, -0.7422], [-0.2461, -0.7148, -0.6602]])
            .to(torch.bfloat16)
            .to(torch_device)
        )

        with torch.no_grad():
            logits = model(dummy_input).logits

        torch.testing.assert_close(logits[0, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)

    def test_small_model_generation(self):
        model_id = "geetu040/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(
            torch_device
        )
        expected_slice = (
            torch.Tensor([[0, 1, 0, 604, 2235, 220, 1664, 1636], [0, 1, 0, 604, 2235, 220, 1664, 1636]])
            .to(torch.int64)
            .to(torch_device)
        )

        outputs = model.generate(dummy_input, max_new_tokens=5, do_sample=False)

        torch.testing.assert_close(outputs, expected_slice, atol=1e-3, rtol=1e-3)
