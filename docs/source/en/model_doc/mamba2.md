<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>

# Mamba 2

[Mamba2](https://huggingface.co/papers/2405.21060) is the second iteration of selective structured state space model (SSMs) by Tri Dao and Albert Gu. It brings many improvements to the original architecture such as better parallelism support and more optimized support for higher dimensionalities. 

Mamba2-based models such as [mistralai/Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1) can be found under the [mistral](https://huggingface.co/mistralai) organization.

> [!TIP]
> Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks.

The example below demonstrates how to generate text with [`Pipeline`], [`AutoModel`], and from the command line.

hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="mistralai/Mamba-Codestral-7B-v0.1",
    torch_dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", torch_dtype=torch.float16, device_map="auto")  
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to("cuda")  

output = model.generate(**input_ids)  
print(tokenizer.decode(output[0], skip_special_tokens=True)
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model mistralai/Mamba-Codestral-7B-v0.1 --device 0
```

</hfoption>
</hfoptions>

## Notes

- Mamba-2 has two different forward passes, `torch_forward` or `cuda_kernels_forward`. The latter uses the original cuda kernels if they are found in your environment, and is slower on the prefill i.e. requires a "warmup run" due to high cpu overhead.

- Without compilation, the `torch_forward` implementation is faster by a factor 3 to 4. Further, there are no positional embeddings in this model, but there is an `attention_mask` and a specific logic to mask out hidden states in two places in the case of batched generation.
 
- Due to this, in addition to the reimplementation of mamba2 kernels, batched generation and cached generation are expected to have slight discrepancies. Further, the results given by the cuda kernels or the torch forward are expected to be slightly different. The SSM algorithm heavily relies on tensor contractions, which have matmul equivalents but the order of operations is slightly different, making the difference greater at smaller precisions. 

- Shutdown of hidden states corresponding to padding tokens is done in 2 places. Right-padding will propagate noise down the line and is not guaranteed to yield satisfactory results. `tokenizer.padding_side = "left"` ensures you are using the correct padding side.

- The example below demonstrates how to fine-tune Mamba with [PEFT](https://huggingface.co/docs/peft).

```python 
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, Mamba2ForCausalLM, TrainingArguments
model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" #enforce padding side left

model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
dataset = load_dataset("Abirate/english_quotes", split="train")
# Without CUDA kernels, batch size of 2 occupies one 80GB device
# but precision can be reduced.
# Experiments and trials welcome!
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()
```


## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
    - forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
    - forward
