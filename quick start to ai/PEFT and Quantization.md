# Using PEFT and Quantization on Hugging Face

**Target audience:** Python-savvy engineers/researchers who know basic PyTorch/Transformers but are not experts in model compression or parameter-efficient fine-tuning (PEFT).

**Goal:** Show how to combine quantization (8-bit / 4-bit) with PEFT (LoRA, Adapters, Prefix Tuning) to perform memory-efficient inference and fine-tuning of large language models (LLMs) using Hugging Face tools.

---

## Table of contents

* [Why combine PEFT + Quantization?](#why-combine-peft--quantization)
* [Key concepts (short primer)](#key-concepts-short-primer)

  * [Quantization](#quantization)
  * [PEFT family (LoRA, Adapters, Prefix)](#peft-family-lora-adapters-prefix)
  * [When to use what](#when-to-use-what)
* [Prerequisites](#prerequisites)
* [Installations and environment tips](#installations-and-environment-tips)
* [Quick overview: flow of operations](#quick-overview-flow-of-operations)
* [Example 1 — Load a quantized model for inference (8-bit and 4-bit)](#example-1---load-a-quantized-model-for-inference-8-bit-and-4-bit)
* [Example 2 — Fine-tune with LoRA on a quantized model (end-to-end)](#example-2---fine-tune-with-lora-on-a-quantized-model-end-to-end)
* [Example 3 — Lightweight inference with LoRA adapters loaded on top of quantized base model](#example-3---lightweight-inference-with-lora-adapters-loaded-on-top-of-quantized-base-model)
* [Saving and loading PEFT adapters / merging weights](#saving-and-loading-peft-adapters--merging-weights)
* [Practical tips & gotchas](#practical-tips--gotchas)
* [Checklist before running on your GPU/ML server](#checklist-before-running-on-your-gpulml-server)
* [Further reading & references (quick list)](#further-reading--references-quick-list)

---

## Why combine PEFT + Quantization?

Large language models quickly exceed the memory of a single GPU when you try to fine-tune or even to run inference with larger sizes (>7B, >13B, etc.). Two complementary solutions:

* **Quantization** reduces memory and compute by storing weights at lower precision (8-bit, 4-bit), letting you *load* a much larger base model onto a single GPU for inference and sometimes for training.
* **PEFT** (Parameter-Efficient Fine-Tuning) updates and stores only a small subset of additional parameters (e.g., LoRA rank matrices) rather than the entire model weights. Typical savings: <1%–5% of full model parameters.

Together: you can *load* a quantized base model (fits on GPU) and *fine-tune* only small adapter matrices (LoRA) — giving you low-memory, fast experimentation.

---

## Key concepts (short primer)

### Quantization

* **8-bit quantization (INT8 / 8-bit)**: Most common practical choice. Libraries such as `bitsandbytes` implement optimized 8-bit matrix multiplications and allow `transformers` to `load_in_8bit=True`.
* **4-bit quantization (NF4, GPTQ, or QLoRA-style)**: More aggressive — can reduce memory further but requires extra care and specialized kernels. QLoRA (quantized LoRA) uses 4-bit quantized base models with LoRA fine-tuning.
* **Tradeoffs:** smaller precision → lower memory and faster compute but slightly lower numerical fidelity (small accuracy drop or sometimes none for many tasks).

### PEFT family (LoRA, Adapters, Prefix)

* **LoRA (Low-Rank Adaptation)**: Insert low-rank matrices into attention / feedforward layers. During training, you only update those low-rank matrices.
* **Adapters**: Small feed-forward modules added to each transformer layer; they are trained instead of the whole model.
* **Prefix Tuning**: Learns virtual tokens prepended to the input in attention space.

**Why LoRA?** It is simple, effective for LLMs, widely supported in the HF `peft` ecosystem, and pairs well with quantization (e.g., QLoRA flow).

---

## Installations and environment tips


```bash
pip install transformers accelerate datasets peft bitsandbytes safetensors sentencepiece
```

Notes:

* `bitsandbytes` provides 8-bit/4-bit kernels and sometimes needs a matching CUDA and PyTorch build.
* `accelerate` helps with device placement and mixed precision.
* `safetensors` recommended for saving/loading adapter weights safely.

If you are using a conda environment and run into `bitsandbytes` issues, follow the project docs for binary compatibility.

---

## Quick overview: flow of operations

1. Choose a base model from Hugging Face (e.g., `meta-llama/Llama-2-7b-chat-hf`, `gpt-neox-20b`).
2. Load the base model with quantization (`load_in_8bit=True` or `load_in_4bit=True`).
3. Prepare the model for k-bit training (helper functions to enable gradient updates for LoRA).
4. Apply a PEFT wrapper (LoRA config) and get a PEFT model.
5. Train only the LoRA parameters using Trainer / accelerate / custom loop.
6. Save the adapter weights. For inference, load the quantized base model and the adapter weights, then run generation.

---

## Example 1 - Load a quantized model for inference (8-bit and 4-bit)

> This example uses `transformers` + `bitsandbytes` for 8-bit and 4-bit loading. Adjust model name to one you have access to.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # example — use a model you are allowed to use

# 8-bit load (common)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_8bit = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
)

# simple inference
prompt = "Write a short haiku about software engineering."
inputs = tokenizer(prompt, return_tensors="pt").to(model_8bit.device)
out = model_8bit.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# 4-bit load (experimental; needs bitsandbytes recent features)
# bitsandbytes supports `load_in_4bit` with additional bnb config. Example:
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",  # or 'float16' depending on your setup
    bnb_4bit_use_double_quant=True,
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# inference similar to above
```

**Notes:**

* `device_map="auto"` with `accelerate`/`transformers` will place model parts across CPU/GPU when needed.
* 4-bit can be much more memory efficient but requires tested `bitsandbytes` + hardware support.

---

## Example 2 — Fine-tune with LoRA on a quantized model (end-to-end)

This example demonstrates a minimal QLoRA-like flow: load a quantized model, prepare it for k-bit training, attach LoRA, and fine-tune only the LoRA parameters. We'll use the `peft` library.

```python
# Minimal QLoRA-style fine-tune example
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset

MODEL = "meta-llama/Llama-2-7b-chat-hf"

# 1) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# 2) BitsAndBytes config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # choose based on your GPU & PyTorch
    bnb_4bit_use_double_quant=True,
)

# 3) Load quantized base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,  # some community models need this
)

# 4) Prepare for k-bit training
model = prepare_model_for_kbit_training(base_model)

# 5) Define LoRA config and wrap
lora_config = LoraConfig(
    r=8,                 # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # depends on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 6) Prepare dataset (toy example)
datasets = load_dataset("json", data_files={"train": "train.json", "validation": "valid.json"})

# Tokenize helper
def tokenize_fn(batch):
    texts = [x + tokenizer.eos_token for x in batch["text"]]
    out = tokenizer(texts, truncation=True, padding=True, max_length=512)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = datasets.map(tokenize_fn, batched=True, remove_columns=datasets["train"].column_names)

# 7) TrainingArguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    output_dir="lora-output",
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("validation"),
)

# 8) Run training (only LoRA params require gradients)
trainer.train()

# 9) Save only PEFT weights
model.save_pretrained("lora-output/adapter")
```

**Important points:**

* `prepare_model_for_kbit_training` modifies the model to enable gradient flow in quantized parameters where required and makes the model compatible with LoRA training.
* `target_modules` varies by architecture — inspect the model's `.named_modules()` to find which modules correspond to query/key/value projections or linear layers you want to adapt.
* Batch sizes are typically small when using large models — rely on `gradient_accumulation_steps`.
* You can find which kinds of layers you can add under `target_modules`, these are usually the linear layers (not attention, encoder or decder)

---

## Example 3 — Lightweight inference with LoRA adapters loaded on top of quantized base model

After training and saving the LoRA adapter, you can load the quantized base model and the adapter for inference.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_PATH = "lora-output/adapter"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

# load LoRA adapter weights
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

prompt = "Explain PEFT in a sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

`PeftModel.from_pretrained` will merge adapter weights on-the-fly for inference without touching the original base model files on disk.

---

## Saving, loading, and merging weights

* `model.save_pretrained("path/to/adapter")` saves only the PEFT adapters (small files).
* To **merge** the adapters into the base model (if you want a single checkpoint):

  * Convert or load the base model and the adapter, then use APIs (or manually apply) to merge weights. In `peft`, depending on versions, there are utilities to merge and unload. Merging creates a full-sized model checkpoint (which may not be quantized) — do this if you need a single distributable model.
* Always prefer `safetensors` for saving if supported to avoid pickling risks.

---

## Practical tips & gotchas

1. **Tokenizer & special tokens**: Ensure `tokenizer.pad_token` exists; some tokenizers lack it and HF generation might error.
2. **Target modules**: `target_modules` in LoRA config depend on implementation details of the transformer blocks. For many models, use `["q_proj","v_proj"]` or `"qkv"` style names; inspect `model.named_modules()`.
3. **Mixed precision**: Use `fp16` or `bf16` where supported. `bnb` compute dtype often pairs with `bfloat16`.
4. **Offloading & device_map**: `device_map='auto'` with `accelerate` is convenient. For constrained GPUs, use CPU offload for some layers.
5. **Reproducibility**: Quantization and mixed-precision can lead to nondeterministic outputs — expect slight differences.
6. **Evaluation**: When evaluating on generation tasks, consider metrics like exact match, BLEU, or human evaluation; token-level losses might not fully reflect generation quality.
7. **Gradient checkpointing**: If memory is tight, consider `model.gradient_checkpointing_enable()` (speeds memory but increases compute).
8. **Confirm compatibility**: Some community models require `trust_remote_code=True` when loading.
9. **Monitor GPU memory**: Use `nvidia-smi` and watch the process while doing `device_map='auto'` loads.

---

## Checklist before running on your GPU/ML server

* `bitsandbytes` installed and compatible with your CUDA/PyTorch.
* Enough host RAM to hold CPU copies of quantized model shards if the device_map places some weights on CPU.
* `safetensors` installed for safer and faster saves.
* If using 4-bit: ensure you tested a small script to load the model before attempting training.
* Have a clean dataset and tokenizer special tokens in place.

---

## Further reading & references (quick list)

* Hugging Face `transformers` docs — model loading, `device_map`, and quantization.
* `bitsandbytes` README — practical install & compatibility.
* `peft` library README — LoRA, Adapters examples.
* QLoRA blog posts and notebooks — examples combining 4-bit quantization with LoRA.

---

## Final notes

Combining quantization and PEFT gives one of the most practical paths to working with larger models on single-GPU machines and to experiment quickly. Start with 8-bit + LoRA for stability, then move to 4-bit + LoRA (QLoRA-style) when you need more memory savings.

If
