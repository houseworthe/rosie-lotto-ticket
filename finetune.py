"""
Autoresearch fine-tuning script for ROSIE SLURM cluster.
Single-GPU, single-file. The agent modifies this file.

Usage: python finetune.py
  or via SLURM: sbatch slurm_template.sh
"""

import os
import json
import time
import random
import datetime
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from eval import evaluate_task

# ---------------------------------------------------------------------------
# Hyperparameters (EDIT THESE — this is what the agent tunes)
# ---------------------------------------------------------------------------

# Task selection
TASK = "trec"  # Options: "trec", "text2sql", "ecommerce"

# Model selection
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Options: Qwen3-0.6B, 1.7B, 4B, 8B

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
USE_LORA = True  # Set False for full fine-tuning (needs more VRAM)

# Training hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
LR_SCHEDULER = "cosine"  # Options: "cosine", "linear", "constant"

# Quantization (for QLoRA)
USE_4BIT = False  # Enable for QLoRA (saves VRAM, may lose some quality)

# Reproducibility
SEED = 42

# Paths
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output")
CHECKPOINT_DIR = os.path.expanduser("~/autoresearch/checkpoints")

# ---------------------------------------------------------------------------
# Task-specific prompt templates
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "trec": {
        "system": "You are a question classifier. Classify the given question into one of these categories: ABBREVIATION, ENTITY, DESCRIPTION, HUMAN, LOCATION, NUMERIC.",
        "input_template": "Classify this question: {text}",
        "output_key": "label_text",
    },
    "text2sql": {
        "system": "You are a SQL expert. Given a natural language question and database schema, generate the correct SQL query.",
        "input_template": "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:",
        "output_key": "sql",
    },
    "ecommerce": {
        "system": "You are a product classifier. Classify the given product description into the correct category.",
        "input_template": "Classify this product: {text}",
        "output_key": "category",
    },
}

# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

def format_example(example, task_config, tokenizer):
    """Format a single example into chat-style input/output for fine-tuning."""
    system_msg = task_config["system"]
    input_text = task_config["input_template"].format(**example)
    output_text = str(example[task_config["output_key"]])

    # Build conversation format
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        full_text = f"<|system|>{system_msg}<|user|>{input_text}<|assistant|>{output_text}"

    return full_text


def format_prompt_only(example, task_config, tokenizer):
    """Format the prompt portion (without the answer) for label masking."""
    system_msg = task_config["system"]
    input_text = task_config["input_template"].format(**example)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": input_text},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"<|system|>{system_msg}<|user|>{input_text}<|assistant|>"

    return prompt_text


def prepare_dataset(dataset, task_config, tokenizer, max_len):
    """Tokenize and prepare dataset for training with proper label masking."""

    def tokenize_fn(examples):
        texts = []
        prompt_texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples.keys()}
            text = format_example(ex, task_config, tokenizer)
            prompt = format_prompt_only(ex, task_config, tokenizer)
            texts.append(text)
            prompt_texts.append(prompt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        # Tokenize prompts to find where the answer starts
        prompt_tokenized = tokenizer(
            prompt_texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        # Mask labels: set prompt tokens to -100, only train on completion tokens
        labels = []
        for ids, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
            label = ids.copy()
            prompt_len = len(prompt_ids)
            # Set all prompt tokens to -100 (ignored by loss)
            for j in range(min(prompt_len, len(label))):
                label[j] = -100
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_gpu_info():
    """Get GPU type and info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        capability = torch.cuda.get_device_capability(0)
        return gpu_name, gpu_mem, capability
    return "cpu", 0, (0, 0)


def get_dtype_for_gpu():
    """Determine correct dtype based on GPU capability. T4 = fp16, V100+/H100 = bf16."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    capability = torch.cuda.get_device_capability(0)[0]
    if capability >= 8:  # Ampere+ (A100, H100)
        return torch.bfloat16, False, True  # dtype, fp16, bf16
    else:  # Turing (T4), Volta (V100)
        return torch.float16, True, False  # dtype, fp16, bf16


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_hyperparameters():
    """Print all hyperparameters for reproducibility."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETERS")
    print("=" * 60)
    params = {
        "task": TASK, "model": MODEL_NAME,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT, "lora_targets": LORA_TARGET_MODULES,
        "use_lora": USE_LORA, "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
        "max_seq_len": MAX_SEQ_LEN, "lr_scheduler": LR_SCHEDULER,
        "use_4bit": USE_4BIT, "seed": SEED,
    }
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")
    return params

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    timestamp = datetime.datetime.now().isoformat()

    # Log all hyperparameters
    hyperparams = log_hyperparameters()

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # GPU info
    gpu_name, gpu_mem, gpu_capability = get_gpu_info()
    torch_dtype, use_fp16, use_bf16 = get_dtype_for_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
        print(f"Compute capability: {gpu_capability}")
        print(f"Using dtype: {torch_dtype} (fp16={use_fp16}, bf16={use_bf16})")

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    # Log model size before LoRA
    total_params_before, trainable_before = count_parameters(model)
    print(f"\nModel size (before LoRA): {total_params_before:,} params ({total_params_before/1e6:.1f}M)")

    # Apply LoRA
    if USE_LORA:
        print(f"Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Full fine-tuning (no LoRA)")

    # Log model size after LoRA
    total_params_after, trainable_after = count_parameters(model)
    print(f"Model size (after LoRA):  {total_params_after:,} params ({total_params_after/1e6:.1f}M)")
    print(f"Trainable parameters:     {trainable_after:,} ({100*trainable_after/total_params_after:.2f}%)")

    # Load dataset
    task_config = TASK_PROMPTS[TASK]
    task_data_dir = os.path.join(DATA_DIR, TASK)
    print(f"\nLoading dataset: {task_data_dir}")

    train_dataset = load_from_disk(os.path.join(task_data_dir, "train"))
    eval_dataset = load_from_disk(os.path.join(task_data_dir, "test"))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Tokenize
    train_dataset = prepare_dataset(train_dataset, task_config, tokenizer, MAX_SEQ_LEN)
    eval_tokenized = prepare_dataset(eval_dataset, task_config, tokenizer, MAX_SEQ_LEN)

    # Training arguments
    run_name = f"{TASK}_{MODEL_NAME.split('/')[-1]}_r{LORA_RANK}_lr{LEARNING_RATE}"
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, run_name),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
        seed=SEED,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_SEQ_LEN,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting fine-tuning: {run_name}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'='*60}\n")

    t_train_start = time.time()
    train_result = trainer.train()
    t_train_end = time.time()
    training_seconds = t_train_end - t_train_start

    # Log training results
    print(f"\nTraining complete in {training_seconds:.1f}s")
    print(f"  Final train loss: {train_result.training_loss:.4f}")

    # Save model
    save_path = os.path.join(CHECKPOINT_DIR, run_name)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")

    # Evaluate using the fixed eval harness
    print(f"\n{'='*60}")
    print(f"Evaluating on {TASK} test set")
    print(f"{'='*60}\n")

    eval_results = evaluate_task(
        model=model,
        tokenizer=tokenizer,
        task=TASK,
        data_dir=DATA_DIR,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # Compute final stats
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**3 * 1024 if torch.cuda.is_available() else 0
    dtype_used = "fp16" if use_fp16 else ("bf16" if use_bf16 else "fp32")

    # Print summary (this is what the agent parses)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"timestamp:        {timestamp}")
    print(f"task:             {TASK}")
    print(f"model:            {MODEL_NAME}")
    print(f"gpu:              {gpu_name}")
    print(f"dtype:            {dtype_used}")
    for metric_name, metric_value in eval_results.items():
        print(f"eval_{metric_name}:    {metric_value:.4f}")
    print(f"train_loss:       {train_result.training_loss:.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"lora_rank:        {LORA_RANK}")
    print(f"lora_alpha:       {LORA_ALPHA}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"num_epochs:       {NUM_EPOCHS}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"grad_accum_steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"effective_batch:  {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"max_seq_len:      {MAX_SEQ_LEN}")
    print(f"lr_scheduler:     {LR_SCHEDULER}")
    print(f"warmup_ratio:     {WARMUP_RATIO}")
    print(f"weight_decay:     {WEIGHT_DECAY}")
    print(f"use_lora:         {USE_LORA}")
    print(f"use_4bit:         {USE_4BIT}")
    print(f"total_params:     {total_params_after:,}")
    print(f"trainable_params: {trainable_after:,}")
    print(f"num_train_samples: {len(train_dataset)}")
    print(f"seed:             {SEED}")
    print("=" * 60)

    # Append to results.tsv
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv")
    header_needed = not os.path.exists(results_path)

    tsv_fields = {
        "timestamp": timestamp,
        "task": TASK,
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "dtype": dtype_used,
        "train_loss": f"{train_result.training_loss:.4f}",
        "peak_vram_mb": f"{peak_vram_mb:.1f}",
        "training_seconds": f"{training_seconds:.1f}",
        "total_seconds": f"{t_end - t_start:.1f}",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_seq_len": MAX_SEQ_LEN,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "use_lora": USE_LORA,
        "use_4bit": USE_4BIT,
        "seed": SEED,
    }
    # Add eval metrics
    for metric_name, metric_value in eval_results.items():
        tsv_fields[f"eval_{metric_name}"] = f"{metric_value:.4f}"

    with open(results_path, "a") as f:
        if header_needed:
            f.write("\t".join(tsv_fields.keys()) + "\n")
        f.write("\t".join(str(v) for v in tsv_fields.values()) + "\n")

    print(f"\nResults appended to: {results_path}")

    # Save JSON summary per experiment
    summary_dir = os.path.join(OUTPUT_DIR, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{run_name}_{int(time.time())}.json")

    summary = {
        "timestamp": timestamp,
        "task": TASK,
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "dtype": dtype_used,
        "eval_results": eval_results,
        "train_loss": train_result.training_loss,
        "training_seconds": training_seconds,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "hyperparameters": hyperparams,
        "total_params": total_params_after,
        "trainable_params": trainable_after,
        "num_train_samples": len(train_dataset),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
