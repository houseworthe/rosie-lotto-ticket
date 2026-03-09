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
import argparse
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


def prepare_dataset(dataset, task_config, tokenizer, max_len):
    """Tokenize and prepare dataset for training."""

    def tokenize_fn(examples):
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            ex = {k: examples[k][i] for k in examples.keys()}
            text = format_example(ex, task_config, tokenizer)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        # For causal LM, labels = input_ids (shifted internally by the model)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]

        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

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
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

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
        bf16=torch.cuda.is_available(),
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
    print(f"{'='*60}\n")

    t_train_start = time.time()
    trainer.train()
    t_train_end = time.time()
    training_seconds = t_train_end - t_train_start

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

    # Print summary (this is what the agent parses)
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"task:             {TASK}")
    print(f"model:            {MODEL_NAME}")
    for metric_name, metric_value in eval_results.items():
        print(f"eval_{metric_name}:    {metric_value:.4f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"lora_rank:        {LORA_RANK}")
    print(f"lora_alpha:       {LORA_ALPHA}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"num_epochs:       {NUM_EPOCHS}")
    print(f"batch_size:       {BATCH_SIZE}")
    print(f"max_seq_len:      {MAX_SEQ_LEN}")
    print(f"use_4bit:         {USE_4BIT}")
    print(f"num_train_samples: {len(train_dataset)}")
    print(f"seed:             {SEED}")


if __name__ == "__main__":
    main()
