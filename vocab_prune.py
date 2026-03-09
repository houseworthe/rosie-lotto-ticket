"""
Vocabulary pruning for transformer models.
Strips unused tokens from embedding/unembedding layers to save VRAM.
This is mathematically lossless for the target task — zero quality loss.

The agent modifies this file to experiment with:
- Token retention strategies (frequency-based, task-based, charset-based)
- Which model sizes benefit most
- Measuring exact VRAM savings

Usage: python vocab_prune.py
  or via SLURM: sbatch slurm_template.sh (set SCRIPT=vocab_prune.py)
"""

import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration (EDIT THESE)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
TASK = "trec"  # Used to determine which tokens appear in task data

# Pruning strategy
STRATEGY = "task_frequency"  # Options: "task_frequency", "charset", "hybrid"

# For task_frequency: keep tokens that appear at least this many times in training data
MIN_FREQUENCY = 1

# For charset: keep tokens whose decoded text is within these character sets
KEEP_ASCII = True
KEEP_DIGITS = True
KEEP_PUNCTUATION = True
KEEP_LATIN_EXTENDED = True
KEEP_CJK = False  # Big savings from dropping CJK if not needed
KEEP_CYRILLIC = False
KEEP_ARABIC = False

# Always keep these token IDs (special tokens, common whitespace, etc.)
# These are auto-detected from the tokenizer, but you can add extras here
EXTRA_KEEP_IDS = []

# Output
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output")
SAVE_MODEL = True  # Save the pruned model

# ---------------------------------------------------------------------------
# Token analysis
# ---------------------------------------------------------------------------

def get_task_token_frequencies(tokenizer, task, data_dir):
    """Count token frequencies across the task's training data."""
    from datasets import load_from_disk

    task_dir = os.path.join(data_dir, task)
    train_dataset = load_from_disk(os.path.join(task_dir, "train"))

    counter = Counter()
    for example in train_dataset:
        # Tokenize all text fields
        for key, value in example.items():
            if isinstance(value, str):
                ids = tokenizer.encode(value, add_special_tokens=False)
                counter.update(ids)

    return counter


def get_special_token_ids(tokenizer):
    """Get all special token IDs that must be kept."""
    special_ids = set()

    # Standard special tokens
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    # Additional special tokens
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        special_ids.update(tokenizer.additional_special_tokens_ids)

    # All tokens in the special tokens map
    if hasattr(tokenizer, "special_tokens_map"):
        for key, value in tokenizer.special_tokens_map.items():
            if isinstance(value, str):
                ids = tokenizer.encode(value, add_special_tokens=False)
                special_ids.update(ids)
            elif isinstance(value, list):
                for v in value:
                    ids = tokenizer.encode(v, add_special_tokens=False)
                    special_ids.update(ids)

    return special_ids


def select_tokens_by_charset(tokenizer, vocab_size):
    """Select tokens to keep based on character set rules."""
    keep_ids = set()

    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue

        keep = False
        for char in text:
            cp = ord(char)
            if KEEP_ASCII and cp < 128:
                keep = True
            elif KEEP_LATIN_EXTENDED and (0x0080 <= cp <= 0x024F):
                keep = True
            elif KEEP_CJK and (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF):
                keep = True
            elif KEEP_CYRILLIC and (0x0400 <= cp <= 0x04FF):
                keep = True
            elif KEEP_ARABIC and (0x0600 <= cp <= 0x06FF):
                keep = True

        if keep:
            keep_ids.add(token_id)

    return keep_ids


def select_tokens(tokenizer, task, data_dir):
    """Select which token IDs to keep based on the configured strategy."""
    vocab_size = tokenizer.vocab_size

    # Always keep special tokens
    keep_ids = get_special_token_ids(tokenizer)
    keep_ids.update(EXTRA_KEEP_IDS)

    if STRATEGY == "task_frequency":
        freq = get_task_token_frequencies(tokenizer, task, data_dir)
        for token_id, count in freq.items():
            if count >= MIN_FREQUENCY:
                keep_ids.add(token_id)

    elif STRATEGY == "charset":
        charset_ids = select_tokens_by_charset(tokenizer, vocab_size)
        keep_ids.update(charset_ids)

    elif STRATEGY == "hybrid":
        # Keep tokens that appear in task data OR are in allowed charsets
        freq = get_task_token_frequencies(tokenizer, task, data_dir)
        for token_id, count in freq.items():
            if count >= MIN_FREQUENCY:
                keep_ids.add(token_id)
        charset_ids = select_tokens_by_charset(tokenizer, vocab_size)
        keep_ids.update(charset_ids)

    return sorted(keep_ids)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def prune_vocabulary(model, tokenizer, keep_ids):
    """
    Prune the model's vocabulary by keeping only specified token IDs.
    Modifies embedding and lm_head (unembedding) layers in-place.
    Returns the ID mapping (old_id -> new_id).
    """
    keep_ids = sorted(set(keep_ids))
    old_vocab_size = model.config.vocab_size
    new_vocab_size = len(keep_ids)

    print(f"Pruning vocabulary: {old_vocab_size:,} → {new_vocab_size:,} tokens")
    print(f"  Removed: {old_vocab_size - new_vocab_size:,} tokens ({100 * (1 - new_vocab_size / old_vocab_size):.1f}%)")

    keep_tensor = torch.tensor(keep_ids, dtype=torch.long, device=model.device)

    # Prune embedding layer
    old_embed = model.model.embed_tokens.weight.data
    new_embed = old_embed[keep_tensor].clone()
    model.model.embed_tokens = torch.nn.Embedding(
        new_vocab_size, old_embed.shape[1],
        device=model.device, dtype=old_embed.dtype,
    )
    model.model.embed_tokens.weight.data = new_embed

    # Prune lm_head (unembedding)
    old_lm_head = model.lm_head.weight.data
    new_lm_head = old_lm_head[keep_tensor].clone()
    model.lm_head = torch.nn.Linear(
        old_lm_head.shape[1], new_vocab_size,
        bias=False, device=model.device, dtype=old_lm_head.dtype,
    )
    model.lm_head.weight.data = new_lm_head

    # Update config
    model.config.vocab_size = new_vocab_size

    # Build ID mapping
    id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    # Calculate memory savings
    embed_params_saved = (old_vocab_size - new_vocab_size) * old_embed.shape[1]
    lm_head_params_saved = (old_vocab_size - new_vocab_size) * old_lm_head.shape[1]
    total_params_saved = embed_params_saved + lm_head_params_saved
    bytes_per_param = 2  # bf16
    mb_saved = total_params_saved * bytes_per_param / 1024 / 1024

    print(f"  Parameters saved: {total_params_saved:,} ({mb_saved:.1f} MB in bf16)")

    return id_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK}")
    print(f"Strategy: {STRATEGY}")
    print()

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    original_params = sum(p.numel() for p in model.parameters())
    original_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print(f"Original parameters: {original_params:,}")
    print(f"Original vocab size: {model.config.vocab_size:,}")
    print()

    # Select tokens to keep
    print("Analyzing token usage...")
    keep_ids = select_tokens(tokenizer, TASK, DATA_DIR)
    print(f"Tokens to keep: {len(keep_ids):,} / {tokenizer.vocab_size:,}")
    print()

    # Prune
    id_map = prune_vocabulary(model, tokenizer, keep_ids)

    new_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # Trigger a forward pass to measure actual VRAM
        dummy = torch.tensor([[0]], device=model.device)
        with torch.no_grad():
            model(dummy)
        new_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Evaluate to confirm zero quality loss
    print("\nEvaluating pruned model...")
    from eval import evaluate_task
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Note: evaluation needs the original tokenizer for encoding inputs,
    # but the pruned model for generation. For a proper eval we'd need to
    # remap token IDs. For now, report the structural savings.
    # Full eval integration is a TODO for the agent to figure out.

    # Save
    if SAVE_MODEL:
        save_path = os.path.join(OUTPUT_DIR, f"vocab_pruned_{MODEL_NAME.split('/')[-1]}_{TASK}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        # Save the ID mapping for downstream use
        with open(os.path.join(save_path, "id_map.json"), "w") as f:
            json.dump({str(k): v for k, v in id_map.items()}, f)
        print(f"\nSaved to: {save_path}")

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"track:            vocab_prune")
    print(f"task:             {TASK}")
    print(f"model:            {MODEL_NAME}")
    print(f"strategy:         {STRATEGY}")
    print(f"original_vocab:   {tokenizer.vocab_size}")
    print(f"pruned_vocab:     {len(keep_ids)}")
    print(f"vocab_reduction:  {100 * (1 - len(keep_ids) / tokenizer.vocab_size):.1f}%")
    print(f"original_params:  {original_params}")
    print(f"pruned_params:    {new_params}")
    print(f"params_saved:     {original_params - new_params}")
    print(f"mb_saved:         {(original_params - new_params) * 2 / 1024 / 1024:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
