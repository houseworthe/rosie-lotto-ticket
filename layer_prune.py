"""
Structured layer/width pruning for transformer models.
Surgical inter-size pruning: prune a larger model to fill gaps in the model family.

Example: Prune Qwen3-27B → ~15-20B to get better-than-9B quality at fraction of 27B cost.

The agent modifies this file to experiment with:
- Which layers to drop (first, last, middle, importance-based)
- How many layers to drop
- Width pruning (reduce hidden dim)
- Combinations of layer + width pruning

Usage: python layer_prune.py
  or via SLURM: sbatch slurm_template.sh (set SCRIPT=layer_prune.py)
"""

import os
import json
import time
import math

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Configuration (EDIT THESE)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-4B"  # Source model to prune

# Layer pruning
PRUNE_LAYERS = True
LAYER_STRATEGY = "importance"  # Options: "uniform", "last", "middle", "importance"
LAYER_DROP_FRACTION = 0.25     # Fraction of layers to remove (0.25 = drop 25%)
# For "uniform": drop every Nth layer
# For "last": drop the last N layers (before final norm)
# For "middle": drop from the middle
# For "importance": estimate layer importance and drop least important

# Width pruning (reduce hidden dimension)
PRUNE_WIDTH = False
WIDTH_TARGET_RATIO = 0.75  # Keep this fraction of hidden dim (0.75 = 25% reduction)

# Evaluation
TASK = "trec"
EVAL_AFTER_PRUNE = True

# Paths
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output")

# Calibration data (for importance-based pruning)
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 512

# ---------------------------------------------------------------------------
# Layer importance estimation
# ---------------------------------------------------------------------------

def estimate_layer_importance(model, tokenizer, data_dir, task, num_samples=128):
    """
    Estimate each layer's importance by measuring perplexity change when skipped.
    Returns list of (layer_idx, importance_score) sorted by importance (ascending).
    """
    from datasets import load_from_disk

    print("Estimating layer importance (this takes a few minutes)...")
    dataset = load_from_disk(os.path.join(data_dir, task, "train"))

    # Prepare calibration texts
    texts = []
    for ex in dataset:
        for key, val in ex.items():
            if isinstance(val, str) and len(val) > 20:
                texts.append(val)
                if len(texts) >= num_samples:
                    break
        if len(texts) >= num_samples:
            break

    # Tokenize
    encodings = tokenizer(
        texts[:num_samples],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    ).to(model.device)

    # Get baseline loss
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        baseline_loss = outputs.loss.item()

    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Measure importance of each layer by ablation
    num_layers = model.config.num_hidden_layers
    importance_scores = []

    for layer_idx in range(num_layers):
        # Hook to skip this layer
        layer = model.model.layers[layer_idx]
        original_forward = layer.forward

        def skip_forward(*args, **kwargs):
            # Return the input unchanged (skip the layer)
            if args:
                return (args[0],) + args[1:] if len(args) > 1 else (args[0],)
            return kwargs.get("hidden_states", None)

        # Monkey-patch
        layer.forward = lambda *a, _orig=original_forward, **kw: (a[0],) + ((None,) * (len(a) - 1)) if len(a) > 1 else (a[0],)

        try:
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings["input_ids"])
                ablated_loss = outputs.loss.item()
        except Exception as e:
            ablated_loss = float("inf")

        # Restore
        layer.forward = original_forward

        importance = ablated_loss - baseline_loss  # Higher = more important
        importance_scores.append((layer_idx, importance))
        print(f"  Layer {layer_idx:3d}: loss change = {importance:+.4f}")

    # Sort by importance (ascending = least important first)
    importance_scores.sort(key=lambda x: x[1])
    return importance_scores


def select_layers_to_drop(model, tokenizer, data_dir, task):
    """Select which layer indices to drop based on the configured strategy."""
    num_layers = model.config.num_hidden_layers
    num_to_drop = max(1, int(num_layers * LAYER_DROP_FRACTION))

    print(f"Selecting {num_to_drop}/{num_layers} layers to drop (strategy: {LAYER_STRATEGY})")

    if LAYER_STRATEGY == "uniform":
        # Drop every Nth layer
        step = num_layers // num_to_drop
        drop_indices = list(range(step - 1, num_layers, step))[:num_to_drop]

    elif LAYER_STRATEGY == "last":
        # Drop the last N layers (before final layer norm)
        drop_indices = list(range(num_layers - num_to_drop, num_layers))

    elif LAYER_STRATEGY == "middle":
        # Drop from the middle
        start = (num_layers - num_to_drop) // 2
        drop_indices = list(range(start, start + num_to_drop))

    elif LAYER_STRATEGY == "importance":
        # Drop least important layers
        scores = estimate_layer_importance(
            model, tokenizer, data_dir, task, NUM_CALIBRATION_SAMPLES
        )
        drop_indices = [idx for idx, _ in scores[:num_to_drop]]

    else:
        raise ValueError(f"Unknown strategy: {LAYER_STRATEGY}")

    print(f"  Dropping layers: {sorted(drop_indices)}")
    return sorted(drop_indices)


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def prune_layers(model, drop_indices):
    """Remove transformer layers at the specified indices."""
    num_layers = model.config.num_hidden_layers
    keep_indices = [i for i in range(num_layers) if i not in drop_indices]

    print(f"Pruning layers: {num_layers} → {len(keep_indices)}")

    # Create new layer list with only kept layers
    new_layers = torch.nn.ModuleList([model.model.layers[i] for i in keep_indices])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(keep_indices)

    # Free old layers
    torch.cuda.empty_cache()

    return model


def prune_width(model, target_ratio):
    """
    Reduce the hidden dimension of the model.
    This is more invasive — requires modifying every linear layer.
    Uses magnitude-based pruning to select which dimensions to keep.
    """
    hidden_size = model.config.hidden_size
    new_hidden_size = int(hidden_size * target_ratio)
    # Round to nearest multiple of 128 for efficiency
    new_hidden_size = max(128, (new_hidden_size // 128) * 128)

    print(f"Pruning width: {hidden_size} → {new_hidden_size}")

    # Compute importance of each hidden dimension using embedding norms
    embed_weight = model.model.embed_tokens.weight.data  # [vocab, hidden]
    dim_importance = embed_weight.float().abs().mean(dim=0)  # [hidden]

    # Keep the most important dimensions
    _, keep_dims = torch.topk(dim_importance, new_hidden_size)
    keep_dims = keep_dims.sort().values

    # This is complex surgery — for now, just handle the embedding layers
    # Full width pruning of attention/MLP layers is a research problem
    # the agent can iterate on

    # Prune embeddings
    old_embed = model.model.embed_tokens.weight.data
    model.model.embed_tokens = torch.nn.Embedding(
        old_embed.shape[0], new_hidden_size,
        device=model.device, dtype=old_embed.dtype,
    )
    model.model.embed_tokens.weight.data = old_embed[:, keep_dims].clone()

    # Prune lm_head
    old_lm = model.lm_head.weight.data
    model.lm_head = torch.nn.Linear(
        new_hidden_size, old_lm.shape[0],
        bias=False, device=model.device, dtype=old_lm.dtype,
    )
    model.lm_head.weight.data = old_lm[:, keep_dims].clone()

    # TODO: Prune attention and MLP layers (agent should iterate on this)
    # For each layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # Need to handle input dim vs output dim correctly

    print(f"  WARNING: Width pruning currently only handles embed/lm_head.")
    print(f"  Attention/MLP layers need matching — model will not run correctly yet.")
    print(f"  The agent should implement full width pruning.")

    model.config.hidden_size = new_hidden_size
    return model, keep_dims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK}")
    print(f"Layer pruning: {PRUNE_LAYERS} (strategy={LAYER_STRATEGY}, drop={LAYER_DROP_FRACTION})")
    print(f"Width pruning: {PRUNE_WIDTH} (target_ratio={WIDTH_TARGET_RATIO})")
    print()

    # Load
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    original_params = sum(p.numel() for p in model.parameters())
    original_layers = model.config.num_hidden_layers
    original_hidden = model.config.hidden_size
    print(f"Original: {original_params:,} params, {original_layers} layers, hidden={original_hidden}")
    print()

    # Layer pruning
    if PRUNE_LAYERS:
        drop_indices = select_layers_to_drop(model, tokenizer, DATA_DIR, TASK)
        model = prune_layers(model, drop_indices)
        print()

    # Width pruning
    keep_dims = None
    if PRUNE_WIDTH:
        model, keep_dims = prune_width(model, WIDTH_TARGET_RATIO)
        print()

    pruned_params = sum(p.numel() for p in model.parameters())

    # Evaluate
    eval_results = {}
    if EVAL_AFTER_PRUNE:
        print("Evaluating pruned model...")
        from eval import evaluate_task
        try:
            eval_results = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=TASK,
                data_dir=DATA_DIR,
                max_seq_len=MAX_SEQ_LEN,
                batch_size=4,
            )
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            eval_results = {"error": str(e)}

    # Save
    save_path = os.path.join(
        OUTPUT_DIR,
        f"layer_pruned_{MODEL_NAME.split('/')[-1]}_{LAYER_STRATEGY}_{LAYER_DROP_FRACTION}"
    )
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    meta = {
        "source_model": MODEL_NAME,
        "strategy": LAYER_STRATEGY,
        "drop_fraction": LAYER_DROP_FRACTION,
        "dropped_layers": drop_indices if PRUNE_LAYERS else [],
        "width_pruning": PRUNE_WIDTH,
        "width_ratio": WIDTH_TARGET_RATIO if PRUNE_WIDTH else 1.0,
    }
    with open(os.path.join(save_path, "prune_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {save_path}")

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"track:            layer_prune")
    print(f"task:             {TASK}")
    print(f"model:            {MODEL_NAME}")
    print(f"strategy:         {LAYER_STRATEGY}")
    print(f"drop_fraction:    {LAYER_DROP_FRACTION}")
    print(f"original_layers:  {original_layers}")
    print(f"pruned_layers:    {model.config.num_hidden_layers}")
    print(f"original_params:  {original_params}")
    print(f"pruned_params:    {pruned_params}")
    print(f"param_reduction:  {100 * (1 - pruned_params / original_params):.1f}%")
    for name, value in eval_results.items():
        if isinstance(value, float):
            print(f"eval_{name}:       {value:.4f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
