"""
Combined pipeline orchestrator. Runs multiple optimization steps in sequence
and measures cumulative quality-per-VRAM.

The agent modifies this file to experiment with:
- Different orderings of techniques
- Which combinations yield best quality/VRAM tradeoff
- Whether fine-tuning can recover quality lost from pruning

Usage: python pipeline.py
  or via SLURM: sbatch slurm_template.sh (set SCRIPT=pipeline.py)
"""

import os
import json
import time
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration (EDIT THESE)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
TASK = "trec"

# Pipeline steps — ordered list of operations to apply
# Options: "vocab_prune", "layer_prune", "finetune", "quantize"
PIPELINE = ["vocab_prune", "finetune"]

# Vocab pruning settings
VOCAB_STRATEGY = "task_frequency"
VOCAB_MIN_FREQUENCY = 1

# Layer pruning settings
LAYER_STRATEGY = "importance"
LAYER_DROP_FRACTION = 0.25

# Fine-tuning settings (LoRA)
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

# Quantization settings
QUANTIZE_BITS = 4  # 4-bit or 8-bit

# Paths
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output")
MAX_SEQ_LEN = 256

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_vocab_prune(model, tokenizer, state):
    """Apply vocabulary pruning."""
    from vocab_prune import (
        select_tokens, prune_vocabulary,
        KEEP_ASCII, KEEP_DIGITS, KEEP_PUNCTUATION, KEEP_LATIN_EXTENDED,
    )

    # Override vocab_prune settings
    import vocab_prune as vp
    vp.STRATEGY = VOCAB_STRATEGY
    vp.MIN_FREQUENCY = VOCAB_MIN_FREQUENCY
    vp.TASK = TASK

    print("\n--- Step: Vocabulary Pruning ---")
    keep_ids = select_tokens(tokenizer, TASK, DATA_DIR)
    id_map = prune_vocabulary(model, tokenizer, keep_ids)

    state["vocab_pruned"] = True
    state["vocab_original"] = tokenizer.vocab_size
    state["vocab_pruned_size"] = len(keep_ids)
    state["id_map"] = id_map

    # Wrap tokenizer so all subsequent steps use remapped IDs
    tokenizer = RemappedTokenizerWrapper(tokenizer, id_map)
    print(f"  Tokenizer wrapped with ID remapping ({len(id_map)} tokens)")

    return model, tokenizer, state


def step_layer_prune(model, tokenizer, state):
    """Apply structured layer pruning."""
    from layer_prune import select_layers_to_drop, prune_layers

    # Override layer_prune settings
    import layer_prune as lp
    lp.LAYER_STRATEGY = LAYER_STRATEGY
    lp.LAYER_DROP_FRACTION = LAYER_DROP_FRACTION
    lp.TASK = TASK

    print("\n--- Step: Layer Pruning ---")
    drop_indices = select_layers_to_drop(model, tokenizer, DATA_DIR, TASK)
    model = prune_layers(model, drop_indices)

    state["layer_pruned"] = True
    state["layers_dropped"] = drop_indices
    state["layers_remaining"] = model.config.num_hidden_layers

    return model, tokenizer, state


def step_finetune(model, tokenizer, state):
    """Apply LoRA fine-tuning."""
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_from_disk
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from finetune import TASK_PROMPTS, prepare_dataset

    print("\n--- Step: Fine-Tuning (LoRA) ---")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    task_config = TASK_PROMPTS[TASK]
    train_dataset = load_from_disk(os.path.join(DATA_DIR, TASK, "train"))
    train_dataset = prepare_dataset(train_dataset, task_config, tokenizer, MAX_SEQ_LEN)

    # If vocab was pruned, remap token IDs in training data
    id_map = state.get("id_map")
    if id_map:
        print("  Remapping training data token IDs for pruned vocabulary...")
        def remap_ids(example):
            example["input_ids"] = [id_map.get(tid, 0) for tid in example["input_ids"]]
            example["labels"] = [id_map.get(tid, -100) if tid != -100 else -100 for tid in example["labels"]]
            return example
        train_dataset = train_dataset.map(remap_ids)

    # Train
    run_name = f"pipeline_{'-'.join(PIPELINE)}_{MODEL_NAME.split('/')[-1]}"
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, run_name),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=4,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=MAX_SEQ_LEN),
    )

    trainer.train()

    # Merge LoRA weights back into base model for clean eval
    model = model.merge_and_unload()

    state["finetuned"] = True
    state["lora_rank"] = LORA_RANK
    state["num_epochs"] = NUM_EPOCHS

    return model, tokenizer, state


def step_quantize(model, tokenizer, state):
    """Apply post-training quantization."""
    print(f"\n--- Step: Quantization ({QUANTIZE_BITS}-bit) ---")

    # Simple dynamic quantization for measurement purposes
    # For production, use GPTQ or AWQ
    if QUANTIZE_BITS == 8:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    elif QUANTIZE_BITS == 4:
        # 4-bit requires bitsandbytes or GPTQ
        # For now, just report what WOULD happen
        print("  4-bit quantization requires bitsandbytes (load model with load_in_4bit=True)")
        print("  Estimated VRAM savings: ~50% of model weights")

    state["quantized"] = True
    state["quantize_bits"] = QUANTIZE_BITS

    return model, tokenizer, state


class RemappedTokenizerWrapper:
    """Wraps a tokenizer to remap token IDs after vocabulary pruning."""

    def __init__(self, tokenizer, id_map):
        self._tokenizer = tokenizer
        self._id_map = id_map  # old_id -> new_id
        self._reverse_map = {v: k for k, v in id_map.items()}  # new_id -> old_id
        # Update special token IDs
        self.pad_token_id = id_map.get(tokenizer.pad_token_id, 0)
        self.eos_token_id = id_map.get(tokenizer.eos_token_id, 0)
        self.bos_token_id = id_map.get(tokenizer.bos_token_id, 0) if tokenizer.bos_token_id is not None else None
        self.vocab_size = len(id_map)

    def __call__(self, *args, **kwargs):
        result = self._tokenizer(*args, **kwargs)
        # Remap input_ids
        remapped = []
        for ids in result["input_ids"]:
            if hasattr(ids, 'tolist'):
                ids_list = ids.tolist()
            else:
                ids_list = list(ids)
            remapped.append([self._id_map.get(tid, 0) for tid in ids_list])
        import torch as _torch
        result["input_ids"] = _torch.tensor(remapped, device=result["input_ids"].device if hasattr(result["input_ids"], 'device') else 'cpu')
        return result

    def decode(self, ids, **kwargs):
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        old_ids = [self._reverse_map.get(tid, tid) for tid in ids]
        return self._tokenizer.decode(old_ids, **kwargs)

    def encode(self, *args, **kwargs):
        ids = self._tokenizer.encode(*args, **kwargs)
        return [self._id_map.get(tid, 0) for tid in ids]

    def apply_chat_template(self, *args, **kwargs):
        return self._tokenizer.apply_chat_template(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


STEP_FUNCTIONS = {
    "vocab_prune": step_vocab_prune,
    "layer_prune": step_layer_prune,
    "finetune": step_finetune,
    "quantize": step_quantize,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK}")
    print(f"Pipeline: {' → '.join(PIPELINE)}")
    print()

    # Load base model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )

    original_params = sum(p.numel() for p in model.parameters())
    state = {"original_params": original_params}

    # Evaluate baseline
    print("\nBaseline evaluation...")
    from eval import evaluate_task
    try:
        baseline_results = evaluate_task(
            model=model, tokenizer=tokenizer, task=TASK,
            data_dir=DATA_DIR, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
        )
        state["baseline"] = baseline_results
    except Exception as e:
        print(f"  Baseline eval failed: {e}")
        baseline_results = {}

    # Run pipeline steps
    for step_name in PIPELINE:
        if step_name not in STEP_FUNCTIONS:
            print(f"  Unknown step: {step_name}, skipping")
            continue
        step_fn = STEP_FUNCTIONS[step_name]
        model, tokenizer, state = step_fn(model, tokenizer, state)

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final_params = sum(p.numel() for p in model.parameters())
    try:
        final_results = evaluate_task(
            model=model, tokenizer=tokenizer, task=TASK,
            data_dir=DATA_DIR, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"  Final eval failed: {e}")
        final_results = {"error": str(e)}

    # Save
    save_path = os.path.join(OUTPUT_DIR, f"pipeline_{'_'.join(PIPELINE)}_{MODEL_NAME.split('/')[-1]}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print("\n---")
    print(f"track:            pipeline")
    print(f"pipeline:         {' -> '.join(PIPELINE)}")
    print(f"task:             {TASK}")
    print(f"model:            {MODEL_NAME}")
    print(f"original_params:  {original_params}")
    print(f"final_params:     {final_params}")
    print(f"param_reduction:  {100 * (1 - final_params / original_params):.1f}%")
    for name, value in baseline_results.items():
        if isinstance(value, float):
            print(f"baseline_{name}:   {value:.4f}")
    for name, value in final_results.items():
        if isinstance(value, float):
            print(f"eval_{name}:       {value:.4f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")


if __name__ == "__main__":
    main()
