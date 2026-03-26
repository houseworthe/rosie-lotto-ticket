"""
Phase 2: Multi-Adapter Loading & Composition Testing

Loads multiple task-specific LoRA adapters into a single base model and tests:
1. Each adapter works independently (sanity check)
2. Naive weight blending (equal weights)
3. Per-adapter evaluation on all tasks (cross-task interference matrix)

Usage:
  python multi_adapter.py                    # Run all tests
  python multi_adapter.py --verify-only      # Just verify adapters load
  python multi_adapter.py --blend-only       # Just test blending
  python multi_adapter.py --cross-eval       # Full cross-task evaluation matrix

Requires: trained LoRA checkpoints in ~/autoresearch/checkpoints/
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from eval import evaluate_task

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("AR_MODEL", "Qwen/Qwen3-0.6B")
CHECKPOINT_DIR = os.path.expanduser("~/autoresearch/checkpoints")
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output/multi_adapter")

# Tasks we have trained adapters for
# Will auto-detect from checkpoint directory
SUPPORTED_TASKS = ["trec", "trec50", "text2sql", "sst2", "agnews", "mnli", "dbpedia"]

SEED = 42

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return gpu_name, gpu_mem
    return "cpu", 0


def get_dtype_for_gpu():
    if not torch.cuda.is_available():
        return torch.float32, False, False
    capability = torch.cuda.get_device_capability(0)[0]
    if capability >= 8:
        return torch.bfloat16, False, True
    else:
        return torch.float16, True, False


def find_adapters(checkpoint_dir, model_name):
    """Auto-detect trained adapter checkpoints."""
    model_short = model_name.split("/")[-1]
    adapters = {}
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return adapters
    
    for entry in sorted(os.listdir(checkpoint_dir)):
        full_path = os.path.join(checkpoint_dir, entry)
        # Check if it's a valid PEFT adapter (has adapter_config.json)
        config_path = os.path.join(full_path, "adapter_config.json")
        if os.path.isdir(full_path) and os.path.exists(config_path):
            # Extract task name from directory name
            # Expected format: {task}_{model}_r{rank}_lr{lr}
            parts = entry.split("_")
            if len(parts) >= 2:
                task = parts[0]
                if task in SUPPORTED_TASKS:
                    # If multiple checkpoints per task, keep the latest
                    if task not in adapters:
                        adapters[task] = full_path
                        print(f"  Found adapter: {task} -> {entry}")
                    else:
                        # Compare modification times, keep newer
                        existing_mtime = os.path.getmtime(adapters[task])
                        new_mtime = os.path.getmtime(full_path)
                        if new_mtime > existing_mtime:
                            adapters[task] = full_path
                            print(f"  Found newer adapter: {task} -> {entry} (replacing)")
    
    return adapters


# ---------------------------------------------------------------------------
# Phase 2a: Verify each adapter independently
# ---------------------------------------------------------------------------

def verify_adapters(base_model, tokenizer, adapters, data_dir):
    """Load each adapter independently and verify it matches original eval scores."""
    print("\n" + "=" * 60)
    print("PHASE 2a: VERIFY INDIVIDUAL ADAPTERS")
    print("=" * 60)
    
    results = {}
    
    for task, adapter_path in adapters.items():
        print(f"\n--- Verifying adapter: {task} ---")
        print(f"    Path: {adapter_path}")
        
        # Check if we have test data for this task
        task_data_dir = os.path.join(data_dir, task)
        if not os.path.exists(os.path.join(task_data_dir, "test")):
            print(f"    SKIP: No test data found at {task_data_dir}/test")
            continue
        
        try:
            # Load adapter on top of base model
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
            
            # Evaluate
            eval_results = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                data_dir=data_dir,
                max_seq_len=256,
                batch_size=2,
            )
            
            results[task] = {
                "adapter_path": adapter_path,
                "metrics": eval_results,
                "status": "ok",
            }
            
            print(f"    Results: {eval_results}")
            
            # Unload adapter to free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results[task] = {
                "adapter_path": adapter_path,
                "error": str(e),
                "status": "error",
            }
    
    return results


# ---------------------------------------------------------------------------
# Phase 2b: Naive weight blending
# ---------------------------------------------------------------------------

def test_naive_blending(base_model, tokenizer, adapters, data_dir):
    """Load multiple adapters and test naive equal-weight blending."""
    print("\n" + "=" * 60)
    print("PHASE 2b: NAIVE WEIGHT BLENDING")
    print("=" * 60)
    
    if len(adapters) < 2:
        print("Need at least 2 adapters for blending. Skipping.")
        return {}
    
    adapter_names = list(adapters.keys())
    adapter_paths = list(adapters.values())
    
    print(f"\nBlending {len(adapter_names)} adapters: {adapter_names}")
    
    # Load first adapter
    print(f"\nLoading base adapter: {adapter_names[0]}")
    model = PeftModel.from_pretrained(
        base_model, 
        adapter_paths[0],
        adapter_name=adapter_names[0],
    )
    
    # Load remaining adapters
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        print(f"Loading adapter: {name}")
        model.load_adapter(path, adapter_name=name)
    
    # Create blended adapter with equal weights
    print(f"\nCreating blended adapter with equal weights ({1.0/len(adapter_names):.3f} each)")
    
    blend_weights = [1.0 / len(adapter_names)] * len(adapter_names)
    
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=blend_weights,
        adapter_name="blended_equal",
        combination_type="linear",
    )
    
    # Activate blended adapter
    model.set_adapter("blended_equal")
    model.eval()
    
    # Evaluate blended model on all tasks that have test data
    results = {}
    for task in adapter_names:
        task_data_dir = os.path.join(data_dir, task)
        if not os.path.exists(os.path.join(task_data_dir, "test")):
            print(f"\n  SKIP {task}: No test data")
            continue
        
        print(f"\n--- Evaluating blended model on: {task} ---")
        try:
            eval_results = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                data_dir=data_dir,
                max_seq_len=256,
                batch_size=2,
            )
            results[task] = {
                "metrics": eval_results,
                "blend_weights": dict(zip(adapter_names, blend_weights)),
                "status": "ok",
            }
            print(f"  Blended results: {eval_results}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[task] = {"error": str(e), "status": "error"}
    
    # Also test individual adapters through the multi-adapter model
    # (sanity check that loading multiple doesn't corrupt any)
    print("\n--- Sanity check: individual adapters after multi-load ---")
    sanity_results = {}
    for name in adapter_names:
        task_data_dir = os.path.join(data_dir, name)
        if not os.path.exists(os.path.join(task_data_dir, "test")):
            continue
        
        model.set_adapter(name)
        model.eval()
        
        try:
            eval_results = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=name,
                data_dir=data_dir,
                max_seq_len=256,
                batch_size=2,
            )
            sanity_results[name] = eval_results
            print(f"  {name} (individual after multi-load): {eval_results}")
        except Exception as e:
            print(f"  {name} ERROR: {e}")
            sanity_results[name] = {"error": str(e)}
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "blended": results,
        "sanity_individual": sanity_results,
    }


# ---------------------------------------------------------------------------
# Phase 2c: Cross-task evaluation matrix
# ---------------------------------------------------------------------------

def cross_task_eval(base_model, tokenizer, adapters, data_dir):
    """Evaluate each adapter on ALL tasks to measure cross-task interference."""
    print("\n" + "=" * 60)
    print("PHASE 2c: CROSS-TASK EVALUATION MATRIX")
    print("=" * 60)
    print("Each row = adapter trained on task X")
    print("Each col = evaluated on task Y")
    print("Diagonal = expected best performance")
    
    matrix = {}
    
    for train_task, adapter_path in adapters.items():
        print(f"\n--- Adapter trained on: {train_task} ---")
        
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        except Exception as e:
            print(f"  ERROR loading adapter: {e}")
            matrix[train_task] = {"error": str(e)}
            continue
        
        matrix[train_task] = {}
        
        for eval_task in adapters.keys():
            task_data_dir = os.path.join(data_dir, eval_task)
            if not os.path.exists(os.path.join(task_data_dir, "test")):
                print(f"  -> {eval_task}: SKIP (no test data)")
                continue
            
            try:
                eval_results = evaluate_task(
                    model=model,
                    tokenizer=tokenizer,
                    task=eval_task,
                    data_dir=data_dir,
                    max_seq_len=256,
                    batch_size=2,
                )
                matrix[train_task][eval_task] = eval_results
                
                marker = " <-- on-task" if train_task == eval_task else ""
                primary_metric = list(eval_results.values())[0]
                print(f"  -> {eval_task}: {primary_metric:.4f}{marker}")
                
            except Exception as e:
                print(f"  -> {eval_task}: ERROR ({e})")
                matrix[train_task][eval_task] = {"error": str(e)}
        
        del model
        torch.cuda.empty_cache()
    
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Adapter Loading")
    parser.add_argument("--verify-only", action="store_true", help="Only verify individual adapters")
    parser.add_argument("--blend-only", action="store_true", help="Only test naive blending")
    parser.add_argument("--cross-eval", action="store_true", help="Only run cross-task evaluation")
    args = parser.parse_args()
    
    # Default: run everything
    run_all = not (args.verify_only or args.blend_only or args.cross_eval)
    
    t_start = time.time()
    timestamp = datetime.datetime.now().isoformat()
    
    # Setup
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    gpu_name, gpu_mem = get_gpu_info()
    torch_dtype, use_fp16, use_bf16 = get_dtype_for_gpu()
    
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Dtype: {torch_dtype}")
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    
    # Find all trained adapters
    print(f"\nScanning for adapters...")
    adapters = find_adapters(CHECKPOINT_DIR, MODEL_NAME)
    
    if not adapters:
        print("\nERROR: No trained adapters found!")
        print(f"Expected checkpoints in: {CHECKPOINT_DIR}")
        print("Run finetune.py for each task first (Phase 1).")
        sys.exit(1)
    
    print(f"\nFound {len(adapters)} adapters: {list(adapters.keys())}")
    
    # Load base model (shared across all tests)
    print(f"\nLoading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Results collector
    all_results = {
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "adapters_found": list(adapters.keys()),
        "adapter_paths": adapters,
    }
    
    # Phase 2a: Verify individual adapters
    if run_all or args.verify_only:
        verify_results = verify_adapters(base_model, tokenizer, adapters, DATA_DIR)
        all_results["verify_individual"] = verify_results
    
    # Phase 2b: Naive weight blending
    if run_all or args.blend_only:
        blend_results = test_naive_blending(base_model, tokenizer, adapters, DATA_DIR)
        all_results["naive_blending"] = blend_results
    
    # Phase 2c: Cross-task evaluation matrix
    if run_all or args.cross_eval:
        matrix = cross_task_eval(base_model, tokenizer, adapters, DATA_DIR)
        all_results["cross_task_matrix"] = matrix
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, f"phase2_results_{int(time.time())}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    t_end = time.time()
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"Adapters tested: {list(adapters.keys())}")
    print(f"Total time: {t_end - t_start:.1f}s")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"Results saved: {results_path}")
    
    # Print comparison table if we have both individual and blended results
    if "verify_individual" in all_results and "naive_blending" in all_results:
        blended = all_results["naive_blending"].get("blended", {})
        individual = all_results["verify_individual"]
        
        print("\n--- Individual vs Blended ---")
        print(f"{'Task':<12} {'Individual':<15} {'Blended':<15} {'Delta':<10}")
        print("-" * 52)
        
        for task in adapters.keys():
            ind_metrics = individual.get(task, {}).get("metrics", {})
            bld_metrics = blended.get(task, {}).get("metrics", {})
            
            if ind_metrics and bld_metrics:
                ind_val = list(ind_metrics.values())[0]
                bld_val = list(bld_metrics.values())[0]
                delta = bld_val - ind_val
                sign = "+" if delta >= 0 else ""
                print(f"{task:<12} {ind_val:<15.4f} {bld_val:<15.4f} {sign}{delta:<10.4f}")
            elif ind_metrics:
                ind_val = list(ind_metrics.values())[0]
                print(f"{task:<12} {ind_val:<15.4f} {'N/A':<15} {'N/A':<10}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
