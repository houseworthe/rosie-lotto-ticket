"""
Evaluation harness for fine-tuned models. DO NOT MODIFY.
This is the fixed metric — the ground truth for all experiments.

Supports: TREC classification, Text2SQL, E-commerce classification.
"""

import os
import re
import torch
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------------------------------------------------------
# Task-specific evaluation
# ---------------------------------------------------------------------------

TREC_LABELS = ["ABBREVIATION", "ENTITY", "DESCRIPTION", "HUMAN", "LOCATION", "NUMERIC"]

TASK_PROMPTS = {
    "trec": {
        "system": "You are a question classifier. Classify the given question into one of these categories: ABBREVIATION, ENTITY, DESCRIPTION, HUMAN, LOCATION, NUMERIC.",
        "input_template": "Classify this question: {text}",
    },
    "text2sql": {
        "system": "You are a SQL expert. Given a natural language question and database schema, generate the correct SQL query.",
        "input_template": "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:",
    },
    "ecommerce": {
        "system": "You are a product classifier. Classify the given product description into the correct category.",
        "input_template": "Classify this product: {text}",
    },
}


def generate_predictions(model, tokenizer, prompts, max_new_tokens=128, batch_size=4):
    """Generate predictions for a list of prompts."""
    model.eval()
    predictions = []

    # Left-pad for batch generation with causal LMs (critical for correct results)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            predictions.append(generated)

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return predictions


def build_prompts(dataset, task_config, tokenizer):
    """Build inference prompts (no answer) for evaluation."""
    prompts = []
    for example in dataset:
        system_msg = task_config["system"]
        input_text = task_config["input_template"].format(**example)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": input_text},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<|system|>{system_msg}<|user|>{input_text}<|assistant|>"

        prompts.append(prompt)

    return prompts


def eval_trec(predictions, references):
    """Evaluate TREC question classification."""
    # Normalize predictions: extract the label from generated text
    pred_labels = []
    for pred in predictions:
        pred_upper = pred.upper().strip()
        matched = None
        for label in TREC_LABELS:
            if label in pred_upper:
                matched = label
                break
        pred_labels.append(matched if matched else pred_upper)

    ref_labels = [r.upper().strip() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


def eval_text2sql(predictions, references):
    """Evaluate Text2SQL (simple string match + normalized comparison)."""
    def normalize_sql(sql):
        sql = sql.strip().lower()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.rstrip(';')
        return sql

    exact_matches = 0
    for pred, ref in zip(predictions, references):
        if normalize_sql(pred) == normalize_sql(ref):
            exact_matches += 1

    accuracy = exact_matches / len(references) if references else 0
    return {"exec_accuracy": accuracy}


def eval_ecommerce(predictions, references):
    """Evaluate e-commerce product classification."""
    pred_labels = [p.strip().lower() for p in predictions]
    ref_labels = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


EVAL_FUNCTIONS = {
    "trec": eval_trec,
    "text2sql": eval_text2sql,
    "ecommerce": eval_ecommerce,
}

REFERENCE_KEYS = {
    "trec": "label_text",
    "text2sql": "sql",
    "ecommerce": "category",
}


def evaluate_task(model, tokenizer, task, data_dir, max_seq_len=512, batch_size=4):
    """
    Run full evaluation for a task. Returns dict of metric_name -> value.

    This function is the ground truth metric. Do not modify.
    """
    task_config = TASK_PROMPTS[task]
    eval_fn = EVAL_FUNCTIONS[task]
    ref_key = REFERENCE_KEYS[task]

    # Load test set
    test_dataset = load_from_disk(os.path.join(data_dir, task, "test"))

    # Build prompts
    prompts = build_prompts(test_dataset, task_config, tokenizer)

    # Generate predictions
    print(f"Generating predictions for {len(prompts)} examples...")
    predictions = generate_predictions(
        model, tokenizer, prompts,
        max_new_tokens=128 if task == "text2sql" else 32,
        batch_size=batch_size,
    )

    # Get references
    references = [str(ex[ref_key]) for ex in test_dataset]

    # Evaluate
    results = eval_fn(predictions, references)

    # Print detailed results
    print(f"\nEvaluation results for {task}:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    return results
