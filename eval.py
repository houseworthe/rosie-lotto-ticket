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

TREC50_LABELS = [
    "abbreviation:abbreviation", "abbreviation:expansion",
    "entity:animal", "entity:body", "entity:color", "entity:creation",
    "entity:currency", "entity:disease", "entity:event", "entity:food",
    "entity:instrument", "entity:language", "entity:letter", "entity:other",
    "entity:plant", "entity:product", "entity:religion", "entity:sport",
    "entity:substance", "entity:symbol", "entity:technique", "entity:term",
    "entity:vehicle", "entity:word", "description:definition",
    "description:description", "description:manner", "description:reason",
    "human:group", "human:individual", "human:title", "human:description",
    "location:city", "location:country", "location:mountain", "location:other",
    "location:state", "numeric:code", "numeric:count", "numeric:date",
    "numeric:distance", "numeric:money", "numeric:order", "numeric:other",
    "numeric:percent", "numeric:period", "numeric:speed", "numeric:temperature",
    "numeric:size", "numeric:weight",
]

TASK_PROMPTS = {
    "trec": {
        "system": "You are a question classifier. Classify the given question into one of these categories: ABBREVIATION, ENTITY, DESCRIPTION, HUMAN, LOCATION, NUMERIC.",
        "input_template": "Classify this question: {text}",
    },
    "trec50": {
        "system": "You are a fine-grained question classifier. Classify the question into one of 50 categories in the format 'coarse:fine' (e.g. 'entity:animal', 'numeric:date', 'human:individual').",
        "input_template": "Classify this question (fine-grained): {text}",
    },
    "text2sql": {
        "system": "You are a SQL expert. Given a natural language question and database schema, generate the correct SQL query.",
        "input_template": "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:",
    },
    "ecommerce": {
        "system": "You are a product classifier. Classify the given product description into the correct category.",
        "input_template": "Classify this product: {text}",
    },
    "sst2": {
        "system": "You are a sentiment analyzer. Classify the given sentence as either positive or negative sentiment.",
        "input_template": "Analyze the sentiment of this sentence: {sentence}",
    },
    "agnews": {
        "system": "You are a news categorizer. Classify the given news article into one of these categories: World, Sports, Business, Technology.",
        "input_template": "Categorize this news article: {text}",
    },
    "mnli": {
        "system": "You are a natural language inference expert. Given a premise and hypothesis, determine if the hypothesis entails, contradicts, or is neutral to the premise.",
        "input_template": "Determine the relationship: {text}",
    },
    "dbpedia": {
        "system": "You are an ontology classifier. Classify the given text into one of these categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.",
        "input_template": "Classify this entity: {text}",
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


def eval_trec50(predictions, references):
    """Evaluate TREC-50 fine-grained question classification."""
    pred_labels = []
    for pred in predictions:
        pred_lower = pred.lower().strip()
        matched = None
        for label in TREC50_LABELS:
            if label in pred_lower:
                matched = label
                break
        pred_labels.append(matched if matched else pred_lower)

    ref_labels = [r.lower().strip() for r in references]

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


def eval_sst2(predictions, references):
    """Evaluate SST-2 sentiment classification."""
    pred_labels = []
    for pred in predictions:
        pred_lower = pred.strip().lower()
        if "positive" in pred_lower:
            pred_labels.append("positive")
        elif "negative" in pred_lower:
            pred_labels.append("negative")
        else:
            pred_labels.append(pred_lower)  # fallback

    ref_labels = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="binary", pos_label="positive", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


def eval_agnews(predictions, references):
    """Evaluate AG News classification."""
    news_labels = ["world", "sports", "business", "technology"]
    
    pred_labels = []
    for pred in predictions:
        pred_lower = pred.strip().lower()
        matched = None
        for label in news_labels:
            if label in pred_lower:
                matched = label
                break
        pred_labels.append(matched if matched else pred_lower)

    ref_labels = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


def eval_mnli(predictions, references):
    """Evaluate MNLI natural language inference."""
    nli_labels = ["entailment", "neutral", "contradiction"]
    
    pred_labels = []
    for pred in predictions:
        pred_lower = pred.strip().lower()
        matched = None
        for label in nli_labels:
            if label in pred_lower:
                matched = label
                break
        pred_labels.append(matched if matched else pred_lower)

    ref_labels = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


def eval_dbpedia(predictions, references):
    """Evaluate DBpedia ontology classification."""
    dbpedia_labels = [
        "company", "educationalinstitution", "artist", "athlete", 
        "officeholder", "meanoftransportation", "building", "naturalplace", 
        "village", "animal", "plant", "album", "film", "writtenwork"
    ]
    
    pred_labels = []
    for pred in predictions:
        pred_lower = pred.strip().lower()
        matched = None
        for label in dbpedia_labels:
            if label in pred_lower:
                matched = label
                break
        pred_labels.append(matched if matched else pred_lower)

    ref_labels = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_labels, pred_labels)
    f1 = f1_score(ref_labels, pred_labels, average="macro", zero_division=0)

    return {"accuracy": accuracy, "f1": f1}


EVAL_FUNCTIONS = {
    "trec": eval_trec,
    "trec50": eval_trec50,
    "text2sql": eval_text2sql,
    "ecommerce": eval_ecommerce,
    "sst2": eval_sst2,
    "agnews": eval_agnews,
    "mnli": eval_mnli,
    "dbpedia": eval_dbpedia,
}

REFERENCE_KEYS = {
    "trec": "label_text",
    "trec50": "label_text",
    "text2sql": "sql",
    "ecommerce": "category",
    "sst2": "label_text",
    "agnews": "label_text",
    "mnli": "label_text",
    "dbpedia": "label_text",
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
