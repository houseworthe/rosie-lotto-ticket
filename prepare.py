"""
One-time dataset preparation for autoresearch fine-tuning experiments.
Downloads and preprocesses task datasets, saves to ~/autoresearch/data/.

Usage: python prepare.py [--tasks trec text2sql ecommerce]

Run on ROSIE management node (no GPU needed).
"""

import os
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/autoresearch/data")

TREC_COARSE_LABELS = {
    0: "ABBREVIATION",
    1: "ENTITY",
    2: "DESCRIPTION",
    3: "HUMAN",
    4: "LOCATION",
    5: "NUMERIC",
}

# ---------------------------------------------------------------------------
# Task preparation functions
# ---------------------------------------------------------------------------

def prepare_trec():
    """Download and prepare TREC question classification dataset."""
    from datasets import load_dataset

    print("Preparing TREC dataset...")
    task_dir = os.path.join(DATA_DIR, "trec")
    os.makedirs(task_dir, exist_ok=True)

    # Load TREC-6 (coarse-grained labels)
    dataset = load_dataset("CogComp/trec", trust_remote_code=True)

    def add_label_text(example):
        example["label_text"] = TREC_COARSE_LABELS.get(
            example["coarse_label"], str(example["coarse_label"])
        )
        return example

    train = dataset["train"].map(add_label_text)
    test = dataset["test"].map(add_label_text)

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  TREC: {len(train)} train, {len(test)} test")
    print(f"  Labels: {list(TREC_COARSE_LABELS.values())}")
    print(f"  Saved to: {task_dir}")


def prepare_text2sql():
    """Download and prepare Text2SQL dataset (WikiSQL subset)."""
    from datasets import load_dataset

    print("Preparing Text2SQL dataset...")
    task_dir = os.path.join(DATA_DIR, "text2sql")
    os.makedirs(task_dir, exist_ok=True)

    # Load WikiSQL
    dataset = load_dataset("wikisql", trust_remote_code=True)

    def format_wikisql(example):
        """Convert WikiSQL format to schema + question + sql."""
        table = example["table"]
        header = table["header"]
        schema = f"Table: table\nColumns: {', '.join(header)}"

        # Build SQL from structured query
        sql_obj = example["sql"]
        agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        cond_ops = ["=", ">", "<"]

        select_col = header[sql_obj["sel"]]
        agg = agg_ops[sql_obj["agg"]]
        if agg:
            select_clause = f"SELECT {agg}({select_col})"
        else:
            select_clause = f"SELECT {select_col}"

        where_clauses = []
        for i, col_idx in enumerate(sql_obj["conds"]["column_index"]):
            col = header[col_idx]
            op = cond_ops[sql_obj["conds"]["operator_index"][i]]
            val = sql_obj["conds"]["condition"][i]
            where_clauses.append(f'{col} {op} "{val}"')

        sql = f"{select_clause} FROM table"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        return {
            "schema": schema,
            "question": example["question"],
            "sql": sql,
        }

    train = dataset["train"].map(format_wikisql, remove_columns=dataset["train"].column_names)
    test = dataset["test"].map(format_wikisql, remove_columns=dataset["test"].column_names)

    # Subsample for speed (WikiSQL is huge)
    if len(train) > 10000:
        train = train.shuffle(seed=42).select(range(10000))
    if len(test) > 2000:
        test = test.shuffle(seed=42).select(range(2000))

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  Text2SQL: {len(train)} train, {len(test)} test")
    print(f"  Saved to: {task_dir}")


def prepare_ecommerce():
    """Download and prepare e-commerce product classification dataset."""
    from datasets import load_dataset

    print("Preparing E-commerce dataset...")
    task_dir = os.path.join(DATA_DIR, "ecommerce")
    os.makedirs(task_dir, exist_ok=True)

    # Use a product classification dataset
    dataset = load_dataset(
        "silicone",  # fallback — replace with better e-commerce dataset
        "dyda_da",
        trust_remote_code=True,
    )

    # Alternative: use Amazon product reviews for classification
    # For now, use a simulated e-commerce task with available data
    try:
        dataset = load_dataset("mteb/amazon_massive_scenario", trust_remote_code=True)

        def format_ecommerce(example):
            return {
                "text": example["text"],
                "category": example["label_text"] if "label_text" in example else str(example["label"]),
            }

        train = dataset["train"].map(format_ecommerce)
        test = dataset["test"].map(format_ecommerce)
    except Exception:
        # Fallback: create a simple classification task from available data
        print("  Warning: Using fallback dataset. Replace with proper e-commerce data.")
        dataset = load_dataset("ag_news", trust_remote_code=True)
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}

        def format_news(example):
            return {
                "text": example["text"],
                "category": label_map[example["label"]],
            }

        train = dataset["train"].map(format_news)
        test = dataset["test"].map(format_news)

    # Subsample
    if len(train) > 10000:
        train = train.shuffle(seed=42).select(range(10000))
    if len(test) > 2000:
        test = test.shuffle(seed=42).select(range(2000))

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  E-commerce: {len(train)} train, {len(test)} test")
    print(f"  Saved to: {task_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_PREPARERS = {
    "trec": prepare_trec,
    "text2sql": prepare_text2sql,
    "ecommerce": prepare_ecommerce,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for fine-tuning")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_PREPARERS.keys()),
        choices=list(TASK_PREPARERS.keys()),
        help="Tasks to prepare (default: all)",
    )
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)
    print()

    for task in args.tasks:
        TASK_PREPARERS[task]()
        print()

    print("Done! Datasets ready for fine-tuning.")
