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

# TREC-50 fine-grained label mapping (coarse:fine)
TREC_FINE_LABELS = {
    0: "abbreviation:abbreviation", 1: "abbreviation:expansion",
    2: "entity:animal", 3: "entity:body", 4: "entity:color",
    5: "entity:creation", 6: "entity:currency", 7: "entity:disease",
    8: "entity:event", 9: "entity:food", 10: "entity:instrument",
    11: "entity:language", 12: "entity:letter", 13: "entity:other",
    14: "entity:plant", 15: "entity:product", 16: "entity:religion",
    17: "entity:sport", 18: "entity:substance", 19: "entity:symbol",
    20: "entity:technique", 21: "entity:term", 22: "entity:vehicle",
    23: "entity:word", 24: "description:definition",
    25: "description:description", 26: "description:manner",
    27: "description:reason", 28: "human:group", 29: "human:individual",
    30: "human:title", 31: "human:description", 32: "location:city",
    33: "location:country", 34: "location:mountain", 35: "location:other",
    36: "location:state", 37: "numeric:code", 38: "numeric:count",
    39: "numeric:date", 40: "numeric:distance", 41: "numeric:money",
    42: "numeric:order", 43: "numeric:other", 44: "numeric:percent",
    45: "numeric:period", 46: "numeric:speed", 47: "numeric:temperature",
    48: "numeric:size", 49: "numeric:weight",
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
    dataset = load_dataset("trec")

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


def prepare_trec50():
    """Download and prepare TREC-50 fine-grained question classification dataset."""
    from datasets import load_dataset

    print("Preparing TREC-50 (fine-grained) dataset...")
    task_dir = os.path.join(DATA_DIR, "trec50")
    os.makedirs(task_dir, exist_ok=True)

    dataset = load_dataset("trec")

    def add_fine_label(example):
        example["label_text"] = TREC_FINE_LABELS.get(
            example["fine_label"], str(example["fine_label"])
        )
        return example

    train = dataset["train"].map(add_fine_label)
    test = dataset["test"].map(add_fine_label)

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    unique_labels = sorted(set(TREC_FINE_LABELS.values()))
    print(f"  TREC-50: {len(train)} train, {len(test)} test")
    print(f"  Labels: {len(unique_labels)} fine-grained classes")
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


def prepare_sst2():
    """Download and prepare SST-2 sentiment classification dataset."""
    from datasets import load_dataset

    print("Preparing SST-2 dataset...")
    task_dir = os.path.join(DATA_DIR, "sst2")
    os.makedirs(task_dir, exist_ok=True)

    # Load SST-2 from GLUE
    dataset = load_dataset("glue", "sst2")

    def add_label_text(example):
        # 0 = negative, 1 = positive
        example["label_text"] = "positive" if example["label"] == 1 else "negative"
        return example

    train = dataset["train"].map(add_label_text)
    test = dataset["validation"].map(add_label_text)  # SST-2 uses validation as test

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  SST-2: {len(train)} train, {len(test)} test")
    print(f"  Labels: positive, negative")
    print(f"  Saved to: {task_dir}")


def prepare_agnews():
    """Download and prepare AG News classification dataset."""
    from datasets import load_dataset

    print("Preparing AG News dataset...")
    task_dir = os.path.join(DATA_DIR, "agnews")
    os.makedirs(task_dir, exist_ok=True)

    # Load AG News
    dataset = load_dataset("ag_news")

    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}

    def add_label_text(example):
        example["label_text"] = label_map[example["label"]]
        return example

    train = dataset["train"].map(add_label_text)
    test = dataset["test"].map(add_label_text)

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  AG News: {len(train)} train, {len(test)} test")
    print(f"  Labels: {list(label_map.values())}")
    print(f"  Saved to: {task_dir}")


def prepare_mnli():
    """Download and prepare MNLI natural language inference dataset."""
    from datasets import load_dataset

    print("Preparing MNLI dataset...")
    task_dir = os.path.join(DATA_DIR, "mnli")
    os.makedirs(task_dir, exist_ok=True)

    # Load MNLI from GLUE
    dataset = load_dataset("glue", "mnli")

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def format_mnli(example):
        # Combine premise and hypothesis
        example["text"] = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
        example["label_text"] = label_map[example["label"]]
        return example

    train = dataset["train"].map(format_mnli)
    # Use matched validation as test set
    test = dataset["validation_matched"].map(format_mnli)

    # Subsample for faster training (MNLI is huge)
    if len(train) > 20000:
        train = train.shuffle(seed=42).select(range(20000))
    if len(test) > 5000:
        test = test.shuffle(seed=42).select(range(5000))

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  MNLI: {len(train)} train, {len(test)} test")
    print(f"  Labels: {list(label_map.values())}")
    print(f"  Saved to: {task_dir}")


def prepare_dbpedia():
    """Download and prepare DBpedia 14-class classification dataset."""
    from datasets import load_dataset

    print("Preparing DBpedia dataset...")
    task_dir = os.path.join(DATA_DIR, "dbpedia")
    os.makedirs(task_dir, exist_ok=True)

    # Load DBpedia 14
    dataset = load_dataset("dbpedia_14")

    # DBpedia labels (14 ontology classes)
    label_map = {
        0: "Company", 1: "EducationalInstitution", 2: "Artist",
        3: "Athlete", 4: "OfficeHolder", 5: "MeanOfTransportation",
        6: "Building", 7: "NaturalPlace", 8: "Village",
        9: "Animal", 10: "Plant", 11: "Album",
        12: "Film", 13: "WrittenWork"
    }

    def format_dbpedia(example):
        # Combine title and content
        example["text"] = f"{example['title']} {example['content']}"
        example["label_text"] = label_map[example["label"]]
        return example

    train = dataset["train"].map(format_dbpedia)
    test = dataset["test"].map(format_dbpedia)

    # Subsample for faster training
    if len(train) > 15000:
        train = train.shuffle(seed=42).select(range(15000))
    if len(test) > 3000:
        test = test.shuffle(seed=42).select(range(3000))

    train.save_to_disk(os.path.join(task_dir, "train"))
    test.save_to_disk(os.path.join(task_dir, "test"))

    print(f"  DBpedia: {len(train)} train, {len(test)} test")
    print(f"  Labels: {len(label_map)} ontology classes")
    print(f"  Saved to: {task_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_PREPARERS = {
    "trec": prepare_trec,
    "trec50": prepare_trec50,
    "text2sql": prepare_text2sql,
    "ecommerce": prepare_ecommerce,
    "sst2": prepare_sst2,
    "agnews": prepare_agnews,
    "mnli": prepare_mnli,
    "dbpedia": prepare_dbpedia,
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
