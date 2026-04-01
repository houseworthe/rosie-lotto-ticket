"""
Phase 3: Adaptive LoRA Router Training

Trains a learned router that decides which LoRA adapters (and which neurons
within each adapter) to activate for a given query. This is Daniel Neugent's
neuron-level composition concept.

Architecture:
  Input query → base model encoder → query embedding
  Query embedding → Router MLP → adapter scores (top-k selection)
  Query embedding → per-adapter Neuron Gates → binary masks per adapter
  Composed forward: h = base(x) + Σ score_i * (mask_i ⊙ ΔW_i) @ x

Training:
  - All LoRA adapter weights are FROZEN
  - Only the router MLP + neuron gate networks train
  - Mixed-task training data (all tasks interleaved)
  - Loss = task_loss + sparsity_reg + load_balance_reg

Usage:
  python router.py                          # Train router
  python router.py --eval-only              # Evaluate trained router
  python router.py --max-train-samples 1000 # Limit training data
  python router.py --epochs 10              # Training epochs
  python router.py --top-k 2               # Number of adapters to activate

Requires: trained LoRA checkpoints from Phase 1 in ~/autoresearch/checkpoints/
"""

import os
import sys
import json
import time
import argparse
import datetime
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model
from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("AR_MODEL", "Qwen/Qwen3-0.6B")
CHECKPOINT_DIR = os.path.expanduser("~/autoresearch/checkpoints")
DATA_DIR = os.path.expanduser("~/autoresearch/data")
OUTPUT_DIR = os.path.expanduser("~/autoresearch/output/router")
ROUTER_SAVE_DIR = os.path.expanduser("~/autoresearch/checkpoints/router")

SUPPORTED_TASKS = ["trec", "trec50", "text2sql", "sst2", "agnews", "mnli", "dbpedia"]

DEFAULT_MAX_TRAIN_SAMPLES = 2000  # per task
DEFAULT_MAX_EVAL_SAMPLES = 500   # per task
DEFAULT_EPOCHS = 5
DEFAULT_TOP_K = 2
DEFAULT_LR = 1e-3
DEFAULT_SPARSITY_LAMBDA = 0.001  # reduced from 0.01 — was collapsing gates to 0
DEFAULT_BALANCE_LAMBDA = 0.01

# ---------------------------------------------------------------------------
# Dataset: mixed-task training data
# ---------------------------------------------------------------------------

class MixedTaskDataset(Dataset):
    """Interleaved dataset from all tasks. Each item includes the task label
    so the router can learn task-specific routing."""
    
    # Task-specific prompt configs (must match finetune.py)
    TASK_CONFIGS = {
        "trec": {
            "system": "You are a question classifier. Classify the given question into one of these categories: ABBREVIATION, ENTITY, DESCRIPTION, HUMAN, LOCATION, NUMERIC.",
            "input_template": "Classify this question: {text}",
            "output_key": "label_text",
        },
        "trec50": {
            "system": "You are a fine-grained question classifier. Classify the question into one of 50 categories.",
            "input_template": "Classify this question (fine-grained): {text}",
            "output_key": "label_text",
        },
        "text2sql": {
            "system": "You are a SQL expert. Given a natural language question and database schema, generate the correct SQL query.",
            "input_template": "Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:",
            "output_key": "sql",
        },
        "sst2": {
            "system": "You are a sentiment analyzer. Classify the given sentence as either positive or negative sentiment.",
            "input_template": "Analyze the sentiment of this sentence: {sentence}",
            "output_key": "label_text",
        },
        "agnews": {
            "system": "You are a news categorizer. Classify the given news article into one of these categories: World, Sports, Business, Technology.",
            "input_template": "Categorize this news article: {text}",
            "output_key": "label_text",
        },
        "mnli": {
            "system": "You are a natural language inference expert. Given a premise and hypothesis, determine if the hypothesis entails, contradicts, or is neutral to the premise.",
            "input_template": "Determine the relationship: {text}",
            "output_key": "label_text",
        },
        "dbpedia": {
            "system": "You are an ontology classifier. Classify the given text into the correct category.",
            "input_template": "Classify this entity: {text}",
            "output_key": "label_text",
        },
    }

    def __init__(self, tokenizer, tasks, max_samples_per_task, split="train", max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = []  # list of (input_ids, attention_mask, labels, task_idx)
        self.task_names = tasks
        
        for task_idx, task in enumerate(tasks):
            # Load data from Arrow format (saved by prepare.py via save_to_disk)
            data_dir = os.path.join(DATA_DIR, task, split)
            if not os.path.exists(data_dir):
                print(f"  ⚠️  No {split} data for {task}, skipping", flush=True)
                continue
            
            try:
                dataset = load_from_disk(data_dir)
            except Exception as e:
                print(f"  ⚠️  Failed to load {split} data for {task}: {e}", flush=True)
                continue
            
            if max_samples_per_task and len(dataset) > max_samples_per_task:
                dataset = dataset.shuffle(seed=42).select(range(max_samples_per_task))
            
            task_config = self.TASK_CONFIGS.get(task, {})
            input_template = task_config.get("input_template", "{text}")
            system_msg = task_config.get("system", "")
            output_key = task_config.get("output_key", "label_text")
            
            print(f"  ✅ {task}: {len(dataset)} samples", flush=True)
            
            for sample in dataset:
                # Format prompt using task config
                try:
                    input_text = input_template.format(**sample)
                except KeyError:
                    input_text = str(sample.get("text", ""))
                
                completion = str(sample.get(output_key, ""))
                
                # Build prompt with chat template
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": input_text},
                ]
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = f"{system_msg}\n{input_text}\n"
                
                text = f"{prompt}{completion}"
                
                encoded = tokenizer(
                    text,
                    max_length=max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # For labels: mask the prompt tokens (set to -100)
                prompt_encoded = tokenizer(
                    prompt,
                    max_length=max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_len = prompt_encoded["input_ids"].shape[1]
                
                labels = encoded["input_ids"].clone()
                labels[0, :prompt_len] = -100  # mask prompt
                # Also mask padding
                labels[labels == tokenizer.pad_token_id] = -100
                
                self.items.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": labels.squeeze(0),
                    "task_idx": task_idx,
                })
            
            print(f"  ✅ {task}: loaded {len([i for i in self.items if i['task_idx'] == task_idx])} samples", flush=True)
        
        # Shuffle everything
        random.shuffle(self.items)
        print(f"  📊 Total training samples: {len(self.items)}", flush=True)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Router Model
# ---------------------------------------------------------------------------

class AdaptiveLoRARouter(nn.Module):
    """Learned router for multi-LoRA neuron-level composition.
    
    Given a query embedding (from the base model's encoder), outputs:
    1. Adapter scores (which adapters to activate, top-k)
    2. Neuron masks per adapter (which neurons within each adapter to use)
    """
    
    def __init__(self, d_model, n_adapters, lora_rank, top_k=2):
        super().__init__()
        self.n_adapters = n_adapters
        self.top_k = min(top_k, n_adapters)
        self.d_model = d_model
        self.lora_rank = lora_rank
        
        # Adapter-level router: query embedding → adapter scores
        self.adapter_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_adapters),
        )
        
        # Neuron-level gates: one per adapter
        # Each gate outputs a mask over the LoRA rank dimensions
        # Initialize with positive bias so gates start OPEN (sigmoid(2) ≈ 0.88)
        # This prevents sparsity penalty from collapsing gates to 0 before learning
        self.neuron_gates = nn.ModuleList()
        for _ in range(n_adapters):
            gate = nn.Sequential(
                nn.Linear(d_model, lora_rank * 2),
                nn.ReLU(),
                nn.Linear(lora_rank * 2, lora_rank),
                nn.Sigmoid(),  # soft mask: 0-1 per neuron
            )
            # Init final layer bias to +2 so gates start open
            nn.init.constant_(gate[2].bias, 2.0)
            nn.init.zeros_(gate[2].weight)  # start near-uniform, let training differentiate
            self.neuron_gates.append(gate)
    
    def forward(self, query_embedding):
        """
        Args:
            query_embedding: (batch, d_model) — mean-pooled hidden states from base model
        
        Returns:
            adapter_scores: (batch, top_k) — scores for selected adapters
            adapter_indices: (batch, top_k) — which adapters were selected
            neuron_masks: dict[adapter_idx] → (batch, lora_rank) — per-neuron masks
            all_scores: (batch, n_adapters) — full score distribution (for regularization)
        """
        # Compute adapter scores
        all_scores = self.adapter_router(query_embedding)  # (batch, n_adapters)
        
        # Top-k selection with straight-through gradient
        topk_scores, topk_indices = torch.topk(all_scores, self.top_k, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)  # normalize selected scores
        
        # Compute neuron masks for ALL adapters (needed for gradient flow)
        # but we'll only use the top-k ones in the forward pass
        neuron_masks = {}
        for i in range(self.n_adapters):
            neuron_masks[i] = self.neuron_gates[i](query_embedding)  # (batch, lora_rank)
        
        return topk_scores, topk_indices, neuron_masks, all_scores


# ---------------------------------------------------------------------------
# Composed Model: Base + Router + Frozen LoRA Adapters
# ---------------------------------------------------------------------------

class ComposedLoRAModel(nn.Module):
    """Wraps the base model with frozen LoRA adapters and a learned router.
    
    During forward pass:
    1. Run input through base model to get query embedding
    2. Router selects top-k adapters and generates neuron masks
    3. For each selected adapter, apply masked LoRA delta to hidden states
    4. Compute loss on the composed output
    """
    
    def __init__(self, base_model, tokenizer, adapter_paths, router, device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.router = router
        
        # Load base model (frozen)
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Extract and store LoRA weights from each adapter (frozen)
        self.adapter_names = list(adapter_paths.keys())
        self.lora_weights = {}  # adapter_name -> {layer_name -> {A: tensor, B: tensor}}
        
        print(f"\n📦 Loading {len(adapter_paths)} LoRA adapters...", flush=True)
        for name, path in adapter_paths.items():
            self._load_adapter_weights(name, path)
            print(f"  ✅ {name}: loaded from {path}", flush=True)
        
        # Move router to device
        self.router = self.router.to(device)
    
    def _load_adapter_weights(self, name, path):
        """Load LoRA A and B matrices from a saved adapter."""
        # Load the adapter state dict
        adapter_model_path = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            from safetensors.torch import load_file
            state_dict = load_file(adapter_model_path)
        else:
            adapter_model_path = os.path.join(path, "adapter_model.bin")
            state_dict = torch.load(adapter_model_path, map_location="cpu")
        
        # Organize by layer
        weights = {}
        for key, tensor in state_dict.items():
            # Keys look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            parts = key.split(".")
            # Find the layer identifier and A/B
            if "lora_A" in key:
                layer_key = key.replace(".lora_A.weight", "").replace(".lora_A.default.weight", "")
                if layer_key not in weights:
                    weights[layer_key] = {}
                weights[layer_key]["A"] = tensor.to(self.device)
            elif "lora_B" in key:
                layer_key = key.replace(".lora_B.weight", "").replace(".lora_B.default.weight", "")
                if layer_key not in weights:
                    weights[layer_key] = {}
                weights[layer_key]["B"] = tensor.to(self.device)
        
        self.lora_weights[name] = weights
    
    def get_query_embedding(self, input_ids, attention_mask):
        """Run input through base model and get mean-pooled hidden state."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Use last hidden state, mean-pooled over sequence
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, d_model)
        # Mask padding
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)
        return pooled
    
    def forward(self, input_ids, attention_mask, labels=None, task_idx=None):
        """
        Forward pass with composed LoRA adapters.
        
        For efficiency, we use a hook-based approach:
        1. Get query embedding from base model
        2. Router decides which adapters and neuron masks
        3. Register forward hooks on target layers to add masked LoRA deltas
        4. Run base model forward pass (hooks inject LoRA contributions)
        5. Compute loss
        """
        batch_size = input_ids.shape[0]
        
        # Step 1: Get query embedding
        query_emb = self.get_query_embedding(input_ids, attention_mask)
        
        # Step 2: Router decision
        topk_scores, topk_indices, neuron_masks, all_scores = self.router(query_emb)
        
        # Step 3: Build per-layer LoRA deltas
        # For each layer that has LoRA weights, compute the composed delta
        composed_deltas = {}  # layer_key -> (batch, out_dim, in_dim)
        
        for layer_key in self._get_all_lora_layers():
            delta = None
            for b in range(batch_size):
                batch_delta = torch.zeros(
                    self.lora_weights[self.adapter_names[0]][layer_key]["B"].shape[0],
                    self.lora_weights[self.adapter_names[0]][layer_key]["A"].shape[1],
                    device=self.device,
                )
                for k in range(topk_scores.shape[1]):
                    adapter_idx = topk_indices[b, k].item()
                    adapter_name = self.adapter_names[adapter_idx]
                    score = topk_scores[b, k]
                    
                    if adapter_name not in self.lora_weights:
                        continue
                    if layer_key not in self.lora_weights[adapter_name]:
                        continue
                    
                    A = self.lora_weights[adapter_name][layer_key]["A"]  # (rank, in_dim)
                    B = self.lora_weights[adapter_name][layer_key]["B"]  # (out_dim, rank)
                    mask = neuron_masks[adapter_idx][b]  # (rank,)
                    
                    # Apply neuron mask to LoRA rank dimension
                    # masked_A = diag(mask) @ A → zero out masked neurons in rank dim
                    masked_A = A * mask.unsqueeze(-1)  # (rank, in_dim) * (rank, 1)
                    
                    # Composed delta for this adapter
                    adapter_delta = B @ masked_A  # (out_dim, in_dim)
                    batch_delta += score * adapter_delta
                
                if delta is None:
                    delta = batch_delta.unsqueeze(0)
                else:
                    delta = torch.cat([delta, batch_delta.unsqueeze(0)], dim=0)
            
            composed_deltas[layer_key] = delta  # (batch, out_dim, in_dim)
        
        # Step 4: Forward pass with hooks
        hooks = []
        
        def make_hook(layer_key):
            def hook_fn(module, input, output):
                if layer_key in composed_deltas:
                    delta = composed_deltas[layer_key]  # (batch, out_dim, in_dim)
                    x = input[0]  # (batch, seq_len, in_dim)
                    # Apply: output += delta @ x (batched matmul)
                    lora_out = torch.bmm(
                        delta,
                        x.transpose(1, 2)
                    ).transpose(1, 2)  # (batch, seq_len, out_dim)
                    return output + lora_out
            return hook_fn
        
        # Register hooks on the actual linear layers
        for layer_key in composed_deltas:
            module = self._get_module_by_lora_key(layer_key)
            if module is not None:
                hooks.append(module.register_forward_hook(make_hook(layer_key)))
        
        # Run forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Compute regularization losses
        sparsity_loss = self._sparsity_loss(neuron_masks)
        balance_loss = self._balance_loss(all_scores)
        
        return outputs, sparsity_loss, balance_loss, {
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "all_scores": all_scores,
        }
    
    def _get_all_lora_layers(self):
        """Get all layer keys that have LoRA weights across any adapter."""
        layers = set()
        for adapter_weights in self.lora_weights.values():
            layers.update(adapter_weights.keys())
        return sorted(layers)
    
    def _get_module_by_lora_key(self, lora_key):
        """Map a LoRA weight key back to the actual nn.Linear module."""
        # lora_key looks like: base_model.model.model.layers.0.self_attn.q_proj
        # We need to navigate the module hierarchy
        parts = lora_key.split(".")
        module = self.base_model
        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, TypeError):
            return None
    
    def _sparsity_loss(self, neuron_masks):
        """Encourage sparse neuron masks (L1 penalty)."""
        total = 0
        count = 0
        for idx, mask in neuron_masks.items():
            total += mask.mean()
            count += 1
        return total / max(count, 1)
    
    def _balance_loss(self, all_scores):
        """Encourage balanced adapter usage across the batch (avoid collapse to one adapter)."""
        # Compute mean adapter selection probability across batch
        probs = F.softmax(all_scores, dim=-1)  # (batch, n_adapters)
        mean_probs = probs.mean(dim=0)  # (n_adapters,)
        # Uniform target
        uniform = torch.ones_like(mean_probs) / mean_probs.shape[0]
        # KL divergence from uniform
        return F.kl_div(mean_probs.log(), uniform, reduction="batchmean")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_router(args):
    """Train the adaptive LoRA router."""
    
    print("=" * 70, flush=True)
    print("🧠 Phase 3: Adaptive LoRA Router Training", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Top-k: {args.top_k}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"LR: {args.lr}", flush=True)
    print(f"Sparsity λ: {args.sparsity_lambda}", flush=True)
    print(f"Balance λ: {args.balance_lambda}", flush=True)
    print(f"Max train samples/task: {args.max_train_samples}", flush=True)
    print(f"Max eval samples/task: {args.max_eval_samples}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Detect available adapters
    # Checkpoints are stored as flat dirs: checkpoints/{task}_Qwen3-0.6B_r16_lr{lr}/
    # Each dir contains adapter_config.json directly (no nesting).
    print(f"\n🔍 Scanning for adapters in {CHECKPOINT_DIR}...", flush=True)
    adapter_paths = {}
    if os.path.isdir(CHECKPOINT_DIR):
        for entry in sorted(os.listdir(CHECKPOINT_DIR)):
            ckpt_path = os.path.join(CHECKPOINT_DIR, entry)
            config_file = os.path.join(ckpt_path, "adapter_config.json")
            if not os.path.isdir(ckpt_path) or not os.path.exists(config_file):
                continue
            # Match task name from directory prefix (longest match first to
            # avoid "trec" matching "trec50_..." before "trec50" gets a chance)
            matched_task = None
            for task in sorted(SUPPORTED_TASKS, key=len, reverse=True):
                if entry.startswith(task + "_"):
                    matched_task = task
                    break
            if matched_task is None:
                continue
            # Verify it's a 0.6B adapter
            try:
                config = json.loads(open(config_file).read())
                base = config.get("base_model_name_or_path", "")
                if "0.6B" in base or "0.5B" in base or MODEL_NAME in base:
                    # Keep the first match per task (sorted order picks canonical config)
                    if matched_task not in adapter_paths:
                        adapter_paths[matched_task] = ckpt_path
            except:
                continue
    
    if len(adapter_paths) < 2:
        print(f"❌ Need at least 2 adapters, found {len(adapter_paths)}", flush=True)
        sys.exit(1)
    
    tasks = sorted(adapter_paths.keys())
    n_adapters = len(tasks)
    print(f"\n✅ Found {n_adapters} adapters: {tasks}", flush=True)
    for task, path in adapter_paths.items():
        print(f"  {task}: {path}", flush=True)
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"📥 Loading base model: {MODEL_NAME}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    base_model.eval()
    
    # Get model dimensions
    d_model = base_model.config.hidden_size
    # Get LoRA rank from first adapter
    first_adapter = list(adapter_paths.values())[0]
    adapter_config = json.loads(open(os.path.join(first_adapter, "adapter_config.json")).read())
    lora_rank = adapter_config.get("r", 16)
    print(f"  d_model={d_model}, lora_rank={lora_rank}", flush=True)
    
    # Create router
    print(f"\n🧠 Creating router (d_model={d_model}, n_adapters={n_adapters}, rank={lora_rank}, top_k={args.top_k})...", flush=True)
    router = AdaptiveLoRARouter(
        d_model=d_model,
        n_adapters=n_adapters,
        lora_rank=lora_rank,
        top_k=args.top_k,
    )
    
    total_params = sum(p.numel() for p in router.parameters())
    print(f"  Router parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
    
    # Create composed model
    composed = ComposedLoRAModel(
        base_model=base_model,
        tokenizer=tokenizer,
        adapter_paths={task: adapter_paths[task] for task in tasks},
        router=router,
        device=device,
    )
    
    # Load training data
    print(f"\n📊 Loading mixed-task training data...", flush=True)
    train_dataset = MixedTaskDataset(
        tokenizer=tokenizer,
        tasks=tasks,
        max_samples_per_task=args.max_train_samples,
        split="train",
        max_seq_len=256,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    # Optimizer — only router parameters
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROUTER_SAVE_DIR, exist_ok=True)
    
    best_loss = float("inf")
    training_log = []
    
    print(f"\n🚀 Starting training ({args.epochs} epochs, {len(train_loader)} batches/epoch)...\n", flush=True)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        router.train()
        total_loss = 0
        total_task_loss = 0
        total_sparsity_loss = 0
        total_balance_loss = 0
        n_batches = 0
        
        # Track router decisions for logging
        adapter_selections = defaultdict(int)
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            task_indices = batch["task_idx"]
            
            optimizer.zero_grad()
            
            outputs, sparsity_loss, balance_loss, routing_info = composed(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task_idx=task_indices,
            )
            
            task_loss = outputs.loss
            loss = task_loss + args.sparsity_lambda * sparsity_loss + args.balance_lambda * balance_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_balance_loss += balance_loss.item()
            n_batches += 1
            
            # Track selections
            for idx in routing_info["topk_indices"].flatten().cpu().tolist():
                adapter_selections[tasks[idx]] += 1
            
            if (batch_idx + 1) % 50 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} (task={total_task_loss/n_batches:.4f}, "
                      f"sparse={total_sparsity_loss/n_batches:.4f}, "
                      f"balance={total_balance_loss/n_batches:.4f})", flush=True)
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(n_batches, 1)
        avg_task = total_task_loss / max(n_batches, 1)
        avg_sparse = total_sparsity_loss / max(n_batches, 1)
        avg_balance = total_balance_loss / max(n_batches, 1)
        
        # Log adapter selection distribution
        total_selections = sum(adapter_selections.values())
        selection_pcts = {k: f"{v/total_selections*100:.1f}%" for k, v in sorted(adapter_selections.items())}
        
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} complete ({epoch_time:.0f}s)", flush=True)
        print(f"  Loss: {avg_loss:.4f} (task={avg_task:.4f}, sparse={avg_sparse:.4f}, balance={avg_balance:.4f})", flush=True)
        print(f"  Adapter selections: {selection_pcts}", flush=True)
        
        # Compute mean neuron mask sparsity
        router.eval()
        with torch.no_grad():
            # Sample a batch to check sparsity
            sample_batch = next(iter(train_loader))
            sample_emb = composed.get_query_embedding(
                sample_batch["input_ids"].to(device),
                sample_batch["attention_mask"].to(device),
            )
            _, _, masks, _ = router(sample_emb)
            for i, task in enumerate(tasks):
                if i in masks:
                    mask_mean = masks[i].mean().item()
                    mask_active = (masks[i] > 0.5).float().mean().item()
                    print(f"  {task} gate: mean={mask_mean:.3f}, active={mask_active*100:.1f}%", flush=True)
        
        epoch_log = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "task_loss": avg_task,
            "sparsity_loss": avg_sparse,
            "balance_loss": avg_balance,
            "time_seconds": epoch_time,
            "adapter_selections": dict(adapter_selections),
        }
        training_log.append(epoch_log)
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "router_state_dict": router.state_dict(),
                "tasks": tasks,
                "adapter_paths": adapter_paths,
                "d_model": d_model,
                "lora_rank": lora_rank,
                "n_adapters": n_adapters,
                "top_k": args.top_k,
                "epoch": epoch + 1,
                "loss": best_loss,
            }, os.path.join(ROUTER_SAVE_DIR, "best_router.pt"))
            print(f"  💾 Saved best router (loss={best_loss:.4f})", flush=True)
        
        print("", flush=True)
    
    # Save final
    torch.save({
        "router_state_dict": router.state_dict(),
        "tasks": tasks,
        "adapter_paths": adapter_paths,
        "d_model": d_model,
        "lora_rank": lora_rank,
        "n_adapters": n_adapters,
        "top_k": args.top_k,
        "epoch": args.epochs,
        "loss": avg_loss,
    }, os.path.join(ROUTER_SAVE_DIR, "final_router.pt"))
    
    # Save training log
    log_path = os.path.join(OUTPUT_DIR, f"router_training_{int(time.time())}.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": vars(args),
            "tasks": tasks,
            "n_adapters": n_adapters,
            "d_model": d_model,
            "lora_rank": lora_rank,
            "router_params": total_params,
            "training_log": training_log,
            "best_loss": best_loss,
        }, f, indent=2)
    
    print(f"=" * 70, flush=True)
    print(f"✅ Router training complete!", flush=True)
    print(f"  Best loss: {best_loss:.4f}", flush=True)
    print(f"  Router saved: {ROUTER_SAVE_DIR}", flush=True)
    print(f"  Training log: {log_path}", flush=True)
    print(f"=" * 70, flush=True)
    
    # Run evaluation if not skipped
    if not args.skip_eval:
        evaluate_router(composed, tasks, tokenizer, device, args)


def evaluate_router(composed, tasks, tokenizer, device, args):
    """Evaluate the trained router on each task individually and report accuracy."""
    from eval import evaluate_task
    
    print(f"\n🧪 Evaluating router on individual tasks...\n", flush=True)
    
    results = {}
    for task in tasks:
        print(f"  Evaluating {task}...", flush=True)
        try:
            acc = evaluate_task(
                model=composed.base_model,  # TODO: need composed forward for eval
                tokenizer=tokenizer,
                task=task,
                data_dir=DATA_DIR,
                max_samples=args.max_eval_samples,
            )
            results[task] = acc
            print(f"  ✅ {task}: {acc:.1f}%", flush=True)
        except Exception as e:
            print(f"  ❌ {task}: {e}", flush=True)
            results[task] = None
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"router_eval_{int(time.time())}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Router evaluation results:", flush=True)
    for task, acc in results.items():
        print(f"  {task}: {acc:.1f}%" if acc else f"  {task}: FAILED", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Adaptive LoRA Router Training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-eval-samples", type=int, default=DEFAULT_MAX_EVAL_SAMPLES)
    parser.add_argument("--sparsity-lambda", type=float, default=DEFAULT_SPARSITY_LAMBDA)
    parser.add_argument("--balance-lambda", type=float, default=DEFAULT_BALANCE_LAMBDA)
    args = parser.parse_args()
    
    train_router(args)


if __name__ == "__main__":
    main()
