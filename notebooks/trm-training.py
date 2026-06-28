import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import random
    import math
    from pathlib import Path
    from collections import defaultdict

    mo.md("# TRM Content Review — Training")
    return Path, json, mo, nn, random, torch


@app.cell
def _(mo):
    mo.md("""
    ## Overview

    Trains a **Tiny Recursive Model (TRM)** — 7M params, 2 layers, no attention —
    on the 25K content review dataset. The TRM learns to detect errors in model
    responses that larger models miss.

    **Target:** ≥3/5 error-injection overrides (currently 0/5 with prompted 8B review).
    **Hardware:** RTX Pro 6000 (96 GB VRAM) — estimated 3-5 hours.
    """)
    return


@app.cell(hide_code=True)
def _(mo, nn, torch):
    # Step 1: TRM Architecture

    class TRMBlock(nn.Module):
        def __init__(self, d_model=128, seq_len=256, expansion=4):
            super().__init__()
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
            self.seq_mlp = nn.Sequential(
                nn.Linear(seq_len, seq_len * expansion),
                nn.GELU(),
                nn.Linear(seq_len * expansion, seq_len),
            )
            self.channel_mlp = nn.Sequential(
                nn.Linear(d_model, d_model * expansion),
                nn.GELU(),
                nn.Linear(d_model * expansion, d_model),
            )

        def forward(self, x):
            residual = x
            x = self.norm1(x)
            x = x.transpose(1, 2)
            x = self.seq_mlp(x)
            x = x.transpose(1, 2)
            x = x + residual
            residual = x
            x = self.norm2(x)
            x = self.channel_mlp(x)
            x = x + residual
            return x


    class TRMClassifier(nn.Module):
        def __init__(self, vocab_size=256, d_model=128, seq_len=256, n_layers=2, n_recursions=6):
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len
            self.n_recursions = n_recursions
        
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
            self.blocks = nn.ModuleList([TRMBlock(d_model, seq_len) for _ in range(n_layers)])
        
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )
        
            n_params = sum(p.numel() for p in self.parameters())
            print(f"TRM Classifier: {n_params:,} parameters ({n_params/1e6:.1f}M)")

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            x = self.embed(input_ids) + self.pos_embed[:, :L, :]
        
            for _ in range(self.n_recursions):
                for block in self.blocks:
                    x = block(x)
        
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        
            return self.classifier(x).squeeze(-1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRMClassifier(vocab_size=256, d_model=128, seq_len=256, n_layers=2, n_recursions=6)
    model = model.to(device)

    mo.md(f"Model initialized on {device}")
    return device, model


@app.cell(hide_code=True)
def _(Path, json, mo, random, torch):
    # Step 2: Load and Tokenize Dataset

    # Load the 25K dataset (upload trm_content_review_25k.jsonl alongside this notebook)
    data_path = Path("trm_content_review_25k.jsonl")
    if not data_path.exists():
        data_path = Path("/home/theyokel/.openframe/harness/trm_training_data/trm_content_review_25k.jsonl")
    examples = [json.loads(l) for l in open(data_path)]

    # Simple byte-level tokenization (no tokenizer dependency)
    def tokenize(text, max_len=256):
        tokens = list(text.encode("utf-8"))[:max_len]
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(mask)

    # Format: "Query: {query} Response: {response}" -> binary label
    inputs = []
    labels = []
    for ex in examples:
        text = f"Query: {ex['query']} Response: {ex['response']}"
        tokens, mask = tokenize(text)
        inputs.append((tokens, mask))
        labels.append(1.0 if ex["is_correct"] else 0.0)

    # Split: 80% train, 20% eval
    n = len(inputs)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    split = int(n * 0.8)

    train_inputs = [inputs[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    eval_inputs = [inputs[i] for i in indices[split:]]
    eval_labels = [labels[i] for i in indices[split:]]

    mo.md(f"Dataset loaded: {len(train_inputs)} train, {len(eval_inputs)} eval")

    return eval_inputs, eval_labels, train_inputs, train_labels


@app.cell(hide_code=True)
def _(
    device,
    eval_inputs,
    eval_labels,
    mo,
    model,
    nn,
    torch,
    train_inputs,
    train_labels,
):
    # Step 3: Train TRM

    import time
    from torch.utils.data import TensorDataset, DataLoader

    # Prepare tensors
    train_tokens = torch.stack([t for t, m in train_inputs])
    train_masks = torch.stack([m for t, m in train_inputs])
    train_labels_t = torch.tensor(train_labels)

    eval_tokens = torch.stack([t for t, m in eval_inputs])
    eval_masks = torch.stack([m for t, m in eval_inputs])
    eval_labels_t = torch.tensor(eval_labels)

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_tokens, train_masks, train_labels_t), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_tokens, eval_masks, eval_labels_t), batch_size=batch_size)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    n_epochs = 10
    best_eval_acc = 0.0
    ema_model = None  # EMA of weights for stability

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
    
        for batch_tokens, batch_masks, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)
        
            optimizer.zero_grad()
            logits = model(batch_tokens, batch_masks)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)
    
        train_acc = train_correct / train_total
    
        # Eval
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
    
        with torch.no_grad():
            for batch_tokens, batch_masks, batch_labels in eval_loader:
                batch_tokens = batch_tokens.to(device)
                batch_masks = batch_masks.to(device)
                batch_labels = batch_labels.to(device)
            
                logits = model(batch_tokens, batch_masks)
                loss = criterion(logits, batch_labels)
            
                eval_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                eval_correct += (preds == batch_labels).sum().item()
                eval_total += batch_labels.size(0)
    
        eval_acc = eval_correct / eval_total
        scheduler.step()
    
        # Save best
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), "/tmp/trm_best.pt")
    
        print(f"Epoch {epoch+1}/{n_epochs} | train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.3f} | eval_loss={eval_loss/len(eval_loader):.4f} eval_acc={eval_acc:.3f}")

    # Load best
    model.load_state_dict(torch.load("/tmp/trm_best.pt"))

    mo.md(f"Training complete. Best eval accuracy: {best_eval_acc:.3f}")

    return


if __name__ == "__main__":
    app.run()
