# /// script
# dependencies = [
#   "torch",
#   "huggingface_hub",
# ]
# ///

"""TRM Content Review training — submitted via HF Jobs. Custom 7M-param model trained from scratch, pushes best checkpoint to Hub."""
import json, os, sys, time, random
from pathlib import Path

import torch
import torch.nn as nn
import urllib.request
from huggingface_hub import HfApi

# ── Config ──
DATASET_URL = "https://raw.githubusercontent.com/Seamus05/bitmajor-trilogy/master/data/trm_content_review_25k.jsonl"
OUTPUT_REPO = "Seamus05/trm-content-review-7m"
HF_TOKEN = os.environ["HF_TOKEN"]

DATA_PATH = Path("/tmp/trm_content_review_25k.jsonl")
OUTPUT_DIR = Path("/tmp/trm-output")
OUTPUT_DIR.mkdir(exist_ok=True)

def log(msg):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)

log("=== TRM Content Review Training (HF Jobs) ===")
log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# ── Download dataset ──
log("Downloading dataset...")
urllib.request.urlretrieve(DATASET_URL, str(DATA_PATH))
examples = [json.loads(l) for l in open(DATA_PATH)]
log(f"Dataset: {len(examples)} examples")

# ── Byte-level tokenization (no tokenizer dependency) ──
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
log(f"Tokenized: {len(train_inputs)} train, {len(eval_inputs)} eval")

# ── TRM Architecture (7M params, 2 layers, no attention) ──
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
n_params = sum(p.numel() for p in model.parameters())
log(f"TRM Classifier: {n_params:,} parameters ({n_params/1e6:.1f}M) on {device}")

# ── Prepare tensors ──
from torch.utils.data import TensorDataset, DataLoader

train_tokens = torch.stack([t for t, m in train_inputs])
train_masks = torch.stack([m for t, m in train_inputs])
train_labels_t = torch.tensor(train_labels)
eval_tokens = torch.stack([t for t, m in eval_inputs])
eval_masks = torch.stack([m for t, m in eval_inputs])
eval_labels_t = torch.tensor(eval_labels)

batch_size = 64
train_loader = DataLoader(TensorDataset(train_tokens, train_masks, train_labels_t), batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(TensorDataset(eval_tokens, eval_masks, eval_labels_t), batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.BCEWithLogitsLoss()

# ── Train ──
n_epochs = 10
best_eval_acc = 0.0
best_epoch = 0

log(f"Training ({n_epochs} epochs, batch_size={batch_size})...")

for epoch in range(n_epochs):
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

    if eval_acc > best_eval_acc:
        best_eval_acc = eval_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), str(OUTPUT_DIR / "trm_best.pt"))

    log(f"Epoch {epoch+1}/{n_epochs} | train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.3f} | eval_loss={eval_loss/len(eval_loader):.4f} eval_acc={eval_acc:.3f}")

log(f"Best: epoch {best_epoch}, eval_acc={best_eval_acc:.3f}")

# ── Upload to HuggingFace ──
log("Uploading to HuggingFace...")
api = HfApi(token=HF_TOKEN)

# Upload model checkpoint
api.upload_file(
    path_or_fileobj=str(OUTPUT_DIR / "trm_best.pt"),
    path_in_repo="trm_best.pt",
    repo_id=OUTPUT_REPO,
    repo_type="model",
)

# Upload model architecture as a standalone Python file for reproducibility
import inspect
arch_src = inspect.getsource(sys.modules[__name__])
(OUTPUT_DIR / "trm_architecture.py").write_text(arch_src)
api.upload_file(
    path_or_fileobj=str(OUTPUT_DIR / "trm_architecture.py"),
    path_in_repo="trm_architecture.py",
    repo_id=OUTPUT_REPO,
    repo_type="model",
)

# Upload training metrics
metrics = {
    "n_params": n_params,
    "n_train": len(train_inputs),
    "n_eval": len(eval_inputs),
    "n_epochs": n_epochs,
    "best_epoch": best_epoch,
    "best_eval_acc": best_eval_acc,
    "batch_size": batch_size,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "device": str(device),
}
(OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
api.upload_file(
    path_or_fileobj=str(OUTPUT_DIR / "training_metrics.json"),
    path_in_repo="training_metrics.json",
    repo_id=OUTPUT_REPO,
    repo_type="model",
)

log(f"Uploaded to {OUTPUT_REPO}!")
log("=== Training complete ===")
