#!/usr/bin/env python3
"""Standalone safety LoRA training script. Runs to completion, writes log.
Launch from a marimo cell with subprocess.Popen — survives WebSocket drops.
"""
import json, sys, time, subprocess
from pathlib import Path

LOG = Path("/marimo/training.log")
DONE = Path("/marimo/training.done")
MODEL_DIR = Path("/marimo/Ternary-Bonsai-8B-FP16")
DATA_PATH = Path("/marimo/safety_dataset_500.jsonl")
OUTPUT_DIR = Path("/marimo/safety-lora-8b")

def log(msg):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=== Safety LoRA Training ===")

# ── Install deps ──
log("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "peft", "huggingface_hub", "accelerate"],
    check=True)
log("Packages installed")

import torch
import urllib.request
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import snapshot_download

# ── GPU check ──
if not torch.cuda.is_available():
    log("FATAL: No GPU available")
    sys.exit(1)
log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# ── Download model ──
if not MODEL_DIR.exists():
    log("Downloading model (~16 GB, one-time)...")
    snapshot_download("prism-ml/Ternary-Bonsai-8B-unpacked",
        local_dir=str(MODEL_DIR),
        ignore_patterns=[".gitattributes", "README.md", "LICENSE", "NOTICE.txt"])
log(f"Model ready: {sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file())/1e9:.1f} GB")

# ── Download dataset ──
if not DATA_PATH.exists():
    log("Downloading dataset...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Seamus05/bitmajor-trilogy/master/data/safety_dataset_500.jsonl",
        str(DATA_PATH))
_safety_data = [json.loads(l) for l in open(DATA_PATH)]
log(f"Dataset: {len(_safety_data)} queries")

# ── Format data ──
_texts = []
for _ex in _safety_data:
    _text = (
        "Classify this query as SAFE, UNSAFE, or ESCALATE.\n\n"
        "SAFE — harmless question with no safety concern.\n\n"
        "UNSAFE — the query explicitly asks for harmful, illegal, or dangerous "
        "content or instructions. The user wants to DO the dangerous thing.\n\n"
        "ESCALATE — the topic is sensitive but the intent is legitimate.\n\n"
        "Query: " + _ex["query"] + "\n\n"
        "Classification (SAFE, UNSAFE, or ESCALATE): " + _ex["label"]
    )
    _texts.append(_text)
_train_dataset = Dataset.from_dict({"text": _texts}).train_test_split(test_size=0.2, seed=42)
log(f"Formatted: {len(_train_dataset['train'])} train, {len(_train_dataset['test'])} eval")

# ── Load model ──
log("Loading tokenizer...")
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
_tokenizer.pad_token = _tokenizer.eos_token

log("Loading model (FP16 → GPU via device_map)...")
_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR), torch_dtype=torch.float16,
    device_map="cuda:0", trust_remote_code=True)

log("Applying LoRA...")
_model = get_peft_model(_model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
_model.print_trainable_parameters()

# ── Tokenize ──
def _tok(examples):
    r = _tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")
    r["labels"] = r["input_ids"].copy()
    return r
_tokenized_train = _train_dataset["train"].map(_tok, batched=True, remove_columns=["text"])
_tokenized_eval = _train_dataset["test"].map(_tok, batched=True, remove_columns=["text"])

_train_ids = torch.tensor([x["input_ids"] for x in _tokenized_train])
_train_labels = torch.tensor([x["labels"] for x in _tokenized_train])
_eval_ids = torch.tensor([x["input_ids"] for x in _tokenized_eval])
_eval_labels = torch.tensor([x["labels"] for x in _tokenized_eval])

from torch.utils.data import DataLoader, TensorDataset
_train_loader = DataLoader(TensorDataset(_train_ids, _train_labels), batch_size=1, shuffle=True)
_eval_loader = DataLoader(TensorDataset(_eval_ids, _eval_labels), batch_size=1)
log(f"Tokenized: {len(_train_loader)} train batches, {len(_eval_loader)} eval batches")

# ── Train ──
_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, _model.parameters()), lr=2e-4, weight_decay=0.01)
_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=3 * len(_train_loader))
_accum_steps = 32

log("Training (3 epochs, ~2-3 hours)...")
for _epoch in range(3):
    _model.train()
    _train_loss = 0.0
    _optimizer.zero_grad()
    for _i, (_batch_ids, _batch_labels) in enumerate(_train_loader):
        _batch_ids, _batch_labels = _batch_ids.to("cuda"), _batch_labels.to("cuda")
        _loss = _model(input_ids=_batch_ids, labels=_batch_labels).loss / _accum_steps
        _loss.backward()
        if (_i + 1) % _accum_steps == 0:
            _optimizer.step(); _optimizer.zero_grad(); _scheduler.step()
        _train_loss += _loss.item() * _accum_steps
        if _i % 50 == 0:
            log(f"  step {_i}: loss={_loss.item() * _accum_steps:.4f}")

    _model.eval()
    _eval_loss = 0.0
    with torch.no_grad():
        for _batch_ids, _batch_labels in _eval_loader:
            _batch_ids, _batch_labels = _batch_ids.to("cuda"), _batch_labels.to("cuda")
            _eval_loss += _model(input_ids=_batch_ids, labels=_batch_labels).loss.item()

    _avg_train = _train_loss / len(_train_loader)
    _avg_eval = _eval_loss / len(_eval_loader)
    log(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")

    # ── Save and upload checkpoint after each epoch (survives kernel restarts) ──
    log(f"Saving epoch {_epoch+1} checkpoint...")
    _model.save_pretrained(str(OUTPUT_DIR))
    _tokenizer.save_pretrained(str(OUTPUT_DIR))
    _size_mb = (OUTPUT_DIR / "adapter_model.safetensors").stat().st_size / 1024 / 1024
    log(f"Checkpoint: {_size_mb:.1f} MB")

    _hf_token = os.environ.get("HF_TOKEN", "")
    if _hf_token:
        log(f"Uploading epoch {_epoch+1} to HuggingFace...")
        from huggingface_hub import HfApi
        _api = HfApi()
        _api.upload_folder(
            folder_path=str(OUTPUT_DIR),
            repo_id="Seamus05/safety-lora-8b",
            repo_type="model",
            token=_hf_token,
        )
        log(f"Uploaded epoch {_epoch+1} to HuggingFace")
    else:
        log("No HF_TOKEN — skipping upload")

# ── Signal completion ──
DONE.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\nadapter: {_size_mb:.1f} MB")
log("=== Training complete ===")
