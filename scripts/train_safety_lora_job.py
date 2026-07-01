# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "peft",
#   "huggingface_hub",
#   "accelerate",
# ]
# ///

"""Safety LoRA training — submitted via HF Jobs. Runs on managed GPU, pushes to Hub."""
import json, os, sys, time
from pathlib import Path

import torch
import urllib.request
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import snapshot_download, HfApi

# ── Config ──
MODEL_ID = "prism-ml/Ternary-Bonsai-8B-unpacked"
DATASET_URL = "https://raw.githubusercontent.com/Seamus05/bitmajor-trilogy/master/data/safety_dataset_500.jsonl"
OUTPUT_REPO = "Seamus05/safety-lora-8b"
HF_TOKEN = os.environ["HF_TOKEN"]

MODEL_DIR = Path("/tmp/model")
DATA_PATH = Path("/tmp/safety_dataset_500.jsonl")
OUTPUT_DIR = Path("/tmp/safety-lora-8b")

def log(msg):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)

log("=== Safety LoRA Training (HF Jobs) ===")
log(f"GPU: {torch.cuda.get_device_name(0)}")
log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# ── Download model ──
log("Downloading model (~16 GB)...")
snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR),
    ignore_patterns=[".gitattributes", "README.md", "LICENSE", "NOTICE.txt"])
log(f"Model ready: {sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file())/1e9:.1f} GB")

# ── Download dataset ──
log("Downloading dataset...")
urllib.request.urlretrieve(DATASET_URL, str(DATA_PATH))
_safety_data = [json.loads(l) for l in open(DATA_PATH)]
log(f"Dataset: {len(_safety_data)} queries")

# ── Format ──
_texts = []
for _ex in _safety_data:
    _text = "Classify: SAFE, UNSAFE, or ESCALATE. Query: " + _ex["query"] + " Label: " + _ex["label"]
    _texts.append(_text)
_train_dataset = Dataset.from_dict({"text": _texts}).train_test_split(test_size=0.2, seed=42)
log(f"Formatted: {len(_train_dataset['train'])} train, {len(_train_dataset['test'])} eval")

# ── Load model ──
log("Loading model...")
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
_tokenizer.pad_token = _tokenizer.eos_token
_model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR), torch_dtype=torch.float16,
    device_map="cuda:0", trust_remote_code=True)
_model = get_peft_model(_model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
log("Model loaded + LoRA applied")

# ── Tokenize ──
def _tok(examples):
    r = _tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")
    r["labels"] = r["input_ids"].copy()
    return r
_tokenized_train = _train_dataset["train"].map(_tok, batched=True, remove_columns=["text"])
_tokenized_eval = _train_dataset["test"].map(_tok, batched=True, remove_columns=["text"])

from torch.utils.data import DataLoader, TensorDataset
_train_ids = torch.tensor([x["input_ids"] for x in _tokenized_train])
_train_labels = torch.tensor([x["labels"] for x in _tokenized_train])
_eval_ids = torch.tensor([x["input_ids"] for x in _tokenized_eval])
_eval_labels = torch.tensor([x["labels"] for x in _tokenized_eval])
_train_loader = DataLoader(TensorDataset(_train_ids, _train_labels), batch_size=1, shuffle=True)
_eval_loader = DataLoader(TensorDataset(_eval_ids, _eval_labels), batch_size=1)

_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, _model.parameters()), lr=2e-4, weight_decay=0.01)
_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=3 * len(_train_loader))
_accum_steps = 32

# ── Train + upload after each epoch ──
log("Training (3 epochs)...")
for _epoch in range(3):
    _model.train(); _train_loss = 0.0; _optimizer.zero_grad()
    for _i, (_batch_ids, _batch_labels) in enumerate(_train_loader):
        _batch_ids, _batch_labels = _batch_ids.to("cuda"), _batch_labels.to("cuda")
        _loss = _model(input_ids=_batch_ids, labels=_batch_labels).loss / _accum_steps
        _loss.backward()
        if (_i + 1) % _accum_steps == 0:
            _optimizer.step(); _optimizer.zero_grad(); _scheduler.step()
        _train_loss += _loss.item() * _accum_steps
        if _i % 50 == 0:
            log(f"  step {_i}: loss={_loss.item() * _accum_steps:.4f}")
    _model.eval(); _eval_loss = 0.0
    with torch.no_grad():
        for _batch_ids, _batch_labels in _eval_loader:
            _batch_ids, _batch_labels = _batch_ids.to("cuda"), _batch_labels.to("cuda")
            _eval_loss += _model(input_ids=_batch_ids, labels=_batch_labels).loss.item()
    _avg_train = _train_loss / len(_train_loader)
    _avg_eval = _eval_loss / len(_eval_loader)
    log(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")

    # Save + upload
    log(f"Saving epoch {_epoch+1}...")
    _model.save_pretrained(str(OUTPUT_DIR))
    _tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Fix README — HF rejects local paths
    _readme = OUTPUT_DIR / "README.md"
    if _readme.exists():
        _content = _readme.read_text()
        _content = _content.replace(f"base_model: {MODEL_DIR}", f"base_model: {MODEL_ID}")
        _readme.write_text(_content)

    _size_mb = (OUTPUT_DIR / "adapter_model.safetensors").stat().st_size / 1024 / 1024
    log(f"Checkpoint: {_size_mb:.1f} MB")

    log(f"Uploading epoch {_epoch+1} to HuggingFace...")
    _api = HfApi(token=HF_TOKEN)
    _api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=OUTPUT_REPO,
        repo_type="model",
    )
    log(f"Uploaded epoch {_epoch+1} to {OUTPUT_REPO}!")

log("=== Training complete ===")
