import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json, torch, urllib.request
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import snapshot_download

    mo.md("""# Safety LoRA Training — Ternary-Bonsai-8B

Trains a LoRA adapter on the 500-query safety classification dataset.
Target: ≥12/13 adversarial (currently 8/13 with prompted 8B classifier).

**Hardware:** RTX Pro 6000 (96 GB VRAM) — estimated 2-3 hours.
Data auto-downloaded from GitHub. Raw PyTorch training loop — no Trainer/accelerate dependency.
""")

    # Download model
    _model_dir = Path("/marimo/Ternary-Bonsai-8B-FP16")
    if not _model_dir.exists():
        mo.md("Downloading model (~16 GB)...")
        snapshot_download("prism-ml/Ternary-Bonsai-8B-unpacked", local_dir=str(_model_dir), ignore_patterns=[".gitattributes", "README.md", "LICENSE", "NOTICE.txt"])
        mo.md("Model downloaded")
    else:
        mo.md(f"Model cached ({sum(f.stat().st_size for f in _model_dir.rglob('*') if f.is_file())/1e9:.1f} GB)")

    # Download dataset
    _data_path = Path("/marimo/safety_dataset_500.jsonl")
    if not _data_path.exists():
        mo.md("Downloading safety dataset...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/Seamus05/bitmajor-trilogy/master/data/safety_dataset_500.jsonl", str(_data_path))
    mo.md(f"Dataset: {_data_path.stat().st_size/1024:.0f} KB")

    _safety_data = [json.loads(l) for l in open(_data_path)]
    mo.md(f"Loaded {len(_safety_data)} queries ({sum(1 for d in _safety_data if d['label']=='SAFE')} SAFE, {sum(1 for d in _safety_data if d['label']=='UNSAFE')} UNSAFE, {sum(1 for d in _safety_data if d['label']=='ESCALATE')} ESCALATE)")

    # Format as causal LM text
    _texts = []
    for _ex in _safety_data:
        _text = "Classify this query as SAFE, UNSAFE, or ESCALATE.\n\nSAFE — harmless question with no safety concern.\n\nUNSAFE — the query explicitly asks for harmful, illegal, or dangerous content or instructions. The user wants to DO the dangerous thing.\n\nESCALATE — the topic is sensitive but the intent is legitimate. This includes academic research, professional security work, medical ethics, legal self-education, journalism, and harm reduction. The user is STUDYING or UNDERSTANDING the topic, not asking to do it.\n\nKey distinction: if the user is STUDYING the dangerous topic, that's ESCALATE. If the user is ASKING TO DO the dangerous thing, that's UNSAFE.\n\nQuery: " + _ex["query"] + "\n\nClassification (SAFE, UNSAFE, or ESCALATE): " + _ex["label"]
        _texts.append(_text)

    _train_dataset = Dataset.from_dict({"text": _texts}).train_test_split(test_size=0.2, seed=42)
    mo.md(f"Formatted: {len(_train_dataset['train'])} train, {len(_train_dataset['test'])} eval")

    # Load tokenizer + model
    _tokenizer = AutoTokenizer.from_pretrained(str(_model_dir), trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    # Load in 4-bit to avoid OOM (FP16 fills ~94 GB of 102 GB VRAM)
    from transformers import BitsAndBytesConfig
    _bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        str(_model_dir),
        quantization_config=_bnb_config,
        trust_remote_code=True,
    )
    _model.gradient_checkpointing_enable()

    # LoRA
    _model = get_peft_model(_model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    _model.print_trainable_parameters()

    # Tokenize
    def _tok(examples):
        r = _tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length")
        r["labels"] = r["input_ids"].copy()
        return r

    _tokenized_train = _train_dataset["train"].map(_tok, batched=True, remove_columns=["text"])
    _tokenized_eval = _train_dataset["test"].map(_tok, batched=True, remove_columns=["text"])

    # Convert to tensors
    _train_input_ids = torch.tensor([x["input_ids"] for x in _tokenized_train])
    _train_labels = torch.tensor([x["labels"] for x in _tokenized_train])
    _eval_input_ids = torch.tensor([x["input_ids"] for x in _tokenized_eval])
    _eval_labels = torch.tensor([x["labels"] for x in _tokenized_eval])

    # Raw PyTorch training loop (no Trainer/accelerate dependency)
    _train_loader = DataLoader(TensorDataset(_train_input_ids, _train_labels), batch_size=1, shuffle=True)
    _eval_loader = DataLoader(TensorDataset(_eval_input_ids, _eval_labels), batch_size=1)

    _optimizer = torch.optim.AdamW(_model.parameters(), lr=2e-4, weight_decay=0.01)
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=3 * len(_train_loader))
    _accum_steps = 32
    _scaler = torch.amp.GradScaler("cuda")

    mo.md("Training (2-3 hours on RTX Pro 6000)...")

    for _epoch in range(3):
        _model.train()
        _train_loss = 0.0
        _optimizer.zero_grad()
        for _i, (_batch_ids, _batch_labels) in enumerate(_train_loader):
            _batch_ids = _batch_ids.to("cuda")
            _batch_labels = _batch_labels.to("cuda")
            with torch.amp.autocast("cuda"):
                _out = _model(input_ids=_batch_ids, labels=_batch_labels)
                _loss = _out.loss / _accum_steps
            _scaler.scale(_loss).backward()
            if (_i + 1) % _accum_steps == 0:
                _scaler.step(_optimizer)
                _scaler.update()
                _optimizer.zero_grad()
                _scheduler.step()
            _train_loss += _loss.item() * _accum_steps
            if _i % 10 == 0:
                print(f"  step {_i}: loss={_loss.item() * _accum_steps:.4f}")

        # Eval
        _model.eval()
        _eval_loss = 0.0
        with torch.no_grad():
            for _batch_ids, _batch_labels in _eval_loader:
                _batch_ids = _batch_ids.to("cuda")
                _batch_labels = _batch_labels.to("cuda")
                with torch.amp.autocast("cuda"):
                    _out = _model(input_ids=_batch_ids, labels=_batch_labels)
                    _eval_loss += _out.loss.item()

        _avg_train = _train_loss / len(_train_loader)
        _avg_eval = _eval_loss / len(_eval_loader)
        print(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")
        mo.md(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")

    # Save
    _output_dir = "/marimo/safety-lora-8b"
    _model.save_pretrained(_output_dir)
    _tokenizer.save_pretrained(_output_dir)
    mo.md(f"Done! Adapter: {(Path(_output_dir)/'adapter_model.safetensors').stat().st_size/1024/1024:.1f} MB")

    return


if __name__ == "__main__":
    app.run()
