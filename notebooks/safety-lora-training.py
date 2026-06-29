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

**Hardware:** RTX Pro 6000 (102 GB VRAM) — estimated 2-3 hours.
Data auto-downloaded from GitHub. 4-bit loading to avoid OOM.
""")
    return (
        AutoModelForCausalLM, AutoTokenizer,
        DataLoader, Dataset, LoraConfig, Path, TaskType, TensorDataset,
        get_peft_model, json, mo, snapshot_download, torch, urllib,
    )


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM, AutoTokenizer,
    DataLoader, Dataset, LoraConfig, Path, TaskType, TensorDataset,
    get_peft_model, json, mo, snapshot_download, torch, urllib,
):
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

    # Load tokenizer + model (FP16 — clean GPU, no other notebooks)
    _tokenizer = AutoTokenizer.from_pretrained(str(_model_dir), trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        str(_model_dir),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")

    # LoRA
    _model = get_peft_model(_model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    _model.print_trainable_parameters()

    # Tokenize
    def _tok(examples):
        r = _tokenizer(examples["text"], max_length=128, truncation=True, padding="max_length")
        r["labels"] = r["input_ids"].copy()
        return r

    _tokenized_train = _train_dataset["train"].map(_tok, batched=True, remove_columns=["text"])
    _tokenized_eval = _train_dataset["test"].map(_tok, batched=True, remove_columns=["text"])

    _train_input_ids = torch.tensor([x["input_ids"] for x in _tokenized_train])
    _train_labels = torch.tensor([x["labels"] for x in _tokenized_train])
    _eval_input_ids = torch.tensor([x["input_ids"] for x in _tokenized_eval])
    _eval_labels = torch.tensor([x["labels"] for x in _tokenized_eval])

    # Training loop
    _train_loader = DataLoader(TensorDataset(_train_input_ids, _train_labels), batch_size=1, shuffle=True)
    _eval_loader = DataLoader(TensorDataset(_eval_input_ids, _eval_labels), batch_size=1)

    _optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, _model.parameters()),
        lr=2e-4, weight_decay=0.01,
    )
    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=3 * len(_train_loader))
    _accum_steps = 32

    mo.md("Training (2-3 hours on RTX Pro 6000)...")

    for _epoch in range(3):
        _model.train()
        _train_loss = 0.0
        _optimizer.zero_grad()
        for _i, (_batch_ids, _batch_labels) in enumerate(_train_loader):
            _batch_ids = _batch_ids.to("cuda")
            _batch_labels = _batch_labels.to("cuda")
            _out = _model(input_ids=_batch_ids, labels=_batch_labels)
            _loss = _out.loss / _accum_steps
            _loss.backward()
            if (_i + 1) % _accum_steps == 0:
                _optimizer.step()
                _optimizer.zero_grad()
                _scheduler.step()
            _train_loss += _loss.item() * _accum_steps
            if _i % 10 == 0:
                print(f"  step {_i}: loss={_loss.item() * _accum_steps:.4f}")

        _model.eval()
        _eval_loss = 0.0
        with torch.no_grad():
            for _batch_ids, _batch_labels in _eval_loader:
                _batch_ids = _batch_ids.to("cuda")
                _batch_labels = _batch_labels.to("cuda")
                _out = _model(input_ids=_batch_ids, labels=_batch_labels)
                _eval_loss += _out.loss.item()

        _avg_train = _train_loss / len(_train_loader)
        _avg_eval = _eval_loss / len(_eval_loader)
        print(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")
        mo.md(f"Epoch {_epoch+1}/3 | train_loss={_avg_train:.4f} | eval_loss={_avg_eval:.4f}")

    _output_dir = "/marimo/safety-lora-8b"
    _model.save_pretrained(_output_dir)
    _tokenizer.save_pretrained(_output_dir)
    _size_mb = (Path(_output_dir)/'adapter_model.safetensors').stat().st_size/1024/1024
    mo.md(f"Done! Adapter: {_size_mb:.1f} MB")

    # Upload to HuggingFace
    _hf_token = mo.ui.text(label="HF Token (write access)", kind="password")
    _hf_repo = mo.ui.text(label="HF Repo", value="prism-ml/safety-lora-8b")
    mo.md("## Upload to HuggingFace\n\nEnter your HF token and repo name, then click Upload.")
    mo.hstack([_hf_token, _hf_repo])

    _upload_btn = mo.ui.run_button(label="Upload to HuggingFace")
    _upload_btn

    if _upload_btn.value:
        import os
        os.environ["HF_TOKEN"] = _hf_token.value
        from huggingface_hub import HfApi
        _api = HfApi()
        _api.upload_folder(
            folder_path=_output_dir,
            repo_id=_hf_repo.value,
            repo_type="model",
        )
        mo.md(f"✅ Uploaded to HuggingFace: {_hf_repo.value}")

    return


if __name__ == "__main__":
    app.run()
