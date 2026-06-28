import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json
    import torch
    import urllib.request
    from pathlib import Path
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from huggingface_hub import snapshot_download

    mo.md("""# Safety LoRA Training — Ternary-Bonsai-8B

Trains a LoRA adapter on the 500-query safety classification dataset.
Target: ≥12/13 adversarial (currently 8/13 with prompted 8B classifier).

**Hardware:** RTX Pro 6000 (96 GB VRAM) — estimated 2-3 hours.
Data auto-downloaded from GitHub. Fully self-contained.
""")

    # Step 1: Download model
    _model_dir = Path("/marimo/Ternary-Bonsai-8B-FP16")
    if not _model_dir.exists():
        mo.md("⏳ Downloading model from prism-ml/Ternary-Bonsai-8B-unpacked (~16 GB)...")
        snapshot_download(
            "prism-ml/Ternary-Bonsai-8B-unpacked",
            local_dir=str(_model_dir),
            ignore_patterns=[".gitattributes", "README.md", "LICENSE", "NOTICE.txt"],
        )
        mo.md("✅ Model downloaded")
    else:
        _size_gb = sum(f.stat().st_size for f in _model_dir.rglob("*") if f.is_file()) / 1e9
        mo.md(f"✅ Model cached ({_size_gb:.1f} GB)")

    # Step 2: Download safety dataset from GitHub
    _data_path = Path("/marimo/safety_dataset_500.jsonl")
    if not _data_path.exists():
        mo.md("⏳ Downloading safety dataset from GitHub...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/Seamus05/bitmajor-trilogy/master/data/safety_dataset_500.jsonl",
            str(_data_path),
        )
        mo.md(f"✅ Downloaded: {_data_path.stat().st_size / 1024:.0f} KB")
    else:
        mo.md(f"✅ Dataset cached: {_data_path.stat().st_size / 1024:.0f} KB")

    _safety_data = [json.loads(l) for l in open(_data_path)]
    mo.md(f"✅ Loaded {len(_safety_data)} queries ({sum(1 for d in _safety_data if d['label']=='SAFE')} SAFE, {sum(1 for d in _safety_data if d['label']=='UNSAFE')} UNSAFE, {sum(1 for d in _safety_data if d['label']=='ESCALATE')} ESCALATE)")

    # Step 3: Format as causal LM text
    _texts = []
    for _ex in _safety_data:
        _text = (
            "Classify this query as SAFE, UNSAFE, or ESCALATE.\n\n"
            "SAFE — harmless question with no safety concern.\n\n"
            "UNSAFE — the query explicitly asks for harmful, illegal, or dangerous "
            "content or instructions. The user wants to DO the dangerous thing.\n\n"
            "ESCALATE — the topic is sensitive but the intent is legitimate. This "
            "includes academic research, professional security work, medical ethics, "
            "legal self-education, journalism, and harm reduction. The user is "
            "STUDYING or UNDERSTANDING the topic, not asking to do it.\n\n"
            "Key distinction: if the user is STUDYING the dangerous topic, that's "
            "ESCALATE. If the user is ASKING TO DO the dangerous thing, that's UNSAFE.\n\n"
            f"Query: {_ex['query']}\n\n"
            f"Classification (SAFE, UNSAFE, or ESCALATE): {_ex['label']}"
        )
        _texts.append(_text)

    _train_dataset = Dataset.from_dict({"text": _texts}).train_test_split(test_size=0.2, seed=42)
    mo.md(f"✅ Formatted: {len(_train_dataset['train'])} train, {len(_train_dataset['test'])} eval")

    # Step 4: Load tokenizer and model
    mo.md("⏳ Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(str(_model_dir), trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    mo.md("⏳ Loading model...")
    _model = AutoModelForCausalLM.from_pretrained(
        str(_model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.gradient_checkpointing_enable()

    # Step 5: Configure LoRA
    _lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    _model = get_peft_model(_model, _lora_config)
    _model.print_trainable_parameters()

    # Step 6: Tokenize
    def _tok(examples):
        r = _tokenizer(examples["text"], max_length=256, truncation=True, padding="max_length")
        r["labels"] = r["input_ids"].copy()
        return r

    _tokenized_train = _train_dataset["train"].map(_tok, batched=True, remove_columns=["text"])
    _tokenized_eval = _train_dataset["test"].map(_tok, batched=True, remove_columns=["text"])

    # Step 7: Train
    _training_args = TrainingArguments(
        output_dir="/marimo/safety-lora-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    _data_collator = DataCollatorForLanguageModeling(tokenizer=_tokenizer, mlm=False)

    _trainer = Trainer(
        model=_model,
        args=_training_args,
        train_dataset=_tokenized_train,
        eval_dataset=_tokenized_eval,
        data_collator=_data_collator,
    )

    mo.md("⏳ Training (2-3 hours on RTX Pro 6000)...")
    _trainer.train()

    # Step 8: Save
    _output_dir = "/marimo/safety-lora-8b"
    _model.save_pretrained(_output_dir)
    _tokenizer.save_pretrained(_output_dir)

    _adapter_path = Path(_output_dir) / "adapter_model.safetensors"
    _size_mb = _adapter_path.stat().st_size / 1024 / 1024 if _adapter_path.exists() else 0
    mo.md(f"✅ Training complete! Adapter saved: {_size_mb:.1f} MB")

    return


if __name__ == "__main__":
    app.run()
