# BITMAJOR Trilogy

Three interactive marimo notebooks for the [alphaXiv/marimo competition](https://marimo.io/pages/events/notebook-competition-2) (deadline July 8, 2026).

**Narrative arc:** we made them fast → we made them smart → we built a system that uses both and found the limits.

## Notebooks

| # | Notebook | What it covers |
|---|----------|---------------|
| 1 | `bitnet-bonsai.py` | BITMAJOR format — 256-wide blocks, bit-major packing, matches TQ2_0 within 3%. Benchmarks on Zen 2/3, KV cache analysis, interactive hardware projector. |
| 2 | `tiny-recursive-reasoning.py` | TRM (Tiny Recursive Model) — 7M params, 2 layers, no attention. Recursive reasoning beats 671B LLMs on Sudoku. Harness test results. |
| 3 | `self-harness.py` | 3-tier cascade — 1.7B router → 4B SIMPLE / 8B REASONING + 8B safety classifier + TRM content review. 105-query test results. |

## Training notebooks (run on molab with GPU)

| Notebook | What it does | Data |
|----------|-------------|------|
| `safety-lora-training.py` | Trains LoRA adapter on 8B model for safety classification (SAFE/UNSAFE/ESCALATE) | `data/safety_dataset_500.jsonl` |
| `trm-training.py` | Trains TRM content review model (7M params) to detect errors in model responses | `data/trm_content_review_25k.jsonl` |

## Running on molab

1. Go to [molab.marimo.io](https://molab.marimo.io)
2. Create a synced notebook from this GitHub repo
3. Toggle GPU on (RTX Pro 6000, 96 GB VRAM)
4. Upload the data file from `data/` through the file sidebar
5. Run the cells

## Models

- **Ternary-Bonsai-1.7B/4B/8B** — Q2_0_BITMAJOR quantized (596 MB / 1.17 GB / 2.29 GB)
- **Base model:** [prism-ml/Ternary-Bonsai-8B-unpacked](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-unpacked) (FP16)
- **Fork:** [github.com/Local-Yokel/llama.cpp-ly](https://github.com/Local-Yokel/llama.cpp-ly)

## License

MIT
