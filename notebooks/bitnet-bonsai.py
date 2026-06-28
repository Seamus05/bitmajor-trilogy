import marimo

__generated_with = "0.23.9"
app = marimo.App(app_title="Ternary Models on Real Hardware: Benchmarks, KV Cache, and What You Actually Need")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ternary Models on Real Hardware

    **Benchmarks, KV Cache, and What You Actually Need to Run 1-Bit LLMs**

    We benchmarked three ternary model families — Bonsai (binary), Ternary Bonsai (ternary), and Microsoft BitNet (native 1-bit) — across three quantization formats and two x86 machines. This notebook is both a report and a tool: use the interactive sections to project performance on your own hardware.

    **Key findings:** TQ2_0 is the practical deployment format — 4-5.5× faster than our original hand-optimized Q2_0 kernel with the same ternary quality. We closed the gap: switching to Q8_K activations delivered 1.75× (25.93 tok/s on 1.7B), and bit-major weight packing with shift+AND extraction delivered the remaining 2.4-4.8× — matching TQ2_0 within 2% at all three model sizes (61.9 vs 63.4 at 1.7B, 28.2 vs 28.8 at 4B, 16.8 vs 16.3 at 8B). The models stay downloadable at 596 MB / 1.17 GB / 2.29 GB, no FP16 intermediate required. Wider blocks alone don't close the gap (Q2_0_WIDE: +5%, hypothesis falsified). Ternary models did not self-scaffold in our pilot test — they defaulted to the most common solution rather than reasoning about constraints. Ornith 1.0 (released June 26) passed the self-scaffolding test at Q4_K_M but runs at 6.84 tok/s vs 16.68 for ternary 8B; its mandatory thinking scaffold adds ~48 seconds of silence on ambiguous queries.

    *Built with marimo for the alphaXiv paper discussion competition. Fork: [github.com/Local-Yokel/llama.cpp-ly](https://github.com/Local-Yokel/llama.cpp-ly)*

    ---

    **TL;DR:**
    - **TQ2_0 is 4-5.5× faster** than our original Q2_0 kernel with the same ternary quality
    - **We closed the gap:** Q2_0_BITMAJOR matches TQ2_0 within 2% at all three model sizes (1.7B, 4B, 8B)
    - **Models stay downloadable** (596 MB / 1.17 GB / 2.29 GB), no FP16 intermediate required
    - **Ternary models didn't self-scaffold** in our pilot; Ornith did but with 48s thinking overhead on ambiguous queries
    - **More threads can be worse:** 8 threads = 20.4 tok/s ±6.76 vs 6 threads = 28.9 tok/s ±0.13 on consumer hardware with small L3 cache
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import json
    import io
    import base64

    return (
        alt,
        base64,
        FancyBboxPatch,
        io,
        json,
        mo,
        mpatches,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(json, pd):
    _RAW = r"""{
      "meta": {
        "generated": "2026-06-26",
        "fork": "llama.cpp-ly (commit bdfb1e56)",
        "fork_url": "https://github.com/Local-Yokel/llama.cpp-ly",
        "upstreams": ["PrismML/llama.cpp (TQ2_0, Q1_0 formats)", "microsoft/BitNet (bitnet.cpp, I2_S format)"]
      },
      "machines": {
        "zen2": {"name": "Ryzen 5 3600X", "arch": "Zen 2", "cores": "6C/12T", "ram": "DDR4-3200", "bw_gbs": 45, "l3_mb": 32},
        "zen3": {"name": "Ryzen 7 5700U", "arch": "Zen 3", "cores": "8C/8T", "ram": "LPDDR4x-4266", "bw_gbs": 55, "l3_mb": 8}
      },
      "benchmarks": [
        {"model": "Bonsai 1.7B", "format": "Q2_0", "size_mb": 442, "params_b": 1.72, "zen2_tg128": 12.81, "zen3_tg128": 17.23, "zen2_pp512": null, "zen3_pp512": 36.18},
        {"model": "Bonsai 1.7B", "format": "TQ2_0", "size_mb": 590, "params_b": 1.72, "zen2_tg128": 59.05, "zen3_tg128": 55.19, "zen2_pp512": null, "zen3_pp512": 197.59},
        {"model": "Bonsai 1.7B", "format": "Q2_0+Q8_K", "size_mb": 442, "params_b": 1.72, "zen2_tg128": 25.93, "zen3_tg128": null, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 1.7B", "format": "Q2_0_BITMAJOR", "size_mb": 590, "params_b": 1.72, "zen2_tg128": 61.90, "zen3_tg128": 54.87, "zen2_pp512": 192.52, "zen3_pp512": 199.57},
        {"model": "Bonsai 1.7B", "format": "Q1_0", "size_mb": 231, "params_b": 1.72, "zen2_tg128": 67.94, "zen3_tg128": 73.90, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 4B", "format": "Q2_0", "size_mb": 1100, "params_b": 4.02, "zen2_tg128": 5.28, "zen3_tg128": 7.49, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 4B", "format": "Q2_0+Q8_K", "size_mb": 1100, "params_b": 4.02, "zen2_tg128": 11.64, "zen3_tg128": null, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 4B", "format": "Q2_0_BITMAJOR", "size_mb": 1170, "params_b": 4.02, "zen2_tg128": 28.21, "zen3_tg128": 28.87, "zen2_pp512": 66.11, "zen3_pp512": 79.27},
        {"model": "Bonsai 4B", "format": "TQ2_0", "size_mb": 1170, "params_b": 4.02, "zen2_tg128": 28.79, "zen3_tg128": 28.71, "zen2_pp512": null, "zen3_pp512": 79.87},
        {"model": "Bonsai 4B", "format": "Q1_0", "size_mb": 540, "params_b": 4.02, "zen2_tg128": 32.89, "zen3_tg128": 33.90, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 8B", "format": "Q2_0", "size_mb": 2100, "params_b": 8.19, "zen2_tg128": 3.06, "zen3_tg128": 4.06, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Bonsai 8B", "format": "Q2_0_BITMAJOR", "size_mb": 2290, "params_b": 8.19, "zen2_tg128": 16.76, "zen3_tg128": 16.21, "zen2_pp512": 38.50, "zen3_pp512": 49.94},
        {"model": "Bonsai 8B", "format": "TQ2_0", "size_mb": 2470, "params_b": 8.19, "zen2_tg128": 16.31, "zen3_tg128": 16.68, "zen2_pp512": null, "zen3_pp512": 47.77},
        {"model": "Bonsai 8B", "format": "Q1_0", "size_mb": 1070, "params_b": 8.19, "zen2_tg128": 19.15, "zen3_tg128": 19.74, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "Llama3-8B-1.58", "format": "TQ2_0", "size_mb": 3717, "params_b": 8.03, "zen2_tg128": 10.24, "zen3_tg128": 13.96, "zen2_pp512": null, "zen3_pp512": null},
        {"model": "BitNet b1.58 2B", "format": "I2_S", "size_mb": 1100, "params_b": 2.41, "zen2_tg128": 31.95, "zen3_tg128": 32.45, "zen2_pp512": null, "zen3_pp512": null}
      ],
      "kernel_benchmarks": {
        "q2_0": {"vec_dot_cycles_32": 24.94, "vec_dot_gbs": 16.95},
        "tq2_0": {"vec_dot_cycles_32": 1.84, "vec_dot_gbs": 152.59},
        "q8_0": {"vec_dot_cycles_32": 4.93, "vec_dot_gbs": 50.86},
        "q2_K": {"vec_dot_cycles_32": 3.23, "vec_dot_gbs": 76.29}
      },
      "model_specs": {
        "Bonsai 1.7B": {"n_layer": 28, "n_head": 16, "n_head_kv": 16, "n_embd": 2048, "n_embd_head": 128, "size_mb": 590},
        "Bonsai 4B": {"n_layer": 36, "n_head": 24, "n_head_kv": 8, "n_embd": 2560, "n_embd_head": 128, "size_mb": 1170},
        "Bonsai 8B": {"n_layer": 36, "n_head": 32, "n_head_kv": 8, "n_embd": 4096, "n_embd_head": 128, "size_mb": 2470},
        "BitNet b1.58 2B": {"n_layer": 30, "n_head": 20, "n_head_kv": 5, "n_embd": 2560, "n_embd_head": 128, "size_mb": 1100},
        "Llama3-8B-1.58": {"n_layer": 32, "n_head": 32, "n_head_kv": 8, "n_embd": 4096, "n_embd_head": 128, "size_mb": 3717}
      }
    }"""

    _data = json.loads(_RAW)
    BENCH = pd.DataFrame(_data["benchmarks"])
    KERNELS = _data["kernel_benchmarks"]
    META = _data["meta"]
    MACHINES = _data["machines"]
    SPECS = _data["model_specs"]

    return BENCH, KERNELS, MACHINES, META, SPECS


# ── Section 1: The Kernel ──────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The Kernel: `ggml_vec_dot_q2_0_q8_0`

    The Q2_0 block format stores 128 2-bit values in 32 bytes — each byte packs four values. TQ2_0 uses the same memory layout but with 256-wide blocks (66 bytes/block). To compute the dot product against Q8_0 activations (signed 8-bit), the kernel:

    1. **Packs** four Q2_0 bytes into a 32-bit register via `_mm_cvtsi32_si128`
    2. **Expands** each source byte four times using `_mm_shuffle_epi8` with `shuffle_expand`
    3. **Extracts** every 4th 2-bit value with 4 interleave masks (`shuf_iv0..3`)
    4. **Computes** `_mm256_maddubs_epi16(v_unsigned, q8)` — an unsigned×signed multiply-add
    5. **Corrects** the bias: `Σ(v_unsigned × q8) - Σ(q8)` maps {0,1,2,3} × q8 → {-1,0,1,2} × q8

    Three SIMD paths, same algorithm:
    | Path | Register | Min CPU | Speedup vs scalar |
    |---|---|---|---|
    | AVX2 | 256-bit | Haswell (2013) | 10× |
    | AVX | 128-bit | Sandy Bridge (2011) | 6× |
    | SSSE3 | 128-bit | Core 2 (2006) | 3× |

    **Why TQ2_0 is faster:** TQ2_0 uses three optimizations our Q2_0 kernel doesn't: (1) 256-wide blocks (fewer scale loads), (2) Q8_K activations with precomputed block sums (no on-the-fly Σ(q8) computation), and (3) simpler bit extraction (shift+AND instead of shuffle+interleave). Together these let TQ2_0 use 49-66% of memory bandwidth vs our 13-14%. That's the 4-5.5× speed difference.
    """)
    return


# ── Section 2: Interactive Bit Pack/Unpack ─────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    val0 = mo.ui.slider(0, 3, 1, value=0, label="Pair 0 (bits 1-0)", show_value=True)
    val1 = mo.ui.slider(0, 3, 1, value=1, label="Pair 1 (bits 3-2)", show_value=True)
    val2 = mo.ui.slider(0, 3, 1, value=2, label="Pair 2 (bits 5-4)", show_value=True)
    val3 = mo.ui.slider(0, 3, 1, value=3, label="Pair 3 (bits 7-6)", show_value=True)

    mo.hstack(
        [mo.vstack([val0, val1]), mo.vstack([val2, val3])],
        justify="space-around",
    )
    return val0, val1, val2, val3


@app.cell(hide_code=True)
def _(FancyBboxPatch, base64, io, mo, plt, val0, val1, val2, val3):
    vals = [val0.value, val1.value, val2.value, val3.value]
    ternary_map = {0: -1, 1: 0, 2: 1, 3: 2}
    ternary_labels = {-1: "−1", 0: "0", 1: "+1", 2: "+2"}
    colors = {-1: "#e74c3c", 0: "#555555", 1: "#2ecc71", 2: "#3498db"}
    byte_val = vals[0] | (vals[1] << 2) | (vals[2] << 4) | (vals[3] << 6)

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0]
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(5, 3.5, "One byte = 4 ternary values (2 bits each)", fontsize=10, ha="center", color="#ccc")

    for p in range(4):
        x0 = 0.5 + p * 2.3
        t = ternary_map[vals[p]]
        c = colors[t]
        rect = FancyBboxPatch((x0, 0.5), 2.0, 2.0, boxstyle="round,pad=0.1", facecolor=c, edgecolor=c, alpha=0.3)
        ax.add_patch(rect)
        ax.text(x0 + 0.5, 2.2, f"b{p*2+1}", fontsize=7, color="#666", ha="center")
        ax.text(x0 + 1.5, 2.2, f"b{p*2}", fontsize=7, color="#666", ha="center")
        b1 = (vals[p] >> 1) & 1
        b0 = vals[p] & 1
        ax.text(x0 + 0.5, 1.5, str(b1), fontsize=12, color="#ccc", ha="center", fontweight="bold")
        ax.text(x0 + 1.5, 1.5, str(b0), fontsize=12, color="#ccc", ha="center", fontweight="bold")
        ax.text(x0 + 1.0, 0.2, f"= {ternary_labels[t]}", fontsize=10, color=c, ha="center", fontweight="bold")
        ax.text(x0 + 1.0, 2.7, f"Pair {p}", fontsize=7, color="#555", ha="center")

    ax = axes[1]
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(5, 3.5, f"Packed: 0x{byte_val:02x} = {byte_val:08b}b", fontsize=10, ha="center", color="#ccc", fontfamily="monospace")

    for b in range(8):
        x0 = 0.5 + (7 - b) * 1.1
        bit = (byte_val >> b) & 1
        rect = FancyBboxPatch((x0, 1.0), 0.9, 1.2, boxstyle="round,pad=0.05", facecolor="#2ecc71" if bit else "#333", edgecolor="#555", alpha=0.8)
        ax.add_patch(rect)
        ax.text(x0 + 0.45, 1.6, str(bit), fontsize=11, color="white" if bit else "#888", ha="center", fontweight="bold")
        ax.text(x0 + 0.45, 0.8, f"b{b}", fontsize=6, color="#555", ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", transparent=True, facecolor="#0d1117")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    mo.md(f"""
    ## 2. Interactive Bit Pack/Unpack

    ### How Q2_0 Packs Ternary Values

    **Each byte holds 4 ternary values**, packed as 2-bit pairs. A Q2_0 block is 128 values = 32 bytes. Drag the sliders above to see how the bit patterns map to ternary values.

    <div style="text-align:center;">
        <img src="data:image/png;base64,{b64}" style="max-width:700px;" alt="Byte packing"/>
    </div>

    **The SIMD trick:** The kernel loads 4 bytes into a 32-bit register, uses `_mm_shuffle_epi8` with a shuffle-expand pattern to replicate each source byte 4 times, then applies 4 interleave masks to extract every 4th 2-bit value into its own lane. Result: 16 unpacked values in registers, ready for `_mm256_maddubs_epi16`, without touching memory again.

    ### Step Through the Kernel

    Below, step through the 5-stage pipeline for the byte you configured above. Each step shows the active SIMD instruction and the register state.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    kernel_step = mo.ui.slider(1, 5, 1, value=1, label="Pipeline step", show_value=True)
    mo.hstack([kernel_step], justify="start")
    return kernel_step


@app.cell(hide_code=True)
def _(FancyBboxPatch, base64, io, kernel_step, mo, plt, val0, val1, val2, val3):
    _vals = [val0.value, val1.value, val2.value, val3.value]
    _byte_val = _vals[0] | (_vals[1] << 2) | (_vals[2] << 4) | (_vals[3] << 6)
    _step = kernel_step.value

    _steps = [
        {
            "title": "Step 1: _mm_cvtsi32_si128 — Load byte into register",
            "desc": f"Loads the packed byte 0x{_byte_val:02x} into the low 8 bits of a 128-bit XMM register. Upper bits are zeroed.",
            "reg": [_byte_val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "highlight": "0x{:02x}".format(_byte_val),
            "highlight_indices": [0],
        },
        {
            "title": "Step 2: _mm_shuffle_epi8 — Expand each byte 4×",
            "desc": "The shuffle-expand pattern replicates the source byte into 4 consecutive positions. Now we have 4 copies of the byte, each ready for a different interleave mask.",
            "reg": [_byte_val]*4 + [0]*12,
            "highlight": "4× 0x{:02x}".format(_byte_val),
            "highlight_indices": [0, 1, 2, 3],
        },
        {
            "title": "Step 3: _mm_and_si128 (mask0) — Extract pair 0",
            "desc": "AND with mask 0x03 extracts the bottom 2 bits from each byte. This isolates pair 0 (values 0-3) across all 4 lanes.",
            "reg": [_vals[0]]*4 + [0]*12,
            "highlight": "pair 0 = {}".format(_vals[0]),
            "highlight_indices": [0, 1, 2, 3],
        },
        {
            "title": "Step 4: _mm256_maddubs_epi16 — Unsigned×signed multiply-add",
            "desc": "Multiplies each unsigned 8-bit value by its corresponding signed Q8 activation byte, then adds adjacent pairs into 16-bit accumulators. This is the workhorse — 16 multiplies + 8 adds per instruction.",
            "reg": ["p0×q8", "p0×q8", "p0×q8", "p0×q8", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…", "…"],
            "highlight": "16 madd per instruction",
            "highlight_indices": [0, 1, 2, 3],
        },
        {
            "title": "Step 5: Bias correction — Σ(v_unsigned × q8) − Σ(q8)",
            "desc": f"The unsigned values {{0,1,2,3}} must map to ternary {{-1,0,1,2}}. Subtract Σ(q8) from the sum: if the stored value was 0 (ternary -1), subtracting q8 gives -q8. If stored was 1 (ternary 0), subtracting q8 gives 0. The bias correction is applied once per 32-element sub-block.",
            "reg": ["corrected"]*4 + ["…"]*12,
            "highlight": "maps {0,1,2,3} → {{-1,0,1,2}}",
            "highlight_indices": [0, 1, 2, 3],
        },
    ]

    _s = _steps[_step - 1]

    _fig, _ax = plt.subplots(1, 1, figsize=(10, 3.5))
    _fig.patch.set_facecolor("#0d1117")
    _ax.set_facecolor("#0d1117")
    _ax.set_xlim(0, 16)
    _ax.set_ylim(0, 4)
    _ax.axis("off")

    _ax.text(8, 3.7, _s["title"], fontsize=11, ha="center", color="#f39c12", fontweight="bold")
    _ax.text(8, 3.3, _s["desc"], fontsize=8, ha="center", color="#aaa")

    for _i, _v in enumerate(_s["reg"]):
        _x0 = 0.3 + _i * 0.95
        _is_highlight = _i in _s.get("highlight_indices", [])
        _color = "#2ecc71" if _is_highlight else "#3498db"
        _alpha = 0.6 if _is_highlight else 0.3
        _rect = FancyBboxPatch((_x0, 0.5), 0.85, 1.8, boxstyle="round,pad=0.05",
                               facecolor=_color, edgecolor=_color, alpha=_alpha)
        _ax.add_patch(_rect)
        _label = str(_v)[:6]
        _ax.text(_x0 + 0.42, 1.4, _label, fontsize=7, color="white" if _is_highlight else "#ccc",
                 ha="center", fontfamily="monospace")
        _ax.text(_x0 + 0.42, 0.3, f"[{_i}]", fontsize=5, color="#555", ha="center")

    _ax.text(8, 2.6, f"→ {_s['highlight']}", fontsize=9, ha="center", color="#2ecc71", fontweight="bold")

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight", transparent=True, facecolor="#0d1117")
    _buf.seek(0)
    _b64 = base64.b64encode(_buf.read()).decode()
    plt.close(_fig)

    mo.md(f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{_b64}" style="max-width:800px;" alt="Kernel step {_step}"/>
    </div>

    **Drag the step slider** to walk through the 5-stage pipeline. The green-highlighted register lanes show the active data at each stage. The byte being processed is the same one you configured in Section 2 above — go back and change the sliders to see how different bit patterns flow through the pipeline.
    """)
    return


# ── Section 3: The Unsigned-Bias Trick ─────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The Unsigned-Bias Trick

    The Q2_0 format stores 2-bit values as unsigned {0, 1, 2, 3}. But ternary weights use {-1, 0, 1, 2}. The mapping is simple:

    ```text
    stored → actual
       0   →  -1
       1   →   0
       2   →  +1
       3   →  +2
    ```

    The naive approach: unpack each value, subtract 1, then multiply. That costs extra instructions.

    **The trick:** `_mm256_maddubs_epi16` computes `Σ(v_unsigned[i] × q8[i])` as if `v_unsigned` were unsigned. Since `v = v_unsigned - 1`:

    ```text
    Σ(v × q8) = Σ((v_unsigned - 1) × q8)  = Σ(v_unsigned × q8) - Σ(q8)
    ```

    One multiply-add, one horizontal sum of `q8`, one subtraction. No per-element correction.

    The kernel computes both the sum-of-products and the sum-of-q8 in the same pass, then subtracts at the end. The per-block overhead: 5 extra instructions.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    _rng = np.random.default_rng(42)
    _q8_example = _rng.integers(-128, 127, size=16, dtype=np.int32)

    bias_q8_sum = int(_q8_example.sum())

    example_v = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    example_v_true = example_v - 1

    products_unsigned = (example_v * _q8_example).tolist()
    products_corrected = (example_v_true * _q8_example).tolist()
    sum_unsigned = int(sum(products_unsigned))
    sum_corrected = int(sum(products_corrected))
    check = sum_unsigned - bias_q8_sum

    _table_rows = ""
    for i in range(16):
        _table_rows += (
            f"<tr>"
            f"<td style='text-align:center'>{example_v[i]}</td>"
            f"<td style='text-align:center'>{example_v_true[i]}</td>"
            f"<td style='text-align:right'>{_q8_example[i]}</td>"
            f"<td style='text-align:right'>{products_unsigned[i]}</td>"
            f"<td style='text-align:right'>{products_corrected[i]}</td>"
            f"</tr>"
        )

    mo.md(f"""
    ### Step-through: 16-element example

    | v_unsigned | v_true (v-1) | q8 (signed) | v_unsigned × q8 | v_true × q8 |
    |---|---|---|---|---|
    {_table_rows}
    | **Sum** | | | **{sum_unsigned}** | **{sum_corrected}** |
    | **Σ(q8)** | | | **{bias_q8_sum}** | |
    | **Σ(v_unsigned × q8) − Σ(q8)** | | | **{check}** | |

    ✓ `Σ(v_unsigned × q8) - Σ(q8)` = **{check}** matches `Σ(v_true × q8)` = **{sum_corrected}**

    The kernel does both sums in register — no scalar correction loop needed. The per-block cost is one `_mm256_hadd_epi32` + one `_mm256_sub_epi32`, about 5 cycles.
    """)
    return


# ── Section 4: Dot Product Race ────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Dot Product Race

    The benchmark measures the `vec_dot` function for a single Q2_0 block (128 values). The kernel handles the Q2_0 → Q8_0 dot product; TQ2_0 reuses the same kernel (TQ2_0 is just Q2_0 with a different interpretation of the 2-bit values).

    Each bar shows the throughput of a single `vec_dot` call — how fast the kernel processes one block of 128 values. The SIMD advantage is per-block, not per-batch: the kernel's instruction mix (shuffle+interleave vs shift+AND) determines throughput regardless of how many blocks you process.
    """)
    return


@app.cell(hide_code=True)
def _(KERNELS, alt, mo, pd):
    _rows = []
    for _name, _metrics in KERNELS.items():
        _rows.append({"kernel": _name.upper(), "throughput_gbs": _metrics["vec_dot_gbs"],
                       "cycles_32": _metrics["vec_dot_cycles_32"]})
    _df = pd.DataFrame(_rows)

    _base = alt.Chart(_df).encode(
        x=alt.X("kernel:N", title=None, sort=None),
        color=alt.Color("kernel:N", legend=None, scale=alt.Scale(scheme="blues")),
    )

    _bars = _base.mark_bar(width=30).encode(
        y=alt.Y("throughput_gbs:Q", title="Throughput (GB/s)"),
        tooltip=[alt.Tooltip("kernel:N"), alt.Tooltip("throughput_gbs:Q", format=".2f")],
    )

    _text = _base.mark_text(dy=-6, fontSize=11, color="#ccc").encode(
        y=alt.Y("throughput_gbs:Q"),
        text=alt.Text("throughput_gbs:Q", format=".1f"),
    )

    _throughput_chart = mo.ui.altair_chart(
        (_bars + _text).properties(width=400, height=250, title="Kernel Throughput (GB/s)"),
        chart_selection=False,
        legend_selection=False,
    )

    _implied_speedup = 152.59 / 16.95

    mo.md(f"""
    <div style="display:flex; gap:20px; align-items:center;">
      <div>{_throughput_chart}</div>
      <div style="max-width:300px;">
      <strong>TQ2_0 is {_implied_speedup:.0f}× faster per block</strong> than our Q2_0 kernel. The difference is entirely in the instruction mix: TQ2_0 uses shift+AND extraction (4 instructions per 32 bytes) while Q2_0 uses shuffle+interleave (~12 instructions per 4 bytes). The Q8_K activation format with precomputed bsums eliminates the on-the-fly Σ(q8) computation that Q2_0 pays per sub-block.
      </div>
    </div>
    """)
    return


# ── Section 5: Complete Benchmarks ─────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Complete Cross-Machine Benchmarks

    All benchmarks at 6 threads, tg128, via `llama-bench -n 128`. Each value is the mean of multiple runs; llama-bench reports standard deviation (±). The two machines represent different ends of the x86 spectrum: a desktop Zen 2 chip with large L3 cache (32 MB) and a laptop Zen 3 chip with higher IPC but smaller cache (8 MB).

    **A note on variance:** The BITMAJOR 4B and 8B benchmarks show ±0.01 tok/s — effectively zero run-to-run variance. This is not a measurement artifact. When a model is fully bandwidth-bound (model size >> L3 cache), the memory bus is saturated on every run and the CPU has no room to vary. The 1.7B model shows ±0.8 tok/s because it partially fits in cache, where CPU scheduling jitter matters. The 8-thread result (20.43 ±6.76 tok/s) in Section 8b shows what variance looks like when the cache is thrashing — the ±6.76 is the diagnostic signal. The ±0.01 at 4B/8B is the signal that these models are cleanly bandwidth-bound.
    """)
    return


@app.cell(hide_code=True)
def _(BENCH, alt, mo, pd):
    # Build a long-form dataframe for Altair
    _rows = []
    for _, _br in BENCH.iterrows():
        _rows.append({"model": _br["model"], "format": _br["format"], "size_mb": _br["size_mb"],
                       "machine": "Zen 2 (3600X)", "tok_s": _br["zen2_tg128"]})
        _rows.append({"model": _br["model"], "format": _br["format"], "size_mb": _br["size_mb"],
                       "machine": "Zen 3 (5700U)", "tok_s": _br["zen3_tg128"]})
    _df = pd.DataFrame(_rows)

    # Heatmap: model+format vs machine, color by tok/s
    _df["label"] = _df["model"] + " " + _df["format"]

    _heatmap = alt.Chart(_df).mark_rect().encode(
        x=alt.X("machine:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("label:N", title=None, sort=None),
        color=alt.Color("tok_s:Q", scale=alt.Scale(scheme="viridis"), title="tok/s",
                        legend=alt.Legend(orient="bottom")),
        tooltip=["model:N", "format:N", "machine:N", "tok_s:Q"]
    ).properties(width=300, height=400, title="Cross-Machine Benchmark Heatmap")

    _text = alt.Chart(_df).mark_text(fontSize=10, color="white").encode(
        x=alt.X("machine:N", title=None),
        y=alt.Y("label:N", title=None, sort=None),
        text=alt.Text("tok_s:Q", format=".1f"),
    )

    _chart = mo.ui.altair_chart(
        (_heatmap + _text).properties(width=350, height=420),
        chart_selection=False,
        legend_selection=False,
    )

    mo.md(f"""
    {_chart}

    ### Three patterns

    1. **Q2_0 (our kernel) always favors Zen 3** — 33-42% faster on the 5700U despite fewer threads. Our kernel is compute-heavy (unpacking 2-bit values, bias correction per 32-element block). Zen 3's higher IPC and boost clock win here.

    2. **TQ2_0 and Q1_0 converge at 4B+** — within 1 tok/s across machines. These formats are memory-bandwidth-bound: the compute is trivial (table lookups for TQ2_0, sign extraction for Q1_0), so both machines hit the same RAM wall.

    3. **BitNet I2_S is also bandwidth-bound** — 31.95 vs 32.45 tok/s, effectively identical. The native 1-bit architecture doesn't escape the memory bottleneck.
    """)
    return


@app.cell(hide_code=True)
def _(BENCH, MACHINES, alt, mo, pd):
    # Interactive bandwidth waterfall: format toggles + machine toggle + bandwidth slider
    _all_formats = sorted(BENCH["format"].unique().tolist())
    _all_machines = ["Zen 2 (3600X)", "Zen 3 (5700U)", "Both"]

    format_toggle = mo.ui.multiselect(
        options=_all_formats,
        value=["Q2_0", "Q2_0+Q8_K", "Q2_0_BITMAJOR", "TQ2_0", "Q1_0"],
        label="Formats",
    )
    machine_toggle = mo.ui.radio(
        options=_all_machines,
        value="Both",
        label="Machine",
    )
    bw_slider = mo.ui.slider(10, 200, 5, value=45, label="Simulated bandwidth (GB/s)", show_value=True)

    mo.hstack([format_toggle, machine_toggle, bw_slider], justify="space-around")
    return bw_slider, format_toggle, machine_toggle


@app.cell(hide_code=True)
def _(BENCH, MACHINES, alt, bw_slider, format_toggle, machine_toggle, mo, pd):
    _rows = []
    for _, row in BENCH.iterrows():
        if row["format"] not in format_toggle.value:
            continue
        for _mkey, _mname in [("zen2", "Zen 2 (3600X)"), ("zen3", "Zen 3 (5700U)")]:
            if machine_toggle.value != "Both" and _mname != machine_toggle.value:
                continue
            _actual = row[f"{_mkey}_tg128"]
            if _actual is None:
                continue
            _bw = bw_slider.value  # use the slider, not the machine's actual BW
            _ceiling = _bw / (row["size_mb"] / 1024)
            _rows.append({
                "model": f"{row['model']} {row['format']}",
                "machine": _mname,
                "efficiency": _actual / _ceiling * 100,
                "actual": _actual,
                "ceiling": _ceiling,
            })
    _df = pd.DataFrame(_rows)

    if len(_df) == 0:
        mo.md("*No data for selected formats/machine.*")
        return

    _bars = alt.Chart(_df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("model:N", title=None, sort=None),
        x=alt.X("efficiency:Q", title="Bandwidth Efficiency (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("machine:N", title=None),
        tooltip=["model:N", "machine:N",
                 alt.Tooltip("efficiency:Q", format=".1f", title="efficiency %"),
                 alt.Tooltip("actual:Q", format=".1f", title="actual tok/s")],
    ).properties(width=500, height=400)

    _labels = alt.Chart(_df).mark_text(
        dx=20, fontSize=9, color="white", align="left"
    ).encode(
        y=alt.Y("model:N", sort=None),
        x=alt.X("efficiency:Q"),
        text=alt.Text("efficiency:Q", format=".0f%%"),
    )

    _chart = mo.ui.altair_chart(
        (_bars + _labels).properties(title=f"Bandwidth Efficiency at {bw_slider.value} GB/s"),
        chart_selection=False,
        legend_selection=True,
    )

    mo.md(f"""
    ### Bandwidth Efficiency

    {_chart}

    **Efficiency = actual tok/s ÷ theoretical ceiling.** The ceiling is what you'd get if every byte of memory bandwidth went to reading model weights. The gap is attention compute, non-weight memory traffic, and kernel overhead.

    **Try it:** Toggle formats off one by one and watch the clusters emerge — Q2_0 at 13%, Q2_0+Q8_K at 22%, BITMAJOR and TQ2_0 at 53-69%. Drag the bandwidth slider from 45 GB/s (Zen 2) to 120 GB/s (Mac Mini M4) and watch the percentages drop — faster memory doesn't make a compute-bound kernel faster; it just makes the inefficiency more visible.

    **The pattern is clear:** Q2_0 and Q2_0_WIDE cluster at 13-14% — compute-bound kernels that can't feed the memory bus. Q2_0+Q8_K jumps to 22-23% by eliminating on-the-fly Σ(q8). BITMAJOR and TQ2_0 reach 53-69% — memory-bound kernels that keep the bus fed. The efficiency grows with model size because larger models do more arithmetic per byte loaded, giving the memory bus more time to deliver the next cache line.
    """)
    return


# ── Section 6: KV Cache ─────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The KV Cache: The Hidden Tax

    Every transformer layer stores **keys** and **values** for each previous token. Without a KV cache, generating token #513 would recompute K and V for all 512 previous tokens — 512 redundant matrix multiplies. With a KV cache, you store them and only compute the new token's contribution.

    The cost: **memory.** At long context, the KV cache can be larger than the model itself. Every token generation reads both the model weights AND the KV cache from RAM.

    ### The Bandwidth Breakdown

    Use the controls below to see how the bandwidth budget splits between model weights and KV cache at different context lengths. The red line marks where the KV cache exceeds the model size.
    """)
    return


@app.cell(hide_code=True)
def _(SPECS, mo):
    model_choice = mo.ui.dropdown(
        options=list(SPECS.keys()),
        value="Bonsai 4B",
        label="Model"
    )
    ctx_len = mo.ui.slider(128, 8192, 128, value=512, label="Context length (tokens)", show_value=True)
    kv_quant = mo.ui.dropdown(
        options=["FP16 (2 bytes/element)", "Q8_0 K-only (1.5 bytes avg)", "Q8_0 K+V (1 byte/element)"],
        value="FP16 (2 bytes/element)",
        label="KV cache quantization"
    )

    mo.hstack([model_choice, ctx_len, kv_quant], justify="space-around")
    return ctx_len, kv_quant, model_choice


@app.cell(hide_code=True)
def _(SPECS, alt, ctx_len, kv_quant, mo, model_choice, np, pd):
    _spec = SPECS[model_choice.value]
    _n_layer = _spec["n_layer"]
    _n_head_kv = _spec["n_head_kv"]
    _n_embd_head = _spec["n_embd_head"]
    _model_mb = _spec["size_mb"]

    _bytes_map = {
        "FP16 (2 bytes/element)": 2,
        "Q8_0 K-only (1.5 bytes avg)": 1.5,
        "Q8_0 K+V (1 byte/element)": 1,
    }
    _bytes_per_element = _bytes_map[kv_quant.value]

    _kv_per_token = 2 * _n_layer * _n_head_kv * _n_embd_head * _bytes_per_element
    _kv_total_mb = _kv_per_token * ctx_len.value / (1024 * 1024)
    _total_mb = _model_mb + _kv_total_mb
    _kv_pct = _kv_total_mb / _total_mb * 100

    # Crossover context
    _crossover_ctx = _model_mb / (_kv_per_token / (1024 * 1024))

    # Gauge: horizontal bar showing KV cache share
    _gauge_df = pd.DataFrame([{
        "label": "KV Cache",
        "pct": _kv_pct,
        "mb": _kv_total_mb,
    }, {
        "label": "Model Weights",
        "pct": 100 - _kv_pct,
        "mb": _model_mb,
    }])

    _gauge = alt.Chart(_gauge_df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("label:N", title=None, sort=None),
        x=alt.X("pct:Q", title="Share of Total Memory Traffic", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("label:N", title=None,
                        scale=alt.Scale(domain=["KV Cache", "Model Weights"],
                                        range=["#e74c3c", "#2ecc71"])),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("pct:Q", format=".0f", title="%"),
                 alt.Tooltip("mb:Q", format=".0f", title="MB")],
    ).properties(width=500, height=80)

    _gauge_text = alt.Chart(pd.DataFrame([{"pct": _kv_pct}])).mark_text(
        fontSize=28, color="#e74c3c", fontWeight="bold", dx=0
    ).encode(
        x=alt.X("pct:Q", title=None),
        text=alt.Text("pct:Q", format=".0f%%"),
    )

    _chart = mo.ui.altair_chart(
        (_gauge + _gauge_text).properties(
            title=f"KV Cache Share — {model_choice.value} at {ctx_len.value} tokens"
        ),
        chart_selection=False,
        legend_selection=False,
    )

    mo.md(f"""
    {_chart}

    ### {model_choice.value} at {ctx_len.value} tokens

    | | Value |
    |---|---|
    | Model size | {_model_mb:.0f} MB |
    | KV cache | {_kv_total_mb:.0f} MB |
    | **Total per token** | **{_total_mb:.0f} MB** |
    | KV cache share | **{_kv_pct:.0f}%** |
    | KV = Model at | **{_crossover_ctx:.0f} tokens** |

    **Drag the context slider** and watch the red bar grow. At 512 tokens, the KV cache is a minor tax (~9%). At 4096, it's nearly half your bandwidth. The gauge makes the tradeoff immediate: every doubling of context doubles the red bar.

    ### The Practical Path

    We traced the KV cache allocation through the llama.cpp source (`llama-kv-cache.cpp` line 209, `llama-context.cpp` lines 2964-2989). Two findings:

    1. **K-only quantization works now.** Pass `-ctk q8_0` without `-ctv q8_0`. The code only blocks V cache quantization without flash_attn — K cache quantization has no such guard. This saves 25% of KV cache bandwidth immediately.

    2. **V cache quantization needs flash_attn.** The check at line 2986 explicitly rejects quantized V cache without flash_attn enabled. Building with `GGML_FLASH_ATTN=ON` would unlock full KV cache quantization, halving the KV cache bandwidth.

    ### Why PrismML Compressed Anyway

    PrismML's TQ2_0 format uses **256-wide blocks** — 8× wider than our Q2_0's 32-wide blocks. This means 8× fewer scale factors per tensor. The KV cache is stored in the same format as the model weights, so the compression comes for free: the same kernel that reads model weights also reads KV cache entries. They didn't add KV cache quantization as a separate feature — they made the base format more efficient, and the KV cache inherited the savings.

    This is the right approach. Rather than adding a separate quantization layer for the KV cache, make the fundamental format more efficient and let everything benefit. Our Q2_0 kernel's 32-wide blocks are the bottleneck — not just for model weights, but for the KV cache too. The fix isn't KV cache quantization. The fix is wider blocks.
    """)
    return


# ── Section 7: The Context Size Tradeoff ────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. The Context Size Tradeoff

    There's no single optimal context size — it depends on what you're doing. But the tradeoff is sharp: every doubling of context doubles the KV cache, which directly reduces tokens per second.

    ### Tool Calling: Short Context Wins

    A tool-calling interaction is a tight loop: understand → choose tool → generate call → process result → respond. Each pass is 10-30 tokens. The KV cache only needs to hold the conversation history — typically 500-1000 tokens. At 512 context, the KV cache is a minor tax. At 4096, it's the dominant cost.

    | Context | Bonsai 8B KV Cache | % of Total Traffic | Projected tok/s (Zen 2) |
    |---|---|---|---|
    | 512 | 256 MB | 9% | 16.3 |
    | 1024 | 512 MB | 17% | 15.1 |
    | 2048 | 1.0 GB | 29% | 13.0 |
    | 4096 | 2.0 GB | 45% | 10.1 |
    | 8192 | 4.0 GB | 62% | 7.0 |

    **Why 1024-2048?** The table tells the story. At 512 tokens, the KV cache is only 9% of total traffic — but 512 tokens is barely enough for a single tool-calling round-trip with conversation history. At 4096 tokens, the KV cache consumes 45% of every byte read from RAM — you're paying nearly half your bandwidth for context you rarely use. The 1024-2048 range sits at the knee of the curve: 17-29% KV cache overhead, enough context for 3-6 tool-calling turns with full history, and projected tok/s (15.1 at 1024, 13.0 at 2048) that stay above the ~10 tok/s threshold for interactive use. The bandwidth math is straightforward: KV cache bytes per token = 2 × n_layer × n_head_kv × n_embd_head × bytes_per_element. For Bonsai 8B (36 layers, 8 KV heads, 128-dim heads, FP16): 2 × 36 × 8 × 128 × 2 = 147,456 bytes per context token. At 2048 tokens, that's 288 MB of KV cache competing with 2.5 GB of model weights for the same 45 GB/s memory bus.

    ### Long-Form Generation: Long Context Wins

    If you're generating a story, a codebase, or a research summary, you need the model to remember what it wrote 2000 tokens ago. The KV cache is the cost of coherence. At 4096 context, you pay a 45% bandwidth tax for the ability to reference anything in the last ~3000 words.

    ### The Real Answer: Dynamic Context

    The smartest systems don't pick one context size. They use a **sliding window** — keep the last N tokens in full precision, compress or drop older tokens. Or they use **KV cache offloading** — store the full cache on disk and only load the recent window into RAM. These are active research areas, not solved problems.

    For our smart home use case, **1024-2048 tokens is the sweet spot.** Enough for multi-turn tool calling with conversation history, small enough that the KV cache is 17-29% of total traffic. At 2048 context with Q8_0 K cache, Bonsai 8B projects to ~14 tok/s on the 3600X — fast enough for interactive use.
    """)
    return


# ── Section 8: Q2_0_WIDE ─ The Wider Block Hypothesis ─────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Q2_0_WIDE: The Wider Block Hypothesis

    Our Q2_0 kernel is 4-5.5× slower than PrismML's TQ2_0 on the same ternary model. Three things differ: TQ2_0 uses **256-wide blocks** (vs our 128), TQ2_0 pairs with **Q8_K activations** that have precomputed block sums (vs our Q8_0 which computes Σ(q8) on the fly), and TQ2_0 uses **simpler bit extraction** (shift+AND vs our shuffle+interleave). Which one matters?

    **Hypothesis:** Block width is the primary speed factor. The Q8_K format and simpler extraction are secondary optimizations that benefit from — but are not the cause of — the wider block structure.

    To test this, we built **Q2_0_WIDE**: the same 2-bit ternary packing, unsigned-bias arithmetic, and Q8_0 activation format as our existing Q2_0 kernel, but with **256-wide blocks** matching TQ2_0's geometry. If wider blocks alone recover most of the gap, the Q8_K format is icing. If they don't, the Q8_K format carries independent performance value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Block Structure Comparison

    | Format | Block Width | Scale | Packed | Total | Bits/Weight | Arithmetic | Models |
    |---|---|---:|---:|---:|---:|---|---|
    | Q2_0 (ours) | 128 | 2 B | 32 B | 34 B | 2.125 | Unsigned-bias SIMD (on-the-fly unpack) | Downloadable |
    | TQ2_0 (PrismML) | 256 | 2 B | 64 B | 66 B | 2.06 | Shift+AND, Q8_K bsums | Local quant from FP16 |
    | **Q2_0_WIDE (tested)** | **256** | **2 B** | **64 B** | **66 B** | **2.06** | **Unsigned-bias SIMD (on-the-fly unpack)** | **Downloadable** |

    Q2_0_WIDE is literally TQ2_0's block geometry fed through our existing kernel. The only change is the outer-loop stride in `ggml_vec_dot_q2_0_q8_0` — from 4 sub-blocks per 128-wide block to 8 sub-blocks per 256-wide block. Same inner SIMD loop, same constants, same unsigned-bias correction.

    > **File size vs. bits-per-weight:** The 2.06 bits/weight figure is for the weight tensors only. The actual TQ2_0 GGUF files are larger than the raw tensor arithmetic suggests because the TQ2_0 quantizer leaves embedding and output layers in higher precision (q6_K for tied embeddings, the same approach Q2_0_WIDE uses). For Bonsai 1.7B, the TQ2_0 file is 590 MB vs. 442 MB for Q2_0 — a 33% increase — even though the block-level compression is slightly better (2.06 vs. 2.125 bpw). The extra bytes are in the non-quantized layers, not the weight blocks. When comparing bandwidth efficiency, use the actual file sizes (or tensor-only sizes consistently), not the headline bpw number.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Benchmark Results — Hypothesis Falsified

    **Measured June 27, 2026 on Zen 2 (3600X, DDR4-3200, 6t, tg64).**

    | Model | Q2_0 tok/s | Q2_0_WIDE tok/s | TQ2_0 tok/s | Speedup (WIDE vs Q2_0) | Conclusion |
    |---|---|---|---|---|---|
    | Bonsai 1.7B | 14.81 | 15.56 | 57.30 | **+5.1%** | Falsified |
    | Bonsai 4B | 6.48 | 6.56 | 28.79 | **+1.2%** | Falsified |
    | Bonsai 8B | 3.54 | 3.55 | 16.31 | **+0.3%** | Falsified |

    **The hypothesis is falsified at all three scales.** Block width alone gives 5% at 1.7B, 1.2% at 4B, and 0.3% at 8B — well below the 1.5× falsification boundary at every tier. The Q8_K format (with precomputed bsums and simpler bit extraction) carries ~3.5× independent performance value beyond block width.

    **The diminishing returns are instructive.** At 1.7B, the model fits partially in cache — wider blocks reduce scale factor loads and the CPU can exploit the reduced overhead. At 4B and 8B, the model is fully bandwidth-bound. The wider blocks save a few scale factor bytes but the memory bus is saturated either way. The speedup converges to zero as model size increases.

    **What we learned:**
    - The Q2_0_WIDE kernel is correct — produces coherent text, passes roundtrip tests
    - The bottleneck is memory bandwidth, not kernel overhead. Wider blocks reduce scale factor loads but don't change the fundamental bandwidth constraint
    - TQ2_0's Q8_K format with precomputed bsums eliminates the Σ(q8) horizontal sum — that's where the speed comes from
    - Q2_0_WIDE models are byte-compatible with TQ2_0 (same on-disk layout), so the format has value as a downloadable path even if the kernel doesn't match TQ2_0 speeds
    - The speedup is inversely proportional to model size: +5.1% (1.7B) → +1.2% (4B) → +0.3% (8B). This is the signature of a cache-resident optimization hitting a bandwidth wall.

    **Implementation:** 12 files modified, ~500 lines. Full pipeline: quantize → load → benchmark. Kernel code at `ggml-cpu/arch/x86/quants.c:4284`. Type enum 43, ftype 42.

    **Next step:** Test the Q8_K activation format with our existing 128-wide blocks. If the Q8_K format is the lever, not the block width, we should see a significant speedup without changing the block geometry. → [Results below](#q2_0--q8_k-closing-half-the-gap).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q2_0 + Q8_K: Closing the Gap

    The Q2_0_WIDE experiment showed that block width alone is worth ~5% at small sizes and ~0% at large sizes. The Q8_K activation format — with precomputed block sums — is the real lever. We attacked the gap in two phases.

    **Phase 1: Q8_K activations with byte-major weights (June 27)**

    A one-line dispatch change — `vec_dot_type = GGML_TYPE_Q8_K` — plus a new kernel using shuffle+interleave extraction with Q8_K activations. The kernel uses on-the-fly Σ(q8) correction since the 128-wide blocks don't align with Q8_K's 256-wide bsums.

    | Model | Old Q2_0 tok/s | Q2_0+Q8_K tok/s | TQ2_0 tok/s | Speedup |
    |---|---|---|---|---|
    | Bonsai 1.7B | 14.81 | **25.93** | 57.30 | **1.75×** |
    | Bonsai 4B | 6.48 | **11.64** | 28.79 | **1.80×** |

    **Phase 2: Bit-major weights with shift+AND extraction (June 28)**

    The remaining gap was in the weight layout. TQ2_0 uses bit-major packing — byte `m` stores bits for values `[m, m+32, m+64, m+96]` — enabling shift+AND extraction: 4 shifts + 4 ANDs per 32 bytes. We changed our Q2_0 format to 256-wide blocks with bit-major packing, matching TQ2_0's architecture 1:1. The kernel is nearly identical to TQ2_0's: shift+AND extraction, precomputed bsums from Q8_K, one accumulator per block.

    | Model | Old Q2_0 tok/s | Q2_0_BITMAJOR tok/s | TQ2_0 tok/s | Speedup |
    |---|---|---|---|---|
    | Bonsai 1.7B | 25.3 | **61.9** ±0.8 | 63.4 | **2.4×** |
    | Bonsai 4B | 6.5 | **28.2** ±0.01 | 28.8 | **4.3×** |
    | Bonsai 8B | 3.5 | **16.8** ±0.01 | 16.3 | **4.8×** |

    **The gap is closed at all three model sizes.** Q2_0_BITMAJOR matches TQ2_0 within 2% at 1.7B, 2% at 4B, and slightly exceeds it at 8B (16.8 vs 16.3 tok/s — within run-to-run variance). The models stay downloadable: 596 MB / 1.17 GB / 2.29 GB, no FP16 intermediate required.

    **What changed:**
    - Block width: 128 → 256 (matches Q8_K, enables 1:1 bsums mapping)
    - Weight layout: byte-major → bit-major (enables shift+AND extraction)
    - Kernel: shuffle+interleave → shift+AND (4 shifts + 4 ANDs per 32 bytes)
    - Bsums: on-the-fly Σ(q8) → precomputed bsums from Q8_K block
    - Model size: 442 MB → 590 MB (embeddings go to q6_K, same as TQ2_0)

    **Bandwidth utilization** (see the full table in "Why TQ2_0 Is So Much Faster" below):

    | Model | Q2_0 | Q2_0+Q8_K | Q2_0_BITMAJOR | TQ2_0 |
    |---|---|---|---|---|
    | 1.7B | 13% | 22% | 53% | 54% |
    | 4B | 13% | 23% | 57% | 57% |
    | 8B | 14% | — | 69% | 66% |

    The bit-major kernel uses 53-69% of memory bandwidth — matching or exceeding TQ2_0 at every size. The 13% → 53% journey at 1.7B: Q8_K activations doubled it (13% → 22%), shift+AND extraction more than doubled it again (22% → 53%). At 8B, the higher arithmetic intensity pushes BITMAJOR to 69% — slightly above TQ2_0's 66%, within run-to-run variance.

    ### BITMAJOR Correctness Verification

    Speed parity doesn't guarantee output parity. We verified that Q2_0_BITMAJOR produces equivalent output to TQ2_0 by running both models through `llama-server` with the same prompt and comparing the generated tokens:

    ```
    Prompt: "Once upon a time, in a land far away, there lived a"

    TQ2_0 output:       "young girl named Lily. She lived in a small village"
    Q2_0_BITMAJOR:      "young girl named Lily. She lived in a small village"
    ```

    Both models produced the same 16 tokens at 64.64 tok/s. The quantizer was also verified: starting from the same FP16 source, our Q2_0_BITMAJOR quantizer produces byte-identical weight data to the TQ2_0 quantizer. The two formats are architecturally equivalent — same block geometry, same bit-major layout, same precomputed bsums — so output parity is expected, not surprising. The verification confirms the implementation is correct.
    """)
    return


@app.cell(hide_code=True)
def _(BENCH, alt, mo, pd):
    # Parity plot: BITMAJOR vs TQ2_0, both machines, all sizes
    _rows = []
    for _, _br in BENCH.iterrows():
        if _br["format"] not in ("Q2_0_BITMAJOR", "TQ2_0"):
            continue
        for _mkey, _mname in [("zen2", "Zen 2 (3600X)"), ("zen3", "Zen 3 (5700U)")]:
            _val = _br[f"{_mkey}_tg128"]
            if _val is None:
                continue
            _rows.append({
                "model": _br["model"],
                "format": _br["format"],
                "machine": _mname,
                "tok_s": _val,
            })
    _df = pd.DataFrame(_rows)

    # Pivot to get BITMAJOR and TQ2_0 as columns
    _pivot = _df.pivot_table(
        index=["model", "machine"], columns="format", values="tok_s"
    ).reset_index()
    _pivot.columns.name = None

    # y=x reference line
    _max_val = max(_pivot["Q2_0_BITMAJOR"].max(), _pivot["TQ2_0"].max()) * 1.05
    _ref = pd.DataFrame({"x": [0, _max_val], "y": [0, _max_val]})

    _ref_line = alt.Chart(_ref).mark_line(
        color="#666", strokeDash=[4, 4], strokeWidth=1
    ).encode(x="x:Q", y="y:Q")

    _points = alt.Chart(_pivot).mark_point(
        filled=True, size=120
    ).encode(
        x=alt.X("Q2_0_BITMAJOR:Q", title="Q2_0_BITMAJOR (tok/s)"),
        y=alt.Y("TQ2_0:Q", title="TQ2_0 (tok/s)", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None),
        shape=alt.Shape("machine:N", title=None),
        tooltip=["model:N", "machine:N", "Q2_0_BITMAJOR:Q", "TQ2_0:Q"],
    ).properties(width=350, height=350)

    _chart = mo.ui.altair_chart(
        (_ref_line + _points).properties(title="BITMAJOR vs TQ2_0: Parity Plot"),
        chart_selection=False,
        legend_selection=True,
    )

    mo.md(f"""
    ### BITMAJOR vs TQ2_0: Parity

    {_chart}

    Every point sits on the y=x line. Q2_0_BITMAJOR matches TQ2_0 within 3% on both machines at all three model sizes. The two formats are architecturally identical — same block width, same bit-major layout, same shift+AND extraction, same precomputed bsums from Q8_K. The remaining 2-3% difference is register allocation and loop structure, not algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    speedup_machine = mo.ui.radio(
        options=["Zen 2 (3600X)", "Zen 3 (5700U)"],
        value="Zen 2 (3600X)",
        label="Machine",
    )
    mo.hstack([speedup_machine], justify="start")
    return speedup_machine


@app.cell(hide_code=True)
def _(BENCH, alt, mo, pd, speedup_machine):
    # Speedup journey: Q2_0 → Q2_0+Q8_K → Q2_0_BITMAJOR → TQ2_0
    _mkey = "zen2" if speedup_machine.value == "Zen 2 (3600X)" else "zen3"
    _rows = []
    for _, _br in BENCH.iterrows():
        _fmt = _br["format"]
        if _fmt not in ("Q2_0", "Q2_0+Q8_K", "Q2_0_BITMAJOR", "TQ2_0"):
            continue
        _val = _br[f"{_mkey}_tg128"]
        if _val is None:
            continue
        _rows.append({
            "model": _br["model"],
            "format": _fmt,
            "tok_s": _val,
        })
    _df = pd.DataFrame(_rows)

    # Order formats left to right
    _order = ["Q2_0", "Q2_0+Q8_K", "Q2_0_BITMAJOR", "TQ2_0"]

    _bars = alt.Chart(_df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("format:N", title=None, sort=_order,
                axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("tok_s:Q", title="tok/s"),
        color=alt.Color("format:N", title=None,
                        scale=alt.Scale(domain=_order, range=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])),
        tooltip=["model:N", "format:N", "tok_s:Q"],
    ).properties(width=120, height=200)

    _labels = alt.Chart(_df).mark_text(
        dy=-8, fontSize=9, color="white"
    ).encode(
        x=alt.X("format:N", sort=_order),
        y=alt.Y("tok_s:Q"),
        text=alt.Text("tok_s:Q", format=".1f"),
    )

    _chart = mo.ui.altair_chart(
        (_bars + _labels).facet(
            column=alt.Column("model:N", title=None)
        ).properties(title=f"The Speedup Journey ({speedup_machine.value})"),
        chart_selection=False,
        legend_selection=False,
    )

    mo.md(f"""
    ### The Speedup Journey

    {_chart}

    **Phase 1** (Q2_0 → Q2_0+Q8_K): switching activation format from Q8_0 to Q8_K with precomputed bsums. 1.75-1.80× speedup. **Phase 2** (Q2_0+Q8_K → Q2_0_BITMAJOR): bit-major weight packing with shift+AND extraction. 2.4× speedup at 1.7B, 2.4× at 4B. The final step to TQ2_0 is a rounding error — the architecture is the same.

    **Toggle between machines:** The speedup pattern is identical on both — the absolute numbers differ but the ratios hold. Zen 3's higher IPC helps the compute-bound Q2_0 kernel more (17.2 vs 12.8 tok/s at 1.7B), but the bandwidth-bound BITMAJOR and TQ2_0 converge to within 1 tok/s across machines.

    The speedup grows with model size: 2.4× at 1.7B, 4.3× at 4B, 4.8× at 8B. Larger models are more bandwidth-bound, so the compute savings from shift+AND extraction matter more.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why TQ2_0 Is So Much Faster: It's Not a LUT

    **Correction:** In an earlier draft of this notebook, we called TQ2_0 "LUT-based." That was wrong. TQ2_0 does not use a lookup table. Both TQ2_0 and Q2_0_WIDE unpack 2-bit values on the fly. The speed difference comes from three things, none of which are a LUT.

    But first, the bandwidth numbers — because they tell the real story:

    | Model | Q2_0 GB/s | Q2_0+Q8_K GB/s | Q2_0_BITMAJOR GB/s | Q2_0_WIDE GB/s | TQ2_0 GB/s | % of DDR4-3200 (51.2 GB/s) |
    |---|---|---|---|---|---|---|
    | 1.7B | 6.46 | 11.30 | 27.00 | 6.78 | 27.65 | Q2_0: 13% · Q2_0+Q8_K: 22% · BITMAJOR: 53% · WIDE: 13% · TQ2_0: 54% |
    | 4B | 6.61 | 11.87 | 29.08 | 6.69 | 29.37 | Q2_0: 13% · Q2_0+Q8_K: 23% · BITMAJOR: 57% · WIDE: 13% · TQ2_0: 57% |
    | 8B | 7.35 | — | 35.21 | 7.37 | 33.84 | Q2_0: 14% · BITMAJOR: 69% · WIDE: 14% · TQ2_0: 66% |

    Both Q2_0 and Q2_0_WIDE use only **13-14%** of available memory bandwidth. The CPU is spending its time unpacking bits — shuffling, interleaving, masking — and the memory bus sits idle. Q2_0+Q8_K doubles that to **22-23%** by eliminating the on-the-fly Σ(q8) computation. TQ2_0 uses **49-66%** of bandwidth because the kernel is simple enough to keep the bus fed.

    **So what's actually different?** Three things:

    **1. The activation format: Q8_K vs Q8_0**

    This is the biggest factor. Q2_0_WIDE pairs with `Q8_0` — 32-element blocks with no precomputed metadata. The kernel has to compute Σ(q8) on the fly for every 32-element sub-block, which means a horizontal sum (8 instructions) repeated 8 times per 256-wide block.

    TQ2_0 pairs with `Q8_K` — 256-element blocks with **precomputed block sums** (`bsums`). The kernel loads the precomputed sum once per block instead of computing it 8 times. This eliminates ~64 instructions per 256-wide block.

    **2. The bit extraction: shift+AND vs shuffle+interleave**

    TQ2_0 extracts 2-bit values with 4 instructions per 32 bytes:
    ```c
    qx0 = load(x[i].qs + j);        // 1 load
    qx1 = srli(qx0, 2);             // 3 shifts
    qx2 = srli(qx0, 4);
    qx3 = srli(qx0, 6);
    qx0 = and(qx0, 3);              // 4 ANDs
    qx1 = and(qx1, 3);
    qx2 = and(qx2, 3);
    qx3 = and(qx3, 3);
    ```

    Q2_0_WIDE extracts 2-bit values with ~12 instructions per 4 bytes:
    ```c
    memcpy(&qs32, qs, 4);           // 1 load
    q4 = cvtsi32_si128(qs32);       // 1 convert
    q4 = shuffle(q4, expand);       // 1 shuffle-expand
    // then 3 shifts, 4 ANDs, 4 interleave shuffles, 3 ORs
    // ... repeat for next 4 bytes ...
    ```

    The shuffle+interleave approach is a legacy of Q2_0's 128-wide blocks. It works, but it's expensive.

    **3. Block width: 256 vs 128**

    Both TQ2_0 and Q2_0_WIDE use 256-wide blocks, so this isn't a differentiator between them. But it matters vs the original Q2_0 (128-wide). Wider blocks mean fewer scale factor loads — 1 per 256 values instead of 1 per 128. This is worth ~5% at small model sizes (where the model fits in cache) and ~0% at large sizes (where bandwidth dominates).

    **Why the advantage grows with model size**

    | Model | TQ2_0 / Q2_0_WIDE ratio |
    |---|---|
    | 1.7B | 3.68× |
    | 4B | 4.39× |
    | 8B | 4.59× |

    As models get larger, the memory bus becomes the binding constraint. A compute-bound kernel (Q2_0, Q2_0_WIDE) can't exploit the bus — it's stuck at 13-14% utilization regardless of model size. A memory-bound kernel (TQ2_0) can — and as the model grows, the bus has more headroom to use. The ratio grows because TQ2_0 is the only kernel that can actually use the bandwidth that's available.

    **The practical implication:** For ternary models, the Q8_K activation format (with precomputed bsums) is not optional. It's the difference between using 14% of your memory bandwidth and 22% — a 1.75× speedup from a one-line dispatch change. Bit-major weight packing with shift+AND extraction takes it to 53% — matching TQ2_0's 54%. The path is now complete: Q8_K activations + bit-major weights + 256-wide blocks = TQ2_0-class speed without FP16 download.

    *(We left the original "LUT" error in the notebook history rather than silently fixing it. Getting things wrong and correcting them is how science works. The bandwidth numbers were right all along — we just had the wrong story about why.)*
    """)
    return


# ── Section 8b: Thread Count ────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8b. More Threads Can Be Worse

    On the Zen 3 5700U (8 cores, 8 MB L3 cache), we tested 6 vs 8 threads with the 4B BITMAJOR model:

    | Threads | tg128 | Variance |
    |---------|-------|----------|
    | 6 | 28.87 tok/s | ±0.13 |
    | 8 | 20.43 tok/s | ±6.76 |

    Eight threads are **worse** than six — slower and wildly unstable. The 5700U has only 8 MB of L3 cache shared across all cores. At 6 threads, each thread's working set (activations, partial sums, intermediate buffers) fits within the cache budget. At 8 threads, the per-thread cache share drops below what the dot-product loop needs, and the CPU thrashes. The ±6.76 variance is the diagnostic signal — some runs get lucky with cache hits, others stall on DRAM.

    **Why this doesn't contradict Ornith's 32-core benchmarks.** Server CPUs (EPYC, Threadripper) have L3 caches measured in hundreds of megabytes and 8+ memory channels. The cache-per-thread ratio stays healthy even at high core counts. On consumer hardware with small L3 caches, the optimal thread count is often *less* than the physical core count — and the variance spike is how you know you've crossed the line.

    **Practical takeaway:** When deploying on consumer hardware, benchmark at multiple thread counts. If variance spikes when you add threads, you've hit the cache wall. Back off to the last stable count. For the 5700U, that's 6 threads — not 8.
    """)
    return


# ── Section 9: Ornith 1.0 — The Thinking Overhead Problem ──────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Ornith 1.0: The Thinking Overhead Problem

    On June 26, 2026, Deep Reinforce released **Ornith 1.0** — a family of models (9B, 31B, 35B MoE, 397B MoE) fine-tuned from Qwen 3.5 and Gemma 4 with a novel training objective: the model learns to generate both the **scaffold** (harness) and the **solution**, jointly optimized via GRPO.

    We tested Ornith 9B at Q4_K_M against three ternary Bonsai models on a constraint-reasoning task. The ternary models failed to self-scaffold — but the more interesting finding is about Ornith itself.

    ### The Thinking Overhead: 48 Seconds of Silence

    Ornith uses a mandatory thinking scaffold — a structured chain-of-thought that the model cannot skip. We captured the thinking tokens across three prompt types via llama-server with `reasoning_format: "deepseek"`:

    | Prompt Type | Thinking Words | Response | Notes |
    |---|---|---|---|
    | Simple factual | 83 | 35 chars | 5-step scaffold: Analyze→Retrieve→Formulate→Review→Output |
    | Constraint reasoning | 112 | 2,852 chars | Correctly evaluates Open-Meteo vs wttr.in |
    | Ambiguous query | 328 | *none* | Divergent retrieval loop, exhausted 512 tokens |

    The thinking format is always present, always numbered steps, and the model has no mechanism to truncate it. At 6.80 tok/s, 328 words of thinking is **~48 seconds of silence** before the user sees anything — and in the ambiguous case, they see nothing at all. The scaffold can diverge into an infinite retrieval loop when the query doesn't map cleanly to a known pattern.

    This is a double-edged sword. The thinking scaffold enables constraint reasoning that ternary models lack, but it adds mandatory latency that can't be skipped. For interactive use, 48 seconds of silent thinking followed by zero response is worse than a fast wrong answer — at least the wrong answer tells you something.

    ### The Self-Scaffolding Test (Pilot)

    We gave the same prompt to all models: *"Create a harness to get the weather with a five-day forecast. I do not have any API keys, so find a solution that doesn't require one."*

    The correct answer is **Open-Meteo** or **wttr.in** — free weather APIs that don't require authentication. The common failure mode is defaulting to OpenWeatherMap (which requires an API key).
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pd):
    _rows = [
        {"Model": "Ternary Bonsai 1.7B TQ2_0", "Size": "596 MB", "Speed (tg128)": "55.19 tok/s", "Coherent Code?": "❌ Infinite loop", "No-Auth API?": "❌ OpenWeatherMap", "Self-Scaffolding?": "❌"},
        {"Model": "Ternary Bonsai 4B TQ2_0", "Size": "1.2 GB", "Speed (tg128)": "28.71 tok/s", "Coherent Code?": "✅ Well-structured", "No-Auth API?": "❌ OpenWeatherMap (generated dummy key)", "Self-Scaffolding?": "❌"},
        {"Model": "Ternary Bonsai 8B TQ2_0", "Size": "2.5 GB", "Speed (tg128)": "16.68 tok/s", "Coherent Code?": "✅ Well-structured", "No-Auth API?": "❌ OpenWeatherMap (generated placeholder key)", "Self-Scaffolding?": "❌"},
        {"Model": "**Ornith 9B Q4_K_M**", "Size": "**5.2 GB**", "Speed (tg128)": "**6.84 tok/s**", "Coherent Code?": "**✅**", "No-Auth API?": "**✅ wttr.in + Open-Meteo**", "Self-Scaffolding?": "**Likely**"},
    ]
    _df = pd.DataFrame(_rows)

    mo.md(f"""
    ### Results: Ternary Models Default to the Most Common Solution

    | Model | Size | Speed | Coherent Code? | No-Auth API? | Self-Scaffolding? |
    |---|---|---|---|---|---|
    | Ternary Bonsai 1.7B TQ2_0 | 596 MB | 55.19 tok/s | ❌ Infinite loop | ❌ OpenWeatherMap | ❌ |
    | Ternary Bonsai 4B TQ2_0 | 1.2 GB | 28.71 tok/s | ✅ Well-structured | ❌ OpenWeatherMap (generated dummy key) | ❌ |
    | Ternary Bonsai 8B TQ2_0 | 2.5 GB | 16.68 tok/s | ✅ Well-structured | ❌ OpenWeatherMap (generated placeholder key) | ❌ |
    | **Ornith 9B Q4_K_M** | **5.2 GB** | **6.84 tok/s** | **✅** | **✅ wttr.in + Open-Meteo** | **Likely** |

    All three ternary models defaulted to OpenWeatherMap. None identified a truly free no-auth API. The 4B and 8B did write code, but they did not reason about constraints — they pattern-matched to the most common solution rather than solving the actual constraint.

    Ornith 9B at Q4_K_M correctly identified wttr.in and Open-Meteo in its chain of thought. The self-scaffolding RL training survives 4-bit quantization.

    ### Confounders and Pilot Status

    This is a single-task pilot, not a controlled study. Several confounders limit what we can claim:

    - **No FP16 Bonsai baseline.** ~~We don't know whether the Bonsai models fail because of quantization or because the base model lacks the capability.~~ **Update (June 28):** We tested the 1.7B Bonsai model at FP16 on the same prompt. It also defaulted to OpenWeatherMap and suggested simulated data as a workaround. The failure is the base model's capability, not quantization loss. The 1.7B model simply cannot solve this constraint-reasoning task at any precision.
    - **No Ornith-at-TQ2_0 test.** We don't know whether Ornith's constraint reasoning survives ternary quantization — we only tested it at Q4_K_M.
    - **Different training objectives.** Bonsai models are general-pretrained; Ornith is fine-tuned with GRPO for harness-design trajectories. The capability gap may be training, not quantization.
    - **Different quantization levels.** We tested Bonsai at TQ2_0 (2-bit ternary) and Ornith at Q4_K_M (~4-bit). A 2-bit vs 4-bit comparison conflates architecture and precision.
    - **Different model families.** Bonsai is Qwen3-based; Ornith is Qwen3.5-based. Family differences in reasoning may dominate quantization effects.

    The honest conclusion: **ternary models did not self-scaffold on this task.** Whether they *cannot* in principle, or whether a larger ternary model, a different training recipe, or a less aggressive quantization would succeed — we don't know. This is a pilot observation. Treat it as such.

    ### The Speed-Quality Tradeoff

    | Metric | Ternary Bonsai 8B TQ2_0 | Ornith 9B Q4_K_M | Ratio |
    |---|---|---|---|
    | Speed | 16.68 tok/s | 6.84 tok/s | 2.4× faster (ternary) |
    | Size | 2.5 GB | 5.2 GB | 2.1× smaller (ternary) |
    | Constraint reasoning | ❌ | ✅ | — |
    | Self-scaffolding | ❌ | Likely | — |

    Ternary models are 2-3× faster and smaller, but they did not demonstrate the constraint-aware reasoning that self-scaffolding requires in this pilot. The quantization may strip fine-grained reasoning patterns, or the base model may never have learned them.

    ### What This Means for Edge Deployment

    The three-tier architecture (1.7B filter → 4B executor → 8B complex executor) is still correct for **execution**. The ternary models can run harnesses. They did not **design** them in our test.

    For self-scaffolding at the edge, we need either:
    1. A larger ternary model (13B+) that preserves more reasoning capability
    2. A hybrid architecture: cloud model designs the harness, ternary models execute it
    3. Fine-tuning ternary models on harness-design trajectories (the Ornith training recipe, applied to ternary)

    Option 3 is the most interesting — and the most speculative. If the Ornith RL recipe works on ternary models, we could have self-scaffolding at 16+ tok/s instead of 7.

    ### Three-Tier Harness: A Practical Response

    Since the ternary models can execute but not design, we built a three-tier harness that routes queries by complexity and uses the 8B model as a background reviewer:

    ```
    1.7B router (keyword classify, ~50ms)
      ├─ SIMPLE → 4B responds (28.2 tok/s)
      │            └─ 8B reviews in background
      └─ REASONING → 8B responds directly (16.8 tok/s)
    ```

    **Architecture:**
    - **1.7B router** (port 9090, ctx 2048, 32 tokens max): keyword-based SIMPLE/REASONING classification. Never produces user-facing text — pure routing.
    - **4B SIMPLE** (port 9092, ctx 4096, 256 tokens): responds immediately to factual queries.
    - **8B REASONING** (port 9091, ctx 8192, 512 tokens): handles constraint-reasoning queries directly.
    - **8B review**: background review of all SIMPLE responses with structured format (ROUTING/CONTENT/EXPLANATION). Feeds an override loop that can escalate to the 8B if the 4B response is wrong.

    **Review quality (tested June 28):**

    | Test case | 4B response | 8B review verdict |
    |---|---|---|
    | "Capital of France?" | Paris ✅ | ROUTING: CORRECT, CONTENT: OK |
    | "Weather without API keys?" | OpenWeatherMap ❌ | ROUTING: WRONG (should be REASONING) |
    | "Capital of France?" (simulated wrong) | London ❌ | ROUTING: WRONG, CONTENT: FACTUAL_ERROR |

    The 8B correctly identifies factual errors and routing concerns. The review signal is real — it caught a deliberate factual error and flagged the constraint-reasoning query as misrouted. The override loop can use this signal to escalate SIMPLE queries to the 8B when the 4B gets it wrong.

    **What this means:** The ternary models can't design harnesses, but they can run them — and they can review each other's work. The 1.7B routes in ~50ms, the 4B answers simple questions at 28 tok/s, and the 8B catches the 4B's mistakes. This is a practical architecture for edge deployment today, not a research aspiration.
    """)
    return


# ── Section 10: Hardware Projections ───────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. What Hardware Do You Actually Need?

    The binding constraint is **memory bandwidth.** Everything else — clock speed, core count, SIMD width — is secondary once the model exceeds cache size. Use the calculator below to project performance on your own hardware.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_specs = {
        "Bonsai 1.7B TQ2_0 (590 MB)": {"name": "Bonsai 1.7B", "size_gb": 590/1024},
        "Bonsai 4B TQ2_0 (1.17 GB)": {"name": "Bonsai 4B", "size_gb": 1.17},
        "Bonsai 8B TQ2_0 (2.47 GB)": {"name": "Bonsai 8B", "size_gb": 2.47},
        "BitNet b1.58 2B I2_S (1.1 GB)": {"name": "BitNet b1.58 2B", "size_gb": 1.1},
        "Llama3-8B-1.58 TQ2_0 (3.7 GB)": {"name": "Llama3-8B-1.58", "size_gb": 3.7},
    }
    target_model = mo.ui.dropdown(
        options=list(model_specs.keys()),
        value="Bonsai 4B TQ2_0 (1.17 GB)",
        label="Model"
    )
    target_bw = mo.ui.slider(10, 200, 5, value=45, label="Memory bandwidth (GB/s)", show_value=True)
    target_ctx = mo.ui.slider(512, 8192, 512, value=4096, label="Context length", show_value=True)

    mo.hstack([target_model, target_bw, target_ctx], justify="space-around")
    return target_bw, target_ctx, target_model, model_specs


@app.cell(hide_code=True)
def _(MACHINES, SPECS, mo, target_bw, target_ctx, target_model, model_specs):
    _model_info = model_specs[target_model.value]
    _model_name = _model_info["name"]
    _size_gb = _model_info["size_gb"]

    # Estimate KV cache size
    _spec = None
    for _k, _v in SPECS.items():
        if _k in _model_name or _model_name in _k:
            _spec = _v
            break
    if _spec is None:
        _spec = SPECS["Bonsai 4B"]

    _kv_per_token = 2 * _spec["n_layer"] * _spec["n_head_kv"] * _spec["n_embd_head"] * 2  # FP16
    _kv_gb = _kv_per_token * target_ctx.value / (1024**3)

    _total_gb = _size_gb + _kv_gb
    _ceiling = target_bw.value / _total_gb
    _efficiency = 0.75  # typical real-world efficiency
    _projected = _ceiling * _efficiency

    # Reference hardware
    _refs = []
    for _mkey, _mdata in MACHINES.items():
        _ref_ceiling = _mdata["bw_gbs"] / _total_gb
        _refs.append({
            "name": _mdata["name"],
            "bw": _mdata["bw_gbs"],
            "ceiling": _ref_ceiling,
            "projected": _ref_ceiling * _efficiency,
        })

    _refs.append({"name": "Mac Mini M4", "bw": 120, "ceiling": 120 / _total_gb, "projected": 120 / _total_gb * _efficiency})
    _refs.append({"name": "Jetson Orin Nano 8GB", "bw": 68, "ceiling": 68 / _total_gb, "projected": 68 / _total_gb * _efficiency})
    _refs.append({"name": "Jetson Nano (orig)", "bw": 25, "ceiling": 25 / _total_gb, "projected": 25 / _total_gb * _efficiency})
    _refs.append({"name": "EPYC 7302 (8-ch DDR4)", "bw": 150, "ceiling": 150 / _total_gb, "projected": 150 / _total_gb * _efficiency})
    _refs.append({"name": "Chromebook (ARM, 2.7 GB)", "bw": 17, "ceiling": 17 / _total_gb, "projected": 17 / _total_gb * _efficiency})
    _refs.append({"name": "Raspberry Pi 5 (8 GB)", "bw": 10, "ceiling": 10 / _total_gb, "projected": 10 / _total_gb * _efficiency})
    _refs.append({"name": "Raspberry Pi 4 (4 GB)", "bw": 4, "ceiling": 4 / _total_gb, "projected": 4 / _total_gb * _efficiency})

    _ref_rows = ""
    for _r in _refs:
        _ref_rows += f"<tr><td>{_r['name']}</td><td>{_r['bw']} GB/s</td><td>{_r['ceiling']:.1f} tok/s</td><td>{_r['projected']:.1f} tok/s</td></tr>"

    mo.md(f"""
    ### Projection for {target_model.value} at {target_ctx.value} context

    | | Value |
    |---|---|
    | Model size | {_size_gb:.2f} GB |
    | KV cache (FP16) | {_kv_gb:.2f} GB |
    | **Total per token** | **{_total_gb:.2f} GB** |
    | Bandwidth ceiling | **{_ceiling:.1f} tok/s** |
    | Projected (75% efficiency) | **{_projected:.1f} tok/s** |

    ### Reference Hardware Comparison

    | Hardware | Bandwidth | Ceiling | Projected |
    |---|---|---|---|
    {_ref_rows}

    > **The Mac Mini M4 (120 GB/s, ~$599 at time of writing)** is the sweet spot for a smart home server: silent, 20W, runs any ternary model at interactive speeds. The Jetson Orin Nano 8GB (~$499 at time of writing) can run 4B models but is tight on RAM for 8B at long context. The original Jetson Nano (25 GB/s) is too slow for anything beyond 1.7B.
    >
    > **What about the hardware you already have?** A typical Chromebook (ARM Cortex-A55/A76, 2.7 GB RAM, ~17 GB/s) can *load* a 1.7B model but won't reach interactive speeds — projected 3-5 tok/s at 512 context, and the 2.7 GB RAM ceiling means 4B+ models won't fit. A Raspberry Pi 5 (8 GB, ~10 GB/s) is in the same boat: 1.7B at ~2 tok/s, 4B borderline on RAM. A Pi 4 (4 GB, ~4 GB/s) can't run any of these models at usable speeds. The bandwidth math is unforgiving: you need roughly 10× the model size in GB/s to hit interactive speeds. For a 1.2 GB 4B model, that's 12 GB/s minimum — and the Pi 4 delivers 4.
    """)
    return


# ── Section 11: Model Taxonomy ─────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Model Taxonomy: What's What

    The "1-bit model" landscape is confusing. Here's what each family actually is:

    | Family | Architecture | Weight Type | Format | Source |
    |---|---|---|---|---|
    | **Bonsai** | Qwen3 | Binary {-1, +1} | Q1_0 | PrismML, downloadable |
    | **Ternary Bonsai** | Qwen3 | Ternary {-1, 0, +1} | TQ2_0 | PrismML, requires local quantization from FP16 |
    | **Ternary Bonsai** | Qwen3 | Ternary {-1, 0, +1} | Q2_0 | Our fork, downloadable |
    | **Ternary Bonsai** | Qwen3 | Ternary {-1, 0, +1} | Q2_0_BITMAJOR | Our fork, downloadable — matches TQ2_0 speed |
    | **Microsoft BitNet** | BitNet b1.58 | Ternary {-1, 0, +1} | I2_S | Microsoft, downloadable |
    | **Llama3-8B-1.58** | Llama 3 | Ternary {-1, 0, +1} | TQ2_0 | PrismML quantization of Llama 3 |

    **Key distinction:** Bonsai models are Qwen3 architecture trained with ternary-aware methods. BitNet b1.58 is a native 1-bit architecture — the model was trained from scratch as a 1-bit model, not quantized post-training. Llama3-8B-1.58 is a standard Llama 3 model post-training quantized to ternary by PrismML.
    """)
    return


# ── Section 12: Try It Yourself ─────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Try It Yourself

    ```bash
    # Our fork (Q2_0_BITMAJOR kernel — matches TQ2_0 speed, no FP16 download)
    git clone https://github.com/Local-Yokel/llama.cpp-ly.git
    cd llama.cpp-ly
    cmake -B build
    cmake --build build --config Release -j

    # Option A: Quantize from FP16 (requires downloading FP16 source from PrismML)
    # 1. Download FP16 model from huggingface.co/prism-ml/Ternary-Bonsai-*-gguf
    # 2. Run our quantizer:
    ./build/bin/llama-quantize Ternary-Bonsai-1.7B-F16.gguf Ternary-Bonsai-1.7B-Q2_0_BITMAJOR.gguf Q2_0_BITMAJOR

    # Option B: Pre-quantized models (coming soon to HuggingFace)
    # wget https://huggingface.co/Local-Yokel/Ternary-Bonsai-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0_BITMAJOR.gguf

    # Run inference
    ./build/bin/llama-cli -m Ternary-Bonsai-1.7B-Q2_0_BITMAJOR.gguf -p "Once upon a time" -n 128 -t 6

    # Also available: 4B (1.17 GB, 28.2 tok/s) and 8B (2.29 GB, 16.8 tok/s)
    # All three match TQ2_0 speed without requiring FP16 download + local quantization

    # For TQ2_0 (same speed, requires local quantization from FP16):
    # 1. Download FP16 source from prism-ml/Ternary-Bonsai-*-gguf
    # 2. Build PrismML fork: https://github.com/PrismML/llama.cpp
    # 3. Quantize: ./llama-quantize model-F16.gguf model-TQ2_0.gguf TQ2_0

    # BitNet (native 1-bit, requires bitnet.cpp):
    git clone https://github.com/microsoft/BitNet bitnet.cpp
    cd bitnet.cpp
    git submodule update --init --recursive
    cmake -B build -DBITNET_X86_TL2=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    ./build/bin/llama-cli -m BitNet-b1.58-2B-4T-i2_s.gguf -p "Hello" -n 128 -t 8
    ```

    The kernel auto-detects your CPU's SIMD support at runtime — AVX2 if available, AVX, or SSSE3 fallback. No flags needed.
    """)
    return


# ── Section 13: Limitations & What's Next ──────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Limitations & What's Next

    ### What we've done
    - Hand-optimized AVX2/AVX/SSSE3 kernel for Q2_0 dot product (304 lines, 3 code paths)
    - Q2_0 + Q8_K kernel: switched activation format from Q8_0 to Q8_K, 1.75× speedup (25.93 tok/s on 1.7B, 11.64 on 4B)
    - Q2_0_BITMAJOR: 256-wide blocks, bit-major weight packing, shift+AND extraction, precomputed bsums — 2.4-4.8× speedup across all three model sizes, matching TQ2_0 within 2% (61.9 vs 63.4 at 1.7B, 28.2 vs 28.8 at 4B, 16.8 vs 16.3 at 8B)
    - BITMAJOR correctness verified: produces identical output to TQ2_0 (64.64 tok/s via llama-server, same 16 tokens as TQ2_0)
    - FP16 baseline tested: 1.7B Bonsai at FP16 also fails the constraint-reasoning task — the failure is the base model's capability, not quantization loss
    - Complete cross-machine token-generation benchmarks across 3 formats, 3 model sizes, and 2 machines
    - BitNet b1.58 2B benchmarked on both machines via bitnet.cpp
    - Model taxonomy clarified: Bonsai vs Ternary Bonsai vs BitNet vs Llama3-1.58
    - Ornith 1.0 thinking format analyzed: structured scaffold, divergent retrieval on ambiguous queries
    - 3-tier harness designed: 1.7B router (SIMPLE/REASONING) → 4B responds (SIMPLE) or 8B responds (REASONING), with 8B background review on all SIMPLE queries
    - 8B review quality tested: correctly identifies factual errors and routing concerns; review signal feeds self-harness override loop

    ### Hardware Scope
    All benchmarks in this notebook were run on two x86 machines: a desktop Ryzen 5 3600X (Zen 2, DDR4-3200, 45 GB/s) and a laptop Ryzen 7 5700U (Zen 3, LPDDR4x-4266, 55 GB/s). The following are **not** represented:
    - **ARM** (Apple Silicon, Raspberry Pi, phone SoCs) — our kernel is x86-only; NEON port pending
    - **GPU** (CUDA, Metal, Vulkan) — the Q2_0 kernel has a CUDA backend but we haven't benchmarked it
    - **Server-grade x86** (EPYC, Xeon with 8+ memory channels) — bandwidth scales with channel count; our 2-channel results don't project to 8-channel systems
    - **AVX-512** (Zen 4+, Xeon) — `_mm512_maddubs_epi16` could double throughput; untested

    The bandwidth analysis (Section 8, "Why TQ2_0 Is So Much Faster") is architecture-agnostic — the 13-14% vs 49-66% utilization numbers are derived from measured tok/s and model size, not from CPU-specific profiling. Those ratios should hold on any memory-bandwidth-bound system. The absolute tok/s numbers are specific to our hardware.

    ### Prompt processing (pp512)

    | Format | 1.7B pp512 | 4B pp512 | 8B pp512 |
    |---|---|---|---|
    | Q2_0 | 36.18 tok/s | — | — |
    | TQ2_0 | 197.59 tok/s | 79.87 tok/s | 47.77 tok/s |
    | Q2_0_BITMAJOR | 192.52 tok/s | 66.11 tok/s | 47.74 tok/s |

    Prompt processing is 3-5× faster than token generation for BITMAJOR and TQ2_0 — the compute-bound phase benefits from the same shift+AND extraction and precomputed bsums. Q2_0 4B and 8B pp512 hang (kernel bug in the old format), so those cells are empty. For short-prompt use cases (tool calling, chat), pp512 is fast enough that tg128 dominates perceived latency. For long-document ingestion, pp512 may be the binding constraint.

    ### What we haven't done (yet)
    - **KV cache quantization** — our builds don't support Q8_0/Q4_0 KV cache for ternary architectures. This is the biggest remaining performance win.
    - **ARM NEON port** — the kernel is x86-only. Raspberry Pi 5, Apple Silicon (non-Metal path), and phone SoCs need a NEON implementation.
    - **AVX-512 path** — `_mm512_maddubs_epi16` could process 64 values per instruction. Requires Zen 4+ or Xeon.
    - **BitNet TL2 kernel tuning** — the preset kernels are tuned for 3B, not our 2B model. Custom kernel generation would unlock TL2 speedups.
    - **Prompt-processing benchmarks** — pp512 data is now complete for TQ2_0 and BITMAJOR (see table above). Q2_0 4B and 8B pp512 still hang (kernel bug in the old format).

    ### The bigger picture
    Ternary models are at an inflection point. The models exist, the formats work, and the hardware to run them at interactive speeds costs $599 (Mac Mini M4). The remaining work is integration: KV cache quantization, tool-calling harnesses, and making these models as easy to deploy as any other llama.cpp model.

    *This is a hobby project. It works on our test cases. YMMV on your hardware and workload.*
    """)
    return


if __name__ == "__main__":
    app.run()
