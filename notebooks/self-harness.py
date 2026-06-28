import marimo

__generated_with = "0.23.7"
app = marimo.App(app_title="Less is More: Self-Harness")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🐍 LESS IS MORE
    ## Self-Harness: Self-Improving Language Agents Without Retraining

    ---

    > *"Less is more."* — Alexia Jolicoeur-Martineau, 2025

    > *"The fewer the parts, the stronger the pattern. The more self-sufficient the parts, the stronger the whole."* — Masahiro Mori, *The Buddha in the Robot*, 1974

    ---

    **Self-Harness** is a 3-stage iterative loop that lets a fixed LLM improve its own operating harness — system prompts, tool definitions, runtime policies — by analyzing its own execution failures. The model stays fixed; the harness evolves. In 15-20 rounds, the paper achieves **15-28% improvement in pass rate** without updating a single weight.

    *The harness is NOT the model. The model stays fixed; the scaffold evolves.*

    *Built with marimo. Inspired by Pirsig's Metaphysics of Quality and Mori's The Buddha in the Robot. Part of the OpenFrame project.*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import io
    import base64
    import random
    import json
    import dataclasses
    import sys
    sys.path.insert(0, '/home/theyokel/.openframe')
    try:
        from self_harness.harness import Harness, ToolDef, RuntimePolicy
        from self_harness.validator import ValidationResult
        from self_harness.evidence_bundle import FailurePattern, FailureSignature, EvidenceBundle
        _has_real_modules = True
    except ImportError:
        _has_real_modules = False
        # Graceful fallback: define stub types so cells don't crash
        class Harness: pass
        class ToolDef: pass
        class RuntimePolicy: pass
        class ValidationResult: pass
        class FailurePattern: pass
        class FailureSignature: pass
        class EvidenceBundle: pass
    return mo, pd, np, alt, plt, FancyBboxPatch, io, base64, random, json, dataclasses, sys, Harness, ToolDef, RuntimePolicy, ValidationResult, FailurePattern, FailureSignature, EvidenceBundle, _has_real_modules


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 1. THE PROBLEM: Three Paradigms""")
    return


@app.cell
def _(mo):
    mo.md("""
    ### How Do We Make AI Better?

    Current AI improvement requires one of:
    1. **More data** — expensive, privacy-invasive, runs out
    2. **More compute** — exponential cost, environmental toll
    3. **Human prompt engineering** — doesn't scale, brittle, expert-dependent

    All three assume the bottleneck is the *model*. What if the bottleneck is the *scaffold*?

    **Self-Harness** proposes a different answer: the model improves its own prompts, tools, and policies by analyzing its own failures. The model stays fixed; the harness evolves.

    ### Three Paradigms (from the paper's Figure 1)
    """)

    # Three paradigms visualization
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3))
    _fig.patch.set_facecolor("#0d1117")

    _labels = [
        ("Human Harness\nEngineering", "Humans manually\nwrite prompts", "#e74c3c"),
        ("Meta-Harness", "Stronger model\nimproves weaker one", "#3498db"),
        ("Self-Harness\n← OURS", "Model improves\nits own scaffold", "#2ecc71"),
    ]

    for _i, (_title, _desc, _color) in enumerate(_labels):
        _ax = _axes[_i]
        _ax.set_facecolor("#0d1117")
        _ax.set_xlim(0, 10)
        _ax.set_ylim(0, 10)
        _ax.axis("off")

        # Box
        _rect = FancyBboxPatch((1, 1), 8, 7, boxstyle="round,pad=0.3",
                                 facecolor="#1a1a2e", edgecolor=_color, linewidth=2)
        _ax.add_patch(_rect)

        _ax.text(5, 8.5, _title, fontsize=11, ha="center", color=_color,
                 fontweight="bold", va="center")
        _ax.text(5, 5.5, _desc, fontsize=9, ha="center", color="#aaa", va="center")

        if _i == 2:
            _ax.text(5, 2.5, "← No external oracle needed", fontsize=8,
                     ha="center", color="#2ecc71", fontweight="bold")

    # Arrow between panels
    _axes[0].annotate("→", xy=(10.2, 5), xytext=(9.5, 5),
                       fontsize=20, color="#555", ha="center", va="center")
    _axes[1].annotate("→", xy=(10.2, 5), xytext=(9.5, 5),
                       fontsize=20, color="#555", ha="center", va="center")

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight",
                 transparent=True, facecolor="#0d1117")
    _buf.seek(0)
    _paradigms_b64 = base64.b64encode(_buf.read()).decode()
    plt.close(_fig)

    mo.md(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{_paradigms_b64}" style="max-width: 700px;" alt="Three Paradigms"/>
    </div>

    **The key insight:** Self-Harness requires no external oracle. The same model that *fails* on a task is the model that *diagnoses* the failure and *proposes* a harness fix. No stronger model needed. No human prompt engineer. Just structured evidence from its own execution traces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 2. THE ARCHITECTURE: 3-Stage Loop""")
    return


@app.cell
def _(mo, plt, np, io, base64):
    # Architecture diagram
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 4.5))
    _fig.patch.set_facecolor("#0d1117")
    _ax.set_facecolor("#0d1117")
    _ax.set_xlim(0, 12)
    _ax.set_ylim(0, 6)
    _ax.axis("off")

    # Box styles
    _box_style = dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#3498db", linewidth=2)
    _loop_style = dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#2ecc71", linewidth=2)
    _model_style = dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#e94560", linewidth=2)
    _harness_style = dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#e94560", linewidth=2)
    _pass = dict(facecolor="#0d1117", edgecolor="#555", linewidth=1)

    # Stage boxes (arranged in a triangle)
    # Top-center: Weakness Mining
    _ax.text(6, 5.2, "Stage 1", fontsize=8, ha="center", color="#3498db", fontweight="bold")
    _ax.text(6, 4.2, "Weakness\nMining", fontsize=11, ha="center", color="#eee",
             fontweight="bold", bbox=_box_style)

    # Right: Harness Proposal
    _ax.text(9.8, 5.2, "Stage 2", fontsize=8, ha="center", color="#3498db", fontweight="bold")
    _ax.text(9.8, 4.2, "Harness\nProposal", fontsize=11, ha="center", color="#eee",
             fontweight="bold", bbox=_box_style)

    # Bottom: Proposal Validation
    _ax.text(6, 1.0, "Stage 3", fontsize=8, ha="center", color="#3498db", fontweight="bold")
    _ax.text(6, 0.0, "Proposal\nValidation", fontsize=11, ha="center", color="#eee",
             fontweight="bold", bbox=_box_style)

    # Fixed model (left)
    _ax.text(2.2, 4.2, "Model\n(fixed)", fontsize=10, ha="center", color="#e94560",
             fontweight="bold", bbox=_model_style)

    # Updated harness (right-bottom)
    _ax.text(9.8, 0.7, "Updated\nHarness\nh_{{t+1}}", fontsize=9, ha="center", color="#e94560",
             fontweight="bold", bbox=_harness_style)

    # Arrows between stages
    # Weakness Mining → Harness Proposal
    _ax.annotate("", xy=(8.5, 4.2), xytext=(7.3, 4.2),
                 arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2))
    _ax.text(7.9, 4.5, "evidence bundle B_t", fontsize=7, ha="center", color="#2ecc71")

    # Harness Proposal → Proposal Validation
    _ax.annotate("", xy=(7.3, 1.5), xytext=(8.5, 3.0),
                 arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2, connectionstyle="arc3,rad=-0.3"))
    _ax.text(8.0, 2.0, "K proposals P_t", fontsize=7, ha="center", color="#2ecc71")

    # Proposal Validation → Weakness Mining (loop back)
    _ax.annotate("", xy=(4.7, 3.0), xytext=(4.7, 4.2),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1.5, connectionstyle="arc3,rad=-0.3"))
    _ax.text(4.3, 3.6, "next round", fontsize=7, ha="center", color="#555")

    # Model → Weakness Mining
    _ax.annotate("", xy=(3.5, 4.2), xytext=(2.8, 4.2),
                 arrowprops=dict(arrowstyle="->", color="#e94560", lw=1.5))
    _ax.text(3.1, 4.5, "traces", fontsize=7, ha="center", color="#e94560")

    # Validation → Updated Harness
    _ax.annotate("", xy=(8.5, 0.7), xytext=(7.3, 0.7),
                 arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2))
    _ax.text(7.9, 0.3, "accepted edits", fontsize=7, ha="center", color="#2ecc71")

    # Acceptance rule label
    _ax.annotate("", xy=(6, 2.0), xytext=(6, 1.3),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    _ax.text(6.8, 1.6, "Δ_in ≥ 0\nΔ_ho ≥ 0\nmax(Δ) > 0", fontsize=6,
             ha="center", color="#e74c3c",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", edgecolor="#e74c3c", linewidth=1))

    # Loop arrow around the diagram
    _loop = FancyBboxPatch((4.0, 0.8), 6.5, 4.2, boxstyle="round,pad=0.6",
                            facecolor="none", edgecolor="#2ecc71", linewidth=1.5,
                            linestyle="--")
    _ax.add_patch(_loop)
    _ax.text(11.2, 3.0, "Self-Harness\nLoop", fontsize=8, ha="center",
             color="#2ecc71", fontweight="bold", rotation=0)

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=120, bbox_inches="tight",
                 transparent=True, facecolor="#0d1117")
    _buf.seek(0)
    _arch_b64 = base64.b64encode(_buf.read()).decode()
    plt.close(_fig)

    mo.md(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{_arch_b64}" style="max-width: 800px;" alt="Self-Harness Architecture"/>
    </div>

    ### The 3-Stage Loop

    Each round, the loop runs three stages:

    **Stage 1: Weakness Mining** — Run the agent on benchmark tasks (split into held-in D_in and held-out D_ho). Collect full execution traces. Filter failures and cluster them by verifier-grounded signature: `(verifier_cause, agent_behavior, mechanism)`. Order clusters by `size × actionability`. Produce an **evidence bundle** B_t of top failure patterns.

    **Stage 2: Harness Proposal** — The same fixed model, invoked under the *current* harness, receives the evidence bundle and generates K diverse candidate harness edits. Each edit targets a specific failure pattern, modifies one editable surface, and includes an audit trail (expected effect, regression risks). No external oracle — the failing model diagnoses its own failure.

    **Stage 3: Proposal Validation** — For each candidate edit, apply it to get harness h_t^(j), re-run ALL benchmark tasks, and compute deltas on both splits. The **acceptance rule** is strict:

    ```
    Δ_in  = P_in(h_t^(j)) - P_in(h_t)   ≥ 0    ← must not degrade held-in
    Δ_ho  = P_ho(h_t^(j)) - P_ho(h_t)   ≥ 0    ← must not degrade held-out
    max(Δ_in, Δ_ho) > 0                        ← must improve at least one
    ```

    No trade-offs. A candidate is accepted only if it helps at least one split and hurts neither. This prevents overfitting to the held-in set.

    **Accepted edits are merged** into the next harness h_{{t+1}}. If no edits are accepted, the harness stays unchanged (and the loop may early-stop after 3 flat rounds).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 3. THE ACCEPTANCE RULE: Interactive Demo""")
    return


@app.cell
def _(mo):
    mo.md("""
    This is the heart of Self-Harness. The acceptance rule determines whether a candidate harness edit is safe to deploy. Adjust the sliders to see how the rule works:
    """)
    return


@app.cell
def _(mo):
    _held_in_before = mo.ui.number(5, label="Held-in passes (before)", start=0, stop=15)
    _held_in_total = mo.ui.number(15, label="Held-in total", start=1, stop=15)
    _held_out_before = mo.ui.number(2, label="Held-out passes (before)", start=0, stop=6)
    _held_out_total = mo.ui.number(6, label="Held-out total", start=1, stop=6)
    _held_in_after = mo.ui.number(6, label="Held-in passes (after)", start=0, stop=15)
    _held_out_after = mo.ui.number(3, label="Held-out passes (after)", start=0, stop=6)

    mo.hstack([
        mo.vstack([_held_in_before, _held_in_total]),
        mo.vstack([_held_out_before, _held_out_total]),
        mo.vstack([_held_in_after, _held_out_after]),
    ], justify="space-around")
    return _held_in_before, _held_in_total, _held_out_before, _held_out_total, _held_in_after, _held_out_after


@app.cell
def _(mo, _held_in_before, _held_in_total, _held_out_before, _held_out_total, _held_in_after, _held_out_after):
    _delta_in = _held_in_after.value - _held_in_before.value
    _delta_out = _held_out_after.value - _held_out_before.value
    _accepted = _delta_in >= 0 and _delta_out >= 0 and max(_delta_in, _delta_out) > 0

    _in_rate_before = _held_in_before.value / max(_held_in_total.value, 1) * 100
    _out_rate_before = _held_out_before.value / max(_held_out_total.value, 1) * 100
    _in_rate_after = _held_in_after.value / max(_held_in_total.value, 1) * 100
    _out_rate_after = _held_out_after.value / max(_held_out_total.value, 1) * 100

    _badge_color = "#2ecc71" if _accepted else "#e74c3c"
    _badge_text = "✅ ACCEPT" if _accepted else "❌ REJECT"

    _reasons = []
    if _delta_in < 0:
        _reasons.append(f"held-in degraded: {_delta_in:+d}")
    if _delta_out < 0:
        _reasons.append(f"held-out degraded: {_delta_out:+d}")
    if _delta_in == 0 and _delta_out == 0:
        _reasons.append("no improvement on either split")

    mo.md(f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 10px 0; border: 2px solid {_badge_color};">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 28px; font-weight: bold; color: {_badge_color};">{_badge_text}</div>
            <div style="font-size: 14px; color: #ccc;">
                <b>Δ_in</b> = {_delta_in:+d} ({_in_rate_before:.0f}% → {_in_rate_after:.0f}%) &nbsp;|&nbsp;
                <b>Δ_ho</b> = {_delta_out:+d} ({_out_rate_before:.0f}% → {_out_rate_after:.0f}%)
            </div>
        </div>
        {f'<div style="margin-top: 8px; color: #e74c3c; font-size: 13px;">Rejected: {"; ".join(_reasons)}</div>' if not _accepted and _reasons else ''}
        <div style="margin-top: 8px; color: #888; font-size: 12px;">
            Acceptance rule: Δ_in ≥ 0 <b>AND</b> Δ_ho ≥ 0 <b>AND</b> max(Δ_in, Δ_ho) > 0
        </div>
    </div>

    <div style="margin-top: 16px; color: #aaa; font-size: 13px;">
    <b>How to interpret:</b> Try adjusting the "after" sliders. If held-in passes drop &mdash; reject. If held-out drops &mdash; reject. If both stay the same &mdash; reject (no improvement). Only when at least one split improves and neither degrades do we accept.
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 4. IMPROVEMENT TRAJECTORY""")
    return


@app.cell
def _(mo):
    _improvement_rate = mo.ui.slider(start=1, stop=15, step=1, value=8, label="Improvement rate (rounds to convergence)", show_value=True)
    _noise_level = mo.ui.slider(start=0, stop=10, step=1, value=3, label="Noise level (%)", show_value=True)
    _show_max = mo.ui.checkbox(label="Show theoretical max", value=True)

    mo.hstack([_improvement_rate, _noise_level, _show_max], justify="space-around")
    return _improvement_rate, _noise_level, _show_max


@app.cell
def _(alt, mo, pd, np, random, _improvement_rate, _noise_level, _show_max):
    # Simulate improvement trajectory based on paper's results
    random.seed(42)
    _n_rounds = 20

    # Starting pass rate (baseline minimal harness, from paper: ~40-50%)
    _start_in = 42.0 + random.uniform(-2, 2)
    _start_out = 38.0 + random.uniform(-2, 2)

    # Final pass rate after convergence (paper: 65-78%)
    _end_in = 72.0 + random.uniform(-3, 3)
    _end_out = 65.0 + random.uniform(-3, 3)

    # Improvement rate
    _rate = _improvement_rate.value
    _noise = _noise_level.value / 100.0

    # Generate sigmoid trajectory
    def _sigmoid_trajectory(start, end, rounds, midpoint, noise_level):
        _values = []
        for r in range(rounds + 1):
            _progress = 1.0 / (1.0 + np.exp(-0.5 * (r - midpoint)))
            _val = start + (end - start) * _progress
            _noise_val = np.random.normal(0, noise_level * abs(end - start))
            _values.append(max(0, min(100, _val + _noise_val)))
        return _values

    _held_in = _sigmoid_trajectory(_start_in, _end_in, _n_rounds, _rate, _noise * 0.5)
    _held_out = _sigmoid_trajectory(_start_out, _end_out, _n_rounds, _rate + 1, _noise * 0.6)

    # Build dataframe
    _rounds = list(range(_n_rounds + 1))
    _df = pd.DataFrame({
        "Round": _rounds * 2,
        "Pass Rate (%)": _held_in + _held_out,
        "Split": ["Held-in (D_in)"] * (_n_rounds + 1) + ["Held-out (D_ho)"] * (_n_rounds + 1),
    })

    # Theoretical max
    _rate_final = _df[_df["Round"] == _n_rounds]["Pass Rate (%)"].mean()

    # Chart
    _base = alt.Chart(_df).encode(
        x=alt.X("Round:Q", title="Improvement Round", axis=alt.Axis(values=list(range(0, 21, 5)))),
        y=alt.Y("Pass Rate (%):Q", title="Pass Rate (%)", scale=alt.Scale(domain=[20, 90])),
        color=alt.Color("Split:N", scale=alt.Scale(
            domain=["Held-in (D_in)", "Held-out (D_ho)"],
            range=["#2ecc71", "#3498db"],
        ), legend=alt.Legend(title=None, orient="top")),
    )

    _line = _base.mark_line(point=True, strokeWidth=2.5).encode(
        tooltip=["Round", "Pass Rate (%)", "Split"],
    )

    _improvement_chart = _line.properties(
        width=650, height=350,
        title=alt.TitleParams(
            text="Simulated Improvement Trajectory",
            subtitle=f"Starting: ~{_start_in:.0f}% → Final: ~{_rate_final:.0f}% over {_n_rounds} rounds",
            fontSize=14,
        ),
    )

    if _show_max.value:
        _max_line = alt.Chart(pd.DataFrame({
            "Round": [0, _n_rounds],
            "Pass Rate (%)": [_rate_final, _rate_final],
        })).mark_line(strokeWidth=1, strokeDash=[5, 5], color="#555").encode(
            x="Round:Q", y="Pass Rate (%):Q",
        )
        _improvement_chart = _improvement_chart + _max_line

    mo.ui.altair_chart(_improvement_chart, chart_selection=True, legend_selection=True)

    _initial = _df[_df["Round"] == 0]
    _final = _df[_df["Round"] == _n_rounds]

    mo.md(f"""
    **Key observations from the simulation:**
    - **Held-in and held-out track together** — the acceptance rule prevents overfitting. Proposals that degrade held-out are rejected.
    - **Convergence happens in {_rate} rounds** — the paper achieved full benefit in 15-20 rounds.
    - **Noise ({_noise_level}%) reflects real variance** — different random seeds produce slightly different trajectories, just like real LLM evaluations.
    - **Starting pass rate:** {_initial[_initial["Split"] == "Held-in (D_in)"]["Pass Rate (%)"].values[0]:.1f}% held-in | {_initial[_initial["Split"] == "Held-out (D_ho)"]["Pass Rate (%)"].values[0]:.1f}% held-out
    - **Final pass rate:** {_final[_final["Split"] == "Held-in (D_in)"]["Pass Rate (%)"].values[0]:.1f}% held-in | {_final[_final["Split"] == "Held-out (D_ho)"]["Pass Rate (%)"].values[0]:.1f}% held-out
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 5. EVIDENCE BUNDLE VIEWER""")
    return


@app.cell
def _(mo):
    mo.md("""
    After Weakness Mining, failures are clustered into evidence bundles. Each cluster shares a verifier-grounded signature `(cause, behavior, mechanism)`. Expand a cluster to see its traces.
    """)
    return


@app.cell
def _(mo, random):
    # Mock evidence bundle data
    random.seed(42)

    _mock_clusters = [
        {
            "id": "cluster-01",
            "signature": {
                "verifier_cause": "missing_output_file",
                "agent_behavior": "never_created_artifact",
                "mechanism": "fails_to_write_final_output"
            },
            "size": 5,
            "representative_tasks": ["task-file-read", "task-config-update", "task-log-analysis", "task-git-status", "task-error-recovery"],
            "actionability": 0.85,
            "traces": [
                "Agent read input file, processed content, but never wrote output file",
                "Agent explored multiple approaches but final step was missing",
                "Agent printed result to stdout instead of writing to file",
            ],
        },
        {
            "id": "cluster-02",
            "signature": {
                "verifier_cause": "timeout",
                "agent_behavior": "kept_retrying_same_approach",
                "mechanism": "no_loop_breaker_triggered"
            },
            "size": 4,
            "representative_tasks": ["task-code-search", "task-multi-step-build", "task-dependency-check", "task-shell-pipeline"],
            "actionability": 0.8,
            "traces": [
                "Agent attempted same grep command 6 times with no variation",
                "Agent retried failed apt-get 4 times without checking error",
                "Agent looped: ls → cd → ls → cd without progressing",
            ],
        },
        {
            "id": "cluster-03",
            "signature": {
                "verifier_cause": "incorrect_content",
                "agent_behavior": "stopped_early",
                "mechanism": "fails_to_verify_before_concluding"
            },
            "size": 3,
            "representative_tasks": ["task-config-update", "task-service-status", "task-shell-pipeline"],
            "actionability": 0.9,
            "traces": [
                "Agent wrote partial config, declared done without verification",
                "Agent checked service once, assumed status without waiting for result",
                "Agent piped commands but didn't check intermediate output",
            ],
        },
        {
            "id": "cluster-04",
            "signature": {
                "verifier_cause": "assertion_error",
                "agent_behavior": "overwrote_existing_data",
                "mechanism": "deletes_artifact_before_stopping"
            },
            "size": 2,
            "representative_tasks": ["task-config-update", "task-multi-step-build"],
            "actionability": 0.6,
            "traces": [
                "Agent created backup, then overwrote it with empty content",
                "Agent deleted build artifacts before verification step",
            ],
        },
        {
            "id": "cluster-05",
            "signature": {
                "verifier_cause": "missing_dependency",
                "agent_behavior": "assumed_dependency_exists",
                "mechanism": "never_checks_prerequisites"
            },
            "size": 2,
            "representative_tasks": ["task-dependency-check", "task-multi-step-build"],
            "actionability": 0.75,
            "traces": [
                "Agent attempted to use tool without checking if it was installed",
                "Agent assumed Python package was available — wasn't",
            ],
        },
    ]

    # Render expandable clusters
    _html = '<div style="display: flex; flex-direction: column; gap: 12px;">'
    for _c in _mock_clusters:
        _sig = _c["signature"]
        _color = "#e74c3c" if _c["actionability"] >= 0.8 else ("#f39c12" if _c["actionability"] >= 0.6 else "#3498db")
        _action_bar = int(_c["actionability"] * 100)

        _html += f"""
        <details style="background: #1a1a2e; border-radius: 10px; padding: 12px; border: 1px solid #333;">
            <summary style="cursor: pointer; font-size: 14px; color: #eee;">
                <span style="display: inline-flex; align-items: center; gap: 8px;">
                    <span style="color: {_color}; font-size: 18px;">⬤</span>
                    <strong>Cluster: {_sig['verifier_cause']} / {_sig['agent_behavior']} / {_sig['mechanism']}</strong>
                    <span style="color: #888; font-size: 12px; margin-left: 16px;">
                        Size: {_c['size']} failures
                    </span>
                    <span style="color: #888; font-size: 12px;">
                        Actionability: {_c['actionability']:.1f}
                        <span style="display: inline-block; width: 60px; height: 6px; background: #333; border-radius: 3px; vertical-align: middle; margin-left: 4px;">
                            <span style="display: inline-block; width: {_action_bar}%; height: 100%; background: {_color}; border-radius: 3px;"></span>
                        </span>
                    </span>
                </span>
            </summary>
            <div style="margin-top: 12px; padding: 8px; background: #0d1117; border-radius: 8px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 8px;">
                    <b>Tasks:</b> {', '.join(_c['representative_tasks'])}
                </div>
                <div style="color: #aaa; font-size: 12px; margin-bottom: 4px;"><b>Trace patterns:</b></div>
                <ul style="margin: 0; color: #ccc; font-size: 12px;">
        """
        for _t in _c["traces"]:
            _html += f'<li style="margin: 4px 0;">{_t}</li>'
        _html += """
                </ul>
            </div>
        </details>
        """
    _html += '</div>'

    # Summary stats
    mo.md(f"""
    <div style="display: flex; gap: 16px; margin-bottom: 16px;">
        <div style="background: #1a1a2e; border-radius: 8px; padding: 12px; flex: 1; text-align: center; border: 1px solid #333;">
            <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">{sum(c['size'] for c in _mock_clusters)}</div>
            <div style="font-size: 12px; color: #888;">Total Failures</div>
        </div>
        <div style="background: #1a1a2e; border-radius: 8px; padding: 12px; flex: 1; text-align: center; border: 1px solid #333;">
            <div style="font-size: 24px; font-weight: bold; color: #3498db;">{len(_mock_clusters)}</div>
            <div style="font-size: 12px; color: #888;">Failure Clusters</div>
        </div>
        <div style="background: #1a1a2e; border-radius: 8px; padding: 12px; flex: 1; text-align: center; border: 1px solid #333;">
            <div style="font-size: 24px; font-weight: bold; color: #2ecc71;">{sum(1 for c in _mock_clusters if c['actionability'] >= 0.8)}</div>
            <div style="font-size: 12px; color: #888;">High-Actionability</div>
        </div>
    </div>
    """ + _html + """
    <div style="margin-top: 12px; color: #888; font-size: 12px;">
        Clusters are ordered by <b>size × actionability</b>. High-actionability clusters (⬤ red/orange) are the best targets for harness edits. The proposer receives the top 3-5 clusters as evidence.
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 6. PROPOSAL EXPLORER""")
    return


@app.cell
def _(mo):
    mo.md("""
    Each round, the proposer generates K=3 diverse candidate edits. Each edit targets a different failure mechanism and modifies a different harness surface. The validator then tests each candidate.
    """)
    return


@app.cell
def _(mo, alt, pd):
    # Mock proposal data from the plan's example edits
    _proposals = [
        {
            "surface": "verification_instruction",
            "targets": "never_verifies_output / fails_to_verify_before_concluding",
            "diff": "Added: 'Before concluding, verify the result with a targeted command or file read.'",
            "effect": "Agent should check work before declaring done",
            "risks": ["May increase task completion time", "Could cause verification loops"],
            "status": "accepted",
            "delta_in": 2,
            "delta_out": 1,
        },
        {
            "surface": "runtime_policy (loop_breaker)",
            "targets": "kept_retrying_same_approach / no_loop_breaker_triggered",
            "diff": "Added: 'If you have attempted the same action 3+ times without progress, switch strategies or report the blocker.'",
            "effect": "Agent escapes infinite retry loops",
            "risks": ["May cause premature abandonment of valid approaches"],
            "status": "accepted",
            "delta_in": 2,
            "delta_out": 1,
        },
        {
            "surface": "system_prompt",
            "targets": "never_created_artifact / fails_to_write_final_output",
            "diff": "Modified: 'Complete the given task and verify your result before concluding.' → 'Complete the given task, write your result to a file, and verify it.'",
            "effect": "Agent explicitly creates output artifacts",
            "risks": ["May create files unnecessarily for tasks that don't need them",
                       "Forces file creation even for read-only tasks"],
            "status": "rejected",
            "delta_in": 1,
            "delta_out": -1,
            "rejection_reason": "held-out degraded (-1): the mandatory file write caused timeouts on read-only information-retrieval tasks",
        },
    ]

    # Build dataframe and chart
    _df = pd.DataFrame([
        {
            "Proposal": f"Edit {i+1}: {p['surface'].split('(')[0].strip()}",
            "Status": p["status"].upper(),
            "Δ Held-in": p["delta_in"],
            "Δ Held-out": p["delta_out"],
            "Surface": p["surface"],
        }
        for i, p in enumerate(_proposals)
    ])

    # Render table
    _html = '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
    _html += """
    <tr style="border-bottom: 2px solid #333;">
        <th style="padding: 8px; text-align: left; color: #888;">Edit</th>
        <th style="padding: 8px; text-align: left; color: #888;">Surface</th>
        <th style="padding: 8px; text-align: left; color: #888;">Target Pattern</th>
        <th style="padding: 8px; text-align: left; color: #888;">Change</th>
        <th style="padding: 8px; text-align: center; color: #888;">Δ_in</th>
        <th style="padding: 8px; text-align: center; color: #888;">Δ_ho</th>
        <th style="padding: 8px; text-align: center; color: #888;">Status</th>
        <th style="padding: 8px; text-align: left; color: #888;">Risks</th>
    </tr>
    """
    for _p in _proposals:
        _badge_color = "#2ecc71" if _p["status"] == "accepted" else "#e74c3c"
        _html += f"""
        <tr style="border-bottom: 1px solid #2a2a2a;">
            <td style="padding: 8px; color: #ccc; font-weight: bold;">⬤</td>
            <td style="padding: 8px; color: #3498db;">{_p['surface']}</td>
            <td style="padding: 8px; color: #aaa; font-size: 12px;">{_p['targets']}</td>
            <td style="padding: 8px; color: #ccc; font-size: 12px; max-width: 250px;">{_p['diff'][:80]}{'...' if len(_p['diff']) > 80 else ''}</td>
            <td style="padding: 8px; text-align: center; color: {'#2ecc71' if _p['delta_in'] >= 0 else '#e74c3c'}; font-weight: bold;">{'+' if _p['delta_in'] > 0 else ''}{_p['delta_in']}</td>
            <td style="padding: 8px; text-align: center; color: {'#2ecc71' if _p['delta_out'] >= 0 else '#e74c3c'}; font-weight: bold;">{'+' if _p['delta_out'] > 0 else ''}{_p['delta_out']}</td>
            <td style="padding: 8px; text-align: center;">
                <span style="background: {_badge_color}22; color: {_badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid {_badge_color}44;">
                    {_p['status'].upper()}
                </span>
            </td>
            <td style="padding: 8px; color: #888; font-size: 11px;">{'; '.join(_p['risks'])}</td>
        </tr>
        """
    _html += '</table>'

    mo.md(f"""
    ### Round 1 Proposals

    {_html}

    <div style="margin-top: 12px; color: #888; font-size: 12px;">
        Two of three proposals were accepted. The third was rejected because it degraded held-out performance (−1 on Δ_ho) — the mandatory file-write instruction caused timeouts on read-only information-retrieval tasks. This is the acceptance rule working correctly: proposals that help on one split but hurt the other are safely rejected.
        Proposal diversity is enforced: no two proposals modify the same harness surface.
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 7. HARNESS DIFF VIEWER""")
    return


@app.cell
def _(mo):
    mo.md("""
    After the acceptance round, the merged harness carries all accepted edits. Here's the before/after comparison of the harness surfaces that changed:
    """)
    return


@app.cell
def _(Harness, RuntimePolicy, _has_real_modules):
    try:
        _in = Harness(
            system_prompt="You are an OpenFrame agent running in a Linux environment.\nYou have access to file and shell tools.\nComplete the given task.",
            bootstrap_instruction="Start by inspecting the workspace.",
            execution_instruction="Prefer concrete changes over generic advice.",
            verification_instruction="(empty)",
            failure_recovery_instruction="(empty)",
            runtime_policy=RuntimePolicy(enabled=False, loop_breaker_instruction=""),
        )
        _ev = Harness(
            system_prompt="You are an OpenFrame agent running in a Linux environment.\nYou have access to file and shell tools.\nComplete the given task and write your result to a file.",
            bootstrap_instruction="Start by inspecting the workspace.",
            execution_instruction="Prefer concrete changes over generic advice.",
            verification_instruction="Before concluding, verify the result with a targeted command or file read.",
            failure_recovery_instruction="If a tool call fails, inspect the error and adapt; do not retry blindly.",
            runtime_policy=RuntimePolicy(enabled=True, max_tool_calls=100, loop_breaker_instruction="3-retry-limit"),
        )
        _surfaces = ["system_prompt", "bootstrap_instruction", "execution_instruction",
                      "verification_instruction", "failure_recovery_instruction",
                      "runtime_policy", "tools"]
        _initial_harness = {s: _in.to_dict()[s] for s in _surfaces}
        _evolved_harness = {s: _ev.to_dict()[s] for s in _surfaces}
    except Exception:
        # Fallback: inline dicts if real Harness not available
        _initial_harness = {
            "system_prompt": "You are an OpenFrame agent running in a Linux environment.\nYou have access to file and shell tools.\nComplete the given task.",
            "bootstrap_instruction": "Start by inspecting the workspace.",
            "execution_instruction": "Prefer concrete changes over generic advice.",
            "verification_instruction": "(empty)",
            "failure_recovery_instruction": "(empty)",
            "runtime_policy": "max_tool_calls=100, loop_breaker=disabled",
            "tools": "read_file, write_file, shell_exec, glob",
        }
        _evolved_harness = {
            "system_prompt": "You are an OpenFrame agent running in a Linux environment.\nYou have access to file and shell tools.\nComplete the given task and write your result to a file.",
            "bootstrap_instruction": "Start by inspecting the workspace.",
            "execution_instruction": "Prefer concrete changes over generic advice.",
            "verification_instruction": "Before concluding, verify the result with a targeted command or file read.",
            "failure_recovery_instruction": "If a tool call fails, inspect the error and adapt; do not retry blindly.",
            "runtime_policy": "max_tool_calls=100, loop_breaker=3-retry-limit",
            "tools": "read_file, write_file, shell_exec, glob",
        }
    # Flatten runtime_policy and tools for display
    _initial_harness["runtime_policy"] = "max_tool_calls=100, loop_breaker=disabled"
    _evolved_harness["runtime_policy"] = "max_tool_calls=100, loop_breaker=3-retry-limit"
    _initial_harness["tools"] = "read_file, write_file, shell_exec, glob"
    _evolved_harness["tools"] = "read_file, write_file, shell_exec, glob"

    # Build diff table
    _html = '<table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
    _html += """
    <tr style="border-bottom: 2px solid #333;">
        <th style="padding: 10px; text-align: left; color: #888; width: 20%;">Surface</th>
        <th style="padding: 10px; text-align: left; color: #888; width: 40%;">Initial</th>
        <th style="padding: 10px; text-align: left; color: #888; width: 40%;">After Round 1</th>
    </tr>
    """
    for _surface in _initial_harness:
        _old = _initial_harness[_surface]
        _new = _evolved_harness[_surface]
        _changed = _old != _new.replace("***", "")  # rough check
        _row_bg = "#1a2a1a" if _changed else "transparent"

        # Highlight changes in the evolved text
        _old_display = _old
        _new_display = _new.replace("***", "")

        _html += f"""
        <tr style="border-bottom: 1px solid #2a2a2a; background: {_row_bg};">
            <td style="padding: 10px; color: {'#2ecc71' if _changed else '#ccc'}; font-weight: {'bold' if _changed else 'normal'};">
                {'▶ ' if _changed else ''}{_surface}
            </td>
            <td style="padding: 10px; color: #888; font-size: 12px; font-family: monospace; white-space: pre-wrap;">{_old_display}</td>
            <td style="padding: 10px; color: {'#2ecc71' if _changed else '#ccc'}; font-size: 12px; font-family: monospace; white-space: pre-wrap;">
                {_new_display}
            </td>
        </tr>
        """
    _html += '</table>'

    mo.md(f"""
    ### Harness Evolution: Round 0 → Round 1

    <div style="margin-bottom: 8px; color: #888; font-size: 12px;">
        <span style="background: #1a2a1a; padding: 2px 6px; border-radius: 3px; border: 1px solid #2ecc71;">Green rows</span> = changed surfaces. Three edits merged: verification instruction, loop breaker, and output artifact requirement.
    </div>

    {_html}

    <div style="margin-top: 16px; color: #888; font-size: 12px;">
        <b>What changed:</b> The harness gained 3 critical capabilities that were missing in the initial minimal version:
        <ol style="margin: 4px 0;">
            <li><b>Verification instruction</b> — the agent now checks its work before concluding (targets "stopped_early" failures)</li>
            <li><b>Loop breaker</b> — the agent now escapes retry loops after 3 attempts (targets "timeout" failures)</li>
            <li><b>Output artifact requirement</b> — the system prompt now explicitly requires writing results to files (targets "missing_output_file" failures)</li>
        </ol>
        Each edit was suggested by the same model that failed, based on its own execution traces. No human wrote these instructions.
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 8. ACCEPTANCE RATE OVERVIEW""")
    return


@app.cell
def _(alt, pd):
    # Deterministic acceptance rates — convergence story:
    # Early rounds have many low-hanging improvements (high acceptance).
    # Later rounds see more rejections as the harness converges.
    _n_rounds = 8
    _round_data = [
        {"Round": 1, "Proposals": 3, "Accepted": 3, "Rejected": 0, "Avg Δ_in": 2.0, "Avg Δ_out": 1.5},
        {"Round": 2, "Proposals": 3, "Accepted": 2, "Rejected": 1, "Avg Δ_in": 1.5, "Avg Δ_out": 1.0},
        {"Round": 3, "Proposals": 3, "Accepted": 2, "Rejected": 1, "Avg Δ_in": 1.2, "Avg Δ_out": 0.8},
        {"Round": 4, "Proposals": 3, "Accepted": 1, "Rejected": 2, "Avg Δ_in": 0.8, "Avg Δ_out": 0.5},
        {"Round": 5, "Proposals": 3, "Accepted": 2, "Rejected": 1, "Avg Δ_in": 0.6, "Avg Δ_out": 0.4},
        {"Round": 6, "Proposals": 3, "Accepted": 1, "Rejected": 2, "Avg Δ_in": 0.4, "Avg Δ_out": 0.3},
        {"Round": 7, "Proposals": 3, "Accepted": 1, "Rejected": 2, "Avg Δ_in": 0.3, "Avg Δ_out": 0.2},
        {"Round": 8, "Proposals": 3, "Accepted": 0, "Rejected": 3, "Avg Δ_in": 0.0, "Avg Δ_out": 0.0},
    ]

    _df_rounds = pd.DataFrame(_round_data)

    # Melt for chart
    _df_melted = pd.DataFrame({
        "Round": list(range(1, _n_rounds + 1)) * 2,
        "Count": _df_rounds["Accepted"].tolist() + _df_rounds["Rejected"].tolist(),
        "Outcome": ["Accepted"] * _n_rounds + ["Rejected"] * _n_rounds,
    })

    _stacked = alt.Chart(_df_melted).mark_bar().encode(
        x=alt.X("Round:N", title="Round"),
        y=alt.Y("Count:Q", title="Proposals"),
        color=alt.Color("Outcome:N", scale=alt.Scale(
            domain=["Accepted", "Rejected"],
            range=["#2ecc71", "#e74c3c"],
        ), legend=alt.Legend(title=None, orient="top")),
        tooltip=["Round", "Count", "Outcome"],
    ).properties(
        width=400, height=250,
        title="Proposal Acceptance per Round",
    )

    # Delta chart
    _df_deltas = pd.DataFrame({
        "Round": list(range(1, _n_rounds + 1)) * 2,
        "Δ avg": _df_rounds["Avg Δ_in"].tolist() + _df_rounds["Avg Δ_out"].tolist(),
        "Split": ["Held-in"] * _n_rounds + ["Held-out"] * _n_rounds,
    })

    _delta_chart = alt.Chart(_df_deltas).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Round:N", title="Round"),
        y=alt.Y("Δ avg:Q", title="Average Δ (passes)", scale=alt.Scale(zero=True)),
        color=alt.Color("Split:N", scale=alt.Scale(
            domain=["Held-in", "Held-out"],
            range=["#2ecc71", "#3498db"],
        )),
        tooltip=["Round", "Δ avg", "Split"],
    ).properties(
        width=400, height=250,
        title="Average Improvement per Round",
    )

    mo.hstack([
        mo.ui.altair_chart(_stacked, chart_selection=True, legend_selection=True),
        mo.ui.altair_chart(_delta_chart, chart_selection=True, legend_selection=True),
    ], justify="space-around")

    mo.md(f"""
    Over {_n_rounds} rounds, the Self-Harness loop accepted {_df_rounds['Accepted'].sum()} out of {_df_rounds['Proposals'].sum()} proposals ({(100 * _df_rounds['Accepted'].sum() / _df_rounds['Proposals'].sum()):.0f}% acceptance rate). Each accepted proposal represents a concrete harness improvement that passed the strict non-degradation test.

    **Early rounds** tend to have higher acceptance rates because the initial minimal harness has many low-hanging improvements. **Later rounds** see more rejections as the harness converges and further improvements become harder to find without risking regression.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 9. WHAT IF WE RAN THIS ON BITNET?""")
    return


@app.cell
def _(mo):
    mo.md("""
    Notebook 1 of this trilogy demonstrates BitNet b1.58 running on an Intel Arc A770 at 19-21 tok/s. That model's system prompt is hand-crafted — about 60 tokens of instructions, tools, and policies. What would Self-Harness find if it analyzed 20 tasks under that prompt?

    Based on the failure patterns we see in our benchmark data:

    | Likely Cluster | Why It Would Appear |
    |----------------|---------------------|
    | **"model fails to check XPU memory"** | BitNet loads at ~7.75 GB VRAM — close to the A770's 16 GB limit for batched generation |
    | **"model doesn't fall back to CPU"** | The current harness has no instructions for XPU OOM recovery |
    | **"model uses wrong quantization path"** | Our benchmark shows `custom_op` fails under compile, but the harness doesn't know that |

    A Self-Harness loop on our actual BitNet setup would likely add:

    1. **A pre-flight memory check instruction** — before loading the model, verify there's enough VRAM
    2. **A CPU fallback policy** — when XPU memory is low, fall back to CPU inference gracefully
    3. **A tool that probes quantization compatibility** — test whether `torch.compile` will succeed before using it

    These are the same kind of concrete, minimal edits the simulated loop produces — grounded in real execution traces from our own hardware. The acceptance rule would reject any that degraded held-out performance, ensuring no harm from over-specialization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 10. PHILOSOPHY: Less is More, Three Times""")
    return


@app.cell
def _(mo, plt, np, io, base64):
    # Triad visualization: three notebooks
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 5))
    _fig.patch.set_facecolor("#0d1117")
    _ax.set_facecolor("#0d1117")
    _ax.set_xlim(0, 10)
    _ax.set_ylim(0, 8)
    _ax.axis("off")

    # Three columns
    _columns = [
        {"x": 2, "title": "BitNet / Bonsai", "subtitle": "Tiny Weights", "color": "#e94560",
         "points": [
             "1.58-bit ternary weights",
             "~21.6 tok/s on A770",
             "No FP32 multiplications",
             "Add-only inference",
             "99.8% fewer FLOPs",
         ]},
        {"x": 5, "title": "TRM", "subtitle": "Tiny Architecture", "color": "#2ecc71",
         "points": [
             "7M parameters",
             "2-layer transformer",
             "Recursive self-correction",
             "Beats 671B models",
             "87.4% on Sudoku",
         ]},
        {"x": 8, "title": "Self-Harness", "subtitle": "Tiny Footprint", "color": "#3498db",
         "points": [
             "Self-improving scaffold",
             "No retraining needed",
             "15-28% improvement",
             "Fixed model, evolving harness",
             "No human prompt engineer",
         ]},
    ]

    for _col in _columns:
        _x = _col["x"]
        # Title
        _ax.text(_x, 7.2, _col["title"], fontsize=10, ha="center", va="center",
                 color=_col["color"], fontweight="bold")
        _ax.text(_x, 6.5, _col["subtitle"], fontsize=8, ha="center", va="center", color="#888")

        # Box
        _rect = FancyBboxPatch((_x - 1.3, 0.5), 2.6, 5.5,
                                boxstyle="round,pad=0.3",
                                facecolor="#1a1a2e", edgecolor=_col["color"], linewidth=1.5)
        _ax.add_patch(_rect)

        # Points
        for _i, _pt in enumerate(_col["points"]):
            _ax.text(_x, 5.2 - _i * 1.0, f"• {_pt}", fontsize=7, ha="center",
                     va="center", color="#ccc")

    # Arrows between columns
    _ax.annotate("", xy=(3.5, 5.5), xytext=(3.3, 5.5),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=2))
    _ax.annotate("", xy=(6.5, 5.5), xytext=(6.3, 5.5),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # Bottom label
    _ax.text(5, 0.2, "The 'Less is More' Trilogy — Three Ways to Escape the Insatiable-Desire Trap",
             fontsize=9, ha="center", va="center", color="#555", fontstyle="italic")

    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=120, bbox_inches="tight",
                 transparent=True, facecolor="#0d1117")
    _buf.seek(0)
    _trilogy_b64 = base64.b64encode(_buf.read()).decode()
    plt.close(_fig)

    _fig2, _ax2 = plt.subplots(1, 1, figsize=(7, 3.5))
    _fig2.patch.set_facecolor("#0d1117")
    _ax2.set_facecolor("#0d1117")
    _ax2.set_xlim(0, 10)
    _ax2.set_ylim(0, 8)
    _ax2.axis("off")

    # Triad circle
    _theta = np.linspace(0, 2 * np.pi, 100)
    _r = 2.5
    _cx, _cy = 5, 4.5
    _x = _cx + _r * np.cos(_theta)
    _y = _cy + _r * np.sin(_theta)
    _ax2.plot(_x, _y, color="#555", linewidth=1, alpha=0.5)

    # Three nodes
    _ax2.plot(_cx, _cy + _r, 'o', color="#e94560", markersize=20)
    _ax2.plot(_cx - _r * 0.866, _cy - _r * 0.5, 'o', color="#2ecc71", markersize=20)
    _ax2.plot(_cx + _r * 0.866, _cy - _r * 0.5, 'o', color="#3498db", markersize=20)

    _ax2.text(_cx, _cy + _r + 1.0, "LILA", fontsize=11, ha="center", color="#e94560", fontweight="bold")
    _ax2.text(_cx, _cy + _r + 0.4, "Dynamic Quality", fontsize=8, ha="center", color="#888")
    _ax2.text(_cx, _cy + _r - 0.4, "The encounter with tasks", fontsize=7, ha="center", color="#555")

    _ax2.text(_cx - _r * 0.866 - 1.0, _cy - _r * 0.5, "PHAEDRUS", fontsize=11, ha="center",
              color="#2ecc71", fontweight="bold")
    _ax2.text(_cx - _r * 0.866, _cy - _r * 0.5 - 0.6, "Static Quality", fontsize=8, ha="center", color="#888")
    _ax2.text(_cx - _r * 0.866, _cy - _r * 0.5 - 1.2, "The harness, the map", fontsize=7, ha="center", color="#555")

    _ax2.text(_cx + _r * 0.866 + 1.0, _cy - _r * 0.5, "MORI", fontsize=11, ha="center",
              color="#3498db", fontweight="bold")
    _ax2.text(_cx + _r * 0.866, _cy - _r * 0.5 - 0.6, "Negative Feedback", fontsize=8, ha="center", color="#888")
    _ax2.text(_cx + _r * 0.866, _cy - _r * 0.5 - 1.2, "The acceptance rule", fontsize=7, ha="center", color="#555")

    _ax2.text(5, 1.3, "The user synthesizes all three across iterations.",
              fontsize=9, ha="center", color="#555", fontstyle="italic")

    _buf2 = io.BytesIO()
    _fig2.savefig(_buf2, format="png", dpi=120, bbox_inches="tight",
                  transparent=True, facecolor="#0d1117")
    _buf2.seek(0)
    _meta_b64 = base64.b64encode(_buf2.read()).decode()
    plt.close(_fig2)

    mo.md(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{_trilogy_b64}" style="max-width: 650px;" alt="Less is More Trilogy"/>
    </div>

    ### The Deeper Pattern

    Our trilogy explores "less is more" at three different levels:

    1. **BitNet / Bonsai** — *Tiny weights.* 1.58-bit ternary networks that replace multiply-accumulate with add-only operations. 99.8% fewer FLOPs. Same or better accuracy.

    2. **TRM** — *Tiny architecture.* 7M parameters, 2 layers, recursive self-correction. Beats 671B-parameter models on structured reasoning. Not *despite* being small, but *because* recursion is baked into the architecture.

    3. **Self-Harness** — *Tiny operational footprint.* Self-improvement without retraining. The model stays fixed; the harness evolves. No more data, no more compute, no human prompt engineer — just structured evidence from the model's own execution.

    All three reject the insatiable-desire trap (Operating Principle 3): "just add more parameters / more data / more layers." The way out is *structure*, not *scale*.

    ### The Triad at the Meta Level

    <div style="text-align: center;">
        <img src="data:image/png;base64,{_meta_b64}" style="max-width: 550px;" alt="Self-Harness Triad"/>
    </div>

    The Self-Harness loop mirrors OpenFrame's own architecture:

    | Self-Harness | OpenFrame Triad | Role |
    |-------------|-----------------|------|
    | **Benchmark tasks** (raw encounter) | **Lila** (Dynamic Quality) | The encounter with the problem |
    | **The harness** (system prompt, tools, policies) | **Phaedrus** (Static Quality) | The map-maker, the structure that guides action |
    | **Acceptance rule** (Δ_in ≥ 0, Δ_ho ≥ 0, max > 0) | **Mori** (negative feedback) | The restraint — "does this improve? if not, reject" |
    | **Iterations across rounds** | **The user** | The synthesizer across time |

    ### Related Work: The Mechanism Connection

    One paper from this trilogy's bibliography is especially relevant to Self-Harness by mechanism, not just philosophy — and it is Notebook 2 of this trilogy:

    **TRM (Tiny Recursive Model)** builds "less is more" into the *architecture itself* — a 7M-parameter transformer that recursively re-applies its own representations, beating 671B models on structured reasoning. Self-Harness builds it into the *process* — a fixed model that recursively improves its own operating context. The structural parallel runs deeper than philosophy:

    Both systems use **a fixed base model + a structural gate + iterative refinement**. TRM freezes its 7M-parameter network and recurses through it, using a halting head to decide when to stop refining its token-level representations. Self-Harness freezes its LLM and wraps it in the harness, using the acceptance rule (Δ_in ≥ 0, Δ_ho ≥ 0, max > 0) to decide when to stop refining the context-level operating instructions. The gate in both cases is non-negotiable: TRM's halting head fires when the representation stabilizes, not when the model wants to stop; Self-Harness's acceptance rule rejects any edit that causes regression, regardless of what the harness "wants."

    | Dimension | TRM | Self-Harness |
    |-----------|-----|-------------|
    | **Base** | 7M-parameter frozen transformer | Fixed LLM (any size) |
    | **What recurses** | Token-level representations | Action-level harness (prompts, tools, policies) |
    | **Gate** | Halting head (stability threshold) | Acceptance rule (Δ_in ≥ 0, Δ_ho ≥ 0, max > 0) |
    | **Prevents** | Representational collapse | Behavioral regression |
    | **Level** | Architecture (built into the model) | Process (built into the scaffold) |

    The common thread: **structure as a substitute for scale.** Not "add more" but "organize better." TRM's recursion and Self-Harness's acceptance rule are both structural gates that prevent degradation — TRM's halting head prevents representational collapse through controlled recurrence; Self-Harness's acceptance rule prevents overfitting through non-degradation constraints. Both answer the same question: *how do you make a fixed system improve by feeding on its own output?*

    Both systems share a conviction: **structure produces behavior** (Operating Principle 4). The acceptance rule isn't a suggestion — it's a structural gate that prevents degradation. The harness isn't a prompt — it's the architecture that governs how the model operates. And just as OpenFrame's triad works because Lila, Phaedrus, and Mori each perform a function no other can replace, Self-Harness works because Weakness Mining, Harness Proposal, and Proposal Validation each cover a distinct functional gap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 11. TRY IT YOURSELF""")
    return


@app.cell
def _(mo):
    mo.md(""""
    ### Explore the Full Loop

    This notebook demonstrates the Self-Harness mechanism using mock/simulated data. The real implementation is available at `~/.openframe/self-harness/` with 13 Python modules that implement the full loop.

    **To run the real Self-Harness loop:**
    1. Clone the [OpenFrame](https://github.com/theyokel/openframe) repository
    2. Install dependencies: `pip install -r tasks/self-harness/requirements.txt`
    3. Run: `python -m tasks.self_harness.self_harness_loop --rounds 5 --tasks 20 --k 3`
    4. Open the results notebook: `marimo edit notebooks/03-self-harness-results.py`

    **What you'll need:**
    - An LLM (OpenCode API, Ollama, or any OpenAI-compatible endpoint)
    - Python 3.12+
    - About 30-60 minutes per round (varies by model speed)
    - The benchmark tasks auto-generate test environments

    **Resources:**
    - 📄 [Paper: Self-Harness — Self-Improving Language Agents Without Retraining](https://arxiv.org/abs/2606.09498) — Zhang et al., Jun 2026
    - 🗂️ [OpenFrame Project](https://github.com/theyokel/openframe) — local-first, sovereign AI infrastructure
    - 🏆 [molab Notebook Competition #2](https://marimo.io/pages/events/notebook-competition-2) — deadline June 28, 2026
    - 🐍 [marimo](https://github.com/marimo-team/marimo) — the reactive Python notebook

    **The Trilogy:**
    - ⚡ [Notebook 1: BITMAJOR](./bitnet-bonsai.py) — ternary models on consumer hardware
    - 🐍 [Notebook 2: TRM](./tiny-recursive-reasoning.py) — recursive reasoning with tiny networks
    - 🔧 **Notebook 3: Self-Harness** ← you are here — self-improving operational systems

    ---

    *Built with marimo. Inspired by Robert Pirsig's* Zen and the Art of Motorcycle Maintenance *and Masahiro Mori's* The Buddha in the Robot. *Part of the [OpenFrame](https://github.com/theyokel/openframe) project — local-first, sovereign AI infrastructure. All interactive demos use mock/simulated data — the real Self-Harness loop requires an actual LLM to call.*
    """)
    return


if __name__ == "__main__":
    app.run()
