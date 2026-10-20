"""Step 3 — Merge Exp 5 / Exp 6 / Exp 7 results into the cross-model deliverables.

Reads:
    data/models_metadata.json   (step 0)
    data/data/exp5_metrics.json        (Exp 5 — per model anisotropy + PR profile;
                                    = Exp-2 hourglass data for all six models)
    data/data/exp6_metrics.json        (Exp 6 — LiM scan, Inversion Contrast)
    data/data/exp7_metrics.json        (Exp 7 — gradient interference G_in/G_out)

Writes into aiir-hourglass/data/:
    cross_model_results.json   — full merged payload (metadata + exp5 + exp6 + exp7 + notes)
    cross_model_summary.md     — Markdown tables: Exp5/6 anatomy table,
                                 Exp-2 PR-profile note, Exp-7 interference table
    05_anisotropy_vs_size.png  — scatter, X = log10(Params), Y = anisotropy raw;
                                 red = tied, green = untied
    06_inversion_contrast_barplot.png — bar per model of Inversion Contrast
    07_gradient_interference.png — per-model head cosine (cos(G_in,G_out) / cos(G_emb,G_lm))
                                   + norm-ratio panel; red = tied, green = untied

(08_pr_profiles_all_models.png is produced by plot_pr_profiles.py from exp5 data.)

CPU-only (no GPU). Models marked "skipped" in the work JSONs get '—' cells.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_common import RESULTS, TVU  # noqa: E402

WORK = os.path.join(TVU, "data")

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def params_of(meta_entry):
    """Float param count for plotting (exact from safetensors; model-card total
    for quantized checkpoints)."""
    p = meta_entry.get("params", {})
    if "exact" in p and p["exact"]:
        return float(p["exact"])
    if "total_params_with_embeddings_B" in p:
        return float(p["total_params_with_embeddings_B"]) * 1e9
    return None


def main():
    meta = load_json(os.path.join(RESULTS, "models_metadata.json")) or {"models": {}}
    exp5 = load_json(os.path.join(WORK, "exp5_metrics.json")) or {}
    exp6 = load_json(os.path.join(WORK, "exp6_metrics.json")) or {}
    exp7 = load_json(os.path.join(WORK, "exp7_metrics.json")) or {}

    models = meta["models"]
    rows = []   # (key, params_B, tied, layers, aniso_raw, neck_level, neck_pr, ic)
    for key, m in models.items():
        e5 = exp5.get(key, {})
        e6 = exp6.get(key, {})
        e7 = exp7.get(key, {})
        tied7 = "status" not in e7 and str(e7.get("protocol", "")).startswith("tied")
        pB = None
        pv = params_of(m)
        if pv:
            pB = round(pv / 1e9, 3)
        rows.append({
            "key": key,
            "params_B": pB,
            "tied": bool(m.get("tie_word_embeddings")),
            "layers": m.get("num_hidden_layers"),
            "dtype": e5.get("dtype", e6.get("dtype")),
            "aniso_raw": e5.get("aniso_raw") if "status" not in e5 else None,
            "aniso_centered": e5.get("aniso_centered") if "status" not in e5 else None,
            "neck_level": e5.get("neck_level") if "status" not in e5 else None,
            "neck_pr": e5.get("neck_pr_centered") if "status" not in e5 else None,
            "inversion_contrast": (e6.get("inversion_contrast")
                                   if "status" not in e6 else None),
            "exp7_head_cos": ((e7.get("cos_gin_vs_gout") if tied7 else e7.get("cos_gemb_vs_glm"))
                              if "status" not in e7 else None),
            "exp7_ratio": ((e7.get("norm_ratio_gout_over_gin") if tied7 else e7.get("norm_ratio_glm_over_gemb"))
                           if "status" not in e7 else None),
            "exp7_proxy": (e7.get("cos_embupdate_vs_qkv0_effect_proxy")
                           if "status" not in e7 else None),
            "exp5_status": e5.get("status"),
            "exp6_status": e6.get("status"),
            "exp7_status": e7.get("status"),
        })
    rows.sort(key=lambda r: (r["params_B"] is None, r["params_B"] or 0))

    # ---- cross_model_results.json
    payload = {
        "task": ("Cross-model extension (Exp 5 anatomy + Exp 6 LiM scan + "
                 "Exp 7 gradient interference) of the aiir-hourglass empirical "
                 "validation; 6 local models, physical GPU 1"),
        "date": "2026-09-16",
        "metadata": {k: v for k, v in models.items()},
        "exp5_anatomy": exp5,
        "exp6_lim_scan": exp6,
        "exp7_gradient_interference": exp7,
        "notes": [
            "Per-model dtype: fp32 where the fp32 peak fits 15.93 GiB VRAM on GPU 1; "
            "bf16 otherwise (qwen2.5-3b — weights + full hidden-state capture exceed "
            "VRAM in fp32; qwen3-4b and gemma — fp32 weights do not fit).",
            "Exp6 lengths per model are capped by max_position_embeddings and VRAM: "
            "8k skipped for qwen2.5-3b / qwen3-4b (fp32 peak exceeds VRAM; user spec "
            "'если не влезает — пропусти').",
            "Exp6 protocol = E4f: positions [0.05, 0.25, 0.45, 0.75, 0.95], 2 reps "
            "(filler offsets [0, 150]); Inversion Contrast = Mean Lift(middle) - "
            "Mean Lift(edges); > 0 inverted-U (middle better), < 0 classic U / LiM.",
            "Exp7 protocol = the ORIGINAL exp1 decomposition (G_total / G_in with a "
            "frozen lm_head clone / G_out from no-grad hidden states + fresh W'), "
            "uniform for all six models: even the gemma composite wrapper's built-in "
            "CE loss is differentiable (probe v2), so out.loss.backward() is used "
            "everywhere; only module resolution differs (gemma text stack = "
            "model.language_model, vision tower excluded). N_GRAD_TOKENS=128.",
            "Original control-pair reference: Qwen2.5-0.5B (tied) cos(G_in,G_out)="
            "+0.018, ||G_out||/||G_in||=3.754; TinyLlama-1.1B (untied) "
            "cos(G_emb,G_lm)=+0.001, ratio 0.998.",
        ],
    }
    out_json = os.path.join(RESULTS, "cross_model_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"written: {out_json}")

    # ---- cross_model_summary.md
    def fmt(v, spec="{:.4f}", dash="—"):
        if v is None:
            return dash
        s = spec.strip()
        if len(s) >= 2 and s.startswith("{") and s.endswith("}"):   # format() wants a bare mini-spec
            body = s[1:-1]
            if body.startswith(":"):
                body = body[1:]
            s = body or ""
        return format(v, s)

    lines = [
        "# Cross-model summary (Exp 5 + Exp 6 + Exp 7), 2026-09-16",
        "",
        "| Model | Params (B) | Tied? | Layers | Aniso (raw) | Neck Level | Neck PR | Inversion Contrast |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['key']} | {fmt(r['params_B'], '{:.3f}')} "
            f"| {'yes' if r['tied'] else 'no'} | {r['layers']} "
            f"| {fmt(r['aniso_raw'])} | {fmt(r['neck_level'], '{}')} "
            f"| {fmt(r['neck_pr'], '{:.1f}')} | {fmt(r['inversion_contrast'])} |")

    lines += [
        "",
        "## Exp 2 — Hourglass PR profile (all six models)",
        "",
        "_The hourglass test is covered for all six models by the Step-1 / Exp-5 data "
        "above: identical protocol to the original exp2 (per-layer centered "
        "participation ratio + neck level = minimal centered PR). Plot: "
        "`08_pr_profiles_all_models.png`._",
        "",
        "## Exp 7 — Gradient interference on the shared matrix (all six models)",
        "",
        "| Model | Tied? | cos(G_in,G_out) / cos(G_emb,G_lm) | Norm ratio (out/emb) | Proxy: emb-update vs qkv0-effect |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['key']} | {'yes' if r['tied'] else 'no'} "
            f"| {fmt(r['exp7_head_cos'])} | {fmt(r['exp7_ratio'], '{:.3f}')} "
            f"| {fmt(r['exp7_proxy'])} |")

    notes = []
    for r in rows:
        sts = [s for s in (r["exp5_status"], r["exp6_status"], r.get("exp7_status")) if s]
        if sts:
            notes.append(f"- **{r['key']}**: " + " / ".join(sts))   # status strings already carry "- " context
        elif r["dtype"]:
            notes.append(f"- {r['key']}: dtype={r['dtype']}")
    lines += ["", "## Notes", ""] + list(notes) + [   # note strings already carry their "- " prefix
        "",
        "_Inversion Contrast > 0 — inverted-U (middle better); < 0 — classic U / "
        "Lost-in-the-Middle. Neck = layer with minimal centered PR._",
        "",
        "_Exp 7: G_in = input-path-only gradient of the shared W (lm_head frozen), "
        "G_out = output-path-only gradient (no-grad hidden states + fresh W'); for "
        "the untied model, G_emb / G_lm are the separate matrices. Proxy = cosine "
        "between layer-0 q/k/v perturbations from the embedding update vs from the "
        "layer-0 weight updates._",
    ]
    out_md = os.path.join(TVU, "docs", "cross_model_summary.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"written: {out_md}")

    # ---- 05_anisotropy_vs_size.png (scatter, red=tied / green=untied)
    pts = [r for r in rows if r["params_B"] and r["aniso_raw"] is not None]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    tied = [p for p in pts if p["tied"]]
    untied = [p for p in pts if not p["tied"]]
    ax.scatter([np.log10(p["params_B"]) for p in tied],
               [p["aniso_raw"] for p in tied], color="#d62728", s=90,
               label="Tied (shared embedding/lm_head)", zorder=3)
    ax.scatter([np.log10(p["params_B"]) for p in untied],
               [p["aniso_raw"] for p in untied], color="#2ca02c", s=90,
               label="Untied (separate lm_head)", zorder=3)
    for p in pts:
        ax.annotate(p["key"], xy=(np.log10(p["params_B"]), p["aniso_raw"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8.5)
    ax.set_xlabel("log10(Parameters)")
    ax.set_ylabel("Embedding anisotropy Mean|cos| (2000 random words, raw)")
    ax.set_title("Exp 5 — Embedding anisotropy vs model size\n"
                 "(red = tied embeddings/lm-head, green = untied)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "05_anisotropy_vs_size.png"), dpi=200)
    plt.close(fig)

    # ---- 06_inversion_contrast_barplot.png
    ic_rows = [r for r in rows if r["inversion_contrast"] is not None]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    xs = np.arange(len(ic_rows))
    colors = ["#d62728" if r["tied"] else "#2ca02c" for r in ic_rows]
    ax.bar(xs, [r["inversion_contrast"] for r in ic_rows], color=colors, width=0.6)
    for i, r in enumerate(ic_rows):
        v = r["inversion_contrast"]
        ax.annotate(f"{v:+.3f}", xy=(i, v), textcoords="offset points",
                    xytext=(0, 8 if v >= 0 else -14), ha="center", fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["key"] for r in ic_rows], rotation=25, ha="right")
    ax.set_ylabel("Inversion Contrast (Mean Lift middle − Mean Lift edges)")
    ax.set_title("Exp 6 — Inverted-U vs Lost-in-the-Middle across models\n"
                 "(>0: inverted-U / middle better; <0: classic U; red = tied, green = untied)",
                 fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS, "06_inversion_contrast_barplot.png"), dpi=200)
    plt.close(fig)

    # ---- 07_gradient_interference.png (Exp 7: head cosine + norm ratio per model)
    g_rows = [r for r in rows if r["exp7_head_cos"] is not None]
    if g_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
        xs = np.arange(len(g_rows))
        colors = ["#d62728" if r["tied"] else "#2ca02c" for r in g_rows]

        ax = axes[0]
        ax.bar(xs, [r["exp7_head_cos"] for r in g_rows], color=colors, width=0.6)
        for i, r in enumerate(g_rows):
            v = r["exp7_head_cos"]
            ax.annotate(f"{v:+.3f}", xy=(i, v), textcoords="offset points",
                        xytext=(0, 8 if v >= 0 else -14), ha="center", fontsize=9)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["key"] for r in g_rows], rotation=25, ha="right")
        ax.set_ylabel("cos(G_in,G_out)  /  cos(G_emb,G_lm)")
        ax.set_title("Exp 7 — head cosine of the interference geometry\n"
                     "(~0: paths orthogonal; red = tied, green = untied)", fontsize=10.5)
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1]
        ratios = [r["exp7_ratio"] for r in g_rows if r["exp7_ratio"] is not None]
        ok = [i for i, r in enumerate(g_rows) if r["exp7_ratio"] is not None]
        ax.bar(ok, ratios, color=[colors[i] for i in ok], width=0.6)
        for pos, i in zip(ok, range(len(ok))):
            v = g_rows[pos]["exp7_ratio"]
            ax.annotate(f"{v:.2f}", xy=(pos, v), textcoords="offset points",
                        xytext=(0, 8 if v >= 0 else -14), ha="center", fontsize=9)
        ax.set_xticks(ok)
        ax.set_xticklabels([g_rows[pos]["key"] for pos in ok], rotation=25, ha="right")
        ax.set_ylabel("||G_out|| / ||G_in||   (untied: ||G_lm|| / ||G_emb||)")
        ax.set_title("Exp 7 — norm ratio of the two gradient paths\n"
                     "(>1: output path dominates; red = tied, green = untied)", fontsize=10.5)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS, "07_gradient_interference.png"), dpi=200)
        plt.close(fig)

    print("\n--- summary table ---")
    for r in rows:
        print(f"{r['key']:<28} {fmt(r['params_B'], '{:.3f}', '—'):>7}B  "
              f"tied={str(r['tied']):<6} aniso_raw={fmt(r['aniso_raw'])}  "
              f"neck_l={r['neck_level']}  IC={fmt(r['inversion_contrast'])}  "
              f"Gcos={fmt(r['exp7_head_cos'], '{:+.4f}', '—')}  "
              f"Gratio={fmt(r['exp7_ratio'], '{:.3f}', '—')}")


if __name__ == "__main__":
    main()
