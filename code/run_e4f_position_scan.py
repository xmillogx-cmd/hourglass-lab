"""E4f follow-up — fine-grained position scan for the lost-in-the-middle question.

Refines E4e: does recall degrade at MID positions of long context (classic
lost-in-the-middle U-shape)? Grid L in {2048, 4096, 8192} x p in
{0.05, 0.15, ..., 0.95} (step 0.1) with 3 independent reps per cell:
different code rotation + different filler phase into the corpus pool.

Same probe and contrast metrics as E4e (dlogit target vs alt same-shape code;
control without needle at same length/filler phase -> lift = pure recall
contribution; p_win; argmax_hit). Aggregates: mean +/- std of lift per
position over 9 samples (3 L x 3 reps) and per length (30 each).

Memory-light as E4e: no_grad, only the last-position logits row is kept on CPU.
Outputs to g3_long_context_h4/data/ (outside the 5-file deliverable set).
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import tvu_common as C  # noqa: E402

TEMPLATE = " The verification code for this record is"
CODES = ["KZVX-9281", "QWRT-3574", "MNPB-6019"]
LENGTHS = [2048, 4096, 8192]
POSITIONS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
REPS = 3
REP_OFFSETS = [0, 150, 300]  # filler phase shifts into the corpus pool


def token_sets(tokenizer):
    probe_ids = tokenizer(TEMPLATE, add_special_tokens=False)["input_ids"]
    needles, targets = {}, {}
    for code in CODES:
        needles[code] = tokenizer(f"{TEMPLATE} {code}.", add_special_tokens=False)["input_ids"]
        targets[code] = set(tokenizer(f" {code}", add_special_tokens=False)["input_ids"])
    return probe_ids, needles, targets


def logsumexp(x):
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def metrics_from_logits(logits, target_set, alt_set):
    Z = logsumexp(logits)
    p_t = float(np.exp(logsumexp(logits[list(target_set)]) - Z))
    p_a = float(np.exp(logsumexp(logits[list(alt_set)]) - Z))
    dlogit = (float(max(logits[i] for i in target_set))
              - float(max(logits[i] for i in alt_set)))
    return {
        "dlogit": round(dlogit, 4),
        "p_win": round(p_t / max(p_t + p_a, 1e-12), 5),
        "argmax_hit": int(np.argmax(logits) in target_set),
    }


def main():
    tok, mod = C.load_model(C.MODEL_A_PATH)

    texts = []
    for path in C.DOC_FILES:
        with open(path, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    pool = np.array(tok(" ".join(texts), add_special_tokens=False)["input_ids"], dtype=np.int64)

    probe_ids, needles, targets = token_sets(tok)
    print(f"probe suffix: {len(probe_ids)} tokens")
    for code in CODES:
        t = sorted(targets[code])
        assert len(t) >= 1 and all(0 <= i < 151936 for i in t), f"bad target set for {code}"
        print(f"  needle '{code}': {len(needles[code])} tokens, target ids = {t}")

    def forward_last_logits(ids):
        with torch.no_grad():
            out = mod(input_ids=torch.tensor([ids], dtype=torch.long).to(C.DEVICE))
            row = out.logits[0, -1].float().cpu().numpy()
        del out
        torch.cuda.empty_cache()
        return row

    cells = {}
    idx = 0
    for L in LENGTHS:
        n_needle_max = max(len(needles[c]) for c in CODES)
        assert REP_OFFSETS[-1] + (L - n_needle_max) <= len(pool), "pool too short for this offset"

        for rep in range(REPS):
            off = REP_OFFSETS[rep]
            control_row = forward_last_logits(list(pool[off: off + L]) + probe_ids)
            ctrl = metrics_from_logits(control_row, set(targets[CODES[0]]), set(targets[CODES[1]]))
            del control_row

            for p in POSITIONS:
                code_t = CODES[idx % 3]
                code_a = CODES[(idx + 1) % 3]
                n_needle = len(needles[code_t])
                filler_len = L - n_needle
                before = int(round(p * filler_len))
                seq = (list(pool[off: off + before]) + needles[code_t]
                       + list(pool[off + before: off + before + (filler_len - before)]))
                row = forward_last_logits(seq + probe_ids)
                m = metrics_from_logits(row, targets[code_t], targets[code_a])
                del row

                cells[f"{L}_p{p}_r{rep}"] = {
                    "length": int(L), "position": p, "rep": rep,
                    "target_code": code_t, "alt_code": code_a,
                    **m,
                    "control_dlogit_generic": ctrl["dlogit"],
                    "lift_vs_control": round(m["dlogit"] - ctrl["dlogit"], 4),
                }
                idx += 1

        print(f"done L={L}: {len(POSITIONS) * REPS} cells")

    C.free_model(tok, mod)

    out_dir = os.path.join(_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    def agg(keys):
        lifts = [cells[k]["lift_vs_control"] for k in keys]
        pwins = [cells[k]["p_win"] for k in keys]
        return {
            "n": len(keys),
            "mean_lift": round(float(np.mean(lifts)), 4),
            "std_lift": round(float(np.std(lifts, ddof=1)), 4),
            "min_lift": round(min(lifts), 4),
            "max_lift": round(max(lifts), 4),
            "mean_p_win": round(float(np.mean(pwins)), 5),
            "argmax_hits": sum(cells[k]["argmax_hit"] for k in keys),
        }

    by_position = {}
    for p in POSITIONS:
        keys = [f"{L}_p{p}_r{rep}" for L in LENGTHS for rep in range(REPS)]
        by_position[str(p)] = agg(keys)
    by_length = {}
    for L in LENGTHS:
        keys = [f"{L}_p{p}_r{rep}" for p in POSITIONS for rep in range(REPS)]
        by_length[str(L)] = agg(keys)

    edge_keys = [k for k in cells if f"_p0.05_" in k or f"_p0.95_" in k]
    mid_keys = [k for k in cells if any(f"_p{q}_r" in k for q in (0.25, 0.35, 0.45, 0.55, 0.65))]
    u_contrast = round(float(np.mean([cells[k]["lift_vs_control"] for k in edge_keys]))
                       - float(np.mean([cells[k]["lift_vs_control"] for k in mid_keys])), 4)

    payload = {
        "task": ("E4f follow-up — fine-grained needle-in-haystack position scan: "
                 "lost-in-the-middle test (U-shape of recall vs needle position), "
                 "3 independent reps per cell"),
        "model": {"name": C.NAME_A, "path": C.MODEL_A_PATH,
                  "note": "tied Qwen2.5-0.5B base, fp32, physical GPU 1"},
        "probe_template": TEMPLATE,
        "codes": CODES,
        "lengths_run": LENGTHS,
        "positions": POSITIONS,
        "reps": REPS,
        "rep_offsets": REP_OFFSETS,
        "cells": cells,
        "summary_by_position": by_position,
        "summary_by_length": by_length,
        "u_shape_contrast_edges_minus_middle": u_contrast,
    }
    out_json = os.path.join(out_dir, "e4f_metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"metrics JSON written: {out_json}")

    print("\n--- summary by position (9 samples each: 3 L x 3 reps) ---")
    for p in POSITIONS:
        s = by_position[str(p)]
        print(f"p={p:<5} lift={s['mean_lift']:+.3f} +- {s['std_lift']} "
              f"[{s['min_lift']} .. {s['max_lift']}]  argmax {s['argmax_hits']}/{s['n']}")
    print("\n--- summary by length (30 samples each) ---")
    for L in LENGTHS:
        s = by_length[str(L)]
        print(f"L={L:<6} lift={s['mean_lift']:+.3f} +- {s['std_lift']} "
              f"[{s['min_lift']} .. {s['max_lift']}]  argmax {s['argmax_hits']}/30")
    print(f"\nU-shape contrast (edges p=0.05/0.95 minus middle p=0.25..0.65): {u_contrast:+.4f}")

    # ---- plot: 1x2 — mean lift per position with error bars; per-length lines
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    xs = np.array(POSITIONS)
    means = [by_position[str(p)]["mean_lift"] for p in POSITIONS]
    stds = [by_position[str(p)]["std_lift"] for p in POSITIONS]
    axes[0].errorbar(xs, means, yerr=stds, marker="o", linewidth=1.8, capsize=3)
    axes[0].axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Needle position p (fraction of context)")
    axes[0].set_ylabel("Δlogit lift over no-needle control")
    axes[0].set_title("Recall vs needle position — lost-in-the-middle test\n"
                      "(mean ± std over 3 lengths x 3 reps)", fontsize=10.5)
    axes[0].grid(True, alpha=0.3)

    cmap = plt.get_cmap("coolwarm")
    for j, L in enumerate(LENGTHS):
        per_p = []
        for p in POSITIONS:
            lifts = [cells[f"{L}_p{p}_r{rep}"]["lift_vs_control"] for rep in range(REPS)]
            per_p.append(float(np.mean(lifts)))
        axes[1].plot(xs, per_p, marker="o", linewidth=1.8,
                     color=cmap(j / (len(LENGTHS) - 1)), label=f"L={L}")
    axes[1].axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Needle position p (fraction of context)")
    axes[1].set_ylabel("Δlogit lift over no-needle control")
    axes[1].set_title("Recall vs needle position per context length\n"
                      "(mean over 3 reps)", fontsize=10.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(f"E4f — fine-grained position scan ({C.NAME_A}, GPU 1, fp32)",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "e4f_position_scan.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
