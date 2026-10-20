"""E4e follow-up — needle-in-haystack functional test (base-model logit probe).

Question: does the model USE far-away context information as length grows?
(Base Qwen2.5-0.5B cannot do instruction QA, so we use a pattern-completion
probe: a unique code string is planted in the context; at the end the same
template is left unfinished and the next token must be the first piece of the
code — which only exists at the planted position.)

Metrics per cell (L, p):
  dlogit_needle = max logit over target-code tokens - max logit over alt-code
                  tokens (same shape) -> cancels template / code-shape priors
  control       = identical context of length L WITHOUT the needle: same probe
  lift          = dlogit_needle - dlogit_control  <- pure recall contribution
  p_win         = softmax share of target set vs alt set at the last position
  argmax_hit    = top-1 token is in the target set

Grid: L in {100,500,1000,2048,4096,8192} x p in {0.1,0.25,0.5,0.75,0.9},
3 codes rotated over cells. Filler = the same corpus as E4b (prefix pool).
Memory-light: no_grad, only the last-position logits row is kept on CPU.

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
LENGTHS = [100, 500, 1000, 2048, 4096, 8192]
POSITIONS = [0.1, 0.25, 0.5, 0.75, 0.9]


def token_sets(tokenizer):
    """Token id sets: probe suffix (template), per-code needle ids and target
    code-token sets."""
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
    n_probe = len(probe_ids)
    print(f"probe suffix: {n_probe} tokens")
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
        assert L - n_needle_max >= 2 * int(POSITIONS[0] * (L - n_needle_max)), "needle too big for L"
        control_row = forward_last_logits(list(pool[:L]) + probe_ids)
        ctrl = metrics_from_logits(control_row, set(targets[CODES[0]]), set(targets[CODES[1]]))
        del control_row

        for p in POSITIONS:
            code_t = CODES[idx % 3]
            code_a = CODES[(idx + 1) % 3]
            n_needle = len(needles[code_t])
            filler_len = L - n_needle
            before = int(round(p * filler_len))
            seq = list(pool[:before]) + needles[code_t] + list(pool[before: before + (filler_len - before)])
            row = forward_last_logits(seq + probe_ids)
            m = metrics_from_logits(row, targets[code_t], targets[code_a])
            del row

            cells[f"{L}_p{p}"] = {
                "length": int(L), "position": p,
                "target_code": code_t, "alt_code": code_a,
                **m,
                "control_dlogit_generic": ctrl["dlogit"],
                "lift_vs_control": round(m["dlogit"] - ctrl["dlogit"], 4),
            }
            print(f"[L={L} p={p}] {code_t}: dlogit={m['dlogit']} lift="
                  f"{cells[f'{L}_p{p}']['lift_vs_control']} p_win={m['p_win']} "
                  f"argmax={m['argmax_hit']}")
            idx += 1

    C.free_model(tok, mod)

    out_dir = os.path.join(_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    by_L = {}
    for L in LENGTHS:
        lifts = [cells[f"{L}_p{p}"]["lift_vs_control"] for p in POSITIONS]
        by_L[str(L)] = {
            "mean_lift": round(float(np.mean(lifts)), 4),
            "min_lift": round(min(lifts), 4),
            "max_lift": round(max(lifts), 4),
            "argmax_hits": sum(cells[f"{L}_p{p}"]["argmax_hit"] for p in POSITIONS),
        }

    payload = {
        "task": ("E4e follow-up — needle-in-haystack functional test (base-model "
                 "pattern-completion logit probe): does the model USE far context?"),
        "model": {"name": C.NAME_A, "path": C.MODEL_A_PATH,
                  "note": "tied Qwen2.5-0.5B base, fp32, physical GPU 1"},
        "probe_template": TEMPLATE,
        "codes": CODES,
        "lengths_run": LENGTHS,
        "positions": POSITIONS,
        "cells": cells,
        "summary_by_length": by_L,
    }
    out_json = os.path.join(out_dir, "e4e_metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"metrics JSON written: {out_json}")

    # ---- plot: 1x2 — lift vs L per position; dlogit_needle vs L per position
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    xs = np.arange(len(LENGTHS))
    tick_labels = [str(L) for L in LENGTHS]
    cmap = plt.get_cmap("coolwarm")
    for axi, key, title, ylabel in (
        (axes[0], "lift_vs_control", "Lift over no-needle control — pure recall contribution",
         "Δlogit lift"),
        (axes[1], "dlogit", "Raw Δlogit (target vs alt code) at the probe position",
         "Δlogit"),
    ):
        for j, p in enumerate(POSITIONS):
            ys = [cells[f"{L}_p{p}"][key] for L in LENGTHS]
            axi.plot(xs, ys, marker="o", linewidth=1.8,
                     color=cmap(j / (len(POSITIONS) - 1)), label=f"p={p}")
        axi.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axi.set_xticks(xs)
        axi.set_xticklabels(tick_labels, fontsize=9)
        axi.set_xlabel("Context length L (tokens)")
        axi.set_ylabel(ylabel)
        axi.set_title(title, fontsize=10.5)
        axi.grid(True, alpha=0.3)
        axi.legend(fontsize=8)
    fig.suptitle(f"E4e — needle-in-haystack logit probe ({C.NAME_A}, GPU 1, fp32)",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "e4e_needle.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
