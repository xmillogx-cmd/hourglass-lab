"""Exp 6 — Cross-model Lost-in-the-Middle scan on the extended model set (GPU 1).

Protocol = E4f (g3_long_context_h4/run_e4f_position_scan.py) generalized to all models,
keeping its exact metric definitions so Qwen2.5-0.5B numbers stay comparable:
  - probe suffix = tokens of TEMPLATE; needle text f"{TEMPLATE} {code}." replaces
    part of the filler (total context length is exactly L); targets[code] = token
    ids of " {code}"; dlogit = max(logits[target]) - max(logits[alt]);
    control = same filler + probe WITHOUT any needle, metric on CODES[0] vs CODES[1];
  - code rotation per cell: code_t = CODES[idx % 3], code_a = CODES[(idx+1) % 3].

Grid (user spec): positions p in [0.05, 0.25, 0.45, 0.75, 0.95]; 2 reps with
filler offsets [0, 150]; per-model lengths from cross_common.MODELS (capped by
max_position_embeddings and VRAM; 8k skipped for the large models).

FINAL METRIC per model:
    Inversion Contrast = Mean Lift(middle p in {0.25, 0.45, 0.75})
                         - Mean Lift(edges  p in {0.05, 0.95})
    > 0 -> inverted-U (middle better); < 0 -> classic U / Lost-in-the-Middle.

Memory-light path: no_grad + only the last-position logits row kept on CPU;
an OOM on any forward marks that model "skipped" and the run continues.
Per-model dtype from cross_common.MODELS (fp32 / bf16).

Writes data/data/exp6_metrics.json incrementally (per model) for crash safety.
"""

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # cross_model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import torch

from cross_common import MODELS  # noqa: E402
import tvu_common as C           # noqa: E402

TEMPLATE = " The verification code for this record is"
CODES = ["KZVX-9281", "QWRT-3574", "MNPB-6019"]
POSITIONS = [0.05, 0.25, 0.45, 0.75, 0.95]   # user spec: 5 points (vs E4f's 10)
REPS = 2
REP_OFFSETS = [0, 150]


def load_with_dtype(path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(C.DEVICE)
    model.eval()
    return tokenizer, model


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


def forward_last_logits(model, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long).to(C.DEVICE))
        row = out.logits[0, -1].float().cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return row


def agg(cells):
    lifts = [c["lift_vs_control"] for c in cells if "lift_vs_control" in c]
    pwins = [c["p_win"] for c in cells if "p_win" in c]
    hits = [c.get("argmax_hit", 0) for c in cells if "argmax_hit" in c]
    return {
        "n": len(lifts),
        "mean_lift": round(float(np.mean(lifts)), 4) if lifts else None,
        "std_lift": round(float(np.std(lifts, ddof=1)), 4) if len(lifts) > 1 else None,
        "min_lift": round(min(lifts), 4) if lifts else None,
        "max_lift": round(max(lifts), 4) if lifts else None,
        "mean_p_win": round(float(np.mean(pwins)), 5) if pwins else None,
        "argmax_hits": int(sum(hits)),
    }


def main():
    # optional CLI filter: python run_exp6_lim_scan.py qwen3-4b[,qwen2.5-0.5b,...]
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    work_json = os.path.join(C.WORK, "exp6_metrics.json")
    results = {}
    if os.path.exists(work_json):   # merge: a later per-model run must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        lengths = m["exp6_lengths"]
        print(f"=== Exp6 {key}  dtype={str(dtype)}  L={lengths}")
        tok = mod = None
        try:
            tok, mod = load_with_dtype(path, dtype)

            texts = []
            for p in C.DOC_FILES:
                with open(p, "r", encoding="utf-8") as fh:
                    texts.append(fh.read())
            pool = np.array(tok(" ".join(texts), add_special_tokens=False)["input_ids"],
                            dtype=np.int64)

            probe_ids, needles, targets = token_sets(tok)
            _tc = getattr(mod.config, "text_config", None)   # composite (multimodal) configs keep vocab in text_config
            vocab_size = (_tc.vocab_size if _tc is not None and hasattr(_tc, "vocab_size")
                          else mod.config.vocab_size)
            for code in CODES:
                t = sorted(targets[code])
                assert len(t) >= 1 and all(0 <= i < vocab_size for i in t), \
                    f"bad target set for '{code}'"
                print(f"  needle '{code}': {len(needles[code])} tokens, targets={t}")

            idx = 0
            cells = {}
            controls = []
            for L in lengths:
                n_needle_max = max(len(needles[c]) for c in CODES)
                assert REP_OFFSETS[-1] + (L - n_needle_max) <= len(pool), \
                    f"pool too short ({len(pool)} tokens) for L={L}"

                for rep in range(REPS):
                    off = REP_OFFSETS[rep]
                    control_row = forward_last_logits(mod, list(pool[off: off + L]) + probe_ids)
                    ctrl = metrics_from_logits(control_row, set(targets[CODES[0]]),
                                               set(targets[CODES[1]]))
                    controls.append({"L": int(L), "rep": rep, "dlogit": ctrl["dlogit"]})
                    del control_row

                    for p in POSITIONS:
                        code_t = CODES[idx % 3]
                        code_a = CODES[(idx + 1) % 3]
                        n_needle = len(needles[code_t])
                        filler_len = L - n_needle
                        before = int(round(p * filler_len))
                        seq = (list(pool[off: off + before]) + needles[code_t]
                               + list(pool[off + before: off + before + (filler_len - before)]))
                        row = forward_last_logits(mod, seq + probe_ids)
                        mm = metrics_from_logits(row, targets[code_t], targets[code_a])
                        del row

                        cells[f"{L}_p{p:.2f}_r{rep}"] = {
                            "length": int(L), "position": p, "rep": rep,
                            "target_code": code_t, "alt_code": code_a,
                            **mm,
                            "control_dlogit_generic": ctrl["dlogit"],
                            "lift_vs_control": round(mm["dlogit"] - ctrl["dlogit"], 4),
                        }
                        idx += 1

                print(f"  done L={L}: {len(POSITIONS) * REPS} cells")

            by_position = {}
            for p in POSITIONS:
                ks = [k for k in cells if f"_p{p:.2f}_" in k]
                by_position[f"p_{p:.2f}"] = agg([cells[k] for k in ks])

            mid_ks = [k for k in cells
                      if any(f"_p{q:.2f}_" in k for q in (0.25, 0.45, 0.75))]
            edge_ks = [k for k in cells
                       if any(f"_p{q:.2f}_" in k for q in (0.05, 0.95))]
            inversion_contrast = round(
                float(np.mean([cells[k]["lift_vs_control"] for k in mid_ks])
                      - np.mean([cells[k]["lift_vs_control"] for k in edge_ks])), 4)

            results[key] = {
                "path": path,
                "dtype": str(dtype),
                "lengths_run": lengths,
                "pool_tokens": int(len(pool)),
                "controls": controls,
                "cells": cells,
                "summary_by_position": by_position,
                "inversion_contrast": inversion_contrast,
            }
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            results[key] = {"path": path, "dtype": str(dtype),
                            "status": f"skipped ({type(e).__name__}: {e})"}
            print(f"  SKIPPED: {e}")

        with open(work_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"  (incremental save -> {work_json})")
        if tok is not None and mod is not None:
            C.free_model(tok, mod)

    print("\n--- Exp6 summary ---")
    for key, r in results.items():
        if "status" in r:
            print(f"{key:<16} SKIPPED: {r['status']}")
        else:
            sp = r["summary_by_position"]
            pos_str = "  ".join(f"{k}:{v['mean_lift']:+.3f}" for k, v in sp.items())
            print(f"{key:<16} InversionContrast={r['inversion_contrast']:+.4f}")
            print(f"{'':<16} {pos_str}")


if __name__ == "__main__":
    main()
