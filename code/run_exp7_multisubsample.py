"""Exp 7 (multi-rep) — statistical power of the gradient-interference geometry.

Reviewer p.1 item: "Exp7 -> 3-5 random corpus subsamples, mean +/- std of cos/ratio".

The ORIGINAL run_exp7_gradient_interference.py ran ONE deterministic window (the
first N_GRAD_TOKENS tokens of the joined corpus) per model and reported a single
cosine / norm-ratio. That is n=1: no uncertainty band on the H1 interference
numbers. This variant reuses that protocol verbatim (imports run_tied/run_untied/
tie_flags/load_with_dtype from run_exp7_gradient_interference so the math cannot
drift) but draws N_REPS distinct RANDOM contiguous windows of the corpus per model
and reports mean +/- std across reps for the head cosine and the norm ratio.

Why contiguous windows rather than scattered token resampling: an autoregressive
CE loss needs coherent local context; a random 128-token window at a random offset
is a genuine "random subsample of the corpus" that still yields meaningful
gradients, while sampling scattered indices would feed incoherent input and give
meaningless cosines. Offsets are deduplicated so reps never repeat a window.

Writes data/data/exp7_multisubsample_metrics.json incrementally (merge pattern + `only`
CLI filter like exp5/exp6). GPU 1, per-model dtype from cross_common.MODELS.

Launch: python run_exp7_multisubsample.py [model-key,...] [n_reps]
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

import numpy as np    # noqa: E402
import torch          # noqa: E402

from cross_common import MODELS   # noqa: E402
import tvu_common as C            # noqa: E402
import run_exp7_gradient_interference as E7  # reuse the exact protocol + helpers


def full_corpus_ids(tokenizer):
    """Tokenize the joined diverse-doc corpus ONCE and return the FULL id list
    (not truncated) so random windows can be drawn from it per rep."""
    texts = []
    for path in C.DOC_FILES:
        with open(path, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    enc = tokenizer(" ".join(texts), add_special_tokens=False)
    return list(enc["input_ids"])


def random_window(full_ids, n_target, rng, used_offsets):
    """Pick a distinct contiguous window of length n_target at a fresh random offset."""
    lo, hi = 0, len(full_ids) - n_target
    if hi <= lo:
        raise ValueError(f"corpus ({len(full_ids)}) shorter than window ({n_target})")
    for _ in range(64):
        s = int(rng.integers(lo, hi + 1))
        if s not in used_offsets:
            used_offsets.add(s)
            return s
    raise RuntimeError("could not find a fresh offset after 64 tries (reps too many?)")


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base_seed = 20_261_809   # distinct from any existing experiment seed

    tie = E7.tie_flags()

    work_json = os.path.join(C.WORK, "exp7_multisubsample_metrics.json")
    results = {}
    if os.path.exists(work_json):   # merge: a later per-model rerun must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        print(f"=== Exp7-multi {key}  reps={n_reps}  tied={bool(tie.get(key))}")
        tok = mod = None
        try:
            tok, mod = E7.load_with_dtype(path, dtype)
            full_ids = full_corpus_ids(tok)

            cos_vals, ratio_vals, per_rep = [], [], []
            used_offsets = set()
            for k in range(n_reps):
                rng = np.random.default_rng(base_seed + k)
                s = random_window(full_ids, E7.N_GRAD_TOKENS, rng, used_offsets)
                ids = torch.tensor([full_ids[s:s + E7.N_GRAD_TOKENS]], dtype=torch.long).to(C.DEVICE)

                if tie.get(key):
                    r = E7.run_tied(mod, ids)
                    cos_val = r["cos_gin_vs_gout"]
                    ratio_val = r["norm_ratio_gout_over_gin"]
                else:
                    r = E7.run_untied(mod, ids)
                    cos_val = r["cos_gemb_vs_glm"]
                    ratio_val = r["norm_ratio_glm_over_gemb"]

                cos_vals.append(cos_val)
                ratio_vals.append(ratio_val)
                per_rep.append({
                    "rep": k, "offset": int(s),
                    "cos": round(float(cos_val), 4),
                    "ratio": round(float(ratio_val), 4),
                    "cos_gtotal_vs_gin": r.get("cos_gtotal_vs_gin"),
                })

            cos_arr = np.asarray(cos_vals, dtype=np.float64)
            ratio_arr = np.asarray(ratio_vals, dtype=np.float64)
            results[key] = {
                "path": path,
                "dtype": str(dtype),
                "n_reps": n_reps,
                "protocol": ("tied G_in/G_out decomposition (multi-rep)" if tie.get(key)
                             else "untied baseline (multi-rep)"),
                "cos_mean": round(float(cos_arr.mean()), 4),
                "cos_std": round(float(cos_arr.std(ddof=1)) if n_reps > 1 else 0.0, 4),
                "ratio_mean": round(float(ratio_arr.mean()), 4),
                "ratio_std": round(float(ratio_arr.std(ddof=1)) if n_reps > 1 else 0.0, 4),
                "per_rep": per_rep,
            }
            print(f"  cos mean+/-std = {cos_arr.mean():+.4f} +/- {cos_arr.std(ddof=1) if n_reps>1 else 0:.4f}")
            print(f"  ratio mean+/-std = {ratio_arr.mean():.3f} +/- {ratio_arr.std(ddof=1) if n_reps>1 else 0:.3f}")
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            results[key] = {"path": path, "dtype": str(dtype),
                            "status": f"skipped ({type(e).__name__}: {e})"}
            print(f"  SKIPPED: {e}")
        finally:
            if tok is not None and mod is not None:
                C.free_model(tok, mod)

        with open(work_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"  (incremental save -> {work_json})")

    print("\n--- Exp7 multi-rep summary ---")
    for key, r in results.items():
        if "status" in r:
            print(f"{key:<16} SKIPPED: {r['status']}")
        else:
            print(f"{key:<16} cos={r['cos_mean']:+.4f}+/-{r['cos_std']:.4f}  "
                  f"ratio={r['ratio_mean']:.3f}+/-{r['ratio_std']:.3f} (n={r['n_reps']})")


if __name__ == "__main__":
    main()
