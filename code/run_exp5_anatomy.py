"""Exp 5 — Cross-model anatomy on the extended model set (GPU 1, physical).

Per model, identical protocol to the original exp1/exp2 so numbers are directly
comparable with the two-model results:

1. Embedding anisotropy — Mean|cos| over 2000 random words (seed=42) from the
   raw embedding matrix; raw and centered variants (tvu_common.embedding_anisotropy).
2. PR profile — forward with output_hidden_states on 2048 tokens of the same
   diverse real-doc corpus, tokenized by each model's own tokenizer; exact SVD
   per hidden-state level; PR = (Σs²)²/Σs⁴ centered (primary); neck_level is
   the layer index with minimal centered PR.

Per-model dtype comes from cross_common.MODELS: fp32 where it fits in 15.93 GiB,
bf16 otherwise (qwen3-4b). Memory-light path: no_grad + per-level CPU transfer;
a CUDA OOM on any forward marks that model "skipped" without killing the run.

Writes data/data/exp5_metrics.json incrementally (per model) for crash safety.
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


def load_with_dtype(path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(C.DEVICE)
    model.eval()
    return tokenizer, model


def pr_profile(model, input_ids):
    """Centered PR per hidden-state level (embeddings + layers), computed on CPU."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    prc = []
    for h in out.hidden_states:
        X = h.detach().squeeze(0).float().cpu().numpy()
        sig = np.linalg.svd((X - X.mean(axis=0)).astype(np.float32), compute_uv=False)
        prc.append(C.participation_ratio(sig))
        del X
    del out
    torch.cuda.empty_cache()
    return np.array(prc)


def main():
    # optional CLI filter: python run_exp5_anatomy.py qwen3-4b[,qwen2.5-0.5b,...]
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    work_json = os.path.join(C.WORK, "exp5_metrics.json")
    results = {}
    if os.path.exists(work_json):   # merge: a later per-model run must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        print(f"=== Exp5 {key}  dtype={str(dtype)}  {path}")
        tok = mod = None
        try:
            tok, mod = load_with_dtype(path, dtype)
            ids = C.make_diverse_tokens(tok, n_target=C.N_TOKENS)

            aniso_raw, aniso_centered = C.embedding_anisotropy(mod)
            print(f"  anisotropy raw={aniso_raw:.4f}  centered={aniso_centered:.4f}")

            prc = pr_profile(mod, ids)
            neck_level = int(np.argmin(prc[1:]) + 1)
            results[key] = {
                "path": path,
                "dtype": str(dtype),
                "aniso_raw": round(float(aniso_raw), 4),
                "aniso_centered": round(float(aniso_centered), 4),
                "pr_centered_per_level": [round(float(x), 2) for x in prc],
                "neck_level": neck_level,
                "neck_pr_centered": round(float(prc[neck_level]), 2),
            }
            print(f"  NECK: level {neck_level}, PR_centered={results[key]['neck_pr_centered']}")
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

    print("\n--- Exp5 summary ---")
    for key, r in results.items():
        if "status" in r:
            print(f"{key:<16} SKIPPED: {r['status']}")
        else:
            print(f"{key:<16} aniso_raw={r['aniso_raw']:.4f}  neck_l={r['neck_level']:>2} "
                  f"neck_PR={r['neck_pr_centered']:>8.1f}")


if __name__ == "__main__":
    main()
