"""Step 0 — Cross-model metadata recon for the 6-model extension (Exp 5 / Exp 6).

For each local model dir under cmq_experiment/models/: read config.json and
extract hidden_size, num_hidden_layers, tie_word_embeddings (critical),
vocab_size (+ max_position_embeddings); compute the exact parameter count from
the safetensors headers (no weight loading; a quantized checkpoint cannot be
counted this way — its model-card numbers are reported instead).

CPU-only script: no GPU involved. Writes data/models_metadata.json.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_common import MODELS, RESULTS  # noqa: E402


def exact_params(model_dir):
    """Sum of numels over all tensors in the safetensors shards (no loading)."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    total = 0
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt") as f:
            for k in f.keys():
                total += math.prod(f.get_slice(k).get_shape())
    return int(total)


def read_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tc = cfg.get("text_config") or {}

    def g(key):  # top-level first, nested text_config fallback (gemma-style)
        v = cfg.get(key)
        return tc.get(key) if v is None else v

    meta = {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "hidden_size": g("hidden_size"),
        "num_hidden_layers": g("num_hidden_layers"),
        "tie_word_embeddings": cfg.get(
            "tie_word_embeddings", tc.get("tie_word_embeddings")),
        "vocab_size": g("vocab_size"),
        "max_position_embeddings": g("max_position_embeddings"),
        "transformers_version_saved_with": cfg.get("transformers_version"),
    }
    return meta, bool(cfg.get("quantization_config"))


def main():
    payload = {
        "task": ("Step 0 — metadata recon for the cross-model extension: "
                 "hidden_size / num_hidden_layers / tie_word_embeddings / vocab_size "
                 "+ max_position_embeddings + exact parameter count per model"),
        "date": "2026-09-16",
        "models": {},
    }

    dirs = [(m["key"], m["path"]) for m in MODELS]

    for key, path in dirs:
        meta, quantized = read_config(path)
        if quantized:
            # packed 2/4-bit tensors — logical param count not derivable from the file;
            # per the HF model card of google/gemma-4-E2B-it (QAT mobile wNa8o8):
            meta["params"] = {
                "note": ("QAT mobile-optimized wNa8o8 checkpoint (quantized 2/4-bit); "
                         "exact logical param count not derivable from packed safetensors; "
                         "numbers per HF model card"),
                "effective_params_B": 2.3,
                "total_params_with_embeddings_B": 5.1,
            }
        else:
            meta["params"] = {"exact": exact_params(path)}
        payload["models"][key] = {**meta, "path": path}

    os.makedirs(RESULTS, exist_ok=True)
    out_json = os.path.join(RESULTS, "models_metadata.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"metadata JSON written: {out_json}")

    print("\n--- metadata table ---")
    for key, v in payload["models"].items():
        p = v["params"]
        if "exact" in p:
            ps = f"{p['exact']/1e9:.3f}B (exact from safetensors)"
        else:
            ps = (f"{p.get('effective_params_B')}B effective / "
                  f"{p.get('total_params_with_embeddings_B')}B total w/ embeddings (model card)")
        print(f"{key:<28} tied={str(v['tie_word_embeddings']):<6} hidden={v['hidden_size']:<6} "
              f"layers={v['num_hidden_layers']:<4} vocab={v['vocab_size']:<7} "
              f"max_pos={v['max_position_embeddings']:<7} {ps}")


if __name__ == "__main__":
    main()
