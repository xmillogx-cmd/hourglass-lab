"""Exp 10c stage A — GPU activation dump for the structural-geometry analysis.

Purpose: Exp10 measured lossy scalar summaries of per-level activation geometry
(PRc / ani / k95) and found them MODEL_SPECIFIC. That does not answer whether the
hidden-space STRUCTURE (token-token relations up to rotation/scale) is shared —
the "same structure from different angles" question. Stage B needs raw activations.

This script re-runs the identical Exp10 control group (wikitext-2 test split,
first 2048 tokens per model) and dumps, per model:
    data/data/exp10c_act_<key>.npz      — arrays h_0..h_L, each [2048, H] float16
                                             (level 0 = embedding; L+1 levels total)
    data/data/exp10c_tokens_<key>.json  — the 2048 token surface forms
    data/data/exp10c_dump_meta.json     — incremental per-model meta

Same loading policy as Exp10: dtype per cross_common.MODELS, VRAM gate before each
load (gemma >= 12.5 GiB, others >= 8.0), no_grad forward with output_hidden_states,
per-level CPU transfer, caller-ref reset + gc + empty_cache after free_model
(the verified Exp10 VRAM-release fix).

Run: <python> [code\]run_exp10c_activation_dump.py [model,filter,...]
(или code\run_exp10c_activation_dump.bat)
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

N_TOKENS = 2048                 # same control length as Exp10 / Exp5
RAW_TEXT_PATH = os.path.join(os.path.dirname(_ROOT), ".wikitext2_raw_text.txt")


def model_layer_count(path):
    """Decoder layer count from config.json; nested text_config fallback (gemma-style)."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    def g(key):
        return cfg.get(key) if cfg.get(key) is not None else cfg.get("text_config", {}).get(key)

    L = g("num_hidden_layers")
    H = g("hidden_size")
    tied = bool(g("tie_word_embeddings"))
    if L is None:
        raise ValueError(f"no num_hidden_layers in {path}/config.json (top-level or text_config)")
    return int(L), (int(H) if H is not None else None), tied


def free_gib_on_gpu1():
    try:
        free, _ = torch.cuda.mem_get_info(1)
        return float(free) / 2**30
    except Exception as e:
        print(f"VRAM probe failed ({e}) — assuming no gate")
        return 99.0


def dump_for_model(model, tok, input_ids):
    """Per-level fp16 activation arrays [n,H] + token surface forms."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    levels = []
    for h in out.hidden_states:
        X = h.detach().squeeze(0).float().cpu().numpy()
        levels.append(np.ascontiguousarray(X.astype(np.float16)))
    del out
    torch.cuda.empty_cache()
    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
    return levels, tokens


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

    if not os.path.isfile(RAW_TEXT_PATH):
        raise SystemExit(f"canonical corpus file missing: {RAW_TEXT_PATH}")
    with open(RAW_TEXT_PATH, encoding="utf-8") as fh:
        raw_text = fh.read()
    print(f"control corpus: {RAW_TEXT_PATH}  ({len(raw_text)} chars)")

    meta_path = os.path.join(C.WORK, "exp10c_dump_meta.json")
    state = {"meta": {
        "corpus": "wikitext-2 test split (quantization control text), first 2048 tokens per model",
        "raw_text_path": RAW_TEXT_PATH,
        "n_tokens": N_TOKENS,
        "note": "stage A of Exp10c — raw per-level activation dump for stage B structural analysis",
    }, "models": {}}
    if os.path.exists(meta_path):   # merge: a later per-model run must not wipe earlier ones
        with open(meta_path, encoding="utf-8") as fh:
            state.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        tok = mod = None
        try:
            L, H, tied = model_layer_count(path)
            required_gib = 12.5 if key == "gemma-4-e2b" else 8.0
            free = free_gib_on_gpu1()
            print(f"=== Exp10c {key}  dtype={str(dtype)}  L={L} H={H} tied={tied}")
            print(f"    GPU1 free VRAM: {free:.2f} GiB (need >= {required_gib})")
            if free < required_gib:
                raise MemoryError(
                    f"insufficient free VRAM on GPU 1 ({free:.2f} < {required_gib} GiB); "
                    "close other GPU processes (LM Studio tray!) and re-run this model only")

            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path)
            mod = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(C.DEVICE)
            mod.eval()

            enc = tok(raw_text, add_special_tokens=False)
            ids = torch.tensor([enc["input_ids"][:N_TOKENS]], dtype=torch.long).to(C.DEVICE)

            levels, tokens = dump_for_model(mod, tok, ids)
            n_levels = len(levels)
            if n_levels - 1 != L:
                print(f"    WARNING: hidden_states levels {n_levels} != expected L+1={L + 1}")

            npz_path = os.path.join(C.WORK, f"exp10c_act_{key}.npz")
            np.savez(npz_path, **{f"h_{i}": lv for i, lv in enumerate(levels)})
            tok_json = os.path.join(C.WORK, f"exp10c_tokens_{key}.json")
            with open(tok_json, "w", encoding="utf-8") as fh:
                json.dump(tokens, fh, ensure_ascii=False)

            state["models"][key] = {
                "path": path,
                "dtype": str(dtype),
                "layers_L": L,
                "hidden_H": H,
                "tied_embeddings": tied,
                "n_levels_seen": n_levels,
                "npz_file": f"exp10c_act_{key}.npz",
                "tokens_json": f"exp10c_tokens_{key}.json",
                "size_mb": round(os.path.getsize(npz_path) / 2**20, 1),
            }
            print(f"    dumped {n_levels} levels -> {os.path.basename(npz_path)} "
                  f"({state['models'][key]['size_mb']} MB)")
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            state["models"][key] = {"path": path, "dtype": str(dtype),
                                    "status": f"skipped ({type(e).__name__}: {e})"}
            print(f"    SKIPPED: {e}")
        finally:
            if tok is not None and mod is not None:
                C.free_model(tok, mod)   # del only drops free_model's locals — the caller
            tok = mod = None             # still holds references until this reset (VRAM bug)
            gc.collect()
            torch.cuda.empty_cache()     # release cached blocks to OS before the next gate probe

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        print(f"    (incremental save -> {meta_path})")

    ok = [k for k, r in state["models"].items() if "status" not in r]
    bad = {k: r["status"] for k, r in state["models"].items() if "status" in r}
    print(f"\n--- Exp10c dump summary ---\ndumped {len(ok)}/8 models; skipped: {bad or 'none'}")


if __name__ == "__main__":
    main()
