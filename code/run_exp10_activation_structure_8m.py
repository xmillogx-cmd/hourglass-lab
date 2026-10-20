"""Exp 10 — cross-model activation-structure comparison on ONE common control group.

Question: do the structural signals (per-level activation geometry) differ strongly
between the 8 models, or is there a common pattern?

Control group = the SAME wikitext-2 test-split raw text used by the quantization
research (host-tree file .wikitext2_raw_text.txt). Every model tokenizes that one
text with its own tokenizer and takes the first 2048 tokens — identical input.

Per hidden-state level (decoder layers l = 1..L), on the token-activation matrix
X [N, H] at that level:
    prc[l]  — centered participation ratio of X (SVD spectrum; tvu_common convention)
    ani[l]  — mean |cos| between RAW token vectors at that level (representation cone)
    k95[l]  — effective rank: smallest k with >= 0.95 cumulative explained variance

Cross-model comparison on depth-normalized curves x = l/L, resampled to a common
64-point grid:
    pairwise Pearson r on prc curves (28 pairs),
    cosine distance on ani curves,
    neck-position spread (depth fraction where centered PR is minimal).

VERDICT CRITERIA — pre-declared in the plan board v0 before this run was launched
(literal criteria; judged exactly as written):
    COMMON_PATTERNS   if  mean pairwise r >= 0.75  AND  min pairwise r >= 0.4
                       AND  std(neck fractions) <= 0.12
    MODEL_SPECIFIC    otherwise

Memory: dtype per cross_common.MODELS policy (validated by Exp5 at N=2048 with full
hidden-state capture); VRAM gate before each load; no_grad + per-level CPU transfer;
a CUDA OOM marks the model "skipped" without killing the run.

Writes data/data/exp10_activation_structure_8m.json incrementally (per model) and
exp10_prc_curves.png / exp10_aniso_curves.png / exp10_k95_curves.png into data/.

Run: <python> [code\]run_exp10_activation_structure_8m.py [model,filter,...]
(или code\run_exp10_activation_structure_8m.bat)
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

N_TOKENS = 2048                 # same control length as Exp5 / tvu_common.N_TOKENS
GRID_N = 64                     # common depth-fraction grid for cross-model curves
RAW_TEXT_PATH = os.path.join(os.path.dirname(_ROOT), ".wikitext2_raw_text.txt")

# pre-declared verdict thresholds (see module docstring)
CRIT_MEAN_R, CRIT_MIN_R, CRIT_NECK_STD = 0.75, 0.4, 0.12


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


def metrics_for_model(model, input_ids):
    """Per-level prc/ani/k95 from one forward with output_hidden_states (CPU SVD)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    prc, ani, k95 = [], [], []
    for h in out.hidden_states:
        X = h.detach().squeeze(0).float().cpu().numpy()
        sig = np.linalg.svd((X - X.mean(axis=0)).astype(np.float32), compute_uv=False)
        prc.append(float(C.participation_ratio(sig)))
        k95.append(int(C.k95_of_spectrum(sig)))
        ani.append(float(C.mean_abs_cos(X)))
        del X, sig
    del out
    torch.cuda.empty_cache()
    return np.array(prc), np.array(ani), np.array(k95)


def resample_curve(curve):
    """Resample a per-layer curve (index 1..L) onto the common depth grid."""
    L = len(curve) - 1
    xs = np.arange(1, L + 1) / float(L)
    ys = curve[1:]
    d = np.linspace(0.0, 1.0, GRID_N)
    return np.interp(d, xs, ys), d


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

    if not os.path.isfile(RAW_TEXT_PATH):
        raise SystemExit(f"canonical corpus file missing: {RAW_TEXT_PATH}")
    with open(RAW_TEXT_PATH, encoding="utf-8") as fh:
        raw_text = fh.read()
    print(f"control corpus: {RAW_TEXT_PATH}  ({len(raw_text)} chars)")

    work_json = os.path.join(C.WORK, "exp10_activation_structure_8m.json")
    state = {"meta": {
        "corpus": "wikitext-2 test split (quantization control text), first 2048 tokens per model",
        "raw_text_path": RAW_TEXT_PATH,
        "n_tokens": N_TOKENS,
        "grid_n": GRID_N,
        "criteria": {
            "mean_pairwise_r_min": CRIT_MEAN_R,
            "min_pairwise_r_min": CRIT_MIN_R,
            "neck_std_max": CRIT_NECK_STD,
        },
    }, "models": {}, "cross_model": None}
    if os.path.exists(work_json):   # merge: a later per-model run must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
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
            print(f"=== Exp10 {key}  dtype={str(dtype)}  L={L} H={H} tied={tied}")
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

            prc, ani, k95 = metrics_for_model(mod, ids)
            n_levels = len(prc)
            if n_levels - 1 != L:
                print(f"    WARNING: hidden_states levels {n_levels} != expected L+1={L + 1}; "
                      f"normalizing by actual level count")
            neck_level = int(np.argmin(prc[1:]) + 1)
            state["models"][key] = {
                "path": path,
                "dtype": str(dtype),
                "layers_L": L,
                "hidden_H": H,
                "tied_embeddings": tied,
                "n_levels_seen": n_levels,
                "prc_per_level": [round(float(x), 3) for x in prc],
                "ani_per_level": [round(float(x), 4) for x in ani],
                "k95_per_level": [int(x) for x in k95],
                "neck_level": neck_level,
                "neck_fraction": round(neck_level / float(n_levels - 1), 3),
                "neck_pr_centered": round(float(prc[neck_level]), 2),
            }
            print(f"    NECK: level {neck_level} (frac {state['models'][key]['neck_fraction']}), "
                  f"PRc={state['models'][key]['neck_pr_centered']}, k95@neck="
                  f"{int(k95[neck_level])}")
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

        with open(work_json, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        print(f"    (incremental save -> {work_json})")

    cross = compute_cross_model_analysis(state["models"])
    state["cross_model"] = cross
    with open(work_json, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)

    print("\n--- Exp10 summary ---")
    for key, r in state["models"].items():
        if "status" in r:
            print(f"{key:<16s} SKIPPED: {r['status']}")
        else:
            print(f"{key:<16s} L={r['layers_L']:>2d}  neck_l={r['neck_level']:>2d} "
                  f"frac={r['neck_fraction']:.2f}  PRc_neck={r['neck_pr_centered']:>7.1f}")
    if cross:
        print(f"\nmean pairwise r (prc) = {cross['mean_pair_r']:.3f}   "
              f"min = {cross['min_pair_r']:.3f}   max = {cross['max_pair_r']:.3f}")
        print(f"neck fractions std = {cross['neck_std']:.4f}   spread(max-min) = {cross['neck_spread_max_min']:.4f}")
        for c in cross["criteria_check"]:
            print(f"  criterion: {c}")
        print(f"\nVERDICT: {cross['verdict']}")

    try:
        make_plots(state)
    except Exception as e:   # plots are a deliverable, but never mask the numbers
        print(f"PLOT FAILED (numbers already saved): {e}")


def compute_cross_model_analysis(models):
    """Pairwise Pearson r on depth-normalized prc curves + cosine distance on ani
    curves + neck-position spread. Literal verdict per the pre-declared criteria."""
    keys = [k for k, r in models.items() if "status" not in r]
    if len(keys) < 2:
        return None

    grid_curves = {}
    for key in keys:
        r = models[key]
        prc_curve, d = resample_curve(np.array(r["prc_per_level"]))
        ani_curve, _ = resample_curve(np.array(r["ani_per_level"]))
        grid_curves[key] = (d, prc_curve, ani_curve)

    n = len(keys)
    r_mat = np.zeros((n, n))
    cos_d_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = keys[i], keys[j]
            _, pa, _ = grid_curves[a]
            _, pb, _ = grid_curves[b]
            r_mat[i, j] = r_mat[j, i] = float(np.corrcoef(pa, pb)[0, 1])
            ca, cb = grid_curves[a][2], grid_curves[b][2]
            cd = 1.0 - C.cosine_similarity(ca, cb)
            cos_d_mat[i, j] = cos_d_mat[j, i] = float(cd)

    off = [r_mat[i, j] for i in range(n) for j in range(i + 1, n)]
    mean_r, min_r, max_r = float(np.mean(off)), float(min(off)), float(max(off))

    neck_fracs = {k: models[k]["neck_fraction"] for k in keys}
    nf = np.array(list(neck_fracs.values()))
    neck_std = float(nf.std())
    neck_spread = float(nf.max() - nf.min())

    c1 = mean_r >= CRIT_MEAN_R
    c2 = min_r >= CRIT_MIN_R
    c3 = neck_std <= CRIT_NECK_STD
    verdict = "COMMON_PATTERNS" if (c1 and c2 and c3) else "MODEL_SPECIFIC"

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"a": keys[i], "b": keys[j],
                          "pearson_r_prc": round(float(r_mat[i, j]), 4),
                          "cosine_distance_ani": round(float(cos_d_mat[i, j]), 4)})

    return {
        "n_models_compared": n,
        "pairwise_pearson_r_prc_matrix": {keys[i]: [round(float(r_mat[i, j]), 4) for j in range(n)]
                                          for i in range(n)},
        "mean_pair_r": round(mean_r, 4),
        "min_pair_r": round(min_r, 4),
        "max_pair_r": round(max_r, 4),
        "ani_cosine_distance_pairs_min": round(float(np.min([p['cosine_distance_ani'] for p in pairs])), 4),
        "ani_cosine_distance_pairs_max": round(float(np.max([p['cosine_distance_ani'] for p in pairs])), 4),
        "pairs": pairs,
        "neck_fractions": {k: v for k, v in neck_fracs.items()},
        "neck_std": round(neck_std, 4),
        "neck_spread_max_min": round(neck_spread, 4),
        "criteria_check": [
            f"mean_pair_r {mean_r:.3f} >= {CRIT_MEAN_R}: {'PASS' if c1 else 'FAIL'}",
            f"min_pair_r {min_r:.3f} >= {CRIT_MIN_R}: {'PASS' if c2 else 'FAIL'}",
            f"neck_std {neck_std:.4f} <= {CRIT_NECK_STD}: {'PASS' if c3 else 'FAIL'}",
        ],
        "verdict": verdict,
    }


def make_plots(state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k, r in state["models"].items() if "status" not in r]
    if len(keys) < 2:
        print("no plots (fewer than 2 successful models)")
        return

    d_grid = np.linspace(0.0, 1.0, GRID_N)
    for metric, fname in (("prc_per_level", "exp10_prc_curves.png"),
                          ("ani_per_level", "exp10_aniso_curves.png"),
                          ("k95_per_level", "exp10_k95_curves.png")):
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=130)
        for key in keys:
            r = state["models"][key]
            L = len(r[metric]) - 1
            xs = np.arange(1, L + 1) / float(L)
            ys = [float(v) if metric != "k95_per_level" else int(v) for v in r[metric][1:]]
            ax.plot(xs, ys, lw=1.6, label=f"{key} (L={r['layers_L']})")
        title = {"prc": "Centered participation ratio of per-level activations",
                 "ani": "Mean |cos| between raw token vectors (representation cone)",
                 "k95": "Effective rank k95 (smallest k with >= 0.95 explained variance)"}[metric[:3]]
        ax.set_xlabel("depth fraction l/L")
        ax.set_ylabel(metric.replace("_per_level", "").replace("prc", "PRc").replace("ani", "mean|cos|"))
        ax.set_title(f"Exp10 — 8 models, common wikitext control group (2048 tokens)\n{title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out = os.path.join(C.RESULTS, fname)
        fig.savefig(out)
        plt.close(fig)
        print(f"plot written: {out}")


if __name__ == "__main__":
    main()
