"""Exp 9 (multi-seed) — statistical power of the layer-sensitivity scan.

Reviewer p.1 item: "Exp9 -> eps normalized on hidden_rms, 10-50 perturbations/layer,
KL(eps) curves". The original run_exp9_hidden_perturbation.py did ONE randn draw per
(layer, eps) with an ABSOLUTE noise scale (eps in {1e-4,1e-3,1e-2} added to H), so:

  (a) the same absolute eps is a very different fraction of the activation scale for
      a layer/model whose hidden-state RMS differs (gemma vs qwen) — curves not
      comparable across models; and
  (b) one draw per cell gives no uncertainty band on the sensitivity neck.

This variant fixes both while reusing the ORIGINAL injection + KL math verbatim
(imported from run_exp9_hidden_perturbation so nothing drifts):

  * EPS NORMALIZED ON HIDDEN_RMS: the noise std for layer i at relative level rel_eps
    is `noise_std = rel_eps * rms[i]`, where rms[i] is that layer's clean per-layer
    hidden-state RMS (captured by base_forward). So rel_eps=1e-2 means "1% of this
    layer's typical activation magnitude" for EVERY model and layer — the KL(rel_eps)
    curves are now on a common scale. The three relative levels {1e-4,1e-3,1e-2} match
    the original's absolute values so results stay directly comparable to exp9.

  * MULTI-SEED: N_PERT (default 10, reviewer floor; bumpable via argv[2] up to 50)
    independent randn draws per (layer, rel_eps), reported as mean +/- std -> the
    sensitivity neck now carries an uncertainty band instead of a single sample.

Same models as the original exp9 set (gemma-4-e2b, qwen2.5-0.5b, tinyllama-1.1b),
same bf16 / L-fallbacks (1024->512->256) / pick_gpu(device with >= 14 GiB free on GPU 1).

Writes data/data/exp9_multiseed_metrics.json after each model (crash-safe, merges into an
existing file). Launch: python g4_dynamics_attractor\run_exp9_multiseed.py [model-key,...] [n_pert]
"""

import gc
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # g4_dynamics_attractor/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
CROSS_MODEL_DIR = os.path.join(TVU, "code")
sys.path.insert(0, CROSS_MODEL_DIR)                        # for cross_common / run_gemma_functional_impact
sys.path.insert(0, TVU)                                   # for tvu_common

import numpy as np                                       # noqa: E402
import torch                                              # noqa: E402
from torch.nn import functional as F                      # noqa: E402

import tvu_common as C                                    # noqa: E402  (make_diverse_tokens, WORK)
from cross_common import MODELS                           # noqa: E402
from run_gemma_functional_impact import pick_gpu, get_text_stack      # noqa: E402
import run_exp9_hidden_perturbation as E9                 # noqa: E402  (base_forward/noisy_forward/kl_per_token)

WORK_JSON = os.path.join(C.WORK, "exp9_multiseed_metrics.json")

SELECTED_MODELS = ["gemma-4-e2b", "qwen2.5-0.5b", "tinyllama-1.1b"]   # same set as original exp9
REL_EPSILONS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)     # relative to hidden_rms (matches original values)
L_FALLBACKS: Tuple[int, ...] = (1024, 512, 256)          # same context OOM fallback as original


def _dump(results: dict) -> None:
    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"  (saved -> {WORK_JSON})")


def scan_multiseed(model, ids, tsm, n_layers, p_base, rms, rel_epsilons, n_pert):
    """Per-layer x relative-eps sweep with N_PERT independent draws each.

    noise_std for layer i at level rel_eps = rel_eps * rms[i] (normalized on the
    layer's own hidden-state RMS). Reuses E9.noisy_forward + E9.kl_per_token so the
    injection and KL math are identical to the original exp9."""
    per_layer: List[Dict] = []
    for i in range(n_layers):
        row: Dict = {"layer": i, "hidden_rms": round(float(rms[i]), 4), "kl_by_rel_eps": {}}
        for re in rel_epsilons:
            noise_std = float(re * rms[i])
            kls = []
            for _ in range(n_pert):
                nl = E9.noisy_forward(model, ids, tsm, i, noise_std)
                kls.append(E9.kl_per_token(nl, p_base))
                del nl
            arr = np.asarray(kls, dtype=np.float64)
            row["kl_by_rel_eps"][f"{re:g}"] = {
                "mean": round(float(arr.mean()), 6),
                "std": round(float(arr.std(ddof=1)), 6) if n_pert > 1 else 0.0,
                "n_perturbations": int(n_pert),
            }
        per_layer.append(row)
        print("  layer %2d/%d: " % (i, n_layers - 1) + "   ".join(
            f"kl({e:g})={row['kl_by_rel_eps'][f'{e:g}']['mean']:.4g}" for e in rel_epsilons))
    return per_layer


def run_model(entry: dict, device: torch.device, n_pert: int) -> dict:
    key, path = entry["key"], entry["path"]
    print(f"\n=== exp9-multiseed [{key}] {path} (bf16 on {device}, n_pert={n_pert}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    model.eval()
    tsm = get_text_stack(model)
    n_layers = len(tsm.layers)

    rec: Dict = {
        "path": path,
        "gpu_used": str(device),
        "dtype": "bf16",
        "n_layers": n_layers,
        "rel_epsilons_normalized_on_hidden_rms": list(REL_EPSILONS),
        "n_perturbations": int(n_pert),
        "L_tokens": None,
        "protocol": ("base forward (no_grad) vs N_PERT independent one-forward hook injections of "
                     "H + (rel_eps*hidden_rms[i])*randn_like(H) at each decoder layer; metric = token-averaged "
                     "KL(softmax(base)||softmax(noisy)) in nats, mean+/-std over perturbations"),
    }

    per_layer: Optional[List[Dict]] = None
    rms: List[float] = []
    for L in L_FALLBACKS:
        ids = C.make_diverse_tokens(tok, n_target=L).to(device)
        p_base: Optional[torch.Tensor] = None
        try:
            base_logits, rms = E9.base_forward(model, ids, tsm)
            p_base = F.softmax(base_logits.float(), dim=-1)   # cached once (fp32), reused by all noisy runs
            del base_logits
            per_layer = scan_multiseed(model, ids, tsm, n_layers, p_base, rms, REL_EPSILONS, n_pert)
            rec["L_tokens"] = L
            print(f"  scan OK at L={L}")
            break
        except torch.OutOfMemoryError as err:
            per_layer = None
            print(f"  OOM at L={L} ({str(err)[:100]}) — retrying shorter context")
        finally:
            if p_base is not None:
                del p_base
            gc.collect()
            torch.cuda.empty_cache()
    if per_layer is None:
        raise SystemExit(f"ABORT [{key}]: context OOM even at L=256 — model skipped.")

    max_re_key = f"{max(REL_EPSILONS):g}"
    neck = max(per_layer, key=lambda r: r["kl_by_rel_eps"][max_re_key]["mean"])
    rec["neck_at_max_rel_eps"] = {
        "layer": neck["layer"],
        "kl_mean": neck["kl_by_rel_eps"][max_re_key]["mean"],
        "kl_std": neck["kl_by_rel_eps"][max_re_key]["std"],
    }
    rec["per_layer"] = per_layer

    del ids, tok, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    n_pert = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    device, free_gib = pick_gpu()
    print(f"exp9 multiseed — device {device} (free {free_gib:.1f} GiB), n_pert={n_pert}")

    results: dict = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    selected = [next(m for m in MODELS if m["key"] == k) for k in SELECTED_MODELS]
    for entry in selected:
        if only is not None and entry["key"] not in only:
            continue
        results[entry["key"]] = run_model(entry, device, n_pert)
        _dump(results)

    print("\n--- exp9 multiseed summary (sensitivity neck at rel_eps=1e-02, mean over perturbations) ---")
    for key in SELECTED_MODELS:
        r = results.get(key)
        if not r or "neck_at_max_rel_eps" not in r:
            continue
        n = r["neck_at_max_rel_eps"]
        print(f"  {key}: L={r.get('L_tokens')} layers={r['n_layers']} -> layer {n['layer']} "
              f"(kl_mean={n['kl_mean']:.4g} +/- {n['kl_std']:.4g})")


if __name__ == "__main__":
    main()
