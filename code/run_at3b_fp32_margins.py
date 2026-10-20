"""AT-3b — fp32 rerun of the AT-3 margin dumps, CPU pass (all 8 models).

Why this pass exists (two problems in the original AT-3 data):
  1. The at3_margins_*.npz files are BROKEN: margins_of indexed vals[:, 0] on a
     (B, S, 2) tensor -> shape (1, 2), so all regularity numbers derived from
     them in the AT-4 audit were garbage (documented in run_at4_theorem_audit.py).
     This pass recomputes margins correctly (vals[..., 0] - vals[..., 1]) in fp32.
  2. The stability groups found in AT-1 correspond PERFECTLY to the run dtype
     (fp32 models stable, bf16 models unstable) — architecture vs bf16 rounding
     cannot be separated from the original data. This pass runs ALL 8 models in
     forced fp32 and extends the delta grid with {1e-4, 3e-4} so the flip
     structure of the four originally-bf16 models is measured at the same
     precision as the stable group — resolving that confound.

Scope (deliberately narrower than AT-3): decision margins + top-k logit dumps +
flip rates only. NO U6 pairs, NO KL / |dlogits| stats, NO fp16 SWD subsamples.

Instantiation: identical to AT-1/AT-3 — state s0 = hidden h^l(x) at decoder
layer l; perturbed neighbor via forward-hook injection H + delta*RMS(H_base)*
randn_like(H), so D(s, s0) = delta exactly (relative to base-state RMS);
program S = rest of the stack; property = per-position top-1 token identity.

Exogenous parameters (paper Table 5 style, NOT fitted): epsilon = 1e-3,
delta_param = 5e-3, gamma = 3e-4 -> eps-gamma = 7e-4, eps+gamma = 1.3e-3.
Delta grid per layer: {1e-4, 3e-4, 7e-4, 1.3e-3} (the first two are the
confound-resolution additions; 7e-4 / 1.3e-3 feed the regularity check).

Per model (sequential loading, forced torch.float32 for ALL models, CPU):
  base forward -> margin [S] fp32 + top-64 logit ids/vals per position [S,64]
  each selected layer l x delta d: noisy forward -> margin [S], top-2 token
  ids+vals [S,2], flip rate vs base. Noise realization is deterministic per
  (layer, delta): torch.manual_seed(2_026_0924 + i*1000 + j*7) before each
  injection, so a rerun reproduces the exact same eta.
  regularity (AT-4 formulas, clean data): ball = base[S] + mar_d[S] at the
  last selected layer; tau_hat = min(ball); L_F_hat = max |mar_d - mar_base| / d
  over d in {7e-4, 1.3e-3}; cond3 = all ball margins >= 0; cond6 = tau > 0 and
  delta_param <= tau / L_F_hat. Same check repeated per layer and combined as
  the full-stack variant (tau_full = min over layers, L_F_full = max).

Writes data/at3b_fp32_margins_metrics.json after EACH model (crash-safe merge;
a rerun skips models whose record already has "singles" AND whose npz exists).
Dump per model: data/at3b_margins_<model>.npz {margin_base [S], topk_ids_base
[S,64] int32 / topk_vals_base [S,64] fp32, l<i>_d<d> margin [S] + _ids [S,2]
int32 + _vals [S,2] fp32 per layer and delta}.

Context: diverse-doc corpus truncated to 512 tokens (RAM OOM fallback 256).
Measured CPU speed (Ryzen 7 5700X, L=512 fp32): smollm-135m 1.2 s/forward,
qwen2.5-0.5b 2.8 s/forward -> all 8 models take ~45-60 min (33 forwards per model).

Run: <python with transformers>=5.x> code\\run_at3b_fp32_margins.py [--models k1,k2]
     or at3b_fp32_margins.bat
"""

import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # aiir-hourglass/code/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys_path_fix = [os.path.join(TVU, "code"), TVU]
import sys
for p in sys_path_fix:
    if p not in sys.path:
        sys.path.insert(0, p)

if os.environ.get("PYTHONIOENCODING") is None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cp1251 console safety
    except Exception:
        pass

import numpy as np                                     # noqa: E402
import torch                                            # noqa: E402

import tvu_common as C                                  # noqa: E402  (make_diverse_tokens, DATA)
from cross_common import MODELS                         # noqa: E402
from run_gemma_functional_impact import get_text_stack  # noqa: E402

WORK_JSON = os.path.join(C.DATA, "at3b_fp32_margins_metrics.json")


def _text_stack(model: torch.nn.Module):
    """Shared resolver + fallback: GPTNeoX names its embedding `embed_in` (not embed_tokens)."""
    try:
        return get_text_stack(model)
    except AttributeError:
        for mod in model.modules():
            if hasattr(mod, "layers") and (hasattr(mod, "embed_tokens") or hasattr(mod, "embed_in")):
                return mod
        raise


# --- exogenous theorem parameters (paper Table 5 style; NOT fitted from data) ---
EPSILON = 1e-3
DELTA_PARAM = 5e-3
GAMMA = 3e-4
MARGIN_DELTAS: Tuple[float, ...] = (EPSILON - GAMMA, EPSILON + GAMMA)   # 7e-4, 1.3e-3 — feed regularity
CONFOND_DELTAS: Tuple[float, ...] = (1e-4, 3e-4)                        # dtype-confound resolution grid
ALL_DELTAS: Tuple[float, ...] = CONFOND_DELTAS + MARGIN_DELTAS          # {1e-4, 3e-4, 7e-4, 1.3e-3}

REL_DEPTHS: Tuple[float, ...] = (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
L_FALLBACKS: Tuple[int, ...] = (512, 256)              # context fallback on RAM OOM
TOPK_SLICE = 64            # per-position logit slice for cross-model shared-vocab comparison

_DEVICE = torch.device("cpu")
_SEED_BASE = 2_026_0924   # deterministic per-(layer, delta) noise realization (see docstring)


def _dump(results: dict) -> None:
    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"  (saved -> {WORK_JSON})")


def selected_layer_indices(n_layers: int) -> List[int]:
    """Unique layer indices at REL_DEPTHS of the stack depth, ascending."""
    idx = sorted({int(round(r * (n_layers - 1))) for r in REL_DEPTHS})
    return [i for i in idx if 0 <= i < n_layers]


def base_forward(model: torch.nn.Module, ids: torch.Tensor, tsm,
                 sel_layers: List[int]) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Clean forward; returns (logits fp32 on CPU, per-selected-layer RMS)."""
    rms: Dict[int, float] = {}

    def _rec(idx):
        def _hook(_mod, _inp, out):
            o = out.detach().float()
            rms[idx] = float(o.pow(2).mean().sqrt())
        return _hook

    handles = [tsm.layers[i].register_forward_hook(_rec(i)) for i in sel_layers]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
    finally:
        for h in handles:
            h.remove()
    return logits, rms


def noisy_forward(model: torch.nn.Module, ids: torch.Tensor, tsm, layer_idx: int,
                  scale: float) -> torch.Tensor:
    """One forward with layers[layer_idx]'s output replaced by H + scale*randn_like(H).

    scale = delta * RMS(h_base^l), so D(s, s0) = delta exactly (relative norm of the
    perturbation against the BASE state — deterministic, not re-measured per run)."""
    layer = tsm.layers[layer_idx]

    def _inject(mod, inp, out):
        eta = scale * torch.randn_like(out)   # identical to AT-1/AT-3 (all-fp32 here)
        return out + eta

    handle = layer.register_forward_hook(_inject)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
    finally:
        handle.remove()
    return logits


def margins_of(logits: torch.Tensor) -> np.ndarray:
    """Per-position decision margin top1 - logit_2nd [S] (CPU fp32).

    NOTE: vals has shape (B, S, 2) — index the LAST dim (the original AT-3 bug was
    vals[:, 0], which returned shape (1, 2))."""
    vals = torch.topk(logits, k=2, dim=-1).values   # (B, S, 2)
    m = (vals[..., 0] - vals[..., 1]).float().cpu().numpy().reshape(-1)
    return m


def topk_slice(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """(ids [S, TOPK_SLICE] int32, vals [S, TOPK_SLICE] fp32) over all positions (batch 0)."""
    tk = torch.topk(logits[0], k=TOPK_SLICE, dim=-1)
    return (tk.indices.cpu().numpy().astype(np.int32), tk.values.float().cpu().numpy())


def top2_slice(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """(ids [S, 2] int32, vals [S, 2] fp32) over all positions (batch 0)."""
    tk = torch.topk(logits[0], k=2, dim=-1)
    return (tk.indices.cpu().numpy().astype(np.int32), tk.values.float().cpu().numpy())


def regularity(mar_base: np.ndarray, mar_7e4: np.ndarray, mar_13e3: np.ndarray) -> dict:
    """AT-4 regularity formulas on clean fp32 full-position data.

    ball = base[S] + mar_{eps-gamma}[S] + mar_{eps+gamma}[S]; tau_hat = min(ball);
    L_F_hat = max_d max_p |mar_d[p] - mar_base[p]| / d; cond3 = all ball margins >= 0;
    cond6 = tau > 0 and delta_param <= tau / L_F_hat (L_F_hat == 0 -> vacuously false)."""
    ball = np.concatenate([mar_base, mar_7e4, mar_13e3])
    tau = float(ball.min())
    lf = max(float(np.abs(mar_d - mar_base).max()) / d for d, mar_d in ((MARGIN_DELTAS[0], mar_7e4),
                                                                        (MARGIN_DELTAS[1], mar_13e3)))
    return {
        "tau_hat": round(tau, 6),
        "lf_hat_max_ratio": round(lf, 6),
        "cond3_all_ball_margins_nonneg": bool((ball >= 0).all()),
        "cond6_delta_le_tau_over_lf": bool(tau > 0 and lf > 0 and DELTA_PARAM <= tau / lf),
    }


def run_model(entry: dict) -> dict:
    key, path = entry["key"], entry["path"]
    print(f"\n=== at3b [{key}] {path} (forced fp32 on {_DEVICE}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(_DEVICE)
    model.eval()
    tsm = _text_stack(model)
    n_layers = len(tsm.layers)
    sel_layers = selected_layer_indices(n_layers)
    last_layer = sel_layers[-1]

    rec: Dict = {
        "path": path,
        "device": str(_DEVICE),
        "dtype_used": "torch.float32 (forced — registry dtype overridden; this is the point of AT-3b)",
        "n_layers": n_layers,
        "selected_layers": sel_layers,
        "params": {
            "epsilon": EPSILON,
            "delta_param": DELTA_PARAM,
            "gamma": GAMMA,
            "eps_minus_gamma": round(EPSILON - GAMMA, 6),
            "eps_plus_gamma": round(EPSILON + GAMMA, 6),
            "delta_minus_2gamma": round(DELTA_PARAM - 2 * GAMMA, 6),
            "deltas_per_layer": list(ALL_DELTAS),
        },
        "protocol": ("AT-3b: fp32 CPU rerun of the AT-3 margin dumps. base forward (no_grad) + "
                     "hook injection H + delta*RMS(H_base)*randn_like(H) at each selected decoder layer; "
                     "D(s,s0)=delta relative to base-state RMS; exogenous epsilon=1e-3, delta_param=5e-3, "
                     "gamma=3e-4 (paper Table 5 style); per-layer delta grid {1e-4, 3e-4, 7e-4, 1.3e-3} "
                     "(first two resolve the AT-1 dtype-vs-stability confound); deterministic noise seeding "
                     "torch.manual_seed(20260924 + i*1000 + j*7) per (layer i, delta index j); property = "
                     "per-position top-1 identity; margins via vals[..., 0] - vals[..., 1] (AT-3's broken "
                     "(1,2)-shape bug fixed). NO U6 pairs / KL / |dlogits| in this pass."),
    }

    singles: Dict[str, dict] = {}
    dump_arrays: Dict[str, np.ndarray] = {"margin_base": None}   # filled below
    base_logits: Optional[torch.Tensor] = None

    for L in L_FALLBACKS:
        ids = C.make_diverse_tokens(tok, n_target=L).to(_DEVICE)
        try:
            base_logits, rms = base_forward(model, ids, tsm, sel_layers)
            mar_base = margins_of(base_logits)
            topk_ids, topk_vals = topk_slice(base_logits)
            dump_arrays["margin_base"] = mar_base.astype(np.float32)
            dump_arrays["topk_ids_base"] = topk_ids
            dump_arrays["topk_vals_base"] = topk_vals

            top1_base = base_logits.argmax(dim=-1)
            rec["L_tokens"] = L
            print(f"  base OK at L={L}, V={base_logits.shape[-1]}")

            # --- per layer x delta: noisy forward, margin + top-2 dump + flip rate ---
            mar_by_layer_delta: Dict[int, Dict[float, np.ndarray]] = {i: {} for i in sel_layers}
            for j, d in enumerate(ALL_DELTAS):
                for i in sel_layers:
                    torch.manual_seed(_SEED_BASE + i * 1000 + j * 7)   # deterministic eta per (layer, delta)
                    nl = noisy_forward(model, ids, tsm, i, d * rms[i])
                    mar = margins_of(nl)
                    flip_rate = float((nl.argmax(dim=-1) != top1_base).float().mean().item())
                    mar_by_layer_delta[i][d] = mar.astype(np.float32)
                    ids2, vals2 = top2_slice(nl)
                    dkey = f"l{i}_d{d:g}"
                    dump_arrays[dkey] = mar.astype(np.float32)
                    dump_arrays[f"{dkey}_ids"] = ids2
                    dump_arrays[f"{dkey}_vals"] = vals2
                    singles.setdefault(str(i), {})[f"{d:g}"] = {"flip_rate": round(flip_rate, 6)}
                    del nl, mar, ids2, vals2
                print("  d=%g done: " % d + "   ".join(
                    f"l{i}:{singles[str(i)][f'{d:g}']['flip_rate']:.4g}" for i in sel_layers))

            np.savez_compressed(os.path.join(C.DATA, f"at3b_margins_{key}.npz"), **dump_arrays)
            rec["dumps"] = {"margins": f"at3b_margins_{key}.npz"}
            rec["singles"] = singles

            # --- regularity (AT-4 formulas; last-layer scope for comparability + full-stack bonus) ---
            reg_last = regularity(mar_base, mar_by_layer_delta[last_layer][MARGIN_DELTAS[0]],
                                  mar_by_layer_delta[last_layer][MARGIN_DELTAS[1]])
            ball_all = [mar_base]
            lf_all = 0.0
            for i in sel_layers:
                m7 = mar_by_layer_delta[i][MARGIN_DELTAS[0]]
                m13 = mar_by_layer_delta[i][MARGIN_DELTAS[1]]
                ball_all.append(np.concatenate([mar_base, m7, m13]))
                lf_all = max(lf_all, float(np.abs(m7 - mar_base).max()) / MARGIN_DELTAS[0],
                             float(np.abs(m13 - mar_base).max()) / MARGIN_DELTAS[1])
            tau_full = float(min(b.min() for b in ball_all))
            reg_full = {
                "tau_hat": round(tau_full, 6),
                "lf_hat_max_ratio": round(lf_all, 6),
                "cond3_all_ball_margins_nonneg": bool(np.concatenate(ball_all).min() >= 0),
                "cond6_delta_le_tau_over_lf": bool(tau_full > 0 and lf_all > 0 and DELTA_PARAM <= tau_full / lf_all),
            }
            reg_last["scope"] = f"last selected layer {last_layer}, full S={L} positions, fp32 (AT-4 used K=128 on broken data)"
            rec["regularity_last_layer"] = reg_last
            rec["regularity_full_stack"] = reg_full

            print(f"  scan OK at L={L}: tau_last={reg_last['tau_hat']}, LF_last={reg_last['lf_hat_max_ratio']}, "
                  f"cond6_last={reg_last['cond6_delta_le_tau_over_lf']}")
            break
        except (MemoryError, torch.OutOfMemoryError) as err:
            singles, dump_arrays = {}, {"margin_base": None}
            print(f"  RAM OOM at L={L} ({str(err)[:100]}) — retrying shorter context")
        finally:
            if base_logits is not None:
                del base_logits
                base_logits = None
            gc.collect()

    if "singles" not in rec:
        raise SystemExit(f"ABORT [{key}]: RAM OOM even at L=256 — model skipped.")

    del ids, tok, model
    gc.collect()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="AT-3b fp32 CPU margin-dump rerun")
    ap.add_argument("--models", default=None, help="comma-separated subset of MODELS keys (default: all 8)")
    args = ap.parse_args()

    targets = MODELS
    if args.models:
        wanted = {k.strip() for k in args.models.split(",") if k.strip()}
        unknown = wanted - {e["key"] for e in MODELS}
        if unknown:
            raise SystemExit(f"unknown model keys: {sorted(unknown)}")
        targets = [e for e in MODELS if e["key"] in wanted]

    print("at3b fp32 CPU margin-dump rerun — device cpu, all models forced torch.float32")
    print(f"corpus files: {[os.path.basename(p) for p in C.DOC_FILES]}")
    print(f"exogenous params: epsilon={EPSILON}, delta_param={DELTA_PARAM}, gamma={GAMMA}")
    print(f"deltas per layer: {list(ALL_DELTAS)}; models: {[e['key'] for e in targets]}")

    results: dict = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for entry in targets:   # sequential loading (RAM-bound on CPU)
        key = entry["key"]
        npz = os.path.join(C.DATA, f"at3b_margins_{key}.npz")
        if (results.get(key) or {}).get("singles") and os.path.exists(npz):
            print(f"\n=== at3b [{key}] already complete — skipping ===")
            continue
        results[key] = run_model(entry)
        _dump(results)

    print("\n--- at3b summary (fp32, CPU) ---")
    for entry in targets:
        key = entry["key"]
        r = results.get(key) or {}
        singles = r.get("singles") or {}
        if not singles:
            print(f"  {key}: no data (skipped?)")
            continue
        n_flip_1e4 = sum(1 for row in singles.values() if float(row[f"{CONFOND_DELTAS[0]:g}"]["flip_rate"]) > 0)
        reg = r.get("regularity_last_layer", {})
        print(f"  {key}: L={r.get('L_tokens')} layers@1e-4 with flip={n_flip_1e4}/{len(singles)} | "
              f"tau_last={reg.get('tau_hat')}, LF_last={reg.get('lf_hat_max_ratio')}, "
              f"cond6_last={reg.get('cond6_delta_le_tau_over_lf')}")


if __name__ == "__main__":
    main()
