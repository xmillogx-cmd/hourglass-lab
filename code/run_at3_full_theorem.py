"""AT-3 — full empirical verification of Nikitin's analogy theorem (Hoare logic) on real corpus data.

Unlike AT-1 (which tested only the operational consequence "D + gamma small -> property
preserved"), this pass collects everything needed to check EVERY statement of the theorem:

  FOL axioms      A3-A7 (metricity — by construction, noted in audit), A8/A9 (stability
                  at delta_param from flip data), A10 (reference similarity — cross-model
                  base agreement from top-k logit dumps), A11/U2/U5 (non-triviality beyond
                  eps+gamma — flips at 1.3e-3)
  Hoare conditions U4 {D <= eps-gamma} S {F <-> L}, U5 {D > eps+gamma & F} {not L},
                  U6 {D(s1,s2) < delta-2gamma} S {F(s1) <-> F(s2)}, C1 (gamma-stability,
                  operationalized via measured output displacement)
  regularity 1-6  threshold functions f_F/f_L = decision margin (top1 - top2 logit gap):
                  tau_hat = min margin inside the ball, L_F/L_L = max |d margin| / D
  FOL statements  (3)/(4) as universal/existential versions of U4/U5

Instantiation (same mapping as AT-1, extended):
  state s0      = hidden state h^l(x) of model m at decoder layer l for prompt x from the
                  diverse-doc corpus; perturbed neighbor s = s0 + eta with
                  ||eta|| / RMS(s_base) = delta  ->  D(s, s0) = delta (relative norm)
  program S     = rest of the stack: layers l+1..L-1 + final norm + lm_head
                  (forward hook replacing layer-l output — same mechanism as AT-1/Exp9)
  properties    = per-position top-1 token identity in base vs perturbed logits
                  (F(φ_S(s)) / L(φ_S(s0)); preserved <=> no flip at that position);
                  decision margin f(s, p) = logit_top1 - logit_2nd at position p

Exogenous parameters (paper Table 5 style — set by the experimenter, NOT fitted):
  epsilon   = 1e-3     delta_param = 5e-3     gamma = 3e-4    (relative state norm)
  -> eps-gamma = 7e-4, eps+gamma = 1.3e-3, delta-2gamma = 4.4e-3
  theorem precondition gamma < min(epsilon, delta/2): 3e-4 < min(1e-3, 2.5e-3) holds

Per model (all 8 in cross_common.MODELS, sequential loading, dtype per registry, physical
GPU 1 via pick_gpu >= 14 GiB free; context = diverse-doc corpus truncated to 512 tokens,
OOM fallback 256):
  base forward   -> dump: logits at a COMMON position subsample (K=128, seed 7) in fp16
                    [for SWD ensembles], top-64 logit ids+values per ALL positions
                    [cross-model A10 / agreement on shared vocab]; margins array [S]
  singles        -> hook injection at each selected layer l (REL_DEPTHS x n_layers) with
                    delta in {7e-4, 1.3e-3}: flip rate vs base top-1, token-averaged KL,
                    mean/max |dlogits|, margins array; at the LAST layer additionally the
                    fp16 logits subsample dump (SWD self-analogy domains)
  U6 pairs       -> per selected layer l and pair scale a in {5e-4, 2e-3}: two independent
                    injections eta1, eta2 ~ N(0, (a*RMS)^2); report D(s1,s2) =
                    RMS(eta1-eta2)/RMS (in expectation sqrt(2)*a < delta-2gamma) and the
                    per-position agreement F(s1) <-> F(s2)

Writes data/at3_full_theorem_metrics.json after EACH model (crash-safe merge into an
existing file; a rerun skips models whose record already has "singles"). Dumps:
  data/at3_base_logits_<model>.npz   {logits_sub [K,V] fp16, topk_ids [S,64], topk_vals}
  data/at3_noisy_last_<model>.npz    {d_0.0007 / d_0.0013 : logits subsample at last layer}
  data/at3_margins_<model>.npz       {base [S], l<idx>_d<delta> [S] per single run}

Run: <python> code\\run_at3_full_theorem.py  or  at3_full_theorem.bat
"""

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

import numpy as np                                     # noqa: E402
import torch                                            # noqa: E402
from torch.nn import functional as F                    # noqa: E402

import tvu_common as C                                  # noqa: E402  (make_diverse_tokens, DATA)
from cross_common import MODELS                         # noqa: E402
from run_gemma_functional_impact import pick_gpu, get_text_stack  # noqa: E402

WORK_JSON = os.path.join(C.DATA, "at3_full_theorem_metrics.json")


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
SINGLE_DELTAS: Tuple[float, ...] = (EPSILON - GAMMA, EPSILON + GAMMA)   # 7e-4, 1.3e-3
PAIR_SCALES: Tuple[float, ...] = (5e-4, 2e-3)    # pair arm scale a; D(s1,s2) ~ sqrt(2)*a

REL_DEPTHS: Tuple[float, ...] = (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
L_FALLBACKS: Tuple[int, ...] = (512, 256)              # context OOM fallback
_KL_CHUNK = 256            # tokens per fp32 chunk (~0.5 GiB transient cap at vocab 262k)
POS_K = 128                # common position subsample for SWD distribution dumps (all models, seed 7)
_POS_SEED = 7
TOPK_SLICE = 64            # per-position logit slice for cross-model shared-vocab comparison


def _dump(results: dict) -> None:
    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"  (saved -> {WORK_JSON})")


def selected_layer_indices(n_layers: int) -> List[int]:
    """Unique layer indices at REL_DEPTHS of the stack depth, ascending."""
    idx = sorted({int(round(r * (n_layers - 1))) for r in REL_DEPTHS})
    return [i for i in idx if 0 <= i < n_layers]


def common_positions(L: int) -> List[int]:
    """Fixed position subsample shared by ALL models (seeded, sorted ascending)."""
    idx = np.random.default_rng(_POS_SEED).choice(L, POS_K, replace=False)
    return sorted(int(i) for i in idx)


def base_forward(model: torch.nn.Module, ids: torch.Tensor, tsm,
                 sel_layers: List[int]) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Clean forward; returns (logits on GPU fp32, per-selected-layer RMS)."""
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
                  scale: float, capture_eta: bool = False) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """One forward with layers[layer_idx]'s output replaced by H + scale*randn_like(H).

    scale = delta * RMS(h_base^l), so D(s, s0) = delta exactly (relative norm of the
    perturbation against the BASE state — deterministic, not re-measured per run).
    With capture_eta=True also returns the injected noise eta as a CPU fp32 array."""
    layer = tsm.layers[layer_idx]
    captured: List[np.ndarray] = []

    def _inject(mod, inp, out):
        eta = scale * torch.randn_like(out)
        if capture_eta:
            # .float() first: numpy has no BFloat16 dtype (bf16 models would crash here)
            captured.append(eta.detach().float().cpu().numpy())
        return out + eta

    handle = layer.register_forward_hook(_inject)
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
    finally:
        handle.remove()
    return logits, (captured[0] if capture_eta else None)


def kl_per_token(noisy_logits: torch.Tensor, p_base: torch.Tensor) -> float:
    """Token-averaged KL(softmax(base) || softmax(noisy)) in nats (chunked fp32)."""
    B, S = noisy_logits.shape[:2]
    total = 0.0
    with torch.no_grad():
        for c in range(0, S, _KL_CHUNK):
            sl, el = c, min(c + _KL_CHUNK, S)
            log_q = F.log_softmax(noisy_logits[:, sl:el], dim=-1)
            per_pos = F.kl_div(log_q, p_base[:, sl:el], reduction="none").sum(dim=-1)
            total += float(per_pos.sum().item())
    return total / (B * S)


def logit_stats(noisy_logits: torch.Tensor, base_logits: torch.Tensor) -> Tuple[float, float]:
    """(mean |delta-logits|, max |delta-logits|), chunked fp32 to cap transients."""
    B, S = noisy_logits.shape[:2]
    total_abs = 0.0
    max_abs = 0.0
    with torch.no_grad():
        for c in range(0, S, _KL_CHUNK):
            sl, el = c, min(c + _KL_CHUNK, S)
            d = (noisy_logits[:, sl:el] - base_logits[:, sl:el]).abs()
            total_abs += float(d.sum().item())
            max_abs = max(max_abs, float(d.max().item()))
    return total_abs / (B * S), max_abs


def margins_of(logits: torch.Tensor) -> np.ndarray:
    """Per-position decision margin top1 - logit_2nd [S] (CPU fp32)."""
    vals = torch.topk(logits, k=2, dim=-1).values   # (B, S, 2) — index LAST dim!
    m = (vals[..., 0] - vals[..., 1]).float().cpu().numpy().reshape(-1)
    return m


def logits_subsample_fp16(logits: torch.Tensor, pos_idx: List[int]) -> np.ndarray:
    """[K, V] fp16 CPU array of the logits at the common position subsample (batch 0)."""
    idx = torch.as_tensor(pos_idx, device=logits.device)
    return logits[0][:, idx].half().cpu().numpy()


def topk_slice(logits: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """(ids [S, TOPK_SLICE] int32, vals [S, TOPK_SLICE] fp32) over all positions (batch 0)."""
    tk = torch.topk(logits[0], k=TOPK_SLICE, dim=-1)
    return (tk.indices.cpu().numpy().astype(np.int32), tk.values.float().cpu().numpy())


def run_model(entry: dict, device: torch.device) -> dict:
    key, path = entry["key"], entry["path"]
    print(f"\n=== at3 [{key}] {path} ({entry['dtype']} on {device}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=entry["dtype"]).to(device)
    model.eval()
    tsm = _text_stack(model)
    n_layers = len(tsm.layers)
    sel_layers = selected_layer_indices(n_layers)
    last_layer = sel_layers[-1]

    rec: Dict = {
        "path": path,
        "gpu_used": str(device),
        "dtype": str(entry["dtype"]),
        "n_layers": n_layers,
        "selected_layers": sel_layers,
        "params": {
            "epsilon": EPSILON,
            "delta_param": DELTA_PARAM,
            "gamma": GAMMA,
            "eps_minus_gamma": round(EPSILON - GAMMA, 6),
            "eps_plus_gamma": round(EPSILON + GAMMA, 6),
            "delta_minus_2gamma": round(DELTA_PARAM - 2 * GAMMA, 6),
            "single_deltas": list(SINGLE_DELTAS),
            "pair_scales": list(PAIR_SCALES),
        },
        "protocol": ("base forward (no_grad) + hook injection of H + delta*RMS(H_base)*randn_like(H) "
                     "at each selected decoder layer; D(s,s0)=delta relative to base-state RMS; "
                     "exogenous epsilon=1e-3, delta_param=5e-3, gamma=3e-4 (paper Table 5 style); "
                     "singles at eps-gamma / eps+gamma with flip rate, KL, |dlogits|, decision margins; "
                     "U6 pairs: two independent injections per layer/scale, D(s1,s2)=RMS(eta1-eta2)/RMS "
                     "and per-position F(s1)<->F(s2) agreement; dumps: fp16 logits at common K=128 "
                     "positions (SWD ensembles), top-64 logit slice per position (cross-model A10)")
    }

    singles: Dict[str, dict] = {}
    u6_pairs: Dict[str, list] = {}
    margins_npz: Dict[str, np.ndarray] = {}

    for L in L_FALLBACKS:
        ids = C.make_diverse_tokens(tok, n_target=L).to(device)
        pos_idx = common_positions(L)
        p_base: Optional[torch.Tensor] = None
        base_logits: Optional[torch.Tensor] = None
        top1_base: Optional[torch.Tensor] = None
        try:
            base_logits, rms = base_forward(model, ids, tsm, sel_layers)

            # --- base dumps ---
            margins_npz["base"] = margins_of(base_logits)
            topk_ids, topk_vals = topk_slice(base_logits)
            sub_base = logits_subsample_fp16(base_logits, pos_idx)
            np.savez_compressed(os.path.join(C.DATA, f"at3_base_logits_{key}.npz"),
                                logits_sub=sub_base.astype(np.float16),
                                topk_ids=topk_ids, topk_vals=topk_vals)

            p_base = F.softmax(base_logits, dim=-1)
            top1_base = base_logits.argmax(dim=-1)
            rec["L_tokens"] = L
            rec["pos_indices"] = pos_idx
            print(f"  base OK at L={L}, V={base_logits.shape[-1]}, subsample K={POS_K}")

            # --- singles sweep: all selected layers x {eps-gamma, eps+gamma} ---
            sub_noisy: Dict[str, np.ndarray] = {}
            for i in sel_layers:
                row: dict = {}
                for delta in SINGLE_DELTAS:
                    dkey = f"{delta:g}"
                    nl = noisy_forward(model, ids, tsm, i, delta * rms[i])[0]
                    kl = kl_per_token(nl, p_base)
                    flips = (nl.argmax(dim=-1) != top1_base).float()
                    flip_rate = float(flips.mean().item())
                    dmean, dmax = logit_stats(nl, base_logits)
                    margins_npz[f"l{i}_d{dkey}"] = margins_of(nl)
                    row[dkey] = {
                        "flip_rate": round(flip_rate, 6),
                        "kl": round(kl, 6),
                        "dlogit_mean": round(dmean, 5),
                        "dlogit_max": round(dmax, 4),
                    }
                    if i == last_layer:
                        sub_noisy[f"d_{dkey}"] = logits_subsample_fp16(nl, pos_idx)
                    del nl
                singles[str(i)] = row
                print("  layer %2d/%d (r=%.2f): " % (i + 1, n_layers, i / max(n_layers - 1, 1))
                      + "   ".join(f"flip({e:g})={row[f'{e:g}']['flip_rate']:.4g}" for e in SINGLE_DELTAS))

            np.savez_compressed(os.path.join(C.DATA, f"at3_noisy_last_{key}.npz"),
                                **{k: v.astype(np.float16) for k, v in sub_noisy.items()})

            # --- U6 pairs: two independent injections per layer and scale a ---
            for i in sel_layers:
                rows = []
                for a in PAIR_SCALES:
                    l1, eta1 = noisy_forward(model, ids, tsm, i, a * rms[i], capture_eta=True)
                    l2, eta2 = noisy_forward(model, ids, tsm, i, a * rms[i], capture_eta=True)
                    d_s1s2 = float(np.sqrt(((eta1 - eta2) ** 2).mean()) / rms[i])
                    agree = int((l1.argmax(dim=-1) == l2.argmax(dim=-1)).sum().item())
                    n_pos = int(l1.shape[1])
                    f1 = float((l1.argmax(dim=-1) != top1_base).float().mean().item())
                    f2 = float((l2.argmax(dim=-1) != top1_base).float().mean().item())
                    rows.append({
                        "a": a,
                        "d_s1s2": round(d_s1s2, 6),
                        "n_pos": n_pos,
                        "n_agree": agree,
                        "agree_rate": round(agree / n_pos, 6),
                        "flip_arm1_vs_base": round(f1, 6),
                        "flip_arm2_vs_base": round(f2, 6),
                    })
                    del l1, l2, eta1, eta2
                u6_pairs[str(i)] = rows
            print("  U6 pairs done")

            np.savez_compressed(os.path.join(C.DATA, f"at3_margins_{key}.npz"), **margins_npz)
            rec["dumps"] = {
                "base": f"at3_base_logits_{key}.npz",
                "noisy_last": f"at3_noisy_last_{key}.npz",
                "margins": f"at3_margins_{key}.npz",
            }
            rec["singles"] = singles
            rec["u6_pairs"] = u6_pairs
            print(f"  scan OK at L={L}")
            break
        except torch.OutOfMemoryError as err:
            singles, u6_pairs = {}, {}
            print(f"  OOM at L={L} ({str(err)[:100]}) — retrying shorter context")
        finally:
            if p_base is not None:
                del p_base
            if base_logits is not None:
                del base_logits
            if top1_base is not None:
                del top1_base
            gc.collect()
            torch.cuda.empty_cache()

    if "singles" not in rec:
        raise SystemExit(f"ABORT [{key}]: context OOM even at L=256 — model skipped.")

    del ids, tok, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    device, free_gib = pick_gpu()
    print(f"at3 full-theorem verification pass — device {device} (free {free_gib:.1f} GiB)")
    print(f"corpus files: {[os.path.basename(p) for p in C.DOC_FILES]}")
    print(f"exogenous params: epsilon={EPSILON}, delta_param={DELTA_PARAM}, gamma={GAMMA}")

    results: dict = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for entry in MODELS:   # all 8 models, sequential loading
        if (results.get(entry["key"]) or {}).get("singles"):
            print(f"\n=== at3 [{entry['key']}] already complete — skipping ===")
            continue
        results[entry["key"]] = run_model(entry, device)
        _dump(results)

    print("\n--- at3 summary (flip rates at eps-gamma / eps+gamma, last selected layer) ---")
    for entry in MODELS:
        key = entry["key"]
        r = results.get(key) or {}
        rows = r.get("singles") or {}
        if not rows:
            print(f"  {key}: no data (skipped?)")
            continue
        last = rows[str(r["selected_layers"][-1])]
        print(f"  {key}: L={r.get('L_tokens')} sel={r.get('selected_layers')} -> last layer "
              f"{list(rows.keys())[-1]}: flip({EPSILON - GAMMA:g})={last[f'{EPSILON - GAMMA:g}']['flip_rate']}, "
              f"flip({EPSILON + GAMMA:g})={last[f'{EPSILON + GAMMA:g}']['flip_rate']}")


if __name__ == "__main__":
    main()
