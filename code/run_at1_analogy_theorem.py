"""AT-1 — empirical test of Nikitin's analogy theorem (Hoare logic) on real corpus data.

Instantiation (mapping theory -> LLM, see the study notes):
  state s     = hidden state h_m^l(x) of model m at decoder layer l for prompt x
                from the diverse-doc corpus (tvu_common.make_diverse_tokens)
  state s0    = s + eta with ||eta|| / RMS(s_base) = delta   ->   D(s, s0) = delta
  program S   = rest of the transformer stack: layers l+1..L-1 + final norm + lm_head
                (forward hook replacing layer-l output — same mechanism as Exp9)
  properties  = top-1 token identity per position in base vs noisy output logits
                (F(S(s)) and L(S(s0)); preserved <=> no flip at that position)
  regularity  = gamma_hat(m, l) = amplification slope dKL/ddelta; AT-2 fits it from
                the delta sweep of this run

The theorem's operational consequence tested here: if D + gamma is small then the
property must be preserved (top-1 unchanged). Systematic violations at small
D + gamma falsify the naive analogy for that model/layer.

Models: all 8 in cross_common.MODELS, sequential loading, dtype per registry
(fp32 smollm/qwen0.5b/tinyllama/phi; bf16 qwen3b/qwen4b/gemma/pythia), physical
GPU 1 via pick_gpu (>= 14 GiB free). Context = diverse-doc corpus truncated to
512 tokens, OOM fallback 256. All forwards under torch.no_grad(), frozen weights.

Selected layers: relative depth r in REL_DEPTHS mapped to the nearest layer index;
r=1.0 (last decoder block) is included on purpose — it isolates the readout
program (norm + lm_head), the candidate for a correct analogy (H-Analogy-4).

Writes data/at1_analogy_perturbation_metrics.json after EACH model (crash-safe
merge into an existing file so reruns of other experiments cannot wipe it; a
rerun skips models whose record already has per_layer). Also
dumps base hidden states at selected layers as fp16 npz per model
(data/at1_act_<model>.npz) for AT-2's cross-model CKA distance D_X.

Run: <python> code\\run_at1_analogy_theorem.py  or  run_at1_analogy_theorem.bat
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

WORK_JSON = os.path.join(C.DATA, "at1_analogy_perturbation_metrics.json")


def _text_stack(model: torch.nn.Module):
    """Shared resolver + fallback: GPTNeoX names its embedding `embed_in` (not embed_tokens)."""
    try:
        return get_text_stack(model)
    except AttributeError:
        for mod in model.modules():
            if hasattr(mod, "layers") and (hasattr(mod, "embed_tokens") or hasattr(mod, "embed_in")):
                return mod
        raise


REL_DEPTHS: Tuple[float, ...] = (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
DELTAS: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)   # relative to RMS(h_base^l)
L_FALLBACKS: Tuple[int, ...] = (512, 256)              # context OOM fallback
_KL_CHUNK = 256            # tokens per fp32 chunk (~0.5 GiB transient cap at vocab 262k)


def _dump(results: dict) -> None:
    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"  (saved -> {WORK_JSON})")


def selected_layer_indices(n_layers: int) -> List[int]:
    """Unique layer indices at REL_DEPTHS of the stack depth, ascending."""
    idx = sorted({int(round(r * (n_layers - 1))) for r in REL_DEPTHS})
    return [i for i in idx if 0 <= i < n_layers]


def base_forward(model: torch.nn.Module, ids: torch.Tensor, tsm,
                 sel_layers: List[int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, np.ndarray]]:
    """Clean forward; returns (logits, per-selected-layer RMS, per-selected-layer
    hidden-state array [S, d] fp32 on CPU).

    Hidden states are captured via temporary forward hooks that move each tensor
    straight to a CPU numpy copy — nothing extra stays in VRAM."""
    rms: Dict[int, float] = {}
    acts: Dict[int, np.ndarray] = {}

    def _rec(idx):
        def _hook(_mod, _inp, out):
            o = out.detach().float()
            rms[idx] = float(o.pow(2).mean().sqrt())
            acts[idx] = o.cpu().numpy()
        return _hook

    handles = [tsm.layers[i].register_forward_hook(_rec(i)) for i in sel_layers]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
    return logits, rms, acts


def noisy_forward(model: torch.nn.Module, ids: torch.Tensor, tsm, layer_idx: int,
                  scale: float) -> torch.Tensor:
    """One forward with layers[layer_idx]'s output replaced by H + scale*randn_like(H).

    scale = delta * RMS(h_base^l), so D(s, s0) = delta exactly (relative norm of the
    perturbation against the BASE state — deterministic, not re-measured per run)."""
    layer = tsm.layers[layer_idx]

    def _inject(mod, inp, out):
        return out + scale * torch.randn_like(out)

    handle = layer.register_forward_hook(_inject)
    try:
        with torch.no_grad():
            return model(input_ids=ids).logits
    finally:
        handle.remove()


def kl_per_token(noisy_logits: torch.Tensor, p_base: torch.Tensor) -> float:
    """Token-averaged KL(softmax(base) || softmax(noisy)) in nats (chunked fp32)."""
    B, S = noisy_logits.shape[:2]
    total = 0.0
    with torch.no_grad():
        for c in range(0, S, _KL_CHUNK):
            sl, el = c, min(c + _KL_CHUNK, S)
            log_q = F.log_softmax(noisy_logits[:, sl:el].float(), dim=-1)
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
            d = (noisy_logits[:, sl:el].float() - base_logits[:, sl:el].float()).abs()
            total_abs += float(d.sum().item())
            max_abs = max(max_abs, float(d.max().item()))
    return total_abs / (B * S), max_abs


def _scan_layers(model: torch.nn.Module, ids: torch.Tensor, tsm, sel_layers: List[int],
                 p_base: torch.Tensor, base_logits: torch.Tensor,
                 rms: Dict[int, float], top1_base: torch.Tensor) -> List[Dict]:
    """Full per-layer x delta sweep; returns the per-row dict list (prints progress)."""
    per_layer: List[Dict] = []
    for i in sel_layers:
        row: Dict = {
            "layer": i,
            "relative_depth": round(i / max(len(tsm.layers) - 1, 1), 4),
            "hidden_rms_base": round(rms[i], 4),
            "kl_by_delta": {},
            "flip_rate_by_delta": {},
            "dlogit_mean_by_delta": {},
            "dlogit_max_by_delta": {},
        }
        for delta in DELTAS:
            scale = delta * rms[i]
            nl = noisy_forward(model, ids, tsm, i, scale)
            kl = kl_per_token(nl, p_base)
            flips = (nl.float().argmax(dim=-1) != top1_base).float()
            flip_rate = float(flips.mean().item())
            dmean, dmax = logit_stats(nl, base_logits)
            key = f"{delta:g}"
            row["kl_by_delta"][key] = round(kl, 6)
            row["flip_rate_by_delta"][key] = round(flip_rate, 6)
            row["dlogit_mean_by_delta"][key] = round(dmean, 5)
            row["dlogit_max_by_delta"][key] = round(dmax, 4)
            del nl
        per_layer.append(row)
        print("  layer %2d/%d (r=%.2f): " % (i + 1, len(tsm.layers), row["relative_depth"])
              + "   ".join(f"kl({e:g})={row['kl_by_delta'][f'{e:g}']:.4g}" for e in DELTAS))
    return per_layer


def run_model(entry: dict, device: torch.device) -> dict:
    key, path = entry["key"], entry["path"]
    print(f"\n=== at1 [{key}] {path} ({entry['dtype']} on {device}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=entry["dtype"]).to(device)
    model.eval()
    tsm = _text_stack(model)
    n_layers = len(tsm.layers)
    sel_layers = selected_layer_indices(n_layers)

    rec: Dict = {
        "path": path,
        "gpu_used": str(device),
        "dtype": str(entry["dtype"]),
        "n_layers": n_layers,
        "selected_layers": sel_layers,
        "delta_grid": list(DELTAS),
        "protocol": ("base forward (no_grad) vs hook injection of H + delta*RMS(H_base)*randn_like(H) "
                     "at each selected decoder layer; D(s,s0)=delta relative to base-state RMS; "
                     "metrics = token-averaged KL(softmax(base)||softmax(noisy)) nats, top-1 flip rate, "
                     "mean/max |dlogits|"),
    }

    per_layer: Optional[List[Dict]] = None
    acts: Dict[int, np.ndarray] = {}
    rms: Dict[int, float] = {}
    for L in L_FALLBACKS:
        ids = C.make_diverse_tokens(tok, n_target=L).to(device)
        p_base: Optional[torch.Tensor] = None
        try:
            base_logits, rms, acts = base_forward(model, ids, tsm, sel_layers)
            p_base = F.softmax(base_logits.float(), dim=-1)   # cached once (fp32), reused by all noisy runs
            top1_base = base_logits.argmax(dim=-1)
            per_layer = _scan_layers(model, ids, tsm, sel_layers, p_base, base_logits, rms, top1_base)
            rec["L_tokens"] = L
            # activation dump for AT-2 cross-model CKA (fp16, compressed)
            npz_path = os.path.join(C.DATA, f"at1_act_{key}.npz")
            np.savez_compressed(npz_path, **{f"h_l{i}": acts[i].astype(np.float16) for i in sel_layers})
            rec["act_dump"] = f"at1_act_{key}.npz"
            del base_logits, top1_base
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

    rec["per_layer"] = per_layer

    del ids, tok, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    device, free_gib = pick_gpu()
    print(f"at1 analogy-theorem perturbation test — device {device} (free {free_gib:.1f} GiB)")
    print(f"corpus files: {[os.path.basename(p) for p in C.DOC_FILES]}")

    results: dict = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for entry in MODELS:   # all 8 models, sequential loading
        if (results.get(entry["key"]) or {}).get("per_layer"):
            print(f"\n=== at1 [{entry['key']}] already complete — skipping ===")
            continue
        results[entry["key"]] = run_model(entry, device)
        _dump(results)

    print("\n--- at1 summary (top-1 flip rate at delta=1e-02, last selected layer) ---")
    for entry in MODELS:
        key = entry["key"]
        r = results.get(key) or {}
        rows = r.get("per_layer") or []
        if not rows:
            print(f"  {key}: no data (skipped?)")
            continue
        last = rows[-1]
        print(f"  {key}: L={r.get('L_tokens')} layers={r['n_layers']} sel={r.get('selected_layers')} "
              f"-> last layer {last['layer']}: flip(1e-02)={last['flip_rate_by_delta'].get('0.01', 'na')}, "
              f"kl(1e-02)={last['kl_by_delta'].get('0.01', 'na')}")


if __name__ == "__main__":
    main()
