"""Exp9 — hidden-state perturbation / layer-sensitivity scan («attractor test»).

For each decoder layer l and noise level eps we replace that layer's output with
H + eps*randn_like(H) during ONE forward pass (a PyTorch forward hook that
returns a tensor replaces the module output — no custom forward, no weight
surgery), then measure how much the final logits move relative to a clean base
run: token-averaged KL(softmax(base_logits) || softmax(noisy_logits)).

Interpretation (README): a sharp rise of KL at some layer = sensitivity «neck»
(downstream representations depend critically on that layer's output); a flat
profile across depth = no single critical layer. Compare the neck against the
PR-profile neck from Exp5 when folding results into PAPER.md.

Models: gemma-4-e2b, qwen2.5-0.5b, tinyllama-1.1b — all bf16 on physical GPU 1
(pick_gpu prefers cuda:1 with >= 14 GiB free; GPU 0 is reserved for the user).
Context = diverse real-doc corpus (tvu_common.make_diverse_tokens) truncated to
1024 tokens, fallback 512 -> 256 on OOM. All forwards under torch.no_grad() —
no optimizer, no training loop, frozen weights only.

Memory hardening for the 16 GiB card: KL is computed in fp32 chunks of
_KL_CHUNK tokens (caps transient at ~0.5 GiB instead of ~3 GiB for full [S,V]
fp32 copies), and the OOM fallback covers the noisy scan too — an OOM mid-scan
restarts the model at a shorter context. The bat redirects stdout to
data/logs/exp9_run.log (tail is printed at the end).

Writes data/data/exp9_hidden_perturbation_metrics.json after EACH model (crash-safe),
merging into an existing file so reruns of other experiments cannot wipe it.

Run (из корня проекта, любой python с torch/transformers>=5): <python> g4_dynamics_attractor\run_exp9_hidden_perturbation.py
(или `g4_dynamics_attractor\run_exp9_hidden_perturbation.bat` — сам подставит %PYTHON%)
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

import torch                                               # noqa: E402
from torch.nn import functional as F                       # noqa: E402

import tvu_common as C                                    # noqa: E402  (make_diverse_tokens, WORK)
from cross_common import MODELS                           # noqa: E402
from run_gemma_functional_impact import pick_gpu, get_text_stack  # noqa: E402

WORK_JSON = os.path.join(C.WORK, "exp9_hidden_perturbation_metrics.json")

SELECTED_MODELS = ["gemma-4-e2b", "qwen2.5-0.5b", "tinyllama-1.1b"]  # interesting model first
EPSILONS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)         # spec values
L_FALLBACKS: Tuple[int, ...] = (1024, 512, 256)          # spec: context OOM fallback
_KL_CHUNK = 256            # tokens per fp32 KL chunk (~0.5 GiB transient cap at vocab 262k)


def _dump(results: dict) -> None:
    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"  (saved -> {WORK_JSON})")


def base_forward(model: torch.nn.Module, ids: torch.Tensor, tsm) -> Tuple[torch.Tensor, List[float]]:
    """Clean forward; returns (logits, per-layer output RMS).

    Per-layer hidden-state RMS is captured via temporary forward hooks that keep
    only the fp32 scalars — no reliance on wrapper-class output_hidden_states
    (multimodal wrappers report hidden states differently), and nothing extra
    stays in VRAM."""
    rms: List[float] = []

    def _rec(_mod, _inp, out):
        rms.append(float(out.detach().float().pow(2).mean().sqrt()))

    handles = [layer.register_forward_hook(_rec) for layer in tsm.layers]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
    if len(rms) != len(tsm.layers):
        raise RuntimeError(f"hook count {len(rms)} != n_layers {len(tsm.layers)}")
    return logits, rms


def noisy_forward(model: torch.nn.Module, ids: torch.Tensor, tsm, layer_idx: int, eps: float) -> torch.Tensor:
    """One forward with layers[layer_idx]'s output replaced by H + eps*randn_like(H).

    A forward hook returning a tensor replaces the module's output — the most
    direct injection mechanism in PyTorch. The hook is removed in `finally` so a
    crash mid-run cannot poison later forwards."""
    layer = tsm.layers[layer_idx]

    def _inject(mod, inp, out):
        return out + eps * torch.randn_like(out)

    handle = layer.register_forward_hook(_inject)
    try:
        with torch.no_grad():
            return model(input_ids=ids).logits
    finally:
        handle.remove()


def kl_per_token(noisy_logits: torch.Tensor, p_base: torch.Tensor) -> float:
    """Token-averaged KL(softmax(base) || softmax(noisy)) in nats.

    Chunked over the sequence: a full [S, V] fp32 copy of a 262k-vocab logit
    tensor is ~1 GiB per transient — three stacked on top of the resident model
    + p_base would OOM a 16 GiB card. Processing _KL_CHUNK tokens at a time
    caps the transient at ~0.5 GiB while keeping fp32 math.

    F.kl_div(log_q, p) computes sum p*(log p - log q) = KL(p || exp(input)), so
    input = log_softmax(noisy), target = the cached p_base."""
    B, S = noisy_logits.shape[:2]
    total = 0.0
    with torch.no_grad():
        for c in range(0, S, _KL_CHUNK):
            sl, el = c, min(c + _KL_CHUNK, S)
            log_q = F.log_softmax(noisy_logits[:, sl:el].float(), dim=-1)
            per_pos = F.kl_div(log_q, p_base[:, sl:el], reduction="none").sum(dim=-1)
            total += float(per_pos.sum().item())
    return total / (B * S)


def _scan_layers(model: torch.nn.Module, ids: torch.Tensor, tsm, n_layers: int,
                 p_base: torch.Tensor, rms: List[float]) -> List[Dict]:
    """Full per-layer x eps sweep; returns the per-row dict list (prints progress)."""
    per_layer: List[Dict] = []
    for i in range(n_layers):
        row: Dict = {"layer": i, "hidden_rms": round(rms[i], 4), "kl_by_eps": {}}
        for eps in EPSILONS:
            nl = noisy_forward(model, ids, tsm, i, eps)
            row["kl_by_eps"][f"{eps:g}"] = round(kl_per_token(nl, p_base), 6)
            del nl
        per_layer.append(row)
        print("  layer %2d/%d: " % (i, n_layers - 1) + "   ".join(
            f"kl({e:g})={row['kl_by_eps'][f'{e:g}']:.4g}" for e in EPSILONS))
    return per_layer


def run_model(entry: dict, device: torch.device) -> dict:
    key, path = entry["key"], entry["path"]
    print(f"\n=== exp9 [{key}] {path} (bf16 on {device}) ===")
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
        "epsilons": list(EPSILONS),
        "L_tokens": None,
        "protocol": ("base forward (no_grad) vs one-forward hook injection of H + eps*randn_like(H) "
                     "at each decoder layer; metric = token-averaged KL(softmax(base)||softmax(noisy)) in nats"),
    }

    per_layer: Optional[List[Dict]] = None
    rms: List[float] = []
    for L in L_FALLBACKS:
        ids = C.make_diverse_tokens(tok, n_target=L).to(device)
        p_base: Optional[torch.Tensor] = None
        try:
            base_logits, rms = base_forward(model, ids, tsm)
            p_base = F.softmax(base_logits.float(), dim=-1)   # cached once (fp32), reused by all noisy runs
            del base_logits
            per_layer = _scan_layers(model, ids, tsm, n_layers, p_base, rms)
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

    max_ep_key = f"{max(EPSILONS):g}"
    neck = max(per_layer, key=lambda r: r["kl_by_eps"][max_ep_key])
    rec["neck_at_max_eps"] = {"layer": neck["layer"], "kl": neck["kl_by_eps"][max_ep_key]}
    rec["per_layer"] = per_layer

    del ids, tok, model
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    device, free_gib = pick_gpu()
    print(f"exp9 hidden-state perturbation — device {device} (free {free_gib:.1f} GiB)")

    results: dict = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    selected = [next(m for m in MODELS if m["key"] == k) for k in SELECTED_MODELS]
    for entry in selected:
        results[entry["key"]] = run_model(entry, device)
        _dump(results)

    print("\n--- exp9 summary (sensitivity neck at eps=1e-02) ---")
    for key in SELECTED_MODELS:
        r = results[key]
        n = r.get("neck_at_max_eps", {})
        print(f"  {key}: L={r.get('L_tokens')} layers={r['n_layers']} -> "
              f"layer {n.get('layer')} (kl={n.get('kl')})")


if __name__ == "__main__":
    main()
