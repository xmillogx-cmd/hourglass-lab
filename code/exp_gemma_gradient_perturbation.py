"""exp_gemma_gradient_perturbation.py — Gemma-4-E2B gradient-direction functional perturbation sweep.

Protocol (per TZ spec):
    CE loss on a fixed token block -> single-step backward -> G_in (gradient of
    embed_tokens.weight) and G_out (gradient of lm_head.weight, input path inert);
    both normalized to unit Frobenius norm; for eps in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
    and sign in {+1, -1}, temporarily add sign*eps*d to the weight matrix and measure
    Delta-L of the CE loss. Gemma-4 ties wte == lm_head (same Parameter), so both arms
    perturb ONE shared matrix along two different directions — exactly Part A's protocol
    in cross_model/run_gemma_functional_impact.py, whose first-order prediction for arm X
    is eps*||G_X|| (cross term negligible: ||G_in|| >> ||G_out||, cos ~ 0.02).

Runtime deviation from spec — documented:
    Spec says bf16 on GPU. That is physically impossible at these epsilons: the per-element
    perturbation magnitude ~ eps/numel-scale is O(1e-5..1e-3) relative, far below bf16's
    ~0.4% quantization resolution for weights of O(0.01-0.1) — the perturbation would round
    to zero. The Part A precedent ran this exact protocol on fp32 CPU (~18.5 GB RAM); this
    script follows it. No training: one backward pass, then no_grad forwards only.

Context — what this rerun adds vs already-answered goal:
    "Is Gemma's input-gradient dominance a scaling artifact?" is answered by Part B (raw
    ratio 47.15 -> stripped 5.84) and the v3 PLE ablation (5.828 intact vs 4.281 zeroed —
    residual NOT PLE-driven, structural 'input tyranny' survives scale removal). This sweep
    adds a finer eps grid + both signs; per Part A's anomaly_note, first-order linearity does
    not hold in this fp32 regime (Delta-L_in flips sign between 1e-4 and 1e-3), so treat the
    ratio curve as descriptive, not discriminating.

Artifacts: data/data/gemma_gradient_perturbation_metrics.json + data/gemma_gradient_perturbation.png
Local sources only; no downloads; OOM handled by halving the token block and retrying.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # g1_gradient_interference/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
PARENT = os.path.dirname(TVU)                             # host tree (here cmq_experiment/) with models/
sys.path.insert(0, TVU)                                  # for the lazy `import tvu_common` in run()

WORK = os.path.join(TVU, "data")
RESULTS = os.path.join(TVU, "data")

MODELS_DIR = os.environ.get("MODELS_DIR") or os.path.join(PARENT, "models")
if not os.path.isdir(MODELS_DIR):
    raise RuntimeError(
        "model directory not found (looked in %s). Set the MODELS_DIR environment variable." % MODELS_DIR)
GEMMA_LOCAL = os.path.join(MODELS_DIR, "gemma-4-e2b")

EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
SIGNS = (+1.0, -1.0)
N_TOKENS = 128        # Part A precedent; autograd retention is the cost driver
WORK_JSON = os.path.join(WORK, "gemma_gradient_perturbation_metrics.json")
PNG_PATH = os.path.join(RESULTS, "gemma_gradient_perturbation.png")

log = logging.getLogger("gemma_grad_pert")


# ---------- architecture helpers (mirrored from cross_model/run_exp7_gradient_interference.py) ----

def text_stack_module(model: torch.nn.Module) -> torch.nn.Module:
    """The module whose forward(input_ids=...) yields .last_hidden_state and holds the
    transformer layers as a direct `.layers` child. For gemma-4-e2b (Gemma4ForConditional
    Generation) that is `language_model`, whose embed_tokens/layers are DIRECT children."""
    for attr in ("language_model", "text_model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
            return inner          # wrapper with an inner text stack (Llama-style)
        if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
            return m              # direct text model (gemma3/4 style)
    return model.model


def get_wte_weight(model: torch.nn.Module) -> torch.Tensor:
    try:
        return model.model.embed_tokens.weight
    except AttributeError:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Embedding) and name.endswith("embed_tokens") \
                    and "vision" not in name:
                return m.weight
        raise AttributeError(f"no text embed_tokens found on {type(model).__name__}")


def get_lm_head(model: torch.nn.Module):
    """(owner_module, lm_head_linear)."""
    if isinstance(getattr(model, "lm_head", None), torch.nn.Linear):
        return model, model.lm_head
    for name, m in model.named_modules():
        if name.endswith("lm_head") and isinstance(m, torch.nn.Linear):
            cur = model
            for p in name.split(".")[:-1]:
                nxt = getattr(cur, p, None)
                if not hasattr(nxt, "lm_head"):
                    break
                cur = nxt
            return (cur if hasattr(cur, "lm_head") else model), m
    raise AttributeError(f"no lm_head Linear found on {type(model).__name__}")


def g_to_np(g: torch.Tensor) -> np.ndarray:
    """Grad tensor -> CPU numpy. float32 when the matrix is big enough that fp64 would be a
    RAM problem (> 200M elements), else float64."""
    if g.numel() > 200_000_000:
        return g.detach().float().cpu().numpy()
    return g.detach().double().cpu().numpy()


def norm_np(x) -> float:
    x = np.asarray(x).ravel()
    if x.dtype == np.float32:
        return float(np.linalg.norm(x.astype(np.float64)))
    return float(np.linalg.norm(x))


def cos_np(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.dtype == np.float32:
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(float(a @ b) / (da * db))


def set_targets(model: torch.nn.Module, targets: set) -> None:
    for p in model.parameters():
        p.requires_grad_(p in targets)


def clear_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def restore_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(True)


def manual_ce_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """HF shift-convention CE from raw logits (G_out phase only)."""
    vocab = logits.size(-1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.reshape(-1, vocab), shift_labels.reshape(-1))


# ---------- G_in / G_out (exp7 decomposition on the tied matrix) -------------------------------

def compute_gin_gout(model: torch.nn.Module, ids: torch.Tensor, device: torch.device) -> dict:
    """Returns {"g_in": np.ndarray, "g_out": np.ndarray} — gradients of CE w.r.t. the shared
    embedding matrix along the input path (output path frozen) and the output path (input path
    inert). The PLE table is deliberately NOT a grad target (its ~4.7 GB buffer would not fit);
    set_targets() forces requires_grad off for every param except wte — exactly exp7's protocol."""
    import torch.nn as nn

    tsm = text_stack_module(model)
    wte = get_wte_weight(model)
    lm_owner, lmh = get_lm_head(model)
    hidden_size = int(lmh.in_features)
    vocab = int(lmh.out_features)
    targets = {wte}

    # --- G_in: output path frozen (lm_head swapped for a frozen clone of W)
    set_targets(model, targets)
    clear_grads(model)
    orig_lm = lm_owner.lm_head
    frozen = nn.Linear(hidden_size, vocab, bias=False).to(device).to(wte.dtype)
    with torch.no_grad():
        frozen.weight.copy_(wte.data)
    for p in frozen.parameters():
        p.requires_grad_(False)
    lm_owner.lm_head = frozen
    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        g_in = g_to_np(wte.grad)
    finally:
        lm_owner.lm_head = orig_lm

    # --- G_out: input path inert (no-grad hidden states + fresh Parameter W')
    clear_grads(model)
    with torch.no_grad():
        base_hidden = tsm(input_ids=ids).last_hidden_state
    w_out = nn.Parameter(wte.data.clone().to(device))
    logits = F.linear(base_hidden, w_out)
    manual_ce_loss(logits, ids).backward()
    g_out = g_to_np(w_out.grad)

    restore_grads(model)
    return {"g_in": g_in, "g_out": g_out}


# ---------- measurement driver ---------------------------------------------------------------

def run(n_tokens: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import tvu_common as C   # local module (cwd = TVU): make_diverse_tokens over DOC_FILES

    log.info("loading gemma-4-e2b fp32 on CPU (~18.5 GB RAM)...")
    tok = AutoTokenizer.from_pretrained(GEMMA_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL, dtype=torch.float32)
    model.eval()

    try:
        ids = C.make_diverse_tokens(tok, n_target=n_tokens).to("cpu")
        wte = get_wte_weight(model)

        log.info("base CE loss (n=%d)", n_tokens)
        out0 = model(input_ids=ids, labels=ids)
        base_loss = float(out0.loss)
        del out0

        log.info("G_in / G_out directions (fp32 CPU backward)")
        res = compute_gin_gout(model, ids, torch.device("cpu"))
        g_in_t = torch.from_numpy(res["g_in"].astype(np.float32)).reshape(wte.shape)
        g_out_t = torch.from_numpy(res["g_out"].astype(np.float32)).reshape(wte.shape)
        n_in = norm_np(g_in_t.numpy().astype(np.float64))
        n_out = norm_np(g_out_t.numpy().astype(np.float64))
        d_in = (g_in_t / n_in).contiguous()
        d_out = (g_out_t / n_out).contiguous()

        def perturb_loss(direction: torch.Tensor, scale: float) -> float:
            w_orig = wte.data.clone()
            try:
                with torch.no_grad():
                    wte.data.add_(scale * direction)
                return float(model(input_ids=ids, labels=ids).loss)
            finally:
                with torch.no_grad():
                    wte.data.copy_(w_orig)
                assert torch.equal(wte.data, w_orig), "weight restore failed"

        sweeps = []
        for eps in EPSILONS:
            for sign in SIGNS:
                log.info("perturbed forwards (eps=%.0e, sign=%+d)", eps, int(sign))
                loss_in = perturb_loss(d_in, sign * eps)
                loss_out = perturb_loss(d_out, sign * eps)
                sweeps.append({
                    "epsilon": eps,
                    "sign": int(sign),
                    "loss_after_gin_direction_perturbation": round(loss_in, 6),
                    "delta_loss_in": round(loss_in - base_loss, 6),
                    "first_order_prediction_delta_in": round(sign * eps * n_in, 6),
                    "loss_after_gout_direction_perturbation": round(loss_out, 6),
                    "delta_loss_out": round(loss_out - base_loss, 6),
                    "first_order_prediction_delta_out": round(sign * eps * n_out, 6),
                })

        # spec'd ratio: (|dL_in+| + |dL_in-|) / (|dL_out+| + |dL_out-|) per epsilon
        by_eps = {}
        for eps in EPSILONS:
            dl_in = sum(abs(s["delta_loss_in"]) for s in sweeps if s["epsilon"] == eps)
            dl_out = sum(abs(s["delta_loss_out"]) for s in sweeps if s["epsilon"] == eps)
            by_eps[f"{eps:g}"] = round(dl_in / dl_out, 3) if dl_out > 0 else None

        return {
            "n_tokens": n_tokens,
            "base_loss": round(base_loss, 6),
            "norm_gin_fp32_cpu": round(n_in, 3),
            "norm_gout_fp32_cpu": round(n_out, 3),
            "cos_gin_vs_gout": round(cos_np(g_in_t.numpy().astype(np.float64),
                                           g_out_t.numpy().astype(np.float64)), 4),
            "epsilon_sweep": sweeps,
            "ratio_by_epsilon": by_eps,
        }
    finally:
        del tok, model
        gc.collect()


def plot_ratio(by_eps: dict) -> None:
    xs = [float(k) for k in by_eps.keys()]
    ys = [by_eps[k] for k in by_eps.keys()]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(xs, ys, "o-", color="tab:blue", lw=1.8, ms=6)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel(r"perturbation size $\varepsilon$ (unit-norm gradient direction)")
    ax.set_ylabel(r"$(|\Delta L_{in}^{+}|+|\Delta L_{in}^{-}|)/(|\Delta L_{out}^{+}|+|\Delta L_{out}^{-}|)$")
    ax.set_title("Gemma-4-E2B: input/output $\Delta L$ ratio by $\\varepsilon$ (fp32 CPU, tied matrix)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    log.info("plot saved -> %s", PNG_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    payload = None
    n = N_TOKENS
    while True:
        try:
            payload = run(n)
            break
        except MemoryError as e:
            if n <= 32:
                raise
            log.warning("MemoryError at n=%d (%s) — halving token block to %d and retrying",
                        n, e, n // 2)
            n //= 2

    result = {
        "model": "gemma-4-e2b",
        "path": GEMMA_LOCAL,
        "runtime": ("fp32 CPU (~18.5 GB RAM). Spec's bf16-GPU form is physically impossible at "
                    "eps<=1e-3: per-element perturbation magnitude is far below bf16's ~0.4% "
                    "quantization resolution for O(0.01-0.1) weights — follows the Part A "
                    "precedent in cross_model/run_gemma_functional_impact.py."),
        **payload,
        "tied_matrix_note": ("wte == lm_head (same Parameter): both arms perturb ONE shared matrix "
                             "along two directions; first-order prediction for arm X = eps*||G_X|| "
                             "(cross term negligible: ||G_in|| >> ||G_out||, cos ~ 0.02) — Part A's convention."),
        "anomaly_notes": [
            ("Known fp32/CPU regime anomaly (Part A): ||G_in||_fp32 = 5405.7 vs 658.9 in bf16/GPU — "
             "an 8x gap too large for dtype rounding; and the sweep breaks first-order linearity, "
             "with delta_loss_in flipping sign between eps=1e-4 (-0.337) and eps=1e-3 (+0.087), "
             "likely tanh softcap 30 saturation dynamics on O(1e2) scaled-embedding hidden states."),
            ("Therefore the ratio curve here is DESCRIPTIVE, not discriminating: it neither confirms "
             "nor refutes functional steepness in this numerical regime."),
        ],
        "context_note": ("Goal already answered by Part B (raw ||G_in||/||G_out|| = 47.15 -> scale-stripped "
                         "5.84: 'NOT explained by the embedding gain — structural input tyranny survives "
                         "scale removal') and the v3 PLE ablation (stripped ratio 5.828 intact vs 4.281 with "
                         "PLE identity zeroed -> residual NOT PLE-driven). This rerun adds a finer eps grid "
                         "(5 values) and both signs; it is not an independent discriminator."),
    }

    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("saved -> %s", WORK_JSON)

    plot_ratio(result["ratio_by_epsilon"])

    print("\n--- gemma gradient perturbation sweep (fp32 CPU) ---")
    print(f"  base_loss = {result['base_loss']}   n_tokens = {result['n_tokens']}")
    print(f"  ||G_in|| = {result['norm_gin_fp32_cpu']}   ||G_out|| = {result['norm_gout_fp32_cpu']}")
    for k, v in result["ratio_by_epsilon"].items():
        print(f"  eps={k:<8} ratio(|dL_in|/|dL_out|) = {v}")


if __name__ == "__main__":
    main()
