"""exp_attractor_subspace_perturbation.py — Qwen2.5-0.5B neck-L4 attractor-subspace perturbation test.

Protocol (per TZ spec):
    Capture the residual-stream hidden state H at the geometric neck L4 (Exp5: PRc 1.17) via a
    forward hook on one no_grad forward of 512 tokens. Center H over tokens, SVD; take the top-10
    right singular vectors as the principal subspace U_par [D, k]; P_par = V@V.T, P_perp = I - P_par.
    Three injection arms (forward hook on layer L4 returning H + eps*dir, removed in `finally`):
        radial     : dir = eta                (random unit direction, seeded)
        parallel   : dir = P_par @ eta        (component inside the principal subspace)
        orthogonal : dir = P_perp @ eta       (component in the orthogonal complement)
    for eps in {1e-4, 1e-3, 1e-2}. Metric: L2 deviation of the FINAL-layer hidden states vs base
    (Frobenius over all tokens + per-token mean). Bar chart at max eps.

Hypothesis under test: if the neck is a dynamical attractor, off-manifold (orthogonal)
perturbations are contracted by downstream dynamics and persist less than in-subspace ones —
expected ordering at max eps: deviation_orthogonal < deviation_parallel (radial ~ mixture; for a
random unit eta over D=896 with k=10, ||P_par@eta||^2 ~ k/D ~ 1.1%, so radial is mostly off-manifold).

Runtime deviation from spec — documented:
    Spec says bf16. The per-element perturbation magnitude of a unit-spread direction eta is
    ~ eps/sqrt(D) = O(1e-4..1e-2)/30; at eps<=1e-3 that is below bf16's ~0.4% relative resolution
    for hidden-state components of O(1), so small arms would round to zero. Qwen2.5-0.5B in fp32
    is ~2 GB — trivially fits the 16 GiB card, so this script runs the model fp32 on GPU.

No training; inference only. Local sources only (tvu_common.make_diverse_tokens over DOC_FILES).
Artifacts: data/data/attractor_subspace_perturbation_metrics.json + data/attractor_subspace_perturbation.png
"""
from __future__ import annotations

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # g4_dynamics_attractor/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                  # for `import tvu_common`

import tvu_common as C                                   # make_diverse_tokens, text stack helpers  noqa: E402

PARENT = os.path.dirname(TVU)                         # host tree (here cmq_experiment/) with models/
WORK = os.path.join(TVU, "data")
RESULTS = os.path.join(TVU, "data")
MODEL_LOCAL = os.path.join(C.MODELS_DIR, "qwen2.5-0.5b")

NECK_L = 4          # Exp5 PRc neck (1.17) for qwen2.5-0.5b
K_SUBSPACE = 10     # spec: top-10 principal subspace
N_TOKENS = 512      # spec: batch of 512 tokens
EPSILONS = (1e-4, 1e-3, 1e-2)   # exp9 scale
MIN_FREE_GIB = 8.0
ETA_SEED = 7
WORK_JSON = os.path.join(WORK, "attractor_subspace_perturbation_metrics.json")
PNG_PATH = os.path.join(RESULTS, "attractor_subspace_perturbation.png")

log = logging.getLogger("attractor_subspace")


def pick_gpu() -> tuple[torch.device, float]:
    for idx in (1, 0):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            continue
        free_gib = free_bytes / (1024 ** 3)
        if free_gib >= MIN_FREE_GIB:
            print(f"[device] GPU {idx}: {free_gib:.1f} GiB free -> selected")
            return torch.device("cuda", idx), free_gib
    for idx in (0, 1):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
            print(f"[device] GPU {idx}: only {free_bytes / (1024 ** 3):.1f} GiB free")
        except Exception as e:
            print(f"[device] GPU {idx}: not available ({e})")
    raise SystemExit("ABORT: neither GPU has >= 8 GiB free VRAM. Close LM Studio (tray!) and rerun.")


def text_stack_module(model: torch.nn.Module) -> torch.nn.Module:
    """Standard causal LM: model.model (has .layers + .embed_tokens)."""
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "layers") and hasattr(m, "embed_tokens"):
        return m
    raise AttributeError(f"no text stack found on {type(model).__name__}")


def base_forward(model: torch.nn.Module, tsm, ids: torch.Tensor) -> dict:
    """One clean no_grad forward; returns H at the neck (level NECK_L+1 in HF's 0-based
    hidden_states tuple) and the final-layer state."""
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    return {"neck": hs[NECK_L + 1].detach().clone(),   # [B, T, D] after block L4
            "final": hs[-1].detach().clone()}


def subspace_from_h(h_neck: torch.Tensor) -> dict:
    """Center over tokens, SVD in float64 on CPU (exact), return the top-K right singular
    vectors V_par [D, K] as an fp32 GPU tensor + energy facts."""
    Hc = h_neck.reshape(-1, h_neck.size(-1)).detach().float().cpu()   # [T, D]
    Hc -= Hc.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Hc.double(), full_matrices=False)      # fp64 for exactness
    v_par = Vh[:K_SUBSPACE].t().float()                                # [D, K] right vectors
    energy_frac_top_k = float((S[:K_SUBSPACE] ** 2).sum() / (S ** 2).sum())
    return {"v_par": v_par, "energy_frac_top_k": round(energy_frac_top_k, 4),
            "top_singular_values": [round(float(x), 4) for x in S[:K_SUBSPACE]]}


def run_arm(model: torch.nn.Module, tsm, ids: torch.Tensor, h_neck_base: torch.Tensor,
           h_final_base: torch.Tensor, direction: torch.Tensor, eps: float) -> dict:
    """One forward with layer NECK_L's output replaced by H0 + eps*direction. The hook is
    removed in `finally` so a crash mid-run cannot poison later forwards (exp9 pattern)."""
    layer = tsm.layers[NECK_L]

    def _inject(_mod, _inp, out):
        return h_neck_base + eps * direction

    handle = layer.register_forward_hook(_inject)
    try:
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        final_pert = hs[-1].detach()
    finally:
        handle.remove()

    diff = (final_pert - h_final_base).float()
    dev_fro = float(diff.norm())                                   # L2 over all tokens x dims
    per_token = float(diff.pow(2).sum(dim=-1).sqrt().mean())      # mean per-token L2
    return {"deviation_frobenius": round(dev_fro, 6), "per_token_mean_l2": round(per_token, 6)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    device, free_gib = pick_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_LOCAL)
    # fp32 on GPU: bf16 cannot resolve eps<=1e-3 unit-spread injections (see docstring)
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL, dtype=torch.float32).to(device)
    model.eval()

    tsm = text_stack_module(model)
    ids = C.make_diverse_tokens(tok, n_target=N_TOKENS).to(device)

    log.info("base forward (512 tokens, fp32 %s)", device)
    base = base_forward(model, tsm, ids)
    h_neck_base = base["neck"]
    h_final_base = base["final"]

    sub = subspace_from_h(h_neck_base)
    v_par_dev = sub["v_par"].to(device).contiguous()               # [D, K] fp32 on GPU

    rng = np.random.default_rng(ETA_SEED)
    D = h_neck_base.size(-1)
    eta = torch.from_numpy(rng.standard_normal(D)).float().to(device)
    eta /= eta.norm()                                              # unit direction
    coeffs = v_par_dev.t() @ eta                                  # [K] coordinates of eta in subspace (V^T@eta)
    p_par_eta = v_par_dev @ coeffs                                # [D] component in subspace (V@(V^T@eta))
    p_perp_eta = eta - p_par_eta                                   # [D] orthogonal complement

    arms = {
        "radial":     {"direction_norm": round(float(eta.norm()), 4)},
        "parallel":   {"direction_norm": round(float(p_par_eta.norm()), 4),
                       "note": "||P_par@eta|| ~ sqrt(k/D) for random eta — the in-subspace share"},
        "orthogonal": {"direction_norm": round(float(p_perp_eta.norm()), 4)},
    }
    dirs = {"radial": eta, "parallel": p_par_eta.contiguous(), "orthogonal": p_perp_eta.contiguous()}

    sweep: list[dict] = []
    for arm_name in ("radial", "parallel", "orthogonal"):
        for eps in EPSILONS:
            log.info("arm=%s eps=%.0e", arm_name, eps)
            res = run_arm(model, tsm, ids, h_neck_base, h_final_base, dirs[arm_name], eps)
            sweep.append({"arm": arm_name, "epsilon": eps, **res})

    max_eps_results = {r["arm"]: r for r in sweep if r["epsilon"] == EPSILONS[-1]}
    dev_par = max_eps_results["parallel"]["deviation_frobenius"]
    dev_orth = max_eps_results["orthogonal"]["deviation_frobenius"]
    verdict = (f"At eps={EPSILONS[-1]:g}: deviation_parallel={dev_par:.4f} vs "
               f"deviation_orthogonal={dev_orth:.4f}. Attractor prediction: orthogonal contracts "
               f"more than parallel (dev_orth < dev_par).")
    if dev_orth < dev_par:
        verdict += " -> consistent with the attractor hypothesis."
    else:
        verdict += " -> NOT consistent with a contracting attractor at this neck; report as negative."

    result = {
        "model": "qwen2.5-0.5b",
        "path": MODEL_LOCAL,
        "device": str(device),
        "gpu_free_gib_at_start": round(free_gib, 1),
        "runtime_note": ("fp32 on GPU (spec said bf16): eps<=1e-3 unit-spread injections are below "
                         "bf16's ~0.4% relative resolution and would round to zero; fp32 0.5B ~ 2 GB fits easily"),
        "neck_layer": NECK_L,
        "k_subspace": K_SUBSPACE,
        "n_tokens": N_TOKENS,
        "eta_seed": ETA_SEED,
        "subspace": {k: v for k, v in sub.items() if k != "v_par"},
        "arms_facts": arms,
        "sweep": sweep,
        "verdict_note": verdict,
    }

    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("saved -> %s", WORK_JSON)

    # ---------- bar chart at max eps ----------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    arm_order = ("radial", "parallel", "orthogonal")
    vals = [max_eps_results[a]["deviation_frobenius"] for a in arm_order]
    colors = ["tab:grey", "tab:blue", "tab:red"]
    ax.bar(range(3), vals, color=colors, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{a}\n(dir norm {arms[a]['direction_norm']:.3f})" for a in arm_order])
    ax.set_ylabel(r"L2 deviation of final hidden states vs base ($\varepsilon$ = %.0g)" % EPSILONS[-1])
    ax.set_title(f"Qwen2.5-0.5B, neck L{NECK_L}: perturbation arms in the top-{K_SUBSPACE} subspace "
                 f"(energy {sub['energy_frac_top_k']:.1%})")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    log.info("plot saved -> %s", PNG_PATH)

    print("\n--- attractor subspace perturbation (qwen2.5-0.5b, neck L%d) ---" % NECK_L)
    print(f"  top-{K_SUBSPACE} energy fraction = {sub['energy_frac_top_k']:.4f}")
    for r in sweep:
        print(f"  arm={r['arm']:<11} eps={r['epsilon']:<6g} dev_fro={r['deviation_frobenius']:.5f} "
              f"per_token={r['per_token_mean_l2']:.5f}")
    print(" ", verdict)


if __name__ == "__main__":
    main()
