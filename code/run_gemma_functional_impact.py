"""Functional verification of the H1 dominance inversion on Gemma-4-E2B.

Context (PAPER.md, H1 observation 3): in gemma-4-e2b the shared-matrix gradient
norms INVERT — ||G_in|| = 664 vs ||G_out|| = 13.97 (ratio ~47.5), while every
other tied model is output-dominant (3.8-7x). Hypothesis under test: the
inversion is an artifact of Gemma-4's scaled text embedding —
Gemma4TextScaledWordEmbedding.forward multiplies the lookup by
embed_scale = sqrt(hidden_size) = sqrt(1536) ~= 39.2, and backprop multiplies
the input-path gradient w.r.t. W by that same factor (chain rule through
h0 = s * E(x)).

The experiment has two parts:

PART B — scale-stripped gradient comparison (THE DISCRIMINATOR, bf16 on GPU):
  Replace tsm.embed_tokens with a plain nn.Embedding sharing the SAME weight
  tensor (forward without the gain), recompute BOTH G_in' and G_out' (G_out'
  differs because base_hidden changes when h0 is unscaled — nonlinearities).
  Decision rule:
    ||G_in'|| / ||G_out'|| < 1   -> inversion fully a scaling artifact
    ~ 1-2                        -> mostly artifact + mild residual input bias
    >> raw ratio (47)            -> NOT an artifact, structural phenomenon

PART A — functional impact sweep at epsilons 1e-5 / 1e-4 / 1e-3 (user-specified
  test, fp32 on CPU; bf16 cannot represent such perturbations and fp32 weights
  ~18.5 GB do not fit any single GPU): base loss with labels, then perturb the shared wte tensor by
  +eps * (G_in/||G_in||) and +eps * (G_out/||G_out||) separately, measure
  Delta-loss each way, restore. NOTE: at small epsilon, Delta-L ~= eps*||G|| to
  first order, so the Delta-ratio reproduces the norm ratio by construction —
  Part A confirms the functional steepness but does NOT discriminate; the
  interpretation note in the JSON states this explicitly.

Also reported: architectural facts of embed_tokens_per_layer (PLE table, scale
sqrt(256) = 16) — Gemma-4 feeds a SECOND scaled per-layer embedding into every
decoder layer. Its GRADIENT is deliberately NOT measured: the PLE weight has
shape [vocab=262144, 35*256=8960] so its bf16 gradient buffer alone is ~4.7 GB
(fp32 ~9.4 GB), which does not fit a single card / modest RAM alongside the
full model. PLE is kept out of the grad target set (matching exp7's protocol);
g_in/g_out on the tied matrix are unaffected and numerically identical to exp7.

GPU selection: prefer cuda:1 (project convention), fall back to cuda:0 if it
lacks >= 14 GiB free VRAM. Aborts with a clear message if neither card has
enough (e.g., LM Studio still running).

Writes data/data/gemma_functional_impact_metrics.json incrementally (Part B saved
before Part A starts, so an interruption keeps the discriminating result).

Run (из cross_model/ или из корня проекта): <python с transformers>=5> [cross_model\]run_gemma_functional_impact.py [--skip-part-a]
(или `cross_model\run_gemma_functional_impact.bat`)
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
from torch.nn import functional as F

from cross_common import GEMMA_LOCAL  # noqa: E402
import tvu_common as C                # noqa: E402
from run_exp7_gradient_interference import (  # noqa: E402
    get_wte_weight, get_lm_head, g_to_np, norm_np, cos_np,
    manual_ce_loss, set_targets, clear_grads, restore_grads,
)

N_TOKENS = 128                    # same protocol as exp7 / original exp1
EPSILONS = (1e-5, 1e-4, 1e-3)     # user-specified sweep of small perturbations
MIN_FREE_GIB = 14.0               # bf16 Gemma peak (weights ~9.5 GB + activations/KV)

WORK_JSON = os.path.join(C.WORK, "gemma_functional_impact_metrics.json")


def get_text_stack(model):
    """Resolve the module that actually owns embed_tokens and decoder layers.

    Plain causal models: model.model (base). Gemma-4 is different: the top
    wrapper's .model is a multimodal shell with NO embed_tokens attribute;
    the real text stack lives at model.language_model.model. Found
    generically by attribute signature instead of assuming a fixed path.
    """
    for mod in model.modules():
        if hasattr(mod, "layers") and hasattr(mod, "embed_tokens"):
            return mod
    raise AttributeError("text stack (module with .layers + .embed_tokens) not found")


# ---------- device selection ---------------------------------------------------

def pick_gpu():
    """Prefer cuda:1; fall back to cuda:0 if it lacks enough free VRAM."""
    for idx in (1, 0):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            continue
        free_gib = free_bytes / (1024 ** 3)
        if free_gib >= MIN_FREE_GIB:
            print(f"[device] GPU {idx}: {free_gib:.1f} GiB free -> selected")
            return torch.device("cuda", idx), free_gib
    # report and abort
    for idx in (0, 1):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
            print(f"[device] GPU {idx}: only {free_bytes / 1024 ** 3:.1f} GiB free")
        except Exception as e:
            print(f"[device] GPU {idx}: not available ({e})")
    raise SystemExit(
        "ABORT: neither GPU has >= 14 GiB free VRAM. Close LM Studio (or other "
        "GPU consumers) and rerun."
    )


# ---------- Part B protocol (mirrors exp7's G_in / G_out, CPU-or-GPU tensors) --

def compute_gin_gout(model, ids, device):
    """exp7 decomposition on the tied matrix: returns (g_in_np, g_out_np)."""
    import torch.nn as nn

    tsm = get_text_stack(model)
    wte = get_wte_weight(model)
    lm_owner, lmh = get_lm_head(model)
    hidden_size = int(lmh.in_features)
    vocab = int(lmh.out_features)
    # The PLE table (vocab x 35*256, ~4.7 GB in bf16) is deliberately NOT a grad
    # target: its gradient buffer would need another ~4.7 GB that does not fit one
    # 16 GiB card alongside the full multimodal weights + frozen lm_head clone.
    # set_targets() below forces requires_grad off for every param except wte —
    # exactly exp7's protocol — so no PLE grad is allocated and g_in/g_out are
    # numerically identical to exp7's. PLE facts (shape, scale) are still reported.
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
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    g_in = g_to_np(wte.grad)
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


def part_b(model, tok):
    """Scale-stripped comparison. Returns dict of metrics."""
    device = next(model.parameters()).device
    tsm = get_text_stack(model)
    emb_mod = tsm.embed_tokens

    s = float(emb_mod.scalar_embed_scale) if hasattr(emb_mod, "scalar_embed_scale") \
        else float(emb_mod.embed_scale.detach().float())
    wte = get_wte_weight(model)
    lmh = get_lm_head(model)[1].weight
    tie_ok = (wte.data_ptr() == lmh.data_ptr())

    ple_mod = getattr(tsm, "embed_tokens_per_layer", None)
    ple_facts = None
    if ple_mod is not None:
        ple_facts = {
            "exists": True,
            "weight_shape": list(ple_mod.weight.shape),
            "params_millions": round(ple_mod.weight.numel() / 1e6, 1),
            "embed_scale": float(ple_mod.scalar_embed_scale) if hasattr(ple_mod, "scalar_embed_scale") else None,
        }

    ids = C.make_diverse_tokens(tok, n_target=N_TOKENS).to(device)

    print("[partB] variant 1: with scale (as shipped)")
    v1 = compute_gin_gout(model, ids, device)
    n_in1, n_out1 = norm_np(v1["g_in"]), norm_np(v1["g_out"])

    # --- swap in a plain nn.Embedding sharing the SAME weight tensor
    print("[partB] variant 2: scale stripped (plain Embedding, shared W)")
    import torch.nn as nn
    orig_emb = tsm.embed_tokens
    plain = nn.Embedding(emb_mod.num_embeddings, emb_mod.embedding_dim,
                         padding_idx=emb_mod.padding_idx)
    plain.weight = emb_mod.weight  # share the Parameter — same matrix, no gain
    tsm.embed_tokens = plain
    try:
        v2 = compute_gin_gout(model, ids, device)
    finally:
        tsm.embed_tokens = orig_emb
        clear_grads(model)

    n_in2, n_out2 = norm_np(v2["g_in"]), norm_np(v2["g_out"])
    ratio_raw = n_in1 / n_out1
    ratio_stripped = n_in2 / n_out2
    implied_ratio = (n_in1 / s) / n_out1

    if ratio_stripped < 1.0:
        decision = ("Scenario B: inversion FULLY a scaling artifact — with the "
                    "gain removed, the output path dominates again.")
    elif ratio_stripped <= 2.0:
        decision = (f"Mostly a scaling artifact + mild residual input bias "
                    f"(stripped ratio {ratio_stripped:.2f} vs raw {ratio_raw:.1f}).")
    else:
        decision = ("NOT explained by the embedding gain — structural "
                    "'input tyranny' survives scale removal.")

    return {
        "architectural_facts": {
            "embed_scale_s": round(s, 4),
            "s_formula": "sqrt(hidden_size) = sqrt(1536)",
            "hidden_size": int(emb_mod.embedding_dim),
            "vocab_size": int(emb_mod.num_embeddings),
            "tie_confirmed_same_data_ptr": bool(tie_ok),
            "ple": ple_facts,
        },
        "with_scale": {
            "norm_gin": round(n_in1, 3), "norm_gout": round(n_out1, 3),
            "ratio_gin_over_gout": round(ratio_raw, 3),
            "cos_gin_vs_gout": round(cos_np(v1["g_in"].ravel(), v1["g_out"].ravel()), 4),
            "ple_grad_note": ("PLE gradient not measured: its ~4.7 GB bf16 buffer does "
                              "not fit one 16 GiB card alongside the full multimodal "
                              "weights; PLE kept out of the grad target set (exp7 protocol)"),
        },
        "scale_stripped": {
            "norm_gin_p": round(n_in2, 3), "norm_gout_p": round(n_out2, 3),
            "ratio_gin_over_gout": round(ratio_stripped, 3),
            "note": ("base_hidden changes when h0 is unscaled (nonlinearities: "
                     "attention softmax, tanh-gelu MLP, PLE context projection) — "
                     "G_out' recomputed on the stripped forward"),
        },
        "implied_by_division_only": {
            "norm_gin_over_s": round(n_in1 / s, 3),
            "vs_norm_gout": round(n_out1, 3),
            "ratio": round(implied_ratio, 3),
            "note": "crude check: ||G_in||/s vs ||G_out|| (ignores nonlinear reflow)",
        },
        "decision_rule_result": decision,
    }


# ---------- Part A (fp32 CPU) ---------------------------------------------------

def part_a():
    """User-specified functional impact test (epsilon sweep), fp32 on CPU.

    NOTE for tied models: lm_head IS wte (same Parameter), so "input" and
    "output" perturbations are two directions in ONE shared matrix; the exact
    first-order Delta-L is eps * <G_total, d> with G_total = G_in + G_out. The
    reported first_order_prediction uses eps * ||G_dir|| — numerically identical
    here because ||G_in|| >> ||G_out|| and cos(G_in, G_out) ~ 0.02.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[partA] loading Gemma fp32 on CPU (~18.5 GB RAM)...")
    tok = AutoTokenizer.from_pretrained(GEMMA_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL, dtype=torch.float32)
    model.eval()

    try:
        ids = C.make_diverse_tokens(tok, n_target=N_TOKENS).to("cpu")
        wte = get_wte_weight(model)

        print("[partA] base loss (labels)")
        out0 = model(input_ids=ids, labels=ids)
        base_loss = float(out0.loss)
        del out0

        # gradient directions in fp32 on CPU (same protocol as exp7 Part B v1)
        print("[partA] G_in / G_out directions (fp32 CPU backward)")
        res = compute_gin_gout(model, ids, torch.device("cpu"))
        g_in_t = torch.from_numpy(res["g_in"].astype(np.float32)).reshape(wte.shape)
        g_out_t = torch.from_numpy(res["g_out"].astype(np.float32)).reshape(wte.shape)
        n_in = norm_np(g_in_t.numpy().astype(np.float64))
        n_out = norm_np(g_out_t.numpy().astype(np.float64))
        d_in = g_in_t / n_in
        d_out = g_out_t / n_out

        def perturb_loss(direction, eps):
            w_orig = wte.data.clone()
            try:
                with torch.no_grad():
                    wte.data.add_(eps * direction)
                return float(model(input_ids=ids, labels=ids).loss)
            finally:
                with torch.no_grad():
                    wte.data.copy_(w_orig)
                assert torch.equal(wte.data, w_orig), "weight restore failed"

        sweeps = []
        for eps in EPSILONS:
            print(f"[partA] perturbed forwards (eps={eps})")
            loss_in = perturb_loss(d_in, eps)
            loss_out = perturb_loss(d_out, eps)
            dl_in = loss_in - base_loss
            dl_out = loss_out - base_loss
            sweeps.append({
                "epsilon": eps,
                "loss_after_gin_direction_perturbation": round(loss_in, 6),
                "loss_after_gout_direction_perturbation": round(loss_out, 6),
                "delta_loss_in": round(dl_in, 6),
                "delta_loss_out": round(dl_out, 6),
                "ratio_delta_in_over_out": round(dl_in / dl_out, 3) if dl_out != 0 else None,
                "first_order_prediction_delta_in": round(eps * n_in, 6),
                "first_order_prediction_delta_out": round(eps * n_out, 6),
            })

        return {
            "base_loss": round(base_loss, 6),
            "epsilon_sweep": sweeps,
            "norm_gin_fp32_cpu": round(float(n_in), 3),
            "norm_gout_fp32_cpu": round(float(n_out), 3),
        }
    finally:
        del tok, model
        gc.collect()


# ---------- driver ----------------------------------------------------------------

def main():
    skip_a = "--skip-part-a" in sys.argv

    results = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    def save():
        with open(WORK_JSON, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"  (saved -> {WORK_JSON})")

    # ---------- PART B on GPU ----------
    device, free_gib = pick_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL, dtype=torch.bfloat16).to(device)
    model.eval()
    try:
        print(f"[partB] running scale-stripped gradient comparison on {device}")
        results["gemma-4-e2b"] = {
            "path": GEMMA_LOCAL,
            "gpu_used": str(device),
            "gpu_free_gib_at_start": round(free_gib, 1),
            **part_b(model, tok),
            "part_a_status": "pending" if not skip_a else "skipped_by_flag",
        }
    finally:
        del tok, model
        gc.collect()
        torch.cuda.empty_cache()
    save()

    results["interpretation_notes"] = [
        ("Part A (epsilon sweep 1e-5..1e-3) is first-order: Delta-L ~= eps*<G_total,d>, "
         "so for the tied matrix its ratio reproduces the gradient norm ratio by "
         "construction — it confirms functional steepness but does NOT discriminate "
         "artifact vs structure. The discriminator is Part B (scale-stripped norms; "
         "see decision_rule_result and implied_by_division_only)."),
        ("Gemma-4 feeds a SECOND scaled embedding table (PLE, scale 16, 2.35B params) "
         "into every decoder layer — the input side gets two scaled channels, not one."),
    ]

    # ---------- PART A on CPU fp32 ----------
    if not skip_a:
        try:
            pa = part_a()
            results["gemma-4-e2b"]["part_a"] = pa
            results["gemma-4-e2b"]["part_a_status"] = "done"
        except Exception as e:
            results["gemma-4-e2b"]["part_a_status"] = f"failed ({type(e).__name__}: {e})"
            print(f"[partA] FAILED: {e}")
    save()

    g = results.get("gemma-4-e2b", {})
    ws, ss = g.get("with_scale", {}), g.get("scale_stripped", {})
    if ws and ss:
        print("\n--- functional impact summary (gemma-4-e2b) ---")
        print(f"  embed_scale s = {g['architectural_facts']['embed_scale_s']}")
        print(f"  raw ratio      ||G_in||/||G_out||    = {ws.get('ratio_gin_over_gout')}")
        print(f"  stripped ratio ||G_in'||/||G_out'|| = {ss.get('ratio_gin_over_gout')}")
        print(f"  => {g.get('decision_rule_result', '')}")


if __name__ == "__main__":
    main()
