"""Experiment 1 — Gradient interference on the shared embedding matrix (the ROOT).

Hypothesis: in a TIED model, gradients reaching the one shared matrix W from its
two roles (input embed_tokens / output lm_head) conflict; in an UNTIED model the
matrices are separate and specialize freely.

Method (DIRECT decomposition into components — no proxies):
  G_total — one forward+backward with labels on the tied model (both paths).
  G_in    — same forward+backward, but lm_head temporarily replaced by a FROZEN
            clone of W: only the input-path gradient flows into wte.
  G_out   — base-model hidden states (no grad) + fresh Parameter initialized to a
            copy of W as unembedding; CE loss with HF shift convention →
            output-path-only gradient.
  By backprop linearity through the shared tensor, G_total = G_in + G_out holds
  exactly, so cos(G_total, G_in/G_out) characterizes the interference geometry.

"Vs internal layer-0 updates" (TZ §2): the wte grad (vocab x hidden) and the
layer-0 q/k/v grads are different-sized objects, so per TZ ("или через прокси")
we use an activation-level proxy: for each context token i, compare how its
layer-0 query/key/value states get perturbed by the embedding update
(W_q dE_i) vs by the layer-0 weight updates (dQ e_i), concatenated over all 128
tokens.

Untied model (TinyLlama): single backward; wte.grad = pure input path,
lm_head.grad = output matrix. Metrics: cos(G_emb, G_lm) as the separation
baseline, ||G_lm||/||G_emb||, same activation-level proxy for G_emb.

Plus v1-style anisotropy of both embedding matrices (raw and centered mean|cos|
over 2000 random words, seed=42) — the "Anisotropy Score" for Tied vs Untied.

Outputs: data/01_gradient_interference.png + data/data/exp1_metrics.json
"""

import os

import matplotlib

matplotlib.use("Agg")

import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # g1_gradient_interference/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                 # for `import tvu_common`

import tvu_common as C                                  # noqa: E402


def set_targets(model, targets):
    for p in model.parameters():
        p.requires_grad_(p in targets)


def clear_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def restore_grads(model):
    for p in model.parameters():
        p.requires_grad_(True)


def qkv0_effect_cos(model, emb_grad_full, input_ids):
    """Activation-level proxy (TZ 'или через прокси'): cosine between the
    perturbation of layer-0 q/k/v states caused by the embedding-matrix update
    rows dE_i = W_q @ dE_i  vs  the perturbation caused by the layer-0 weight
    updates dQ e_i, concatenated over all context tokens (both sides become
    vectors of size n_tokens * hidden)."""
    l0 = model.model.layers[0].self_attn
    E_cpu = (model.model.embed_tokens.weight.detach().float()
             .cpu().numpy())
    rows = input_ids.squeeze(0).cpu().numpy()
    e_rows = torch.from_numpy(E_cpu[rows].astype(np.float32)).to(C.DEVICE)   # [n_tok, hidden]
    gE_rows = torch.from_numpy(emb_grad_full[rows].astype(np.float32)).to(C.DEVICE)  # [n_tok, hidden]

    parts_a, parts_b = [], []
    for name in ("q_proj", "k_proj", "v_proj"):
        W = getattr(l0, name).weight.detach().float()              # [out_dim, hidden]
        Gw = getattr(l0, name).weight.grad.detach().float()        # same shape
        a = gE_rows @ W.T     # dE_i then W_q  -> perturbation from emb update
        b = e_rows @ Gw.T     # e_i then dQ    -> perturbation from qkv update
        parts_a.append(a.reshape(-1))
        parts_b.append(b.reshape(-1))
    va = torch.cat(parts_a).cpu().numpy().astype(np.float64)
    vb = torch.cat(parts_b).cpu().numpy().astype(np.float64)
    return C.cosine_similarity(va, vb)


def run_tied(model, input_ids):
    """Qwen2.5-0.5B (tie_word_embeddings=True)."""
    import torch.nn as nn

    hidden_size = model.config.hidden_size
    vocab = model.config.vocab_size
    wte = model.model.embed_tokens.weight
    l0attn = model.model.layers[0].self_attn
    targets = {wte}
    for name in ("q_proj", "k_proj", "v_proj"):
        targets.add(getattr(l0attn, name).weight)

    # --- G_total: both paths at once (normal forward+backward with labels)
    set_targets(model, targets)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_total = wte.grad.clone().detach().cpu().numpy().astype(np.float64)

    # --- G_in: output path frozen (lm_head swapped for a frozen clone of W)
    clear_grads(model)
    orig_lm = model.lm_head
    frozen = nn.Linear(hidden_size, vocab, bias=False).to(C.DEVICE)
    with torch.no_grad():
        frozen.weight.copy_(wte.data)
    for p in frozen.parameters():
        p.requires_grad_(False)
    model.lm_head = frozen
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_in = wte.grad.clone().detach().cpu().numpy().astype(np.float64)
    cos_emb_vs_qkv0 = qkv0_effect_cos(model, g_in, input_ids)  # while l0 grads alive
    model.lm_head = orig_lm

    # --- G_out: input path inert (no-grad base hidden states + fresh Parameter W')
    clear_grads(model)
    with torch.no_grad():
        base_hidden = model.model(input_ids=input_ids).last_hidden_state
    w_out = nn.Parameter(wte.data.clone().to(C.DEVICE))
    logits = F.linear(base_hidden, w_out)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, vocab), shift_labels.reshape(-1)
    )
    loss.backward()
    g_out = w_out.grad.clone().detach().cpu().numpy().astype(np.float64)

    restore_grads(model)
    return {
        "cos_gin_vs_gout": C.cosine_similarity(g_in.ravel(), g_out.ravel()),
        "norm_ratio_gout_over_gin": C.norm_of(g_out) / C.norm_of(g_in),
        "norm_gin": C.norm_of(g_in),
        "norm_gout": C.norm_of(g_out),
        "cos_gtotal_vs_gin": C.cosine_similarity(g_total.ravel(), g_in.ravel()),
        "cos_gtotal_vs_gout": C.cosine_similarity(g_total.ravel(), g_out.ravel()),
        "cos_embupdate_vs_qkv0_effect_proxy": cos_emb_vs_qkv0,
    }


def run_untied(model, input_ids):
    """TinyLlama-1.1B (tie_word_embeddings=False)."""
    wte = model.model.embed_tokens.weight
    lmh = model.lm_head.weight
    l0attn = model.model.layers[0].self_attn
    targets = {wte, lmh}
    for name in ("q_proj", "k_proj", "v_proj"):
        targets.add(getattr(l0attn, name).weight)

    set_targets(model, targets)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_emb = wte.grad.clone().detach().cpu().numpy().astype(np.float64)
    g_lm = lmh.grad.clone().detach().cpu().numpy().astype(np.float64)
    cos_emb_vs_qkv0 = qkv0_effect_cos(model, g_emb, input_ids)

    restore_grads(model)
    return {
        "cos_gemb_vs_glm": C.cosine_similarity(g_emb.ravel(), g_lm.ravel()),
        "norm_ratio_glm_over_gemb": C.norm_of(g_lm) / C.norm_of(g_emb),
        "norm_gemb": C.norm_of(g_emb),
        "norm_glm": C.norm_of(g_lm),
        "cos_embupdate_vs_qkv0_effect_proxy": cos_emb_vs_qkv0,
    }


def main():
    import gc

    results = {}

    print(f"[A] TIED model: {C.MODEL_A_PATH}")
    tok_a, mod_a = C.load_model(C.MODEL_A_PATH)
    ids_a = C.make_diverse_tokens(tok_a, n_target=128)
    r_a = run_tied(mod_a, ids_a)
    raw_a, cen_a = C.embedding_anisotropy(mod_a)
    r_a["anisotropy_raw"] = raw_a
    r_a["anisotropy_centered"] = cen_a
    results[C.NAME_A] = r_a
    print(f"  cos(G_in,G_out) = {r_a['cos_gin_vs_gout']:.4f} | "
          f"norm ratio out/in = {r_a['norm_ratio_gout_over_gin']:.3f} | "
          f"aniso raw/centered = {raw_a:.4f}/{cen_a:.4f}")
    C.free_model(tok_a, mod_a)

    print(f"[B] UNTIED model: {C.MODEL_B_PATH}")
    tok_b, mod_b = C.load_model(C.MODEL_B_PATH)
    ids_b = C.make_diverse_tokens(tok_b, n_target=128)
    r_b = run_untied(mod_b, ids_b)
    raw_b, cen_b = C.embedding_anisotropy(mod_b)
    r_b["anisotropy_raw"] = raw_b
    r_b["anisotropy_centered"] = cen_b
    results[C.NAME_B] = r_b
    print(f"  cos(G_emb,G_lm) = {r_b['cos_gemb_vs_glm']:.4f} | "
          f"norm ratio lm/emb = {r_b['norm_ratio_glm_over_gemb']:.3f} | "
          f"aniso raw/centered = {raw_b:.4f}/{cen_b:.4f}")
    C.free_model(tok_b, mod_b)
    gc.collect()

    # ---- plot: the four headline cosines + annotation box with ratios/anisotropy
    bars = [
        ("Tied Qwen: G_in vs G_out\n(conflict on the shared matrix W)",
         r_a["cos_gin_vs_gout"], "#d64545"),
        ("Tied Qwen: G_total vs G_in\n(total update vs input path)",
         r_a["cos_gtotal_vs_gin"], "#f2a09b"),
        ("Untied TinyLlama: G_emb vs G_lm\n(separate matrices — baseline)",
         r_b["cos_gemb_vs_glm"], "#2e9e5b"),
        ("Untied TinyLlama: emb-update vs qkv_L0 effect (proxy)",
         r_b["cos_embupdate_vs_qkv0_effect_proxy"], "#8fcf9f"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
    for yi, v in zip(y, vals):
        off = 0.03 if abs(v) > 0.9 else (0.02 if v >= 0 else -0.02)
        ax.text(v + (off if v >= 0 else -abs(off)), yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_xlabel("Cosine similarity of update vectors")
    ax.set_title("Experiment 1 — Gradient interference: Tied vs Untied\n"
                 "(G_in/G_out = input/output-path components on the shared matrix)")
    note = (
        f"Tied Qwen:      ||G_out||/||G_in|| = {r_a['norm_ratio_gout_over_gin']:.3f}  "
        f"| cos(G_total,G_out) = {r_a['cos_gtotal_vs_gout']:+.3f}\n"
        f"                anisotropy raw/cen = {r_a['anisotropy_raw']:.4f}/{r_a['anisotropy_centered']:.4f}  "
        f"| emb-vs-qkv0 proxy = {r_a['cos_embupdate_vs_qkv0_effect_proxy']:+.3f}\n"
        f"Untied TinyLlama: ||G_lm||/||G_emb|| = {r_b['norm_ratio_glm_over_gemb']:.3f}  "
        f"| anisotropy raw/cen = {r_b['anisotropy_raw']:.4f}/{r_b['anisotropy_centered']:.4f}"
    )
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#f5f8fc", ec="#c8d2e0"))
    plt.tight_layout()
    out_png = os.path.join(C.RESULTS, "01_gradient_interference.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    C.save_work_json("exp1_metrics.json", results)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
