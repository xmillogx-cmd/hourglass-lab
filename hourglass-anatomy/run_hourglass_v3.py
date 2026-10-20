"""
Hourglass Anatomy v3 — follow-up checks on the two ambiguous points from v2.

1. Per-layer effective rank (EXACT full SVD, no randomized PCA):
   participation ratio d_eff = (sum s^2)^2 / sum s^4 and k95 at every
   hidden-state level. Tests whether the hourglass shape (U-shaped collapse)
   holds with a finer metric than top-3 variance, and verifies the odd dip
   seen in v2's TruncatedSVD top-3 curve at layers 22-23.

2. Residual isotropy done properly: project mid-layer activations onto the
   FULL cone subspace (top-k95 right-singular vectors, ~60 dims per v2)
   instead of only top-3; compare residual mean |cos| with the white-noise
   expectation sqrt(2/(pi*d)).

Same local model and same diverse context as v1/v2. CPU-only, fp32.

Outputs (results/):
    03_per_layer_effective_rank.png
    hourglass_summary_v3.json
"""

import gc
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, no plt.show()

import numpy as np
import torch
from matplotlib import pyplot as plt

from run_hourglass import RESULTS, make_contexts

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # repo root
MODEL_PATH = os.path.join(os.environ.get("MODELS_DIR") or os.path.join(ROOT, "models"), "qwen2.5-0.5b")


def load_model_v3():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def k95_of_spectrum(sigma):
    cumvar = np.cumsum((sigma**2) / float(np.sum(sigma**2)))
    return int(np.searchsorted(cumvar, 0.95) + 1)


def participation_ratio(sigma):
    s2 = float(np.sum(sigma**2))
    s4 = float(np.sum(sigma**4))
    return (s2**2) / s4


def mean_abs_cos(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = M / (norms + 1e-8)
    cs = Mn @ Mn.T
    np.fill_diagonal(cs, 0.0)
    return float(np.mean(np.abs(cs)))


def part_a_per_layer(hidden_states):
    print("\n" + "=" * 60)
    print("CHECK A — PER-LAYER EFFECTIVE RANK (exact SVD)")
    print("=" * 60)

    d_effs = []
    k95s = []
    for i, hidden in enumerate(hidden_states):
        Xc = hidden - np.mean(hidden, axis=0)
        sigma = np.linalg.svd(Xc.astype(np.float32), compute_uv=False)
        de = participation_ratio(sigma)
        k95 = k95_of_spectrum(sigma)
        d_effs.append(de)
        k95s.append(k95)
        mark = ""
        if i in (0, num_layers // 2, num_layers - 1):
            mark = "   <-- sampled in v2 console"
        print(f"layer {i:>2}: d_eff = {de:7.1f}   k95 = {k95:>4}{mark}")

    neck_idx = int(np.argmin(d_effs))  # most compressed layer by d_eff
    print(
        f"MOST COMPRESSED (min d_eff): layer {neck_idx}, "
        f"d_eff={d_effs[neck_idx]:.1f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(num_layers), d_effs, marker="o", color="red")
    axes[0].axvspan(num_layers * 0.3, num_layers * 0.7, alpha=0.1, color="gray",
                    label="Middle Layers (The Neck)")
    axes[0].set_title("Effective Dimensionality per Layer\n"
                      "(Participation Ratio, exact SVD)")
    axes[0].set_xlabel("Layer Index (Depth)")
    axes[0].set_ylabel("d_eff")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(num_layers), k95s, marker="o", color="purple")
    axes[1].axvspan(num_layers * 0.3, num_layers * 0.7, alpha=0.1, color="gray")
    axes[1].set_title("k95 per Layer (dims holding 95% of energy)")
    axes[1].set_xlabel("Layer Index (Depth)")
    axes[1].set_ylabel("k95")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "03_per_layer_effective_rank.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)

    return {
        "d_eff_per_layer": [round(x, 1) for x in d_effs],
        "k95_per_layer": k95s,
        "most_compressed_layer": neck_idx,
        "most_compressed_d_eff": round(d_effs[neck_idx], 1),
    }


def part_b_residual_isotropy(hidden_states):
    print("\n" + "=" * 60)
    print("CHECK B — RESIDUAL ISOTROPY (full cone subspace, not just top-3)")
    print("=" * 60)

    mid_layer_idx = num_layers // 2
    X_mid = hidden_states[mid_layer_idx]
    Xc = X_mid - np.mean(X_mid, axis=0)

    U, S, Vt_full = np.linalg.svd(
        Xc.astype(np.float32), full_matrices=False
    )
    K = min(k95_of_spectrum(S), Xc.shape[1] - 1)
    Q = Vt_full[:K].T @ Vt_full[:K]  # projector onto the cone subspace
    Residual = Xc.astype(np.float64) - Xc.astype(np.float64) @ Q

    cos_cone_raw = mean_abs_cos(X_mid)
    cos_res = mean_abs_cos(Residual)
    d_resid = Xc.shape[1] - K
    isotropic_expectation = float(np.sqrt(2.0 / (np.pi * d_resid)))

    print(f"Mid layer {mid_layer_idx}: cone dim K = {K} (its k95)")
    print(f"mean |cos| raw activations:   {cos_cone_raw:.4f}")
    print(f"mean |cos| residual (post-K): {cos_res:.4f}")
    print(
        f"white-noise expectation in d={d_resid}: "
        f"{isotropic_expectation:.4f}  -> ratio = {cos_res / isotropic_expectation:.2f}"
    )

    return {
        "mid_layer_index": mid_layer_idx,
        "cone_dim_K": int(K),
        "mean_abs_cos_cone_raw": round(cos_cone_raw, 4),
        "mean_abs_cos_residual_topK": round(cos_res, 4),
        "isotropic_expectation_d_minus_K": round(isotropic_expectation, 4),
        "residual_to_isotropic_ratio": round(
            cos_res / isotropic_expectation, 2
        ),
    }


num_layers = None  # set in main() before use (module-level for part functions)


def main():
    global num_layers

    print(f"Loading local model: {MODEL_PATH}")
    tokenizer, model = load_model_v3()
    _, diverse_t, n_target = make_contexts(tokenizer)
    print(f"Diverse real-text context (same as v1/v2 B): {n_target} tokens")

    with torch.no_grad():
        outputs = model(input_ids=diverse_t, output_hidden_states=True)
    hidden_states = [
        h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states
    ]
    num_layers = len(hidden_states)
    del outputs, diverse_t, tokenizer, model
    gc.collect()

    summary = {
        "model": "qwen2.5-0.5b (local)",
        "context": f"diverse real docs, {n_target} tokens",
    }
    summary["check_A_per_layer_effective_rank"] = part_a_per_layer(hidden_states)
    summary["check_B_residual_isotropy_full_cone"] = part_b_residual_isotropy(
        hidden_states
    )

    out = os.path.join(RESULTS, "hourglass_summary_v3.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
