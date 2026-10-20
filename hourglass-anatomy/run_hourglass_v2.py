"""
Hourglass Anatomy v2 — "Anatomy of the Attractor and Structureless Noise".

Verifies the three core claims on the LOCAL Qwen2.5-0.5B (../models/qwen2.5-0.5b),
CPU-only, fp32, no HF downloads:

1. Hourglass neck   — top-3 PC variance + mean |cos| per layer across depth;
                      expect collapse of the trajectory at middle layers.
2. Law vs Trajectory — SVD spectrum of a mid-layer MLP weight (gate_proj) vs the
                      activation spectrum entering that layer: dense (weights) vs cliff (activations).
3. Structureless noise — residual after projecting mid-layer activations onto top-3 PCs;
                      expect isotropic noise (mean |cos| ~ 0).

Context: the same diverse real text as v1 context B (project docs, 2006 tokens) —
avoids the repetitive-input artifact found in v1.

Outputs (results/):
    01_hourglass_neck.png
    02_law_vs_trajectory.png
    hourglass_summary_v2.json
"""

import gc
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, no plt.show()

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD

from run_hourglass import RESULTS, make_contexts

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # repo root
MODEL_PATH = os.path.join(os.environ.get("MODELS_DIR") or os.path.join(ROOT, "models"), "qwen2.5-0.5b")


def load_model_v2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # fp32 for stable CPU math (weights ship as bf16, which numpy() rejects)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()
    hidden_dim = int(getattr(model.config, "hidden_size", 0))
    return tokenizer, model, hidden_dim


def k95_of_spectrum(sigma):
    """Smallest k such that top-k singular values hold 95% of the energy."""
    cumvar = np.cumsum((sigma**2) / float(np.sum(sigma**2)))
    return int(np.searchsorted(cumvar, 0.95) + 1)


def experiment_1(hidden_states):
    """The hourglass neck: top-3 PC variance and anisotropy per layer."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — HOURGLASS NECK (per-layer compression)")
    print("=" * 60)

    num_layers = len(hidden_states)
    top3_var_pct = []
    mean_abs_cos = []

    for i, hidden in enumerate(hidden_states):
        X = hidden.astype(np.float64)
        Xc = X - np.mean(X, axis=0)
        pca = TruncatedSVD(n_components=min(10, X.shape[1]), random_state=42)
        pca.fit(Xc)
        top3_var_pct.append(float(np.sum(pca.explained_variance_ratio_[:3])) * 100.0)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / (norms + 1e-8)
        cs = Xn @ Xn.T
        np.fill_diagonal(cs, 0.0)
        mean_abs_cos.append(float(np.mean(np.abs(cs))))

    neck_layer = int(np.argmax(top3_var_pct))
    for i in [0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]:
        print(
            f"layer {i:>2}: top-3 var = {top3_var_pct[i]:6.2f}%   "
            f"mean |cos| = {mean_abs_cos[i]:.4f}"
        )
    print(f"NECK (max top-3 variance): layer {neck_layer} at {top3_var_pct[neck_layer]:.2f}%")

    # ---- plot ----
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        range(num_layers), top3_var_pct, marker="o", color="red",
        label="Top-3 PC Variance (%)",
    )
    ax1.set_xlabel("Layer Index (Depth)")
    ax1.set_ylabel("Variance in Top-3 PCs (%)", color="red")
    ax1.axvspan(
        num_layers * 0.3, num_layers * 0.7, alpha=0.1, color="gray",
        label="Middle Layers (The Neck)",
    )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        range(num_layers), mean_abs_cos, marker="x", color="blue",
        label="Mean |Cosine| (Anisotropy)",
    )
    ax2.set_ylabel("Anisotropy", color="blue")
    ax2.legend(loc="upper right")

    plt.title("The Hourglass Neck: Trajectory Collapse in Middle Layers")
    plt.tight_layout()
    out = os.path.join(RESULTS, "01_hourglass_neck.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)

    return {
        "n_hidden_states": num_layers,
        "top3_var_pct_per_layer": [round(x, 2) for x in top3_var_pct],
        "mean_abs_cos_per_layer": [round(x, 4) for x in mean_abs_cos],
        "neck_layer_index": neck_layer,
        "neck_top3_var_pct": round(top3_var_pct[neck_layer], 2),
    }


def experiment_2(model, hidden_states):
    """Law (weights) vs Trajectory (activations) — mid-layer spectra."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — LAW VS TRAJECTORY (mid-layer SVD spectra)")
    print("=" * 60)

    num_layers = len(hidden_states)
    mid_layer_idx = num_layers // 2  # hidden_states index of the mid output
    layer_module = model.model.layers[mid_layer_idx - 1]  # block feeding it

    W_gate = layer_module.mlp.gate_proj.weight.detach().float().cpu().numpy()
    X_act = hidden_states[mid_layer_idx - 1].astype(np.float64)  # input to that block
    print(
        f"Mid block index: {mid_layer_idx - 1}; "
        f"W_gate shape: {W_gate.shape}; activations: {X_act.shape}"
    )

    S_w = np.linalg.svd(W_gate, full_matrices=False)[1]
    Xc = X_act - np.mean(X_act, axis=0)
    S_x = np.linalg.svd(Xc, full_matrices=False)[1]

    S_w_norm = (S_w / np.sum(S_w)).astype(np.float64)
    S_x_norm = (S_x / np.sum(S_x)).astype(np.float64)

    k95_w = k95_of_spectrum(S_w)
    k95_x = k95_of_spectrum(S_x)
    top3_e_w = float(np.sum((S_w**2)[:3]) / np.sum(S_w**2))
    top3_e_x = float(np.sum((S_x**2)[:3]) / np.sum(S_x**2))

    print(f"WEIGHTS (Law):       k95 = {k95_w} of {S_w.size}, "
          f"top-3 energy share = {top3_e_w:.4f}")
    print(f"ACTIVATIONS (Track): k95 = {k95_x} of {S_x.size}, "
          f"top-3 energy share = {top3_e_x:.4f}")

    # ---- plot ----
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.clip(S_w_norm, 1e-12, None), color="black", linewidth=2,
        label=f"Weights (The Law) — k95={k95_w}",
    )
    plt.plot(
        np.clip(S_x_norm, 1e-12, None), color="red", linewidth=2,
        label=f"Activations (The Trajectory) — k95={k95_x}",
    )
    plt.yscale("log")
    plt.title(f"Layer {mid_layer_idx - 1}: Law (Dense) vs Trajectory (Cone)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "02_law_vs_trajectory.png")
    plt.savefig(out, dpi=300)
    plt.close()

    return {
        "mid_block_index": mid_layer_idx - 1,
        "weight_matrix_shape": list(W_gate.shape),
        "activation_matrix_shape": [int(x) for x in X_act.shape],
        "weights_k95": k95_w,
        "weights_n_sv": int(S_w.size),
        "weights_top3_energy_share": round(top3_e_w, 4),
        "activations_k95": k95_x,
        "activations_n_sv": int(S_x.size),
        "activations_top3_energy_share": round(top3_e_x, 4),
    }


def experiment_3(hidden_states):
    """Structureless noise: residual after removing the top-3 PC cone."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — STRUCTURELESS NOISE (residual isotropy)")
    print("=" * 60)

    num_layers = len(hidden_states)
    mid_layer_idx = num_layers // 2
    X_mid = hidden_states[mid_layer_idx].astype(np.float64)
    Xc = X_mid - np.mean(X_mid, axis=0)

    # exact top-3 reconstruction via the SVD right-singular subspace projector
    Vt3 = np.linalg.svd(Xc, full_matrices=False)[2][:3]  # (3, D)
    Q = Vt3.T @ Vt3  # (D, D) projector onto span(top-3 PCs)
    Residual = Xc - Xc @ Q

    def mean_abs_cos(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        Mn = M / (norms + 1e-8)
        cs = Mn @ Mn.T
        np.fill_diagonal(cs, 0.0)
        return float(np.mean(np.abs(cs)))

    cos_cone = mean_abs_cos(X_mid)
    cos_res = mean_abs_cos(Residual)

    print(f"Средний |cos| в исходных активациях (Конус): {cos_cone:.4f}")
    print(f"Средний |cos| в Остатке (Шум):               {cos_res:.4f}")
    print("Если в остатке cos ~ 0, значит шум изотропен. Геометрических паттернов для CHMC там нет.")

    return {
        "mid_layer_index": mid_layer_idx,
        "mean_abs_cos_cone_raw": round(cos_cone, 4),
        "mean_abs_cos_residual": round(cos_res, 4),
    }


def main():
    print(f"Loading local model: {MODEL_PATH}")
    tokenizer, model, hidden_dim = load_model_v2()
    n_layers_cfg = int(getattr(model.config, "num_hidden_layers", 0))
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    print(
        f"Model loaded. D={hidden_dim} layers={n_layers_cfg} "
        f"tie_word_embeddings={tie}"
    )

    _, diverse_t, n_target = make_contexts(tokenizer)
    print(f"Diverse real-text context (same as v1 B): {n_target} tokens")

    with torch.no_grad():
        outputs = model(input_ids=diverse_t, output_hidden_states=True)
    hidden_states = [
        h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states
    ]
    del outputs, diverse_t, tokenizer
    gc.collect()

    summary = {
        "model": "qwen2.5-0.5b (local)",
        "hidden_dim": hidden_dim,
        "num_hidden_layers": n_layers_cfg,
        "tie_word_embeddings": tie,
        "context": f"diverse real docs, {n_target} tokens (same as v1 context B)",
    }

    summary["experiment_1_hourglass_neck"] = experiment_1(hidden_states)
    summary["experiment_2_law_vs_trajectory"] = experiment_2(
        model, hidden_states
    )
    del model
    gc.collect()
    summary["experiment_3_structureless_noise"] = experiment_3(hidden_states)

    out = os.path.join(RESULTS, "hourglass_summary_v2.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
