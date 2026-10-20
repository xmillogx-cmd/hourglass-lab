"""
Hourglass Anatomy v4 — quantifying the structure OUTSIDE the cone.

v3 check B: after removing the full 95%-energy cone subspace (K = k95 = 63
at mid layer 12), the residual is still 1.74x more anisotropic than white
noise (mean |cos| = 0.0481 vs expected 0.0276). This check decomposes that
residual: exact SVD of the post-cone residual -> d_eff, k95, top-3 energy
share. Low-rank residual spectrum => secondary sub-cone; flat spectrum
(d_eff in the hundreds) => the 1.74x is diffuse finite-sample correlation,
not an exploitable geometry.

Same local model and same diverse context as v1-v3. CPU-only, fp32.

Outputs (results/):
    04_residual_spectrum.png
    hourglass_summary_v4.json
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


def load_model_v4():
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


def main():
    print(f"Loading local model: {MODEL_PATH}")
    tokenizer, model = load_model_v4()
    _, diverse_t, n_target = make_contexts(tokenizer)
    print(f"Diverse real-text context (same as v1-v3): {n_target} tokens")

    with torch.no_grad():
        outputs = model(input_ids=diverse_t, output_hidden_states=True)
    hidden_states = [
        h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states
    ]
    del outputs, diverse_t, tokenizer, model
    gc.collect()

    num_layers = len(hidden_states)
    mid_layer_idx = num_layers // 2
    print(f"\nMid layer index: {mid_layer_idx}")

    X_mid = hidden_states[mid_layer_idx]
    Xc = X_mid - np.mean(X_mid, axis=0)

    _, S_raw, Vt_full = np.linalg.svd(
        Xc.astype(np.float32), full_matrices=False
    )
    K = min(k95_of_spectrum(S_raw), Xc.shape[1] - 1)
    Q = Vt_full[:K].T @ Vt_full[:K]  # projector onto the cone subspace
    Residual = Xc.astype(np.float64) - Xc.astype(np.float64) @ Q

    _, S_res, _ = np.linalg.svd(Residual.astype(np.float32), full_matrices=False)

    d_eff_res = participation_ratio(S_res)
    k95_res = k95_of_spectrum(S_res)
    top3_e_res = float(np.sum((S_res**2)[:3]) / np.sum(S_res**2))
    top10_e_res = float(np.sum((S_res**2)[:10]) / np.sum(S_res**2))

    print(f"cone dim K = {K} (k95 of raw mid-layer activations)")
    print(
        f"RESIDUAL spectrum: d_eff = {d_eff_res:.1f}, k95 = {k95_res}, "
        f"top-3 energy = {top3_e_res:.4f}, top-10 energy = {top10_e_res:.4f}"
    )
    if k95_res < 200:
        print("=> low-rank residual: a SECONDARY sub-cone exists outside the main cone.")
    else:
        print("=> flat residual spectrum: structureless diffuse noise, no exploitable geometry.")

    # ---- plot: raw vs residual spectra (log scale) ----
    S_raw_norm = np.clip(S_raw / np.sum(S_raw), 1e-12, None).astype(np.float64)
    S_res_norm = np.clip(S_res / np.sum(S_res), 1e-12, None).astype(np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(np.clip(S_raw_norm, 1e-12, None), color="black", linewidth=2,
             label=f"Raw mid-layer activations (K={K})")
    plt.plot(
        np.arange(len(S_res)) + K, S_res_norm, color="red", linewidth=2,
        label=f"Residual after removing top-{K} cone subspace",
    )
    plt.yscale("log")
    plt.title(f"Layer {mid_layer_idx}: Cone vs Outside-of-Cone Residual Spectrum")
    plt.xlabel("Singular Value Index (residual indices offset by K)")
    plt.ylabel("Normalized Singular Value (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "04_residual_spectrum.png")
    plt.savefig(out, dpi=300)
    plt.close()

    summary = {
        "model": "qwen2.5-0.5b (local)",
        "context": f"diverse real docs, {n_target} tokens",
        "mid_layer_index": mid_layer_idx,
        "cone_dim_K": int(K),
        "residual_d_eff": round(d_eff_res, 1),
        "residual_k95": k95_res,
        "residual_top3_energy_share": round(top3_e_res, 4),
        "residual_top10_energy_share": round(top10_e_res, 4),
    }
    out_json = os.path.join(RESULTS, "hourglass_summary_v4.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSummary written to {out_json}")


if __name__ == "__main__":
    main()
