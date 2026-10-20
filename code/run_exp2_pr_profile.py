"""Experiment 2 — Hourglass profile: Participation Ratio per depth level, both models.

Hypothesis (TZ): the conflict at the edges makes the network build a compensatory
attractor in the middle; effective dimensionality should dip in mid layers
(U/V-shaped curve).

Method: forward with output_hidden_states on 2048 tokens of the diverse real-doc
context (per model's own tokenizer). For every hidden-state level l (embeddings +
layers): X_l = seq x dim matrix, centered Xc = X - mean(X, axis=0); exact SVD;
PR = (sum s^2)^2 / sum s^4 on singular values. Both CENTERED (primary — raw PR is
dominated by the single DC direction, see hourlass_anatomy v3) and RAW (as written
in the TZ: C_l = X_l^T X_l without centering) are reported.

Outputs: data/02_pr_profile.png + data/data/exp2_metrics.json
"""

import os

import matplotlib

matplotlib.use("Agg")

import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # g2_geometry_hourglass/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                 # for `import tvu_common`

import tvu_common as C                                  # noqa: E402


def level_spectrum(X):
    """(PR centered, PR raw) for one hidden-state matrix X [seq, dim]."""
    sig_c = np.linalg.svd((X - X.mean(axis=0)).astype(np.float32), compute_uv=False)
    sig_r = np.linalg.svd(X.astype(np.float32), compute_uv=False)
    return C.participation_ratio(sig_c), C.participation_ratio(sig_r)


def profile(model, input_ids):
    # no_grad: without it the autograd graph would retain every intermediate
    # activation (incl. 512 MB attention-score tensors per layer) → OOM on Llama
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = [h.detach().squeeze(0).float().cpu().numpy() for h in out.hidden_states]
    prc, prr = [], []
    n_levels = len(hs)
    for i, X in enumerate(hs):
        pc, pr = level_spectrum(X)
        prc.append(pc)
        prr.append(pr)
        print(f"  level {i:>2}: PR_centered = {pc:8.1f}   PR_raw = {pr:8.1f}")
    del out
    return n_levels, np.array(prc), np.array(prr)


def main():
    import gc

    results = {}

    for name, path in ((C.NAME_A, C.MODEL_A_PATH), (C.NAME_B, C.MODEL_B_PATH)):
        print(f"Model {name}: {path}")
        tok, mod = C.load_model(path)
        ids = C.make_diverse_tokens(tok, n_target=C.N_TOKENS)
        n_levels, prc, prr = profile(mod, ids)

        # neck = most compressed LAYER (levels 1..n-1; level 0 is embeddings)
        neck_level = int(np.argmin(prc[1:]) + 1)
        mid_level = n_levels // 2
        results[name] = {
            "model_path": path,
            "hidden_size": mod.config.hidden_size,
            "num_hidden_layers": mod.config.num_hidden_layers,
            "n_tokens": C.N_TOKENS,
            "pr_centered_per_level": [round(x, 2) for x in prc],
            "pr_raw_per_level": [round(x, 2) for x in prr],
            "neck_level": neck_level,
            "neck_pr_centered": round(float(prc[neck_level]), 2),
            "pr_layer0": round(float(prc[0]), 2),
            "pr_mid": round(float(prc[mid_level]), 2),
            "pr_last": round(float(prc[-1]), 2),
        }
        print(f"  NECK (min PR over layers): level {neck_level}, "
              f"PR_centered = {results[name]['neck_pr_centered']}")
        C.free_model(tok, mod)

    # ---- plot: PR vs depth for both models, centered solid + raw dashed, log-y
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x_a = np.arange(len(results[C.NAME_A]["pr_centered_per_level"]))
    x_b = np.arange(len(results[C.NAME_B]["pr_centered_per_level"]))

    ax.plot(x_a, results[C.NAME_A]["pr_centered_per_level"], color="#1f6fd6",
            marker="o", markersize=4, linewidth=2, label=f"{C.NAME_A} (centered)")
    ax.plot(x_a, results[C.NAME_A]["pr_raw_per_level"], color="#7fb0e8",
            linestyle="--", linewidth=1.5, label=f"{C.NAME_A} (raw)")
    ax.plot(x_b, results[C.NAME_B]["pr_centered_per_level"], color="#e8871a",
            marker="^", markersize=4, linewidth=2, label=f"{C.NAME_B} (centered)")
    ax.plot(x_b, results[C.NAME_B]["pr_raw_per_level"], color="#f5c98e",
            linestyle="--", linewidth=1.5, label=f"{C.NAME_B} (raw)")

    for name, xc in ((C.NAME_A, x_a), (C.NAME_B, x_b)):
        nk = results[name]["neck_level"]
        ax.axvline(nk, color="gray", linestyle=":", linewidth=1.2)
        ax.annotate(f"neck {name.split('_')[0]} @ {nk}", xy=(nk, 3), fontsize=8.5,
                    ha="center", color="#444")

    ax.set_yscale("log")
    ax.set_xlabel("Hidden-state level (0 = embeddings, N = final layer)")
    ax.set_ylabel("Participation Ratio PR = (Σs²)²/Σs⁴  (log scale)")
    ax.set_title("Experiment 2 — Effective dimensionality per depth: Tied vs Untied\n"
                 "(solid = centered states [primary], dashed = raw, as in TZ C_l = X_lᵀX_l)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(C.RESULTS, "02_pr_profile.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    C.save_work_json("exp2_metrics.json", results)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
