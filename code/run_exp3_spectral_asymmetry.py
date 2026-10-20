"""Experiment 3 — Spectral asymmetry at the neck: Law (weights) vs Trajectory.

Hypothesis (TZ): at the most compressed layer, the weight matrices are dense /
full-rank ("the Law"), while the activations passing through it form a low-rank
cliff ("the Trajectory" = the attractor cone).

Method: neck level per model is read from data/data/exp2_metrics.json (argmin of
centered PR over layers 1..N). Re-run forward with output_hidden_states on the
same 2048-token diverse context. X = centered hidden-state matrix at the neck
level; W = gate_proj weights of that layer's MLP block. Exact SVD for both;
normalized spectra sigma_i / sum(sigma) plotted log-scale (as in the TZ); plus
k95 and top-3/top-10 energy shares per spectrum for the numbers table.

Outputs: data/03_spectral_asymmetry.png + data/data/exp3_metrics.json
"""

import json
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


def asymmetry(model, input_ids, neck_level):
    # no_grad: the autograd graph would otherwise retain all intermediates → OOM
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = [h.detach().squeeze(0).float().cpu().numpy() for h in out.hidden_states]
    X = hs[neck_level]
    Xc = (X - X.mean(axis=0)).astype(np.float32)

    layer_idx = max(neck_level - 1, 0)  # block whose output is at the neck level
    W = model.model.layers[layer_idx].mlp.gate_proj.weight.detach().float().cpu().numpy()
    S_W = np.linalg.svd(W.astype(np.float32), compute_uv=False)
    S_X = np.linalg.svd(Xc, compute_uv=False)

    def stats(S):
        e = (S**2) / float(np.sum(S**2))
        return {
            "d_eff": round(C.participation_ratio(S), 1),
            "k95": C.k95_of_spectrum(S),
            "top3_energy": round(float(e[:3].sum()), 4),
            "top10_energy": round(float(e[:10].sum()), 4),
        }

    return {
        "neck_level": int(neck_level),
        "block_layer_index_1based": layer_idx + 1,
        "weights_shape": list(W.shape),
        "activations_shape": list(Xc.shape),
        "S_W": S_W.astype(np.float64).tolist(),
        "S_X": S_X.astype(np.float64).tolist(),
        "W_stats": stats(S_W),
        "X_stats": stats(S_X),
    }


def main():
    import gc

    with open(os.path.join(C.WORK, "exp2_metrics.json"), encoding="utf-8") as fh:
        exp2 = json.load(fh)

    results = {}
    for name, path in ((C.NAME_A, C.MODEL_A_PATH), (C.NAME_B, C.MODEL_B_PATH)):
        neck = int(exp2[name]["neck_level"])
        print(f"Model {name}: neck level = {neck} (from exp2)")
        tok, mod = C.load_model(path)
        ids = C.make_diverse_tokens(tok, n_target=C.N_TOKENS)
        r = asymmetry(mod, ids, neck)
        results[name] = r
        print(f"  W: k95={r['W_stats']['k95']}/{min(r['weights_shape'])} "
              f"top3={r['W_stats']['top3_energy']} | X: k95={r['X_stats']['k95']}"
              f"/{len(r['S_X'])} top3={r['X_stats']['top3_energy']}")
        C.free_model(tok, mod)

    # ---- plot: 2 subplots (A | B), weights black vs activations red, log-y
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, name in zip(axes, (C.NAME_A, C.NAME_B)):
        r = results[name]
        S_W = np.array(r["S_W"])
        S_X = np.array(r["S_X"])
        nW, nX = len(S_W), len(S_X)
        ax.plot(np.arange(nW), np.clip(S_W / S_W.sum(), 1e-12, None), color="black",
                linewidth=2, label=f"gate_proj weights (Law): k95={r['W_stats']['k95']}")
        ax.plot(np.arange(nX), np.clip(S_X / S_X.sum(), 1e-12, None), color="#d64545",
                linewidth=2, label=f"entering activations (Trajectory): k95={r['X_stats']['k95']}")
        ax.set_yscale("log")
        ax.set_title(f"{name}\nneck level {r['neck_level']}, block layer "
                     f"{r['block_layer_index_1based']}: W{tuple(r['weights_shape'])} vs X{tuple(r['activations_shape'])}")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Normalized singular value σᵢ/Σσ (log)")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Experiment 3 — Law vs Trajectory at the neck: dense weights, low-rank activations",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(C.RESULTS, "03_spectral_asymmetry.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    C.save_work_json("exp3_metrics.json", results)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
