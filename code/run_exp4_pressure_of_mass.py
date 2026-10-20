"""Experiment 4 — Pressure of mass: does a longer context squeeze tokens toward
the center of mass (Janssen effect, as hypothesized in the TZ)?

Method: Model A (tied Qwen2.5-0.5B) only, per the TZ. Three prefixes of the same
diverse real-doc context: 100 / 1000 / 4096 tokens (Qwen's native context is
32768, so no extrapolation). Last-layer hidden states H [seq, dim]; center of
mass C = mean over all tokens. Primary metric (as in the TZ): mean cosine of each
token to C on RAW hidden states. Secondary check (methodological, per
hourlass_anatomy v3 lesson that raw |cos| is DC-dominated): mean pairwise |cos|
between CENTERED last-layer states — if it stays ~white-noise level while the raw
metric grows, the growth is a common-direction effect, not individuality loss.

Outputs: data/04_pressure_of_mass.png + data/data/exp4_metrics.json
"""

import os

import matplotlib

matplotlib.use("Agg")

import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # g3_long_context_h4/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                 # for `import tvu_common`

import tvu_common as C                                  # noqa: E402

LENGTHS = [100, 1000, 4096]


def main():
    print(f"Model A (tied): {C.MODEL_A_PATH}")
    tok, mod = C.load_model(C.MODEL_A_PATH)
    ids_full = C.make_diverse_tokens(tok, n_target=max(LENGTHS))

    raw_cos, cen_cos, stds = [], [], []
    for L in LENGTHS:
        ids = ids_full[:, :L]
        # memory-light path: last layer only (no output_hidden_states, no autograd
        # graph) — 4096 tokens would not fit alongside a co-runner on one GPU
        with torch.no_grad():
            H = mod.model(input_ids=ids).last_hidden_state.squeeze(0) \
                .float().cpu().numpy()

        C_ = H.mean(axis=0)
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        cos_i = (H / (norms + 1e-8)) @ (C_ / (np.linalg.norm(C_) + 1e-8))
        raw_cos.append(float(cos_i.mean()))
        stds.append(float(cos_i.std()))

        Hc = H - H.mean(axis=0)
        cen_cos.append(C.mean_abs_cos(Hc))

        print(f"len={L:>5}: mean cos to CoM (raw) = {cos_i.mean():.4f} "
              f"(std {cos_i.std():.4f}) | centered pairwise |cos| = {cen_cos[-1]:.4f}")

    C.free_model(tok, mod)

    # ---- plot: raw cos to CoM vs context length (solid black), centered
    # pairwise |cos| as the DC-free check (dashed red)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.array(LENGTHS)
    ax.plot(xs, raw_cos, color="black", marker="o", linewidth=2, markersize=7,
            label="Mean cosine to center of mass (RAW last-layer states)")
    for x, v in zip(xs, raw_cos):
        ax.annotate(f"{v:.4f}", xy=(x, v), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9.5)
    ax.plot(xs, cen_cos, color="#d64545", linestyle="--", marker="^", linewidth=1.8,
            markersize=6, label="Centered pairwise |cos| (DC-free check)")
    for x, v in zip(xs, cen_cos):
        ax.annotate(f"{v:.4f}", xy=(x, v), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=9.5, color="#d64545")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Experiment 4 — Pressure of mass: Tied Qwen2.5-0.5B\n"
                 "diverse real-doc context, last-layer hidden states")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(C.RESULTS, "04_pressure_of_mass.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload = {
        "model": C.NAME_A,
        "context": f"diverse real docs (same corpus as exp1-3), prefixes of length {LENGTHS}",
        "lengths": LENGTHS,
        "raw_cos_to_com_mean": [round(x, 4) for x in raw_cos],
        "raw_cos_to_com_std": [round(x, 4) for x in stds],
        "centered_pairwise_abs_cos": [round(x, 4) for x in cen_cos],
    }
    C.save_work_json("exp4_metrics.json", payload)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
