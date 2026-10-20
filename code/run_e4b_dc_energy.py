"""E4b follow-up — direct test of the "pressure of mass" quantity.

Reformulated H4: as context grows (or topic interference grows), token states
get pulled toward a common direction -> the DC component's energy share must
grow with N, and the effective dimensionality of the token matrix must drop.

Metrics per (condition, length) on Model A (tied Qwen2.5-0.5B):
  E_DC          = N*||mean||^2 / sum_i ||v_i||^2   <- THE hypothesis quantity
  raw PR/k95    of the token matrix H (N x hidden)
  centered PR/k95 of H - mean(H)                   <- DC-free individuality
  norm_ratio    = mean_i ||v_i - mean|| / ||v_i||  (non-DC fraction)
  + exp4-compatible: raw cos to CoM, centered pairwise |cos|

Conditions:
  prefix — prefixes of the same diverse corpus (as in exp4)
  mixed  — 256-token blocks from the 4 docs shuffled (seed 42): topic interference

Memory-light path (no_grad, last_hidden_state only). Outputs go to
g3_long_context_h4/data/ so the original 5-file deliverable set stays intact.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import tvu_common as C  # noqa: E402


def make_mixed_tokens(tokenizer, n_target, block=256, seed=42):
    """Shuffled 256-token blocks from each doc file (topic-interference context)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for path in C.DOC_FILES:
        with open(path, "r", encoding="utf-8") as fh:
            ids = tokenizer(fh.read(), add_special_tokens=False)["input_ids"]
        for i in range(0, max(len(ids) - block + 1, 0), block):
            blocks.append(np.array(ids[i : i + block], dtype=np.int64))
    rng.shuffle(blocks)
    out = []
    for b in blocks:
        if len(out) >= n_target:
            break
        out.extend(b.tolist())
    return torch.tensor([out[:n_target]], dtype=torch.long).to(C.DEVICE)


def metrics_for(H):
    """H: (N, hidden) numpy float32 — last-layer states of N tokens."""
    N = H.shape[0]
    norms2 = np.sum(H**2, axis=1)
    total_energy = float(norms2.sum())
    xbar = H.mean(axis=0)
    e_dc = (N * float(np.dot(xbar, xbar))) / total_energy

    Hc = H - xbar
    S_raw = np.linalg.svd(H.astype(np.float32), compute_uv=False, full_matrices=False)
    S_cen = np.linalg.svd(Hc.astype(np.float32), compute_uv=False, full_matrices=False)

    norms = np.sqrt(norms2)
    dc_norms = np.linalg.norm(Hc, axis=1)

    Mn = H / (norms[:, None] + 1e-8)
    cos_com = Mn @ (xbar / (np.linalg.norm(xbar) + 1e-8))

    return {
        "n_tokens": int(N),
        "e_dc": round(e_dc, 5),
        "raw_pr": round(C.participation_ratio(S_raw), 3),
        "raw_k95": C.k95_of_spectrum(S_raw),
        "centered_pr": round(C.participation_ratio(S_cen), 3),
        "centered_k95": C.k95_of_spectrum(S_cen),
        "norm_ratio_mean": round(float(np.mean(dc_norms / (norms + 1e-8))), 5),
        "raw_cos_to_com_mean": round(float(cos_com.mean()), 4),
        "raw_cos_to_com_std": round(float(cos_com.std()), 4),
        "centered_pairwise_abs_cos": C.mean_abs_cos(Hc),
    }


def main():
    lengths = [100, 1000, 4096, 8192]
    tok, mod = C.load_model(C.MODEL_A_PATH)

    # how many tokens does the full corpus give under this tokenizer?
    texts = []
    for path in C.DOC_FILES:
        with open(path, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    full_len = len(tok(" ".join(texts), add_special_tokens=False)["input_ids"])
    print(f"full corpus under Qwen tokenizer: {full_len} tokens")
    lengths = [n for n in lengths if n <= full_len]
    print(f"sweeping lengths: {lengths}")

    results = {}
    with torch.no_grad():
        for cond, maker in (
            ("prefix", lambda n: C.make_diverse_tokens(tok, n_target=n)),
            ("mixed", lambda n: make_mixed_tokens(tok, n)),
        ):
            results[cond] = {}
            for n in lengths:
                ids = maker(n)
                H = mod.model(input_ids=ids).last_hidden_state.squeeze(0).float().cpu().numpy()
                m = metrics_for(H)
                del ids
                torch.cuda.empty_cache()
                results[cond][str(n)] = m
                print(f"[{cond} N={n}] E_DC={m['e_dc']}  PRc={m['centered_pr']} "
                      f"k95c={m['centered_k95']}  ratio={m['norm_ratio_mean']}")

    C.free_model(tok, mod)

    out_dir = os.path.join(_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "task": ("E4b follow-up — pressure of mass: DC energy share + token-matrix "
                 "PR/k95 vs context length; prefix (coherent doc) vs mixed "
                 "(topic-interference) conditions"),
        "model": {"name": C.NAME_A, "path": C.MODEL_A_PATH,
                  "note": "tied Qwen2.5-0.5B, fp32, physical GPU 1"},
        "full_corpus_tokens": full_len,
        "lengths_run": lengths,
        "conditions": results,
    }
    out_json = os.path.join(out_dir, "e4b_metrics.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"metrics JSON written: {out_json}")

    # ---- plot: 2x2 panels, prefix (black) vs mixed (red), x = length
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    xs = np.arange(len(lengths))
    tick_labels = [str(n) for n in lengths]

    def panel(ax, key, ylabel, title):
        for cond, color in (("prefix", "black"), ("mixed", "#d64545")):
            ys = [results[cond][str(n)][key] for n in lengths]
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=cond)
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, alpha=0.3)

    panel(axes[0][0], "e_dc", "E_DC = N·‖x̄‖²/Σ‖vᵢ‖² (DC energy share)",
          "Pull toward the common direction — THE hypothesis quantity")
    panel(axes[0][1], "centered_pr", "PR of centered token matrix",
          "Effective dimensionality across tokens (DC-free)")
    panel(axes[1][0], "centered_k95", "k95 of centered token matrix",
          "Dimension holding 95% of DC-free energy")
    panel(axes[1][1], "norm_ratio_mean", "mean ‖vᵢ−x̄‖/‖vᵢ‖ (non-DC fraction)",
          "How much of each token is NOT the common part")

    for ax in axes.ravel():
        ax.legend(fontsize=8.5)
    fig.suptitle(f"E4b — pressure of mass, direct DC-energy test ({C.NAME_A}, GPU 1, fp32)",
                 fontsize=12.5, y=1.0)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "e4b_dc_energy.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"PNG written: {out_png}")


if __name__ == "__main__":
    main()
