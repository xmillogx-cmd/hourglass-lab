"""Exp-2 presentation for all six models: per-layer centered participation-ratio
profiles ("hourglass" shape) from data/data/exp5_metrics.json — the SAME data already
collected in Step 1 / Exp 5 (identical protocol to the original exp2), rendered
here as one cross-model plot.

CPU-only (matplotlib + numpy, no torch/CUDA): safe to run while a GPU job is on
physical GPU 1. Writes data/08_pr_profiles_all_models.png.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # cross_model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tvu_common as C  # noqa: E402

LABELS = {
    "smollm-135m":   "SmolLM-135M (tied, 0.13B)",
    "qwen2.5-0.5b":  "Qwen2.5-0.5B (tied, 0.49B)",
    "tinyllama-1.1b":"TinyLlama-1.1B (untied, 1.1B)",
    "qwen2.5-3b":    "Qwen2.5-3B (tied, 3.0B)",
    "qwen3-4b":      "Qwen3-4B (tied, 4.0B)",
    "gemma-4-e2b":   "Gemma-4-E2B (tied, 5.1B)",
}


def main():
    work = os.path.join(C.WORK, "exp5_metrics.json")
    with open(work, encoding="utf-8") as fh:
        data = json.load(fh)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=140)
    for i, (key, r) in enumerate(data.items()):
        if "status" in r or "pr_centered_per_level" not in r:
            continue
        levels = list(range(len(r["pr_centered_per_level"])))
        prc = r["pr_centered_per_level"]
        c = plt.cm.tab10(i / 10.0)
        ax.plot(levels, prc, lw=1.4, color=c, label=f"{LABELS[key]} — neck L{r['neck_level']} (PR {r['neck_pr_centered']:.2f})")
        ax.scatter([r["neck_level"]], [r["neck_pr_centered"]], s=38, color=c, zorder=5, edgecolors="k", linewidths=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("Hidden-state level (0 = embeddings, k+1 = layer k)")
    ax.set_ylabel("Participation ratio of centered hidden states (log scale)")
    ax.set_title("Hourglass PR profile — all six models (Exp 2 protocol, Step 1 / Exp 5 data)\n"
                 "dots: neck level (minimum centered PR) per model")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    out = os.path.join(C.RESULTS, "08_pr_profiles_all_models.png")
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
