"""Exp 6 — Inversion Contrast (IC) confidence interval + test vs zero, from the
ALREADY STORED per-cell lifts in data/data/exp6_metrics.json. No GPU needed:

    IC = mean(lift_vs_control over mid positions p in {0.25, 0.45, 0.75})
       - mean(lift_vs_control over edge positions p in {0.05, 0.95})

exactly as computed by run_exp6_lim_scan.py (the stored scalar is reproduced here
as a sanity check). A nonparametric bootstrap resamples the mid lifts and the
edge lifts independently with replacement, yielding a distribution of IC from
which we read: point estimate, 95% percentile CI, bootstrap std, and a two-sided
p-value vs zero (fraction of bootstraps whose |IC*| reaches |IC_obs|).

Reviewer answer to p.1 / Exp6 ("CI for inversion contrast + test vs zero"): the
contrast is estimated with an uncertainty band and tested against 0 using only
the replicate cells already recorded in the experiment — no extra forward passes.

Writes data/data/exp6_ic_ci.json (CPU, stdlib+numpy).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cross_model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import tvu_common as C  # noqa: E402

MID = (0.25, 0.45, 0.75)
EDGE = (0.05, 0.95)
N_BOOT = 10_000
SEED = 1337


def key_matches(key, positions):
    return any(f"_p{q:.2f}_" in key for q in positions)


def main():
    src = os.path.join(C.WORK, "exp6_metrics.json")
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)

    rng = np.random.default_rng(SEED)
    out = {}
    for key, r in data.items():
        if "cells" not in r or "inversion_contrast" not in r:
            continue  # skipped models have no cells
        cells = r["cells"]
        mid_l = np.array([cells[k]["lift_vs_control"] for k in cells if key_matches(k, MID)])
        edge_l = np.array([cells[k]["lift_vs_control"] for k in cells if key_matches(k, EDGE)])
        ic_obs = float(mid_l.mean() - edge_l.mean())

        # Nonparametric bootstrap: resample each group with replacement.
        boot = np.empty(N_BOOT)
        n_mid, n_edge = mid_l.size, edge_l.size
        for b in range(N_BOOT):
            mid_b = mid_l[rng.integers(0, n_mid, n_mid)]
            edge_b = edge_l[rng.integers(0, n_edge, n_edge)]
            boot[b] = mid_b.mean() - edge_b.mean()

        lo, hi = np.percentile(boot, [2.5, 97.5])
        # two-sided p vs zero: fraction of the bootstrap distribution at least as
        # extreme (in |IC|) than the observed contrast.
        p_two = float(np.mean(np.abs(boot) >= abs(ic_obs)))

        out[key] = {
            "n_mid": int(n_mid),
            "n_edge": int(n_edge),
            "ic_point_estimate": round(ic_obs, 4),
            "ic_stored_in_metrics": r["inversion_contrast"],
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
            "bootstrap_std": round(float(boot.std()), 4),
            "excludes_zero_at_005": bool((ic_obs > 0 and hi < 0) or (ic_obs < 0 and lo > 0)),
            "p_two_sided_vs_zero": round(p_two, 4),
        }

    out_path = os.path.join(C.WORK, "exp6_ic_ci.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"work JSON written: {out_path}")

    for key, v in out.items():
        sig = "SIGNIFICANT (CI excludes 0)" if v["excludes_zero_at_005"] else "not significant at .05"
        print(f"{key:<16} IC={v['ic_point_estimate']:+.4f} (stored {v['ic_stored_in_metrics']:+.4f}) "
              f"CI95=[{v['ci95_low']:+.3f},{v['ci95_high']:+.3f}] p={v['p_two_sided_vs_zero']:.3f} {sig}")


if __name__ == "__main__":
    main()
