"""Null models for reviewer points 4 & 5 (CPU).

(4) Orthogonality null: two independent standard-normal vectors in R^d, d = vocab*hidden
    per tied model -> distribution/scale of cos. Analytic std ~ 1/sqrt(d); Monte-Carlo
    confirmation at reduced dims; z-score + tail probability for the measured |cos|.
(5) Low-dim baseline: participation ratio (same definition as Exp-5 PRc) of centered
    i.i.d. Gaussian data N x hidden -> shows real necks (PRc~1.1) are far below random.

Writes data/data/null_model_metrics.json + prints summary. numpy required (venv python).
"""
import json, math, os
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
TVU = _ROOT
rng = np.random.default_rng(0)

# model -> (d=vocab*hidden, measured cos of G_in vs G_out from data/data/exp7_metrics.json)
TIED_DIMS = {
    "smollm-135m":  (28_311_552, 0.0155),
    "qwen2.5-0.5b": (136_134_656, 0.0183),
    "qwen2.5-3b":   (311_164_928, -0.0061),
    "qwen3-4b":     (388_956_160, -0.0036),
    "gemma-4-e2b":  (402_653_184, 0.0229),
}


def phi(x):  # standard normal CDF via erf (no scipy)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---- (4) orthogonality null -------------------------------------------------
orth = {}
for name, (d, c_meas) in TIED_DIMS.items():
    std_null = 1.0 / math.sqrt(d)              # leading-order std of cos under the null
    z = abs(c_meas) / std_null                 # how many null-sigma the measured |cos| is
    p_tail = 2.0 * (1.0 - phi(abs(c_meas) / std_null))   # P(|cos| >= c_meas) under null
    orth[name] = {
        "d": d,
        "measured_cos": round(float(c_meas), 4),
        "measured_abs_cos": round(abs(c_meas), 4),
        "null_std_cos_1_over_sqrt_d": float(f"{std_null:.6e}"),
        "z_score_vs_null_sigma": round(z, 1),
        "p_tail_under_null": (float(p_tail) if p_tail > 0 else "< 5e-324"),
    }

# Monte-Carlo confirmation of std(cos) ~ 1/sqrt(d) at reduced dims (validates the analytic model)
mc = {}
for d_mc in (1_000, 10_000, 100_000):
    n_pairs = 4000
    coses = []
    for _ in range(n_pairs):
        a = rng.standard_normal(d_mc)
        b = rng.standard_normal(d_mc)
        ca = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        coses.append(ca)
    arr = np.array(coses)
    mc[f"d={d_mc}"] = {
        "n_pairs": n_pairs,
        "std_empirical": round(float(arr.std()), 6),
        "std_analytic_1_over_sqrt_d": round(1.0 / math.sqrt(d_mc), 6),
        "mean_abs_cos": round(float(np.abs(arr).mean()), 6),
    }

# ---- (5) PRc random-matrix baseline -----------------------------------------
HIDDEN = {"smollm-135m": 576, "qwen2.5-0.5b": 896, "tinyllama-1.1b": 2048,
          "qwen2.5-3b": 2048, "qwen3-4b": 2560, "gemma-4-e2b": 1536}
N_TOK = 2048


def participation_ratio(H):
    # H: N x h. center columns; covariance eigenvalues; PR = (sum)^2 / sumsq  (Exp-5 PRc def)
    Hc = H - H.mean(axis=0)
    C = (Hc.T @ Hc) / max(N_TOK - 1, 1)        # h x h
    lam = np.linalg.eigvalsh(C)                # ascending
    s = float(lam.sum())
    ssq = float((lam ** 2).sum())
    return s * s / ssq if ssq > 0 else float("inf")


prc_base = {}
for name, h in HIDDEN.items():
    vals = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        H = r.standard_normal((N_TOK, h))
        vals.append(participation_ratio(H))
    prc_base[name] = {
        "hidden": h, "n_tokens": N_TOK, "seeds": 5,
        "prc_mean": round(float(np.mean(vals)), 2),
        "prc_min": round(float(np.min(vals)), 2),
        "prc_max": round(float(np.max(vals)), 2),
    }

out = {"orthogonality_null_p4": orth, "mc_confirmation": mc, "prc_random_baseline_p5": prc_base}
out_path = os.path.join(TVU, "data", "null_model_metrics.json")
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)

print("=== (4) Orthogonality null: cos of 2 independent random vectors in R^d ===")
for name, o in orth.items():
    print(f"{name:16s} d={o['d']:>9,} meas_cos={o['measured_cos']:+.4f} "
          f"|meas|={o['measured_abs_cos']:.4f} null_std=1/sqrt(d)={o['null_std_cos_1_over_sqrt_d']:.2e} "
          f"z={o['z_score_vs_null_sigma']}sigma")
print("\n--- MC confirmation: std(cos) vs 1/sqrt(d) ---")
for k, m in mc.items():
    print(f"{k:>10s}  std_empirical={m['std_empirical']:.3e} analytic=1/sqrt(d)={m['std_analytic_1_over_sqrt_d']:.3e}")
print("\n=== (5) PRc random-matrix baseline (N=2048 i.i.d. Gaussian, 5 seeds) ===")
for name, p in prc_base.items():
    print(f"{name:16s} h={p['hidden']:>4d}  PRc_mean={p['prc_mean']:.1f} (min {p['prc_min']} max {p['prc_max']})")
print(f"\nsaved -> {out_path}")
