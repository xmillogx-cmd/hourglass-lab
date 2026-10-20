"""AT-2 — CPU audit: analogy-theorem checks from existing metrics + the AT-1 run.

Inputs (all in data/):
  exp5_metrics.json                       PRc profile per layer, all 8 models
  exp6_ic_ci.json                         IC point estimates / CIs, all 8 models
  exp9_hidden_perturbation_metrics.json   KL sensitivity profiles (3 models, absolute eps)
  at1_analogy_perturbation_metrics.json   AT-1 sweep (all 8 models, relative delta; may be absent -> partial audit)

Checks computed:
  T1 regularity->stability : Spearman/Pearson between gamma_hat(m,l) = dKL/ddelta and the
                             breakdown point delta*(l) = first delta with top-1 flips.
                             The theorem's regularity precondition predicts a NEGATIVE
                             correlation (irregular S breaks the property sooner).
  T2 margin monotonicity   : per row, flip_rate(delta) should be non-decreasing in delta;
                             violations counted and reported per model/layer.
  T3 cross-model analogy   : D_G(m,n) = RMS-normalized Euclidean distance of PRc profiles
                             resampled to a relative-depth grid (K=32); agreement indicators
                             (IC sign match, KL-profile similarity where data exist);
                             pairwise table + correlations over the 28 pairs (exploratory, n small).
  T4 last-layer readout    : exploratory PRc_last <-> IC-sign association with leave-one-model-out
                             threshold stability (H-Analogy-4 candidate).

Outputs:
  data/at2_analogy_audit.json
  figures (exactly 3): analogy_fig01_gamma_profile.png,
                       analogy_fig02_preserved_curves.png   (only when AT-1 data present),
                       analogy_fig03_crossmodel_matrix.png
Run under .venv-gemma4 python: <python> code\\at2_analogy_audit.py
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # aiir-hourglass/code/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT

import numpy as np      # noqa: E402

DATA = os.path.join(TVU, "data")


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resample_profile(profile, n_layers, grid=32):
    """Resample a per-layer scalar profile (uniform in layer index) to `grid` points of relative depth [0,1]."""
    r = np.linspace(0.0, 1.0, len(profile))
    g = np.linspace(0.0, 1.0, grid)
    return np.interp(g, r, np.asarray(profile, dtype=np.float64)), g


def resample_by_r(values, rs, grid=32):
    """Resample values given at explicit relative depths rs (non-uniform, e.g. AT-1 selected layers)."""
    g = np.linspace(0.0, 1.0, grid)
    return np.interp(g, np.asarray(rs, dtype=np.float64), np.asarray(values, dtype=np.float64)), g


def rms_normalize(v):
    v = np.asarray(v, dtype=np.float64)
    rm = float(np.sqrt(np.mean(v ** 2)))
    return (v - v.mean()) / (rm + 1e-12) if rm > 0 else v - v.mean()


def euclid(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return None if den == 0 else float((xc * yc).sum() / den)


def ranks(v):
    order = np.argsort(v)
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(1, len(v) + 1)
    return r


def spearman(x, y):
    if len(np.asarray(x)) < 3:
        return None
    return pearson(ranks(np.asarray(x, dtype=np.float64)), ranks(np.asarray(y, dtype=np.float64)))


# -------------------------------------------------------------------------- data

MODELS = ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b", "qwen2.5-3b",
          "qwen3-4b", "gemma-4-e2b", "pythia-2.8b", "phi-1_5"]

exp5 = load("exp5_metrics.json") or {}
ic_ci = load("exp6_ic_ci.json") or {}
exp9 = load("exp9_hidden_perturbation_metrics.json") or {}
at1 = load("at1_analogy_perturbation_metrics.json")


def prc_profile(key):
    m = exp5.get(key)
    if not m:
        return None, None
    prof = m.get("pr_centered_per_level")
    return np.asarray(prof, dtype=np.float64), len(prof) - 1


# ------------------------------------------------- T1/T2: gamma_hat + breakdown

def at1_rows(key):
    r = (at1 or {}).get(key)
    if not r:
        return []
    out = []
    for row in r.get("per_layer", []):
        deltas, kls, flips = [], [], []
        for dkey, kl in row["kl_by_delta"].items():
            d = float(dkey)
            deltas.append(d)
            kls.append(kl)
            flips.append(row["flip_rate_by_delta"][dkey])
        if len(deltas) < 3:
            continue
        slope = float(np.polyfit(np.asarray(deltas), np.asarray(kls, dtype=np.float64), 1)[0])
        first_flip = next((float(dk) for dk in row["flip_rate_by_delta"]
                           if row["flip_rate_by_delta"][dk] > 0.0), None)
        out.append({"layer": row["layer"], "relative_depth": row["relative_depth"],
                    "gamma_hat": slope, "delta_star": first_flip})
    return out


def exp9_rows(key):
    """gamma_hat from the Exp9 sweep: absolute eps converted to relative via hidden_rms_base."""
    r = exp9.get(key)
    if not r:
        return []
    out = []
    for row in r.get("per_layer", []):
        rms_l = float(row.get("hidden_rms") or 0.0)
        if rms_l <= 0:
            continue
        deltas, kls = [], []
        for ekey, kl in row["kl_by_eps"].items():
            eps = float(ekey)
            deltas.append(eps / rms_l)
            kls.append(kl)
        if len(deltas) < 3:
            continue
        slope = float(np.polyfit(np.asarray(deltas), np.asarray(kls, dtype=np.float64), 1)[0])
        out.append({"layer": row["layer"],
                    "relative_depth": round(row["layer"] / max(r.get("n_layers", 1) - 1, 1), 4),
                    "gamma_hat": slope, "delta_star": None})
    return out


t1_t2 = {}
for key in MODELS:
    rows = at1_rows(key) or exp9_rows(key)
    src = "at1" if (at1 or {}).get(key) else ("exp9" if exp9.get(key) else None)
    t1_t2[key] = {"source": src, "rows": rows}

all_gamma, all_dstar = [], []
for key in MODELS:
    for row in t1_t2[key]["rows"]:
        if row["delta_star"] is not None and row["gamma_hat"] > 0:
            all_gamma.append(row["gamma_hat"])
            all_dstar.append(row["delta_star"])

t1_result = {
    "n_rows": len(all_gamma),
    "spearman_gamma_vs_dstarr": spearman(all_gamma, all_dstar) if len(all_gamma) >= 3 else None,
    "pearson_loggamma_vs_logdstarr": (
        pearson(np.log10(np.asarray(all_gamma)), np.log10(np.asarray(all_dstar)))
        if len(all_gamma) >= 3 else None),
}

t2_violations = {}
if at1:
    for key in MODELS:
        r = (at1 or {}).get(key) or {}
        bad = []
        for row in r.get("per_layer", []):
            fr = [row["flip_rate_by_delta"][f"{d:g}"] for d in sorted(float(k) for k in row["flip_rate_by_delta"])]
            # non-decreasing check over the ascending delta grid
            for i in range(1, len(fr)):
                if fr[i] + 1e-9 < fr[i - 1]:
                    bad.append({"layer": row["layer"], "delta_at": f"{sorted(float(k) for k in row['flip_rate_by_delta'])[i]:g}"})
        t2_violations[key] = bad

# ------------------------------------------------- T3: cross-model matrix (8x8)

GRID = 32
prc_grid = {}
for key in MODELS:
    prof, _n = prc_profile(key)
    if prof is not None:
        prc_grid[key] = rms_normalize(resample_profile(prof, len(prof), GRID)[0])

def kl_profile_grid(k):
    """KL-at-max-eps profile resampled to the relative-depth grid (AT-1 rows carry
    explicit non-uniform depths; Exp9 rows are uniform over all layers)."""
    a = (at1 or {}).get(k)
    if a and a.get("per_layer"):
        max_d_key = f"{max(float(x) for x in a['per_layer'][0]['kl_by_delta']):g}"
        vals = [row["kl_by_delta"][max_d_key] for row in a["per_layer"]]
        rs = [row["relative_depth"] for row in a["per_layer"]]
        return resample_by_r(vals, rs, GRID)[0]
    e9 = exp9.get(k)
    if e9 and e9.get("per_layer"):
        max_eps_key = f"{max(float(x) for x in e9['per_layer'][0]['kl_by_eps']):g}"
        prof = [row["kl_by_eps"][max_eps_key] for row in e9["per_layer"]]
        return resample_profile(prof, len(e9["per_layer"]), GRID)[0]
    return None


kl_grid = {k: rms_normalize(kl_profile_grid(k)) for k in MODELS if kl_profile_grid(k) is not None}

D_G, D_F, ic_sign_match = {}, {}, {}
for m in MODELS:
    for n in MODELS:
        pair = f"{m}__{n}"
        if m in prc_grid and n in prc_grid:
            D_G[pair] = round(euclid(prc_grid[m], prc_grid[n]), 4)
        # KL-profile similarity only where Exp9/AT-1 data exist (3+8 models)
        if m in kl_grid and n in kl_grid:
            D_F[pair] = round(euclid(kl_grid[m], kl_grid[n]), 4)
        icm, icn = ic_ci.get(m), ic_ci.get(n)
        if icm and icn:
            sm = (icm["ic_point_estimate"] > 0) == (icn["ic_point_estimate"] > 0)
            ic_sign_match[pair] = bool(sm)

pairs = []
for m in MODELS:
    for n in MODELS:
        if m < n and f"{m}__{n}" in D_G:
            pairs.append({"pair": f"{m} | {n}", "D_G": D_G[f"{m}__{n}"],
                          "D_F_KL": D_F.get(f"{m}__{n}"),
                          "ic_sign_match": ic_sign_match.get(f"{m}__{n}")})

t3_result = {"pairs": pairs,
             "corr_DG_vs_icSign": (pearson([p["D_G"] for p in pairs],
                                          [int(p["ic_sign_match"]) for p in pairs])
                                  if len(pairs) >= 3 else None),
             "note": "exploratory: n=28 pairs, no multiple-testing correction"}

# ------------------------------------------------- T4: last-layer PRc <-> IC sign (LOO)

prc_last = {k: float(np.asarray(exp5[k]["pr_centered_per_level"])[-1]) for k in MODELS if exp5.get(k)}
ic_sign = {k: int(ic_ci[k]["ic_point_estimate"] > 0) for k in MODELS if ic_ci.get(k)}

def best_threshold(ks):
    """Threshold maximizing sign separation of prc_last vs ic_sign over ks."""
    vals = sorted((prc_last[k], ic_sign[k]) for k in ks)
    best_t, best_acc = None, 0.0
    for i in range(len(vals) - 1):
        if vals[i][1] == vals[i + 1][1]:
            continue
        t = (vals[i][0] + vals[i + 1][0]) / 2.0
        acc = sum(1 for v, s in vals if int(v >= t) == s) / len(vals)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


t4_result = {"full_n8": {}, "leave_one_out": []}
ks = sorted(prc_last.keys())
t_full, acc_full = best_threshold(ks)
t4_result["full_n8"] = {"threshold": t_full, "accuracy": acc_full}
for drop in ks:
    rest = [k for k in ks if k != drop]
    t_loo, acc_loo = best_threshold(rest)
    ok_drop = None
    if t_loo is not None and drop in prc_last:
        pred = int(prc_last[drop] >= t_loo)
        ok_drop = bool(pred == ic_sign[drop])
    t4_result["leave_one_out"].append({"dropped": drop, "threshold_on_rest": t_loo,
                                       "accuracy_on_rest": acc_loo, "dropped_correctly_classified": ok_drop})

# -------------------------------------------------------------------------- JSON out

audit = {
    "task": "AT-2 — analogy-theorem audit from existing metrics + AT-1 sweep",
    "date": "2026-09-23",
    "at1_present": bool(at1),
    "protocol": ("T1: gamma_hat = linear slope dKL/ddelta per model/layer (AT-1 relative delta; Exp9 absolute eps "
                 "converted to relative via hidden_rms_base); delta* = first delta with top-1 flips; theorem regularity "
                 "precondition predicts negative correlation. T2: flip_rate non-decreasing in delta, violations listed. "
                 "T3: D_G = RMS-normalized Euclidean distance of PRc profiles resampled to K=32 relative-depth grid; "
                 "D_F_KL likewise from KL-at-max-eps profiles where data exist; IC sign match from exp6_ic_ci point estimates. "
                 "T4: best single threshold separating last-layer PRc vs IC sign, with leave-one-model-out stability."),
    "t1_regularity_stability": t1_result,
    "t2_margin_monotonicity": t2_violations if at1 else {"note": "AT-1 data absent — skipped"},
    "gamma_hat_by_model": {k: v["rows"] for k, v in t1_t2.items()},
    "t3_crossmodel_analogy": t3_result,
    "prc_last_layer": prc_last,
    "ic_sign_point_estimate": ic_sign,
    "t4_lastlayer_readout_loo": t4_result,
}

OUT_JSON = os.path.join(DATA, "at2_analogy_audit.json")
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(audit, fh, ensure_ascii=False, indent=2)
print(f"audit JSON written: {OUT_JSON}")

# -------------------------------------------------------------------------- figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402


def fig1_gamma():
    has = [k for k in MODELS if t1_t2[k]["rows"]]
    if not has:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in has:
        rows = t1_t2[k]["rows"]
        rd = [r["relative_depth"] for r in rows]
        gh = [max(r["gamma_hat"], 1e-9) for r in rows]
        ax.plot(rd, gh, marker="o", ms=4, label=f"{k} ({t1_t2[k]['source']})")
    ax.set_yscale("log")
    ax.set_xlabel("relative depth l/(L-1)")
    ax.set_ylabel(r"gamma_hat = dKL/ddelta (nats per rel. unit)")
    ax.set_title("AT regularity: amplification slope by relative depth")
    ax.legend(fontsize=7, loc="upper left")
    p = os.path.join(DATA, "analogy_fig01_gamma_profile.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig2_preserved():
    if not at1:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for k in MODELS:
        r = (at1 or {}).get(k)
        if not r or not r.get("per_layer"):
            continue
        for row in r["per_layer"]:
            rd = row["relative_depth"]
            is_last = abs(rd - 1.0) < 1e-6
            ax = axes[1] if is_last else (axes[0] if abs(rd - 0.5) < 0.08 else None)
            if ax is None:
                continue
            ds = sorted(float(x) for x in row["flip_rate_by_delta"])
            pr = [1.0 - row["flip_rate_by_delta"][f"{d:g}"] for d in ds]
            ax.semilogx(ds, pr, marker="o", ms=3, label=k)
    axes[0].set_title("P(top-1 preserved) vs delta — mid-stack (r~0.5)")
    axes[1].set_title("P(top-1 preserved) vs delta — last layer (readout)")
    for ax in axes:
        ax.set_xlabel("delta (relative to RMS(h_base))")
        ax.set_ylabel("preserved fraction of positions")
        ax.legend(fontsize=6, loc="lower left")
    p = os.path.join(DATA, "analogy_fig02_preserved_curves.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig3_matrix():
    idx = {k: i for i, k in enumerate(MODELS)}
    M_G = np.full((8, 8), np.nan)
    M_S = np.zeros((8, 8))
    for pair, v in D_G.items():
        m, n = pair.split("__")
        M_G[idx[m], idx[n]] = v
        M_G[idx[n], idx[m]] = v
    for pair, v in ic_sign_match.items():
        m, n = pair.split("__")
        M_S[idx[m], idx[n]] = 1.0 if v else 0.0
        M_S[idx[n], idx[m]] = 1.0 if v else 0.0
    np.fill_diagonal(M_G, 0.0)
    np.fill_diagonal(M_S, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(M_G, cmap="viridis")
    axes[0].set_title("D_G — PRc-profile distance (RMS-normalized)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    im2 = axes[1].imshow(M_S, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("IC sign match (point estimates)")
    SHORT = ["smollm", "qwen.5b", "tinylama", "qwen3b", "qwen4b", "gemma4", "pythia", "phi1.5"]
    for ax in axes:
        ax.set_xticks(range(8), SHORT, rotation=60, fontsize=7)
        ax.set_yticks(range(8), SHORT, fontsize=7)
    p = os.path.join(DATA, "analogy_fig03_crossmodel_matrix.png")
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


figs = [f for f in (fig1_gamma(), fig2_preserved(), fig3_matrix()) if f]
print("figures written:", *figs, sep="\n  ")

# -------------------------------------------------------------------------- console summary
print("\n--- AT-2 summary ---")
print(f"AT-1 present: {bool(at1)}")
print(f"T1 gamma vs delta*: n={t1_result['n_rows']} spearman={t1_result['spearman_gamma_vs_dstarr']} "
      f"pearson(log)={t1_result['pearson_loggamma_vs_logdstarr']}")
print(f"T3 cross-model: {len(pairs)} pairs, corr(D_G, IC sign match)={t3_result['corr_DG_vs_icSign']}, "
      f"D_F_KL available for {sum(1 for p in pairs if p['D_F_KL'] is not None)} pairs")
print(f"T4 full-n8 threshold={t_full} acc={acc_full}; LOO correct="
      f"{sum(1 for r in t4_result['leave_one_out'] if r['dropped_correctly_classified'])}/"
      f"{len(t4_result['leave_one_out'])}")
