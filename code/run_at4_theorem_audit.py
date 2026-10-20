"""AT-4 — CPU audit of EVERY statement of Nikitin's analogy theorem on AT-1/AT-3 data.

No GPU: everything is computed from the JSON records and npz dumps written by AT-1/AT-3,
plus CPU tokenizer loads for the shared-vocabulary cross-model checks.

Statements checked (exogenous parameters epsilon=1e-3, delta_param=5e-3, gamma=3e-4):
  A3-A7   metricity of D: relative RMS Euclidean norm -> by construction; empirical
          triangle check from U6 pairs (D(s1,s2) <= D(s1,s0)+D(s0,s2) = 2a)
  A8/A9   flip rates at delta_param: exact 5e-3 not measured on GPU, so verdicts are
          bracketed by the AT-1/AT-3 grid (3e-3 / 1e-2): PASS if both zero, FAIL if the
          inner bound is nonzero, BRACKETED otherwise
  A10     reference similarity: trivially true in self-analogy reading (F == L); measured
          cross-model on shared vocabulary (restricted top-64 argmax agreement)
  A11     existence beyond delta: flips at 1e-2 > 0 per model
  U4      {D <= eps-gamma} S {F <-> L}: flip rate at 7e-4 == 0?
  U5/U2   existence beyond eps+gamma: flips at 1.3e-3 > 0?
  U6      {D(s1,s2) < delta-2gamma} S {F(s1) <-> F(s2)}: pair agreement from AT-3
  C1      gamma-stability proxy: mean Hellinger distance base vs noisy(7e-4) at the last
          layer (measured program displacement; units differ from state-norm gamma — noted)
  reg 1-6 f = decision margin (top1 - logit_2nd): tau_hat = min margin inside the ball
          (base + 7e-4 + 1.3e-3 runs), L_F = max_p |d margin| / D over both noisy runs,
          condition 6: delta_param <= tau_hat / L_F
  FOL(3)  universal preservation at eps=1e-3 (AT-1 data); FOL(4) existence beyond delta
  SWD     sliced Wasserstein between base/noisy output ensembles at the last layer
          (K=128 common positions, softmax probabilities): W_hat, bootstrap sigma_W
          (B resamples), eps_swd = W + z*sigma with beta=0.05; probabilistic bound check:
          max flip fraction at D <= eps vs 2*beta; cross-model SWD on shared vocab

Writes data/at4_theorem_audit.json and three figures in data/
(fulltheorem_fig01_stability_map.png, fulltheorem_fig02_u6_crossmodel.png,
fulltheorem_fig03_swd_regularity.png).

Run: <python> code\\run_at4_theorem_audit.py   (CPU only; ~10-15 min)
"""

import json
import os
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # aiir-hourglass/code/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys_path_fix = [os.path.join(TVU, "code"), TVU]
import sys
for p in sys_path_fix:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                     # noqa: E402
import tvu_common as C                                  # noqa: E402  (DATA)
from cross_common import MODELS                         # noqa: E402

AT1_JSON = os.path.join(C.DATA, "at1_analogy_perturbation_metrics.json")
AT3_JSON = os.path.join(C.DATA, "at3_full_theorem_metrics.json")
OUT_JSON = os.path.join(C.DATA, "at4_theorem_audit.json")
CKPT_JSON = os.path.join(C.DATA, "at4_models_checkpoint.json")   # per-model crash-safe resume
CKPT_VERSION = 1
FIG_DIR = C.DATA

SEED = 2026
R_PCA = 32               # principal directions kept per model before cross-model CCA (r << S)
Z_BETA = 1.96            # two-sided beta=0.05 quantile for the SWD estimate
B_SELF = 64              # bootstrap resamples, self-analogy SWD
S_SELF = 64              # random projection slices, self-analogy SWD
B_PAIR = 32              # bootstrap resamples, cross-model SWD
S_PAIR = 32              # random projection slices, cross-model SWD

EPSILON, DELTA_PARAM, GAMMA = 1e-3, 5e-3, 3e-4
GRID_KEYS: List[Tuple[float, str]] = [
    (1e-4, "0.0001"), (3e-4, "0.0003"), (7e-4, "0.0007"),
    (1e-3, "0.001"), (1.3e-3, "0.0013"), (3e-3, "0.003"), (1e-2, "0.01"),
]


def sliced_wasserstein(P: np.ndarray, Q: np.ndarray, n_slices: int, rng: "np.random.Generator") -> float:
    """SWD between ensembles P, Q of shape [K, d]: mean over random unit projections
    of the 1-W distance between the projected empirical distributions."""
    K = P.shape[0]
    total = 0.0
    for _ in range(n_slices):
        u = rng.standard_normal(P.shape[1])
        u /= np.linalg.norm(u)
        pu = np.sort(P @ u)
        qu = np.sort(Q @ u)
        total += float(np.abs(pu - qu).sum() / K)
    return total / n_slices


def swd_bootstrap(P: np.ndarray, Q: np.ndarray, n_slices: int, n_boot: int) -> Tuple[float, float]:
    """SWD estimate + bootstrap std over position resamples (with replacement).
    One shared rng so projection directions differ across resamples."""
    K = P.shape[0]
    rng = np.random.default_rng(SEED + 1)
    w0 = sliced_wasserstein(P, Q, n_slices, rng)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, K, size=K)
        boots.append(sliced_wasserstein(P[idx], Q[idx], n_slices, rng))
    return w0, float(np.std(boots))


def softmax_rows(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    X -= X.max(axis=1, keepdims=True)
    E = np.exp(X)
    return (E / E.sum(axis=1, keepdims=True)).astype(np.float32)


def hellinger_mean(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean per-position Hellinger distance between probability ensembles [K, d]."""
    bc = np.sqrt(P * Q).sum(axis=1)
    return float(np.sqrt(np.maximum(0.0, 1.0 - bc)).mean())


def cca_rho(Hm: np.ndarray, Hn: np.ndarray, r_pca: int = R_PCA) -> float:
    """Top canonical correlation between the leading-r_pca principal subspaces of two
    per-position hidden-state ensembles [S,d_m] and [S,d_n] (dimensions may differ).

    Two traps avoided here: (1) paired linear least-squares 'Procrustes' is under-
    determined when d > S positions — lstsq interpolates exactly and returns D~0 for ANY
    two models (first-run values ~3e-6 were that artifact); a single shared orthogonal
    Procrustes matrix needs equal dims. (2) Full-dimension sample CCA with d > S is also
    degenerate: independent random matrices give rho~1, so the state ensembles are first
    PCA-truncated to r_pca directions (r << S=512), which keeps the top sample CC well-
    posed. Returns rho in [0, 1]; higher = more aligned dominant state structure."""
    M = Hm - Hm.mean(axis=0, keepdims=True)      # [S, d_m]
    N = Hn - Hn.mean(axis=0, keepdims=True)      # [S, d_n]

    def _trunc(H):
        r = min(r_pca, H.shape[0] - 1, H.shape[1])
        _, _, Vt = np.linalg.svd(H, full_matrices=False)   # Vt[min(S,d), d]
        P = Vt[:r].T                                      # [d, r] leading directions
        return (H @ P).astype(np.float32)

    Mr, Nr = _trunc(M), _trunc(N)
    s = float(M.shape[0] - 1)
    A = (Mr.T @ Mr) / s                                  # [r_m, r_m]
    Bc = (Nr.T @ Nr) / s                                 # [r_n, r_n]
    C = (Mr.T @ Nr) / s                                  # [r_m, r_n]
    Um, sm, _ = np.linalg.svd(A)
    Un, sn, _ = np.linalg.svd(Bc)
    eps = 1e-9 * max(float(sm.max()), float(sn.max()))
    inv_sqrt_m = (Um / np.sqrt(np.maximum(sm, eps))[:, None]) @ Um.T   # A^-1/2 (pseudo)
    inv_sqrt_n = (Un / np.sqrt(np.maximum(sn, eps))[:, None]) @ Un.T   # B^-1/2 (pseudo)
    W = inv_sqrt_m @ C @ inv_sqrt_n                      # canonical correlations = svd(W).S
    _, sv, _ = np.linalg.svd(W)
    return float(min(1.0, max(float(sv[0]), 0.0)))


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def merge_flip_grid(at1_rec: dict, at3_rec: dict) -> Dict[int, Dict[str, float]]:
    """Per selected layer: union of AT-1 and AT-3 flip-rate grids keyed by %g delta."""
    sel = at3_rec["selected_layers"]
    grid: Dict[int, Dict[str, float]] = {i: {} for i in sel}
    a1_rows = {(r["layer"]): r for r in (at1_rec.get("per_layer") or [])}
    for i in sel:
        row = a1_rows.get(i)
        if row and "flip_rate_by_delta" in row:
            grid[i].update(row["flip_rate_by_delta"])
        srow = (at3_rec.get("singles") or {}).get(str(i))
        if srow:
            for dkey, vals in srow.items():
                grid[i][dkey] = vals["flip_rate"]
    return grid


def audit_model(key: str, at1_rec: dict, at3_rec: dict) -> dict:
    sel = at3_rec["selected_layers"]
    L = at3_rec.get("L_tokens", 512)

    grid = merge_flip_grid(at1_rec, at3_rec)

    # --- Hoare / FOL verdicts from the flip-rate union grid ---
    u4: Dict[int, dict] = {}
    fol3: Dict[int, dict] = {}
    a8v: Dict[int, dict] = {}
    for i in sel:
        g = grid[i]
        r_u4 = g.get("0.0007")
        r_eps = g.get("0.001")
        r_lo, r_hi = g.get("0.003"), g.get("0.01")
        u4[i] = {"flip_rate": r_u4, "pass": bool(r_u4 == 0.0) if r_u4 is not None else False}
        fol3[i] = {"flip_rate": r_eps, "pass": bool(r_eps == 0.0) if r_eps is not None else False}
        if r_lo is None or r_hi is None:
            a8v[i] = {"status": "NO_DATA"}
        elif r_lo == 0.0 and r_hi == 0.0:
            a8v[i] = {"status": "PASS", "flip_3e-03": r_lo, "flip_1e-02": r_hi}
        elif r_lo > 0.0:
            a8v[i] = {"status": "FAIL", "flip_3e-03": r_lo, "flip_1e-02": r_hi}
        else:
            a8v[i] = {"status": "BRACKETED", "flip_3e-03": r_lo, "flip_1e-02": r_hi}

    u5_exist = max((grid[i].get("0.0013") or 0.0) for i in sel)
    a11_exist = max((grid[i].get("0.01") or 0.0) for i in sel)

    # --- U6 pairs + empirical triangle check (A3-A7 proxy on state space) ---
    u6_rows, tri_ratios = [], []
    for i in sel:
        for pr in (at3_rec.get("u6_pairs") or {}).get(str(i), []):
            d12 = pr["d_s1s2"]
            ok_ball = bool(d12 < DELTA_PARAM - 2 * GAMMA)
            u6_rows.append({
                "layer": i, "a": pr["a"], "d_s1s2": d12,
                "inside_delta_minus_2gamma": ok_ball,
                "n_agree": pr["n_agree"], "n_pos": pr["n_pos"],
                "agree_rate": pr["agree_rate"],
                "pass": bool(ok_ball and pr["n_agree"] == pr["n_pos"]),
            })
            tri_ratios.append(d12 / (2 * pr["a"]))

    # --- regularity (operationalized via decision margins) — LAST LAYER ONLY: the AT-3 margins
    #     npz is broken (margins_of indexed vals[:,0] on a (B,S,2) tensor -> shape (1,2)), so per-layer
    #     margin arrays are absent. Derived instead from intact dumps: base margin = top1-top2 of the
    #     top-64 slice over all S positions; noisy margins from the fp16 logits dumps at the K=128
    #     common positions (aligned via pos_indices). ---
    base_npz = np.load(os.path.join(C.DATA, at3_rec["dumps"]["base"]))
    noisy_npz = np.load(os.path.join(C.DATA, at3_rec["dumps"]["noisy_last"]))

    def _ens(arr):  # AT-3 dumped (1, ...) — squeeze the batch axis
        return np.asarray(arr).astype(np.float32).squeeze(axis=0)

    tv = _ens(base_npz["topk_vals"])                        # [S, 64] fp32
    mar_base_all = (tv[:, 0] - tv[:, 1]).astype(np.float64)   # margins over all positions
    pos_arr = np.asarray(at3_rec.get("pos_indices", []), dtype=np.int64)

    def _margin_from_logits(arr):     # (1, K, V) fp16 -> per-position top1 - top2 [K]
        a = _ens(arr)
        t = np.sort(a, axis=-1)[..., ::-1][:, :2]          # descending [K, 2]
        return (t[:, 0] - t[:, 1]).astype(np.float64)

    mar_b = mar_base_all[pos_arr] if len(pos_arr) else mar_base_all
    mar_7e4 = _margin_from_logits(noisy_npz["d_0.0007"])
    mar_13e3 = _margin_from_logits(noisy_npz["d_0.0013"])
    ball = np.concatenate([mar_b, mar_7e4, mar_13e3])      # margins inside the D <= eps+gamma ball
    tau = float(ball.min())
    neg = int((ball < 0).sum())
    lf = max(
        float(np.abs(mar_7e4 - mar_b).max()) / (EPSILON - GAMMA),
        float(np.abs(mar_13e3 - mar_b).max()) / (EPSILON + GAMMA),
    )
    cond6 = bool(tau > 0 and DELTA_PARAM <= tau / lf) if lf > 0 else False
    reg = {
        "scope": ("last layer only, K=128 common positions — AT-3 margins npz broken "
                  "(per-layer margin arrays absent); base margin from top-64 slice over all S"),
        "tau_hat": round(tau, 4),
        "L_F_hat_max_ratio": round(lf, 2),
        "cond3_margin_nonneg_in_ball": bool((ball >= 0).all()),
        "negative_margin_positions_noisy": neg,
        "cond6_delta_le_tau_over_L": cond6,
    }

    # --- SWD self-analogy + C1 proxy at the last layer (from fp16 logits dumps) ---
    P0 = softmax_rows(_ens(base_npz["logits_sub"]))
    swd_self, c1_proxy = {}, {}
    for dkey in ("d_0.0007", "d_0.0013"):
        Qk = softmax_rows(_ens(noisy_npz[dkey]))
        w, sig = swd_bootstrap(P0, Qk, S_SELF, B_SELF)
        swd_self[dkey] = {"W_hat": round(w, 6), "sigma_W": round(sig, 6),
                          "eps_swd": round(w + Z_BETA * sig, 6)}
    c1_proxy["hellinger_base_vs_7e-04_last_layer"] = round(hellinger_mean(P0, softmax_rows(_ens(noisy_npz["d_0.0007"]))), 6)

    # --- probabilistic bound check: max flip fraction at D <= eps vs 2*beta ---
    prob_bound = {
        "max_flip_at_eps_over_layers": max((grid[i].get("0.001") or 0.0) for i in sel),
        "bound_2_beta": round(2 * 0.05, 4),
        "pass": bool(max((grid[i].get("0.001") or 0.0) for i in sel) <= 2 * 0.05),
    }

    return {
        "L_tokens": L,
        "selected_layers": sel,
        "flip_grid_by_layer": {str(i): grid[i] for i in sel},
        "u4_eps_minus_gamma": {str(i): u4[i] for i in sel},
        "fol3_at_epsilon": {str(i): fol3[i] for i in sel},
        "a8_a9_at_delta_param": {str(i): a8v[i] for i in sel},
        "u5_existence_beyond_eps_plus_gamma": {"max_flip_1p3e-03": round(u5_exist, 6),
                                               "satisfied": bool(u5_exist > 0)},
        "a11_fol4_existence_beyond_delta": {"max_flip_1e-02": round(a11_exist, 6),
                                            "satisfied": bool(a11_exist > 0)},
        "u6_pairs": u6_rows,
        "triangle_check_max_ratio_d_s1s2_over_2a": round(max(tri_ratios), 6) if tri_ratios else None,
        "regularity": reg,
        "swd_self_last_layer": swd_self,
        "c1_proxy_gamma_stability": c1_proxy,
        "probabilistic_bound_check": prob_bound,
    }


def cross_model_audit(at3: dict) -> dict:
    """Shared-vocabulary A10 agreement + CCA rho matrix + SWD between model pairs."""
    from transformers import AutoTokenizer

    keys = [e["key"] for e in MODELS]
    paths = {e["key"]: e["path"] for e in MODELS}
    print("  loading tokenizers (CPU) ...")
    vocabs: Dict[str, dict] = {}
    topk: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    subs: Dict[str, np.ndarray] = {}
    acts_last: Dict[str, np.ndarray] = {}
    for key in keys:
        tok = AutoTokenizer.from_pretrained(paths[key])
        vocabs[key] = tok.get_vocab()
        bnpz = np.load(os.path.join(C.DATA, f"at3_base_logits_{key}.npz"))
        # AT-3 dumped (1, ...) arrays — squeeze the batch axis before use
        tids = np.asarray(bnpz["topk_ids"]).astype(np.int32).squeeze(axis=0)
        tvals = np.asarray(bnpz["topk_vals"]).astype(np.float32).squeeze(axis=0)
        topk[key] = (tids, tvals)
        # AT-3 dumped (1, K, V) — squeeze the batch axis before softmaxing rows
        subs[key] = softmax_rows(np.asarray(bnpz["logits_sub"]).astype(np.float32).squeeze(axis=0))
        anpz = np.load(os.path.join(C.DATA, f"at1_act_{key}.npz"))
        last_key = f"h_l{at3[key]['selected_layers'][-1]}"
        acts_last[key] = np.asarray(anpz[last_key]).astype(np.float32).squeeze(axis=0)
        print(f"    {key}: V={len(vocabs[key])}, sub {subs[key].shape}")

    pairs: List[dict] = []
    for mi in range(len(keys)):
        for ni in range(mi + 1, len(keys)):
            m, n = keys[mi], keys[ni]
            common = sorted(set(vocabs[m]) & set(vocabs[n]))
            idx_m = np.array([vocabs[m][w] for w in common], dtype=np.int64)
            idx_n = np.array([vocabs[n][w] for w in common], dtype=np.int64)

            # A10: restricted top-64 argmax agreement at all positions (coverage-filtered);
            # shared vocabulary is defined by token STRING identity, mapped to per-model ids
            word_by_id_m = {int(i): w for w, i in zip(common, idx_m)}
            word_by_id_n = {int(i): w for w, i in zip(common, idx_n)}
            ids_m, vals_m = topk[m]
            ids_n, vals_n = topk[n]
            agree = cov = 0
            for p in range(ids_m.shape[0]):
                best_m = max(((int(i), v) for i, v in zip(ids_m[p], vals_m[p]) if int(i) in word_by_id_m),
                             key=lambda t: t[1], default=None)
                best_n = max(((int(i), v) for i, v in zip(ids_n[p], vals_n[p]) if int(i) in word_by_id_n),
                             key=lambda t: t[1], default=None)
                if best_m is None or best_n is None:
                    continue
                cov += 1
                if word_by_id_m[best_m[0]] == word_by_id_n[best_n[0]]:
                    agree += 1

            # SWD on shared-vocab ensembles (K common positions)
            Pm = subs[m][:, idx_m] / subs[m][:, idx_m].sum(axis=1, keepdims=True)
            Qn = subs[n][:, idx_n] / subs[n][:, idx_n].sum(axis=1, keepdims=True)
            w_pair, sig_pair = swd_bootstrap(Pm.astype(np.float32), Qn.astype(np.float32),
                                             S_PAIR, B_PAIR)

            # CCA top canonical correlation on last-layer hidden states (symmetric)
            rho = cca_rho(acts_last[m], acts_last[n])

            pairs.append({
                "pair": [m, n],
                "shared_vocab_size": len(common),
                "a10_agreement_rate": round(agree / cov, 6) if cov else None,
                "a10_coverage_positions": int(cov),
                "swd_W_hat": round(w_pair, 6),
                "swd_sigma": round(sig_pair, 6),
                "swd_eps_est": round(w_pair + Z_BETA * sig_pair, 6),
                "cca_rho": round(rho, 6),
            })
            print(f"    {m} x {n}: shared-V={len(common)}, a10={pairs[-1]['a10_agreement_rate']}, "
                  f"swd={w_pair:.4g}, cca_rho={rho:.4f}")

    return {"pairs": pairs}


def make_figures(at4: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [e["key"] for e in MODELS]
    models = at4["models"]

    # --- fig 01: stability map (worst-case flip over layers vs delta, param lines) ---
    xlog = np.log10([d for d, _ in GRID_KEYS])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    cmap = plt.get_cmap("tab10")
    for ki, key in enumerate(keys):
        mrec = models[key]
        sel = mrec["selected_layers"]
        worst = [max((mrec["flip_grid_by_layer"][str(i)].get(k) or 0.0) for i in sel)
                 for d, k in GRID_KEYS if any(mrec["flip_grid_by_layer"][str(i)].get(k) is not None for i in sel)]
        xs = [d for d, k in GRID_KEYS if any(mrec["flip_grid_by_layer"][str(i)].get(k) is not None for i in sel)]
        ax.plot(np.log10(xs), worst, "o-", color=cmap(ki % 10), label=key, ms=4, lw=1.2)
    for v, lab, c in [(EPSILON - GAMMA, r"$\varepsilon-\gamma$", "green"),
                      (EPSILON, r"$\varepsilon$", "blue"),
                      (EPSILON + GAMMA, r"$\varepsilon+\gamma$", "orange"),
                      (DELTA_PARAM - 2 * GAMMA, r"$\delta-2\gamma$", "purple"),
                      (DELTA_PARAM, r"$\delta$", "red")]:
        ax.axvline(np.log10(v), color=c, ls="--", lw=0.8)
        ax.text(np.log10(v), 0.97, lab, transform=ax.get_xaxis_transform(),
                rotation=90, fontsize=7, va="top", ha="right", color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D(s,s_0)$ (relative state norm)")
    ax.set_ylabel("worst-case top-1 flip rate over selected layers")
    ax.set_title("AT-4 stability map — full analogy theorem, 8 models\n"
                 f"(exogenous $\\varepsilon$=1e-3, $\\delta$=5e-3, $\\gamma$=3e-4)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fulltheorem_fig01_stability_map.png"))
    plt.close(fig)

    # --- fig 02: U6 disagreement heatmap + CCA rho matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    ax = axes[0]
    n_cols = 8   # every model has exactly len(REL_DEPTHS)=8 selected layers, same order
    M = np.zeros((len(keys), n_cols))
    for ki, key in enumerate(keys):
        sel_key = models[key]["selected_layers"]
        for pr in models[key]["u6_pairs"]:
            if abs(pr["a"] - 2e-3) < 1e-9:
                j = sel_key.index(pr["layer"])
                M[ki, j] = pr["n_pos"] - pr["n_agree"]
    im = ax.imshow(M, aspect="auto", cmap="Reds")
    ax.set_xticks(range(n_cols), [f"r={r:.2f}" for r in (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)], fontsize=7)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_title("U6 disagreements at a=2e-3 (D(s1,s2)$\\approx$2.8e-3 < $\\delta-2\\gamma$)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    P = np.zeros((len(keys), len(keys)))
    for pr in at4["cross_model"]["pairs"]:
        i, j = keys.index(pr["pair"][0]), keys.index(pr["pair"][1])
        P[i, j] = P[j, i] = pr.get("cca_rho", 0.0)
    im = ax.imshow(P, cmap="viridis")
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if P[i, j] > 0:
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Cross-model top canonical correlation (CCA rho)\non last-layer hidden states")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fulltheorem_fig02_u6_crossmodel.png"))
    plt.close(fig)

    # --- fig 03: regularity cond-6 + SWD estimates ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    ax = axes[0]
    for ki, key in enumerate(keys):
        r = models[key]["regularity"]   # last-layer-only (AT-3 margins npz broken — see JSON scope note)
        if r.get("L_F_hat_max_ratio", 0) <= 0:
            continue
        ax.scatter([r["L_F_hat_max_ratio"]], [max(r["tau_hat"], 1e-6) / r["L_F_hat_max_ratio"]],
                   s=45, color=cmap(ki % 10), alpha=0.9, label=key, zorder=3)
    ax.axhline(DELTA_PARAM, color="red", ls="--", lw=1)
    ax.text(ax.get_xlim()[1] * 0.98 if ax.get_xlim()[1] > 0 else 1, DELTA_PARAM,
            r"$\delta$ (cond. 6)", va="bottom", ha="right", color="red", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L_F$ (Lipschitz of the margin)")
    ax.set_ylabel(r"$\tau / L_F$ — allowed ball radius")
    ax.set_title("Regularity condition 6: $\\delta \\leq \\tau/L_F$ per model, last layer\n"
                 "(points above the red line pass; K=128 positions)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    labels, vals, cols = [], [], []
    for key in keys:
        swd = models[key]["swd_self_last_layer"]
        w = swd["d_0.0007"]["W_hat"]
        labels.append(f"{key}\n@{EPSILON - GAMMA:g}")
        vals.append(w)
        cols.append(cmap(keys.index(key) % 10))
    xp = np.arange(len(vals))
    ax.bar(xp, vals, color=cols, alpha=0.85)
    ax.set_xticks(xp, labels, fontsize=6.5)
    ax.set_ylabel("SWD(base, noisy) on last-layer output ensembles")
    ax.set_title(r"Sliced Wasserstein: $W$ at $\varepsilon-\gamma$ (self-analogy,\n"
                 f"{S_SELF} slices, {B_SELF}-bootstrap; see JSON for sigma and eps_swd)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fulltheorem_fig03_swd_regularity.png"))
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # log file must not die on non-cp1251 chars
    except Exception:
        pass
    np.random.seed(SEED)
    print("at4 full-theorem audit (CPU only)")
    at1 = load_json(AT1_JSON)
    at3 = load_json(AT3_JSON)

    # crash-safe checkpoint of the per-model audits (the slow part, ~60+ min total); a rerun skips done models
    models_out: dict = {}
    if os.path.exists(CKPT_JSON):
        with open(CKPT_JSON, encoding="utf-8") as fh:
            ckpt = json.load(fh)
        if ckpt.get("ckpt_version") == CKPT_VERSION:
            for k, v in (ckpt.get("models") or {}).items():
                if isinstance(v, dict) and "u4_eps_minus_gamma" in v:
                    models_out[k] = v
    for entry in MODELS:
        key = entry["key"]
        if key not in at3 or "singles" not in (at3.get(key) or {}):
            print(f"  [{key}] missing AT-3 data — skipped")
            continue
        if key in models_out:
            print(f"\n=== audit [{key}] already checkpointed — skipping ===")
            continue
        print(f"\n=== audit [{key}] ===")
        models_out[key] = audit_model(key, at1.get(key) or {}, at3[key])
        with open(CKPT_JSON, "w", encoding="utf-8") as fh:
            json.dump({"ckpt_version": CKPT_VERSION, "models": models_out},
                      fh, ensure_ascii=False, indent=2)

    print("\n=== cross-model audit (28 pairs) ===")
    cm = cross_model_audit(at3)

    # --- per-statement summary verdicts ---
    n_models = len(models_out)
    u4_pass = sum(1 for m in models_out.values()
                  if all(m["u4_eps_minus_gamma"][str(i)]["pass"] for i in m["selected_layers"]))
    fol3_pass = sum(1 for m in models_out.values()
                    if all(m["fol3_at_epsilon"][str(i)]["pass"] for i in m["selected_layers"]))
    a8_status = {}
    for m in models_out.values():
        for v in m["a8_a9_at_delta_param"].values():
            a8_status[v["status"]] = a8_status.get(v["status"], 0) + 1
    u6_pass = sum(1 for m in models_out.values() for pr in m["u6_pairs"] if pr["pass"])
    u6_tot = sum(len(m["u6_pairs"]) for m in models_out.values())
    cond6_pass = sum(1 for m in models_out.values()
                     if (m.get("regularity") or {}).get("cond6_delta_le_tau_over_L"))
    cond6_tot = n_models
    prob_pass = sum(1 for m in models_out.values() if m["probabilistic_bound_check"]["pass"])

    at4 = {
        "params": {"epsilon": EPSILON, "delta_param": DELTA_PARAM, "gamma": GAMMA,
                   "eps_minus_gamma": round(EPSILON - GAMMA, 6),
                   "eps_plus_gamma": round(EPSILON + GAMMA, 6),
                   "delta_minus_2gamma": round(DELTA_PARAM - 2 * GAMMA, 6),
                   "beta": 0.05, "z_beta": Z_BETA,
                   "note": ("exogenous parameters (paper Table 5 style); exact delta_param=5e-3 flip not measured on GPU — A8/A9 verdicts bracketed by the 3e-3/1e-2 grid; C1 checked via Hellinger proxy in output space (units differ from state-norm gamma); AT-3 margins npz broken (margins_of indexed vals[:,0] of a (B,S,2) tensor -> shape (1,2)) — regularity cond 3/6 derived at the last layer from intact logits dumps (K=128 common positions); cross-model state similarity reported as top canonical correlation cca_rho on the leading R_PCA=32 principal directions of last-layer hidden states (paired linear least-squares 'Procrustes' is under-determined when hidden dim > S=512 positions and returns D~0 for any two models — first-run values ~3e-6 were that artifact, not a result; cca_rho~0.9 across all model pairs reflects shared low-frequency positional trend of per-position states in the dominant PCA directions, i.e. common sequential structure rather than fine-grained architectural similarity)")},
        "axioms": {
            "A3-A7_metricity": ("D = relative RMS Euclidean norm on hidden states — metric BY CONSTRUCTION; "
                                f"empirical triangle check from U6 pairs: max D(s1,s2)/(2a) = "
                                + str(max((m['triangle_check_max_ratio_d_s1s2_over_2a'] or 0.0) for m in models_out.values()))),
            "A10_reference_similarity": ("TRIVIAL in self-analogy reading (F == L); cross-model measured on shared vocabulary — see cross_model.pairs[].a10_agreement_rate"),
        },
        "models": models_out,
        "cross_model": cm,
        "summary_verdicts": {
            "U4_eps_minus_gamma": f"PASS in {u4_pass}/{n_models} models (all selected layers, flip(7e-4)==0)",
            "FOL3_universal_at_epsilon": f"PASS in {fol3_pass}/{n_models} models (flip(1e-3)==0 all layers)",
            "A8_A9_stability_at_delta_param": str(a8_status) + " (per layer; exact 5e-3 bracketed by grid)",
            "U5_existence_beyond_eps_plus_gamma": f"satisfied in {sum(1 for m in models_out.values() if m['u5_existence_beyond_eps_plus_gamma']['satisfied'])}/{n_models} models",
            "A11_FOL4_existence_beyond_delta": f"satisfied in {sum(1 for m in models_out.values() if m['a11_fol4_existence_beyond_delta']['satisfied'])}/{n_models} models",
            "U6_local_stability": f"PASS in {u6_pass}/{u6_tot} pairs (D(s1,s2) < delta-2gamma and F(s1)<->F(s2))",
            "C1_gamma_stability_proxy": "see models[].c1_proxy_gamma_stability (Hellinger, output space)",
            "regularity_cond6_delta_le_tau_over_L": f"PASS in {cond6_pass}/{cond6_tot} models (last layer, K=128 positions; margins npz broken — per-layer coverage absent)",
            "probabilistic_bound_flip_at_eps_le_2beta": f"PASS in {prob_pass}/{n_models} models (max flip at eps <= 0.1)",
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(at4, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {OUT_JSON}")

    make_figures(at4)
    print("figures saved:")
    for f in ("fulltheorem_fig01_stability_map.png", "fulltheorem_fig02_u6_crossmodel.png",
              "fulltheorem_fig03_swd_regularity.png"):
        print(f"  data/{f}")

    print("\n--- summary ---")
    for k, v in at4["summary_verdicts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
