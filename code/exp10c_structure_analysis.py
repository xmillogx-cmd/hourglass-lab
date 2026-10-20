"""Exp 10c stage B — structural-geometry analysis of the dumped activations (CPU only).

Question: do the 8 models' hidden spaces share ONE structure "from different angles"
(invariant to rotation/scale of the space), or is the structure itself model-specific?
(Exp10's scalar curves were MODEL_SPECIFIC; that does not answer this question.)

Per model, per level l=0..L: centered token matrix Xc [n,H]; rows unit-normalized;
kernel K_l = X̂ X̂ᵀ (token-token cosine similarity); centered kernel C_l. Row
normalization makes the comparison invariant to per-vector scale — it measures the
ANGULAR token-token geometry, which is what "structure" means for cross-model
comparison (scales differ wildly between models).

1) Pairwise LINEAR CKA of centered kernels at common depth fractions f=0.05..0.95
   (nearest level per model). LCKA is invariant to orthogonal transforms and scale —
   the direct probe for "same structure from different angles". Values in [0, 1].
2) Identity-preservation profile: LCKA(C_l, C_0) vs LCKA(C_l, C_L), l=1..L per model —
   how much depth-l token geometry still resembles embedding-space / final-layer-space.
3) Rank-32 PCA coords + Procrustes residual (optimal orthogonal alignment) per pair/fraction.
4) PCA 2D snapshots colored by lexical class of the tokens (upper/lower/digit/other).

Reads: data/data/exp10c_act_*.npz, exp10c_tokens_*.json (+ neck levels from
data/data/exp10_activation_structure_8m.json).
Writes: data/data/exp10c_structure.json + 4 PNGs into data/.

Run: <python> exp10c_structure_analysis.py   (CPU only; any venv with numpy+matplotlib)
"""

import json
import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
_DATA = os.path.join(_ROOT, "data")
_RES = os.path.join(_ROOT, "data")
FRACS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95 (19 points)
K_PCA = 32


def load_models():
    models = {}
    for f in sorted(os.listdir(_DATA)):
        if not (f.startswith("exp10c_act_") and f.endswith(".npz")):
            continue
        key = f[len("exp10c_act_"):-len(".npz")]
        z = np.load(os.path.join(_DATA, f))
        n_levels = max(int(k[2:]) for k in z.files if k.startswith("h_")) + 1
        levels = [z[f"h_{i}"].astype(np.float32) for i in range(n_levels)]
        with open(os.path.join(_DATA, f"exp10c_tokens_{key}.json"), encoding="utf-8") as fh:
            toks = json.load(fh)
        models[key] = {"levels": levels, "L": n_levels - 1, "tokens": toks}
    return models


def centered_kernel(X):
    """Centered kernel of row-unit-normalized centered activations (fp16 on disk in RAM)."""
    Xc = (X - X.mean(axis=0)).astype(np.float32)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = (Xc / norms).astype(np.float32)
    K = Xn @ Xn.T
    C = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
    return C.astype(np.float16), Xc


def lcka(A, B):
    """Linear CKA between two centered kernels (float64 accumulation)."""
    a = A.astype(np.float64)
    b = B.astype(np.float64)
    num = float(np.einsum("ij,ij->", a, b))
    den = float(np.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))))
    return num / den if den > 0 else 0.0


def pca_coords(Xc):
    """Top-K_PCA PCA coords [n,K] (U*sqrt(s) form, i.e. Xc @ V_kᵀ) + singular values."""
    U, s, _ = np.linalg.svd(Xc.astype(np.float32), full_matrices=False)
    k = min(K_PCA, len(s))
    return (U[:, :k] * s[:k]).astype(np.float32), s


def procrustes_residual(Pa, Pb):
    """Normalized Procrustes distance after optimal orthogonal alignment of PCA coords."""
    pa = Pa.astype(np.float64)
    pb = Pb.astype(np.float64)
    M = pa.T @ pb
    _, sv, _ = np.linalg.svd(M)
    na = float(pa.ravel() @ pa.ravel())
    nb = float(pb.ravel() @ pb.ravel())
    fit = 2.0 * float(sv.sum())
    res = max(0.0, na + nb - fit) ** 0.5 / (na * nb) ** 0.5
    return float(res)


def token_classes(toks):
    cls = []
    for t in toks:
        s = t.lstrip("#")
        if not s or s.isspace():
            cls.append("other")
        elif s[0].isupper():
            cls.append("upper")
        elif s[0].isdigit():
            cls.append("digit")
        elif s[0].islower():
            cls.append("lower")
        else:
            cls.append("other")
    return cls


def main():
    models = load_models()
    keys = list(models.keys())
    n = len(keys)
    print(f"loaded {n} models from dumps: {keys}")
    if n < 2:
        raise SystemExit("need at least 2 dumped models")

    with open(os.path.join(_DATA, "exp10_activation_structure_8m.json"), encoding="utf-8") as fh:
        exp10 = json.load(fh)
    necks = {k: r.get("neck_level") for k, r in exp10["models"].items() if "status" not in r}

    # --- centered kernels + PCA coords per model/level ----------------------
    kernels, pcas, Xcs = {}, {}, {}
    for key in keys:
        L = models[key]["L"]
        ks, ps, cs = [], [], []
        for l in range(L + 1):
            C_l, Xc = centered_kernel(models[key]["levels"][l])
            P_l, _s = pca_coords(Xc)
            ks.append(C_l)
            ps.append(P_l)
            cs.append(Xc)
        kernels[key], pcas[key], Xcs[key] = ks, ps, cs
        print(f"  {key:<14s} L={L:>2d}: {L + 1} centered kernels [n x n] built")

    def level_of(key, f):
        return max(1, min(models[key]["L"], int(round(f * models[key]["L"]))))

    # --- 1) pairwise CKA at common fractions ---------------------------------
    cka_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = keys[i], keys[j]
            vals = [lcka(kernels[a][level_of(a, f)], kernels[b][level_of(b, f)]) for f in FRACS]
            cka_pairs[f"{a} x {b}"] = [round(float(v), 4) for v in vals]
    mean_cka = {p: round(float(np.mean(v)), 4) for p, v in cka_pairs.items()}
    print(f"\nCKA (linear, centered kernels): mean over pairs = "
          f"{float(np.mean(list(mean_cka.values()))):.3f}   min pair = "
          f"{min(p for p in mean_cka if mean_cka[p] == min(mean_cka.values()))} "
          f"({min(mean_cka.values()):.3f})   max pair = "
          f"{max(p for p in mean_cka if mean_cka[p] == max(mean_cka.values()))} "
          f"({max(mean_cka.values()):.3f})")

    # --- 2) identity-preservation profile ------------------------------------
    identity = {}
    for key in keys:
        L = models[key]["L"]
        vs_emb = [lcka(kernels[key][l], kernels[key][0]) for l in range(1, L + 1)]
        vs_fin = [lcka(kernels[key][l], kernels[key][L]) for l in range(1, L + 1)]
        identity[key] = {
            "vs_embedding": [round(float(v), 4) for v in vs_emb],
            "vs_final": [round(float(v), 4) for v in vs_fin],
        }

    # --- 3) pairwise Procrustes at common fractions ---------------------------
    proc_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = keys[i], keys[j]
            vals = [procrustes_residual(pcas[a][level_of(a, f)], pcas[b][level_of(b, f)])
                    for f in FRACS]
            proc_pairs[f"{a} x {b}"] = [round(float(v), 4) for v in vals]
    mean_proc = {p: round(float(np.mean(v)), 4) for p, v in proc_pairs.items()}

    out = {
        "note": ("stage B of Exp10c — structural geometry (linear CKA on centered kernels of "
                 "row-unit-normalized centered activations; Procrustes at rank-32 PCA coords). "
                 "CKA high across pairs => shared structure up to rotation/scale."),
        "n_models": n,
        "fractions": FRACS,
        "cka_pairs_by_fraction": cka_pairs,
        "mean_cka_per_pair": mean_cka,
        "identity_profile_vs_embedding_or_final": identity,
        "procrustes_pairs_by_fraction": proc_pairs,
        "mean_procrustes_per_pair": mean_proc,
        "neck_levels_from_exp10": {k: v for k, v in necks.items() if v is not None},
    }
    out_path = os.path.join(_DATA, "exp10c_structure.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")

    make_plots(models, keys, necks, cka_pairs, mean_cka, identity, Xcs)


def make_plots(models, keys, necks, cka_pairs, mean_cka, identity, Xcs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- CKA curves ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=130)
    order = sorted(mean_cka.items(), key=lambda kv: -kv[1])
    for p in [order[0][0], order[-1][0]] + [kv[0] for kv in order[1:-1]]:
        lw = 2.4 if p == order[0][0] else (1.8 if p == order[-1][0] else 0.9)
        ax.plot(FRACS, cka_pairs[p], lw=lw, label=p)
    ax.set_xlabel("depth fraction f")
    ax.set_ylabel("linear CKA (pairwise)")
    ax.set_title("Exp10c — pairwise structural similarity (CKA) vs depth\n"
                 "thick = max pair / 2nd thick = min pair; CKA is invariant to rotation & scale")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.4, ncol=2, loc="lower right")
    fig.tight_layout()
    out1 = os.path.join(_RES, "exp10c_cka_curves.png")
    fig.savefig(out1)
    plt.close(fig)
    print(f"plot written: {out1}")

    # --- identity-preservation profile ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=130)
    for ax, ref, title in ((axes[0], "vs_embedding", "CKA vs embedding level (l=0)"),
                           (axes[1], "vs_final", "CKA vs final layer (l=L)")):
        for key in keys:
            L = models[key]["L"]
            xs = np.arange(1, L + 1) / float(L)
            ax.plot(xs, identity[key][ref], lw=1.3, label=key)
        ax.set_xlabel("depth fraction l/L")
        ax.set_ylabel("linear CKA")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    out2 = os.path.join(_RES, "exp10c_identity_profile.png")
    fig.savefig(out2)
    plt.close(fig)
    print(f"plot written: {out2}")

    # --- PCA snapshots --------------------------------------------------------
    CLASS_COLORS = {"upper": "#d62728", "lower": "#1f77b4", "digit": "#ff7f0e", "other": "#999999"}

    def pca_fig(frac, fname, neck_mode=False):
        fig, axes = plt.subplots(3, ceil_div(len(keys), 3), figsize=(6.2 * ceil_div(len(keys), 3), 5.4), dpi=110)
        axes = np.atleast_1d(axes).ravel() if len(keys) > 1 else [fig.add_axes([0, 0, 1, 1])]
        for ax, key in zip(axes, keys):
            L = models[key]["L"]
            l = necks.get(key) if neck_mode else max(1, min(L, int(round(frac * L))))
            U, s, _ = np.linalg.svd(Xcs[key][l], full_matrices=False)
            pts = (U[:, :2] * s[:2]).astype(np.float64)
            cls = token_classes(models[key]["tokens"])
            for c in ("upper", "lower", "digit", "other"):
                m = np.array([x == c for x in cls])
                ax.scatter(pts[m, 0], pts[m, 1], s=2.5, alpha=0.55, color=CLASS_COLORS[c], label=c)
            ax.set_title(f"{key} — level {l}" + (" (neck)" if neck_mode else f" (frac≈{frac})"), fontsize=8.5)
            ax.legend(fontsize=5.5, loc="best")
            ax.grid(alpha=0.25)
        for ax in axes[len(keys):]:   # hide leftover panels of the 3xN grid
            ax.axis("off")
        fig.tight_layout()
        out = os.path.join(_RES, fname)
        fig.savefig(out)
        plt.close(fig)
        print(f"plot written: {out}")

    def ceil_div(a, b):
        return -(-a // b)

    pca_fig(0.5, "exp10c_pca_mid.png")
    pca_fig(None, "exp10c_pca_neck.png", neck_mode=True)


if __name__ == "__main__":
    main()
