"""Margin-degeneracy diagnostic on AT-3b fp32 dumps (pure numpy, no model load).

User hypothesis B under test: gemma is unstable even in uniform fp32 because its
stored weights (natively bfloat16) leave systematically near-degenerate top1-top2
decision margins that any perturbation flips. Also characterizes flip locality:
do flips concentrate exactly where the base margin is smallest?

Inputs : data/at3b_margins_<model>.npz  (AT-3b, L=512, per-layer delta grid)
         models/<model>/config.json     (architecture facts for hypothesis A/B)
Output : stdout table only (re-runnable; no new result files)

Keys in the npz: margin_base [S], topk_ids_base [S,64]; per (layer i, delta d):
l{i}_d{d:g} margin [S], l{i}_d{d:g}_ids [S,2] int32, l{i}_d{d:g}_vals [S,2].
"""

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
TVU = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(TVU, "data")
MODELS_DIR = os.path.join(os.path.dirname(TVU), "models")

ALL_DELTAS = (1e-4, 3e-4, 7e-4, 1.3e-3)

LAYERS = {
    "smollm-135m": [2, 6, 10, 14, 19, 23, 26, 29],
    "qwen2.5-0.5b": [2, 5, 8, 12, 15, 18, 21, 23],
    "tinyllama-1.1b": [2, 4, 7, 10, 14, 17, 19, 21],
    "qwen2.5-3b": [3, 7, 12, 18, 23, 28, 32, 35],
    "qwen3-4b": [3, 7, 12, 18, 23, 28, 32, 35],
    "gemma-4-e2b": [3, 7, 12, 17, 22, 27, 31, 34],
    "pythia-2.8b": [2, 6, 11, 16, 20, 25, 28, 31],
    "phi-1_5": [2, 5, 8, 12, 15, 18, 21, 23],
}

MODELS = list(LAYERS)


def cfg_facts(key):
    """Architecture facts from config.json (hypothesis A/B support)."""
    p = os.path.join(MODELS_DIR, key, "config.json")
    if not os.path.isfile(p):
        return {"note": "no config"}
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    tc = c.get("text_config", {}) or {}
    out = {
        "dtype_stored": c.get("dtype", tc.get("dtype")),
        "tie_word_embeddings": c.get("tie_word_embeddings", tc.get("tie_word_embeddings")),
    }
    for fld in ("enable_moe_block", "num_experts", "hidden_size_per_layer_input",
                "num_kv_shared_layers", "final_logit_softcapping", "sliding_window"):
        if fld in c or fld in tc:
            out[fld] = c.get(fld, tc.get(fld))
    if "untie_embeddings" in c:  # GPTNeoX-style (pythia)
        out["tie_word_embeddings"] = not c["untie_embeddings"]
    return out


def base_stats(m):
    p05, p1, p5, p50 = np.percentile(m, [0.5, 1, 5, 50])
    return {
        "min": float(m.min()), "p0.5": float(p05), "p1": float(p1),
        "p5": float(p5), "med": float(p50), "mean": float(m.mean()),
        "max": float(m.max()),
        "frac<0.05": float((m < 0.05).mean()),
        "frac<0.2": float((m < 0.2).mean()),
        "frac<0.5": float((m < 0.5).mean()),
    }


def flip_cell(z, base_ids1, i, d):
    """(flip_mask, mar_base, mar_pert) for one (layer, delta) cell."""
    key = "l%d_d%g" % (i, d)
    if key not in z.files:
        return None
    ids = z[key + "_ids"][:, 0]
    mask = base_ids1 != ids
    return mask, z["margin_base"].copy(), z[key].astype(np.float64)


def main():
    print("=" * 78)
    print("AT-3b margin-degeneracy diagnostic (pure numpy on saved dumps)")
    print("=" * 78)

    # --- architecture facts table -------------------------------------------
    print("\n--- config.json facts (hypotheses A/B support) ---")
    for key in MODELS:
        f = cfg_facts(key)
        bits = []
        if "dtype_stored" in f and f["dtype_stored"]:
            bits.append("stored=%s" % f["dtype_stored"])
        if f.get("tie_word_embeddings") is not None:
            bits.append("tied_emb=%s" % f["tie_word_embeddings"])
        for fld in ("enable_moe_block", "num_experts", "hidden_size_per_layer_input",
                    "num_kv_shared_layers", "final_logit_softcapping", "sliding_window"):
            if fld in f:
                bits.append("%s=%s" % (fld, f[fld]))
        print("  %-14s %s" % (key, "; ".join(bits)))

    # --- base margin distributions ------------------------------------------
    print("\n--- base top1-top2 logit-gap distribution (S=512 positions) ---")
    hdr = ("model            min     p0.5    p1      p5      med     mean"
           "   max  frac<0.05 frac<0.2 frac<0.5")
    print(hdr)
    base_margins = {}
    for key in MODELS:
        z = np.load(os.path.join(DATA, "at3b_margins_%s.npz" % key))
        m = z["margin_base"].astype(np.float64)
        base_margins[key] = (z, m)
        s = base_stats(m)
        print("%-14s %7.3f %8.4f %7.4f %7.4f %7.4f %7.4f"
              "  %6.2f  %.3f     %.3f   %.3f"
              % (key, s["min"], s["p0.5"], s["p1"], s["p5"], s["med"],
                 s["mean"], s["max"], s["frac<0.05"], s["frac<0.2"], s["frac<0.5"]))

    # --- flip grid per model + margin at flips -------------------------------
    print("\n--- flip cells: rate%% | base-margin med @flips / @non-flips ---")
    for key in MODELS:
        z, m = base_margins[key]
        base_ids1 = z["topk_ids_base"][:, 0]
        rows = []
        any_flip = False
        for i in LAYERS[key]:
            cells = []
            for d in ALL_DELTAS:
                cell = flip_cell(z, base_ids1, i, d)
                if cell is None or not cell[0].any():
                    cells.append("  -   ")
                    continue
                mask, mb, mp = cell
                nf = int(mask.sum())
                any_flip = True
                med_f = float(np.median(mb[mask]))
                med_n = float(np.median(mb[~mask])) if (~mask).any() else float("nan")
                cells.append("%3d%% %5.3f/%5.3f" % (100.0 * nf / len(mask), med_f, med_n))
            rows.append("  L%-2d | %s%s | %s" % (i, "", " ".join(cells[:2]), " ".join(cells[2:])))
        print("  %s%s" % (key, ("" if any_flip else " (no flips)")))
        for r in rows:
            if "%" in r:
                print(r)

    # --- binned flip probability vs base margin (locality check) -------------
    def binned(key, i, d):
        z, m = base_margins[key]
        cell = flip_cell(z, z["topk_ids_base"][:, 0], i, d)
        if cell is None or not cell[0].any():
            print("  %s L%d @%g: no flips" % (key, i, d))
            return
        mask, mb, mp = cell
        qs = np.quantile(mb, [0, .2, .4, .6, .8, 1.0])
        print("  %s L%d @%g: flip%% by base-margin quantile bin (bin range -> rate)" % (key, i, d))
        for b in range(5):
            sel = (mb >= qs[b]) & (mb < qs[b + 1]) if b < 4 else (mb >= qs[4])
            n_b = int(sel.sum())
            r = 100.0 * mask[sel].sum() / max(n_b, 1)
            print("    [%7.3f .. %7.3f]  n=%-4d flip=%.1f%%" % (qs[b], qs[b + 1], n_b, r))

    print("\n--- locality: flips vs base-margin quantile bins ---")
    for key, i, d in [("gemma-4-e2b", 3, 1e-4), ("gemma-4-e2b", 7, 1e-4),
                      ("gemma-4-e2b", 12, 1e-4), ("pythia-2.8b", 2, 1e-4)]:
        binned(key, i, d)

    # control: a model that is stable at this delta but flips later
    zc, mc = base_margins["qwen2.5-0.5b"]
    cell = flip_cell(zc, zc["topk_ids_base"][:, 0], 5, 3e-4)
    if cell and cell[0].any():
        mask, mb, mp = cell
        print("\n  control qwen2.5-0.5b L5 @3e-4: flip=%d%%; base-margin med"
              " @flips=%.4f vs @non-flips=%.4f; |mar_pert-mar_base| mean at flips=%.4f"
              % (100.0 * mask.mean(), np.median(mb[mask]), np.median(mb[~mask]),
                 float(np.mean(mp[mask] - mb[mask]))))

    # gemma early-layer flip margin collapse detail
    print("\n--- gemma L3 @1e-4: margin change at flips ---")
    z, m = base_margins["gemma-4-e2b"]
    cell = flip_cell(z, z["topk_ids_base"][:, 0], 3, 1e-4)
    mask, mb, mp = cell
    df = mp[mask] - mb[mask]
    print("  n_flips=%d; base-margin at flips: min=%.4f med=%.4f max=%.4f"
          % (int(mask.sum()), mb[mask].min(), np.median(mb[mask]), mb[mask].max()))
    print("  margin delta (pert-base) at flips: min=%.4f med=%.4f mean=%.4f; "
          "frac negative=%d%%" % (df.min(), np.median(df), df.mean(), int(100.0 * (df < 0).mean())))
    print("  non-flips margin delta: med=%.4f (context)" % float(np.median(mp[~mask] - mb[~mask])))


if __name__ == "__main__":
    main()
