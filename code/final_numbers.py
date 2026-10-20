# -*- coding: utf-8 -*-
"""Final precise numbers for the correlation-findings proofreading doc (CPU-only, ASCII output).

Prints: 1) full-precision feature table (markdown), 2) group means tied/untied,
3) selected correlations overall / within-tied / within-untied.
"""
import json, io, os, math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
BASE = _ROOT

def load(p):
    with io.open(os.path.join(BASE, p), encoding="utf-8") as f:
        return json.load(f)

exp5   = load("data/exp5_metrics.json")
icci   = load("data/exp6_ic_ci.json")
mr     = load("data/exp7_multisubsample_metrics.json")
zones  = load("data/neck_zone_stats.json")
boot   = load("data/exp5_neck_bootstrap_metrics.json")
meta   = load("data/models_metadata.json")["models"]

MODELS = list(exp5.keys())

def pearson(pairs):
    n = len(pairs)
    if n < 3: return None
    mx = sum(x for x,_ in pairs)/n; my = sum(y for _,y in pairs)/n
    sxy = sum((x-mx)*(y-my) for x,y in pairs)
    sx = math.sqrt(sum((x-mx)**2 for x,_ in pairs))
    sy = math.sqrt(sum((y-my)**2 for _,y in pairs))
    return sxy/(sx*sy) if sx>0 and sy>0 else None

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0]*len(v); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
        avg = (i+j)/2 + 1.0
        for k in range(i, j+1): rk[order[k]] = avg
        i = j+1
    return rk

def spearman(pairs):
    p = [(float(x), float(y)) for x,y in pairs if None not in (x,y)]
    if len(p) < 3: return None, len(p)
    r = pearson(list(zip(ranks([a for a,_ in p]), ranks([b for _,b in p]))))
    return r, len(p)

rows = {}
for m in MODELS:
    e5, zt, mt = exp5[m], zones[m]["strict_T2"], meta[m]
    r7 = mr.get(m) or {}
    prof = list(e5["pr_centered_per_level"])
    rows[m] = {
        "L": mt["num_hidden_layers"], "hidden": mt["hidden_size"],
        "log10P": math.log10(mt["params"]["exact"]),
        "tying": 1.0 if mt["tie_word_embeddings"] else 0.0,
        "aniso_raw": e5["aniso_raw"], "aniso_cen": e5["aniso_centered"],
        "emb_pr": prof[0], "last_pr": prof[-1],
        "neck_frac": zt["neck_depth_frac"], "neck_pr": e5["neck_pr_centered"],
        "n_low_t2": float(zt["n_low"]), "low_frac": zt["low_frac_of_stack"],
        "ratio_mean": r7.get("ratio_mean"),
        "cos_abs": abs(r7["cos_mean"]) if r7.get("cos_mean") is not None else None,
        "ic": icci[m]["ic_point_estimate"],
        "boot_match": boot[m]["fraction_match_reference"],
        "boot_std": boot[m]["neck_std"],
    }

print("="*70)
print("FEATURE TABLE (full precision)")
print("="*70)
hdr = ["model","L","h","log10P","tying","aniso_raw","aniso_cen","emb_pr","last_pr",
       "neck_frac","neck_pr","n_low_t2","low_frac","ratio_mean","cos_abs","ic","boot_match","boot_std"]
print("| " + " | ".join(hdr) + " |")
for m in MODELS:
    r = rows[m]
    cells = [m, str(r["L"]), str(r["hidden"]), f'{r["log10P"]:.3f}',
             "tied" if r["tying"]==1 else "untied",
             f'{r["aniso_raw"]:.4f}', f'{r["aniso_cen"]:.4f}',
             f'{r["emb_pr"]:.2f}', f'{r["last_pr"]:.2f}',
             f'{r["neck_frac"]:.3f}', f'{r["neck_pr"]:.2f}',
             str(int(r["n_low_t2"])), f'{r["low_frac"]:.3f}']
    for k in ("ratio_mean","cos_abs"):
        v = r[k]
        cells.append("-" if v is None else f"{v:.4g}")
    cells += [f'{r["ic"]:+.3f}', f'{r["boot_match"]:.2f}', f'{r["boot_std"]:.3f}']
    print("| " + " | ".join(cells) + " |")

print()
print("="*70)
print("GROUP MEANS (full precision)")
print("="*70)
for f in ["aniso_raw","aniso_cen","ratio_mean","neck_pr","ic","emb_pr"]:
    t = [rows[m][f] for m in MODELS if rows[m]["tying"]==1.0 and rows[m][f] is not None]
    u = [rows[m][f] for m in MODELS if rows[m]["tying"]==0.0 and rows[m][f] is not None]
    tm = sum(t)/len(t) if t else float("nan")
    um = sum(u)/len(u) if u else float("nan")
    print(f"{f:<12} tied(n={len(t)})={tm:.4f}   untied(n={len(u)})={um:.4f}")

print()
print("="*70)
print("SELECTED CORRELATIONS: overall / within-tied / within-untied")
print("="*70)
KEY = [("aniso_raw","emb_pr"),("hidden","aniso_cen"),("last_pr","ic"),
       ("tying","ratio_mean"),("tying","aniso_raw"),("tying","emb_pr"),("tying","last_pr"),
       ("aniso_raw","cos_abs"),("last_pr","ratio_mean"),("ratio_mean","ic")]

def sel(fx, fy, subset):
    xs = [(rows[m][fx], rows[m][fy]) for m in MODELS if m in subset]
    return [p for p in xs if None not in p]

tied   = [m for m in MODELS if rows[m]["tying"]==1.0]
untied = [m for m in MODELS if rows[m]["tying"]==0.0]
def fmt(v): return "-" if v is None else f"{v:+.3f}"

for fx, fy in KEY:
    po = sel(fx,fy,MODELS); pt = sel(fx,fy,tied); pu = sel(fx,fy,untied)
    ro_s, no = spearman(po); rp_o = pearson(po)
    rt_s, nt  = spearman(pt); rp_t = pearson(pt)
    ru_s, nu  = spearman(pu); rp_u = pearson(pu)
    print(f"{fx} x {fy}")
    print(f"   overall(n={no}): pearson={fmt(rp_o)}  spearman={fmt(ro_s)}")
    print(f"   tied   (n={nt}): pearson={fmt(rp_t)}  spearman={fmt(rt_s)}")
    print(f"   untied (n={nu}): pearson={fmt(rp_u)}  spearman={fmt(ru_s)}")

print()
print("="*70)
print("delta_aniso x ratio_mean")
print("="*70)
for m in MODELS:
    d = rows[m]["aniso_raw"] - rows[m]["aniso_cen"]
    print(f"{m:<14} delta={d:+.4f}")
pairs_o = [(rows[m]["aniso_raw"]-rows[m]["aniso_cen"], rows[m]["ratio_mean"]) for m in MODELS if rows[m]["ratio_mean"] is not None]
print(f"overall(n={len(pairs_o)}): pearson={pearson(pairs_o):+.3f} spearman={spearman(pairs_o)[0]:+.3f}")
pairs_t = [(rows[m]["aniso_raw"]-rows[m]["aniso_cen"], rows[m]["ratio_mean"]) for m in tied if rows[m]["ratio_mean"] is not None]
print(f"tied  (n={len(pairs_t)}): pearson={pearson(pairs_t):+.3f} spearman={'-' if spearman(pairs_t)[0] is None else f'{spearman(pairs_t)[0]:+.3f}'}")
