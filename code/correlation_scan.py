# -*- coding: utf-8 -*-
"""Correlation scan across the 8 cross-model results (CPU-only, no GPU).

Reads data/data/exp5_metrics.json, exp6_ic_ci.json, exp7_multisubsample_metrics.json,
neck_zone_stats.json, exp5_neck_bootstrap_metrics.json + data/models_metadata.json.
Builds one feature row per model, computes Pearson + Spearman for all pairs,
group means (tied vs untied), and pairwise profile-shape correlations.

Output is ASCII-only on purpose (cmd.exe console). Interpretation goes in chat.
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
assert len(MODELS) == 8, MODELS

rows = {}
for m in MODELS:
    e5, zt, mt = exp5[m], zones[m]["strict_T2"], meta[m]
    zl = zones[m].get("loose_T10") or {}
    r7 = mr.get(m) or {}
    b  = boot[m]
    prof = list(e5["pr_centered_per_level"])
    rows[m] = {
        "n_layers": mt["num_hidden_layers"],
        "hidden":   mt["hidden_size"],
        "vocab":    mt["vocab_size"],
        "log10P":   math.log10(mt["params"]["exact"]),
        "tying":    1.0 if mt["tie_word_embeddings"] else 0.0,
        "aniso_raw": e5["aniso_raw"],
        "aniso_cen": e5["aniso_centered"],
        "emb_pr":   prof[0],
        "last_pr":  prof[-1],
        "neck_frac": zt["neck_depth_frac"],
        "neck_pr":  e5["neck_pr_centered"],
        "n_low_t2": float(zt["n_low"]),
        "low_frac": zt["low_frac_of_stack"],
        "zone_w_t10": float(zl.get("zone", {}).get("width") or 0),
        "onset_t10": zl.get("zone", {}).get("onset_frac"),
        "centering_t10": zl.get("zone", {}).get("centering"),
        "ratio_mean": r7.get("ratio_mean"),          # None for gemma (OOM skip)
        "cos_abs": abs(r7["cos_mean"]) if r7.get("cos_mean") is not None else None,
        "ic": icci[m]["ic_point_estimate"],
        "boot_frac_match": b["fraction_match_reference"],
        "boot_neck_std": b["neck_std"],
        "_profile": prof,
    }

FEATS = ["n_layers","hidden","vocab","log10P","tying","aniso_raw","aniso_cen",
         "emb_pr","last_pr","neck_frac","neck_pr","n_low_t2","low_frac",
         "zone_w_t10","ratio_mean","cos_abs","ic","boot_frac_match","boot_neck_std"]

def valid_pairs(fx, fy):
    xs = [(rows[m][fx], rows[m][fy]) for m in MODELS]
    return [p for p in xs if p[0] is not None and p[1] is not None]

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
    rk = [0.0]*len(v)
    i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
        avg = (i + j)/2 + 1.0
        for k in range(i, j+1): rk[order[k]] = avg
        i = j+1
    return rk

def spearman(pairs):
    p = [(float(x), float(y)) for x,y in pairs if x is not None and y is not None]
    if len(p) < 3: return None, len(p)
    r = pearson(list(zip(ranks([x for x,_ in p]), ranks([y for _,y in p]))))
    return r, len(p)

print("="*78)
print("FEATURE TABLE (n=8 models)")
print("="*78)
hdr = "model".ljust(14) + "".join(f[:9].rjust(10) for f in FEATS)
print(hdr)
for m in MODELS:
    line = m.ljust(14)
    for f in FEATS:
        v = rows[m][f]
        line += ("  " if v is None else f"{v:.3g}").rjust(10)
    print(line)

print()
print("="*78)
print("PAIRWISE CORRELATIONS (sorted by |Spearman|, n>=6 valid pairs)")
print("="*78)
res = []
for i in range(len(FEATS)):
    for j in range(i+1, len(FEATS)):
        fx, fy = FEATS[i], FEATS[j]
        pr = valid_pairs(fx, fy)
        if len(pr) < 6: continue
        r_s, n = spearman(pr)
        r_p = pearson(pr)
        res.append((abs(r_s or 0), fx, fy, r_p, r_s, n))
res.sort(reverse=True)
for a, fx, fy, rp, rs, n in res:
    if a < 0.25 and n == 8: continue   # trim weak full-set pairs from output
    print(f"{fx:>14} x {fy:<14}  pearson={rp:+.3f}  spearman={rs:+.3f}  (n={n})")

print()
print("="*78)
print("GROUP MEANS: tied vs untied")
print("="*78)
for f in ["aniso_raw","aniso_cen","ratio_mean","neck_pr","ic","emb_pr"]:
    t = [rows[m][f] for m in MODELS if rows[m]["tying"]==1.0 and rows[m][f] is not None]
    u = [rows[m][f] for m in MODELS if rows[m]["tying"]==0.0 and rows[m][f] is not None]
    tm = sum(t)/len(t) if t else float("nan")
    um = sum(u)/len(u) if u else float("nan")
    print(f"{f:<12} tied(n={len(t)})={tm:8.4g}   untied(n={len(u)})={um:8.4g}")

print()
print("="*78)
print("PROFILE SHAPE MATRIX (Spearman of per-level PRc, resampled to 25 depth pts)")
print("="*78)
def resample(prof, n=25):
    L = len(prof)-1
    out = []
    for k in range(n):
        pos = (k+0.5)*L/n   # midpoints of equal depth bins
        i = int(pos); frac = pos-i
        v = prof[i]*(1-frac) + (prof[min(i+1, len(prof)-1)]*frac)
        out.append(v)
    return out

profs = {m: resample(rows[m]["_profile"]) for m in MODELS}
print("model".ljust(14) + "".join(x[:9].rjust(10) for x in MODELS))
for a in MODELS:
    line = a.ljust(14)
    for b2 in MODELS:
        rs, _ = spearman(list(zip(profs[a], profs[b2])))
        line += (f"{rs:+.3f}").rjust(10) if rs is not None else "  -".rjust(10)
    print(line)

# regime split: collapse (n_low_t2>0) vs floor (n_low_t2==0)
coll = [m for m in MODELS if rows[m]["n_low_t2"] > 0]
flr  = [m for m in MODELS if rows[m]["n_low_t2"] == 0]
def mean_pair(a_list, b_list):
    vals = []
    for x in a_list:
        for y in b_list:
            rs,_ = spearman(list(zip(profs[x], profs[y])))
            if rs is not None: vals.append(rs)
    return sum(vals)/len(vals) if vals else float("nan")
print()
print(f"regime groups: collapse={coll}")
print(f"               floor  ={flr}")
print(f"mean within-collapse profile corr = {mean_pair(coll, coll):+.3f}")
print(f"mean within-floor profile corr    = {mean_pair(flr, flr):+.3f}")
print(f"mean cross-regime profile corr    = {mean_pair(coll, flr):+.3f}")
