# -*- coding: utf-8 -*-
"""Targeted follow-up checks on the correlation scan (CPU-only, ASCII output).

1) delta_aniso = aniso_raw - aniso_centered  ("tying component") vs ratio_mean / tying.
2) IC sign split by last-level PRc (threshold search).
3) Profile-shape correlations grouped by NECK PLACEMENT family, not regime:
   early-neck {qwen0.5b, tinyllama, qwen3b, pythia, phi} vs distributed {smollm, gemma}.
"""
import json, io, os, math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
BASE = _ROOT

def load(p):
    with io.open(os.path.join(BASE, p), encoding="utf-8") as f:
        return json.load(f)

exp5  = load("data/exp5_metrics.json")
icci  = load("data/exp6_ic_ci.json")
mr    = load("data/exp7_multisubsample_metrics.json")
meta  = load("data/models_metadata.json")["models"]
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

print("="*78)
print("BONUS 1: delta_aniso = raw - centered ('tying component')")
print("="*78)
rows = {}
for m in MODELS:
    e5 = exp5[m]
    d = e5["aniso_raw"] - e5["aniso_centered"]
    ratio = mr.get(m, {}).get("ratio_mean") if "n_reps" in mr.get(m, {}) else None
    rows[m] = (d, meta[m]["tie_word_embeddings"], ratio)
for m in MODELS:
    d, t, r = rows[m]
    print(f"{m:<14} delta_aniso={d:+.4f}  tying={'yes' if t else 'no':<3}  ratio_mean={'-' if r is None else f'{r:.2f}'}")
pairs_d_r = [(rows[m][0], rows[m][2]) for m in MODELS if rows[m][2] is not None]
rp, _ = spearman(pairs_d_r)
print(f"corr(delta_aniso, ratio_mean): pearson={pearson(pairs_d_r):+.3f}  spearman={rp:+.3f} (n={len(pairs_d_r)})")

# within tied only
pairs_t = [(rows[m][0], rows[m][2]) for m in MODELS if meta[m]["tie_word_embeddings"] and rows[m][2] is not None]
print(f"  within TIED: pearson={pearson(pairs_t):+.3f} (n={len(pairs_t)})")

print()
print("="*78)
print("BONUS 2: IC sign vs last-level PRc")
print("="*78)
tab = []
for m in MODELS:
    prof = list(exp5[m]["pr_centered_per_level"])
    tab.append((prof[-1], icci[m]["ic_point_estimate"], m))
tab.sort()
signs_ok = True
pos_min = min(x[0] for x in tab if x[1] > 0)
neg_max = max(x[0] for x in tab if x[1] < 0)
for lp, ic, m in tab:
    print(f"{m:<14} last_pr={lp:8.2f}   IC={ic:+.3f}")
print(f"max last_pr among IC<0 = {neg_max:.2f}; min last_pr among IC>0 = {pos_min:.2f}")
if neg_max < pos_min:
    print("SPLIT IS CLEAN: every model with last_pr > %.1f has IC>0, below it IC<0" % ((neg_max+pos_min)/2))

print()
print("="*78)
print("BONUS 3: profile corr by NECK-PLACEMENT family (early vs distributed)")
print("="*78)
def resample(prof, n=25):
    L = len(prof)-1
    out = []
    for k in range(n):
        pos = (k+0.5)*L/n
        i = int(pos); frac = pos-i
        out.append(prof[i]*(1-frac) + prof[min(i+1, len(prof)-1)]*frac)
    return out
profs = {m: resample(list(exp5[m]["pr_centered_per_level"])) for m in MODELS}

EARLY  = ["qwen2.5-0.5b","tinyllama-1.1b","qwen2.5-3b","pythia-2.8b","phi-1_5"]
DISTR  = ["smollm-135m","gemma-4-e2b"]

def mean_pair(A, B):
    vals = []
    for x in A:
        for y in (A if len(B)>1 else B):
            if x != y:
                rs,_ = spearman(list(zip(profs[x], profs[y])))
                if rs is not None: vals.append(rs)
    return sum(vals)/len(vals)

w_early  = mean_pair(EARLY, EARLY)
w_dist   = spearman(list(zip(profs[DISTR[0]], profs[DISTR[1]])))[0]
cross    = []
for a in EARLY:
    for b in DISTR:
        rs,_ = spearman(list(zip(profs[a], profs[b])))
        if rs is not None: cross.append((a[:9], b[:9], rs))
print(f"mean within early-neck cluster (10 pairs)  = {w_early:+.3f}")
print(f"distributed pair smollm x gemma            = {w_dist:+.3f}")
print(f"mean early vs distributed ({len(cross)} pairs)      = {sum(c[2] for c in cross)/len(cross):+.3f}   per-pair: " +
      ", ".join(f"{a}x{b}={c:+.2f}" for a,b,c in cross))

# p.2 models vs their early-neck Qwen/TinyLlama relatives specifically
print()
print("p.2 models vs early-neck relatives (same 'early dip + expansion' shape):")
for rel in ["qwen2.5-0.5b","tinyllama-1.1b","qwen2.5-3b"]:
    for p2 in ["pythia-2.8b","phi-1_5"]:
        rs,_ = spearman(list(zip(profs[p2], profs[rel])))
        print(f"  {p2} x {rel:<14} = {rs:+.3f}")
