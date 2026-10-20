"""Zone statistics over the Exp-5 centered PR profiles (CPU only).

For each model we characterize the low-dimensional ("compressed") region of the
stack instead of a single neck layer:

  * n_low      -- number of stack levels (1..L, embeddings excluded) with PRc < T
  * zone       -- maximal contiguous run of sub-T levels containing the neck
                  (start level, end level, width in layers, onset/end fraction
                  of depth, centering = |zone_center - L/2| / (L/2))
  * local_mins -- local minima of PRc below T inside the stack (secondary
                  compression dips; a "double-dip" waist has >1)

T = 2.0 is the strict threshold (PR ~ 2 means an effective rank near 2);
T = 10.0 is a loose one used to catch moderately compressed final blocks
(gemma's neck PRc = 6.0).

Output: data/data/neck_zone_stats.json + a printed table. No GPU, no model loading.
"""

import json
import os

_TV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT


def analyze(pr, T):
    L = len(pr) - 1  # stack levels are 1..L (level 0 = embeddings)
    low = {i for i in range(1, L + 1) if pr[i] < T}
    n_low = len(low)

    neck = min(range(1, L + 1), key=lambda i: pr[i])
    # contiguous run around the neck (only if the neck itself is below T)
    zone = None
    if neck in low:
        a = b = neck
        while a - 1 >= 1 and (a - 1) in low:
            a -= 1
        while b + 1 <= L and (b + 1) in low:
            b += 1
        zone = {
            "start": a,
            "end": b,
            "width": b - a + 1,
            "onset_frac": round(a / L, 3),
            "end_frac": round(b / L, 3),
            "centering": round(abs((a + b) / 2 - (L + 1) / 2) / ((L + 1) / 2), 3),
        }

    # local minima below T: pr[i] <= both neighbors, strictly minimal in window
    lmins = []
    for i in range(2, L):
        if pr[i] < T and pr[i] < pr[i - 1] and pr[i] <= pr[i + 1]:
            lmins.append({"level": i, "pr": round(pr[i], 3)})

    return {
        "T": T,
        "L": L,
        "neck_level": neck,
        "neck_pr": round(pr[neck], 3),
        "neck_depth_frac": round(neck / L, 3),
        "n_low": n_low,
        "low_frac_of_stack": round(n_low / L, 3),
        "zone": zone,
        "local_mins_below_T": lmins,
    }


def main():
    exp5 = json.load(open(os.path.join(TVU, "data", "exp5_metrics.json"), encoding="utf-8"))
    out = {}
    rows = []
    for name, d in exp5.items():
        pr = d["pr_centered_per_level"]
        s2 = analyze(pr, 2.0)
        s10 = analyze(pr, 10.0) if s2["n_low"] == 0 else None
        out[name] = {"strict_T2": s2, "loose_T10": s10}
        z = s2["zone"] or (s10 or {}).get("zone")
        rows.append(
            f"{name:16s} L={s2['L']:2d} neck=L{s2['neck_level']}({s2['neck_depth_frac']:.2f}) "
            f"PRc={s2['neck_pr']} n_low(<2)={s2['n_low']} ({s2['low_frac_of_stack']:.0%}) "
            + (f"zone L{z['start']}-L{z['end']} w={z['width']} onset={z['onset_frac']:.2f} centering={z['centering']:.2f}" if z else "no zone <2")
            + f" lmins={[m['level'] for m in s2['local_mins_below_T']]}"
        )
    out_path = os.path.join(TVU, "data", "neck_zone_stats.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print("\n".join(rows))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
