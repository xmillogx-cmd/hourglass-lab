"""Point 5 (reviewer): robustness of the low-dim "neck zone" to the PRc threshold.

Recomputes the Exp-5 zone statistics at T in {1.0, 1.25, 1.5, 2.0, 2.5, 3.0}
using EXISTING per-level centered PR profiles (data/data/exp5_metrics.json) and the
EXACT same `analyze()` logic as cross_model/compute_zone_stats.py (no GPU, no model
loading). Shows that the identified neck level is threshold-invariant and the zone
onset/width are stable across 1.5 / 2.0 / 3.0 -> justifies that T=2.0 is not an
arbitrary choice.

Writes data/data/point5_threshold_sensitivity.json + prints a table. Stdlib only.
"""
import json
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
TVU = _ROOT


def analyze(pr, T):
    L = len(pr) - 1  # stack levels are 1..L (level 0 = embeddings)
    low = {i for i in range(1, L + 1) if pr[i] < T}
    n_low = len(low)

    neck = min(range(1, L + 1), key=lambda i: pr[i])
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

    lmins = []
    for i in range(2, L):
        if pr[i] < T and pr[i] < pr[i - 1] and pr[i] <= pr[i + 1]:
            lmins.append({"level": i, "pr": round(pr[i], 3)})

    return {
        "T": T,
        "L": L,
        "neck_level": neck,
        "neck_pr": round(pr[neck], 3),
        "n_low": n_low,
        "low_frac_of_stack": round(n_low / L, 3),
        "zone": zone,
        "local_mins_below_T": lmins,
    }


THRESHOLDS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def main():
    exp5 = json.load(open(os.path.join(TVU, "data", "exp5_metrics.json"), encoding="utf-8"))
    out = {}
    for name, d in exp5.items():
        pr = d["pr_centered_per_level"]
        rows = [analyze(pr, T) for T in THRESHOLDS]
        necks = sorted({r["neck_level"] for r in rows})
        onset = [ (r["zone"]["onset_frac"] if r["zone"] else None) for r in rows ]
        out[name] = {
            "L": pr and len(pr) - 1,
            "per_threshold": rows,
            "neck_level_stable_across_T": necks[0] if len(necks) == 1 else None,
            "neck_levels_seen": necks,
        }

    out_path = os.path.join(TVU, "data", "point5_threshold_sensitivity.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)

    print(f"{'model':16s} " + "  ".join(f"T={t:<4}" for t in THRESHOLDS))
    for name, o in out.items():
        cells = []
        for r in o["per_threshold"]:
            z = r["zone"]
            if z:
                cells.append(f"L{r['neck_level']} nlow={r['n_low']:2d} w={z['width']:2d} on={z['onset_frac']:.2f}")
            else:
                cells.append(f"L{r['neck_level']} nlow= 0  --")
        stable = "STABLE" if o["neck_level_stable_across_T"] is not None else f"drifts {o['neck_levels_seen']}"
        print(f"{name:16s} " + " | ".join(cells) + f"   [neck {stable}]")

    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
