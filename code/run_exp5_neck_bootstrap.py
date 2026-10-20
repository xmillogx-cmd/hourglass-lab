"""Exp 5 (bootstrap) — stability of the neck-level argmin under corpus resampling.

Reviewer p.1 item: "Exp5 -> token bootstrap, neck-argmin stability". The original
run_exp5_anatomy.py computed a single centered PR profile on ONE deterministic
window (the first C.N_TOKENS tokens) and read neck_level = argmin(prc[1:])+1 — an
n=1 estimate with no uncertainty. This variant reuses that exact pr_profile
(imported from run_exp5_anatomy so the SVD/PR math cannot drift) but draws N_BOOT
distinct RANDOM contiguous windows of the corpus (same n_target = C.N_TOKENS, so
the PRc scale matches the original and neck levels are directly comparable),
records each window's argmin, and reports how concentrated that distribution is:

  - reference_neck        : the deterministic neck from data/data/exp5_metrics.json
  - fraction_match_ref    : share of boots whose argmin == reference neck
  - neck_mode / mode_frac : most common boot neck level and its frequency
  - neck_distribution     : {layer_index -> count} over all boots
  - neck_min / neck_max   : range of observed neck levels

A high fraction_match_ref means the mid-stack zone location is robust to which
part of the corpus is fed; a scattered distribution would indicate the "neck" is a
sampling artifact. Contiguous windows (not scattered-token resampling) keep local
context coherent for an autoregressive model — see run_exp7_multisubsample.py.

Writes data/data/exp5_neck_bootstrap_metrics.json incrementally. GPU 1, per-model dtype
from cross_common.MODELS. Launch: python run_exp5_neck_bootstrap.py [model-key,...] [n_boot]
"""

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # cross_model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

from collections import Counter  # noqa: E402

import numpy as np    # noqa: E402
import torch          # noqa: E402

from cross_common import MODELS   # noqa: E402
import tvu_common as C            # noqa: E402
import run_exp5_anatomy as E5     # reuse pr_profile + load_with_dtype verbatim
from run_exp7_multisubsample import full_corpus_ids, random_window  # shared window helpers


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    base_seed = 30_518_726

    exp5_path = os.path.join(C.WORK, "exp5_metrics.json")
    exp5 = {}
    if os.path.exists(exp5_path):
        with open(exp5_path, encoding="utf-8") as fh:
            exp5 = json.load(fh)

    work_json = os.path.join(C.WORK, "exp5_neck_bootstrap_metrics.json")
    results = {}
    if os.path.exists(work_json):   # merge: a later per-model rerun must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        print(f"=== Exp5-bootstrap {key}  boots={n_boot}")
        tok = mod = None
        try:
            tok, mod = E5.load_with_dtype(path, dtype)
            full_ids = full_corpus_ids(tok)

            necks = []
            used_offsets = set()
            for b in range(n_boot):
                rng = np.random.default_rng(base_seed + b)
                s = random_window(full_ids, C.N_TOKENS, rng, used_offsets)
                ids = torch.tensor([full_ids[s:s + C.N_TOKENS]], dtype=torch.long).to(C.DEVICE)
                prc = E5.pr_profile(mod, ids)
                necks.append(int(np.argmin(prc[1:]) + 1))

            counter = Counter(necks)
            ref_neck = exp5.get(key, {}).get("neck_level")
            match_frac = (sum(1 for n in necks if n == ref_neck) / len(necks)) \
                if ref_neck is not None else None
            mode_lvl, mode_cnt = counter.most_common(1)[0]

            results[key] = {
                "path": path,
                "dtype": str(dtype),
                "n_boot": n_boot,
                "reference_neck": int(ref_neck) if ref_neck is not None else None,
                "fraction_match_reference": round(float(match_frac), 4) if match_frac is not None else None,
                "neck_mode": int(mode_lvl),
                "mode_fraction": round(mode_cnt / len(necks), 4),
                "neck_mean": round(float(np.mean(necks)), 3),
                "neck_std": round(float(np.std(necks, ddof=1)) if n_boot > 1 else 0.0, 3),
                "neck_min": int(min(necks)),
                "neck_max": int(max(necks)),
                "neck_distribution": {str(k): v for k, v in sorted(counter.items())},
            }
            print(f"  reference neck={ref_neck}  match_frac={results[key]['fraction_match_reference']}")
            print(f"  mode L{mode_lvl} ({mode_cnt}/{n_boot})  range L{min(necks)}-L{max(necks)}")
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            results[key] = {"path": path, "dtype": str(dtype),
                            "status": f"skipped ({type(e).__name__}: {e})"}
            print(f"  SKIPPED: {e}")
        finally:
            if tok is not None and mod is not None:
                C.free_model(tok, mod)

        with open(work_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"  (incremental save -> {work_json})")

    print("\n--- Exp5 bootstrap summary ---")
    for key, r in results.items():
        if "status" in r:
            print(f"{key:<16} SKIPPED: {r['status']}")
        else:
            fr = r.get("fraction_match_reference")
            fr_s = f"{fr:.2f}" if isinstance(fr, float) else "n/a"
            print(f"{key:<16} ref_neck={r['reference_neck']} match_frac={fr_s} "
                  f"mode=L{r['neck_mode']} range=L{r['neck_min']}-L{r['neck_max']} (n={r['n_boot']})")


if __name__ == "__main__":
    main()
