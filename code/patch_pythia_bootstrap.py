"""One-shot CPU fix for the degraded pythia entry in exp5_neck_bootstrap_metrics.json.

The GPU run (2026-09-19 15:56) started BEFORE the anatomy retry finished (18:13), so
reference_neck was None and fraction_match_reference could not be computed at run time.
Both fields are now pure functions of already-recorded data:

    reference_neck            = exp5_metrics.json["pythia-2.8b"]["neck_level"]
    fraction_match_reference  = neck_distribution[str(ref)] / n_boot

run_exp5_neck_bootstrap.py draws each boot window with a deterministic seed
(base_seed=30_518_726, rng = default_rng(base_seed + b), used_offsets reset per run),
so a GPU re-run would reproduce the identical neck distribution {5:1, 6:22, 32:2} and
write exactly these two values. All other fields are untouched.
"""

import json
import os

_TV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
WORK = os.path.join(_TV_ROOT, "data")   # raw per-experiment metrics (was data/data/)
exp5p = os.path.join(WORK, "exp5_metrics.json")
bootp = os.path.join(WORK, "exp5_neck_bootstrap_metrics.json")

with open(exp5p, encoding="utf-8") as fh:
    exp5 = json.load(fh)
e5 = exp5["pythia-2.8b"]
assert "status" not in e5 and e5.get("neck_level") is not None, \
    f"anatomy entry not valid yet: {e5}"
ref = int(e5["neck_level"])

with open(bootp, encoding="utf-8") as fh:
    boot = json.load(fh)
e = boot["pythia-2.8b"]
assert "status" not in e and e.get("reference_neck") is None, \
    f"unexpected entry state (ref={e.get('reference_neck')}): patch twice?"
n_boot = e["n_boot"]
cnt = e["neck_distribution"].get(str(ref))
assert cnt is not None, f"no boots recorded at reference level {ref}"

frac = round(cnt / n_boot, 4)
e["reference_neck"] = ref
e["fraction_match_reference"] = frac

with open(bootp, "w", encoding="utf-8") as fh:
    json.dump(boot, fh, ensure_ascii=False, indent=2)

print(f"pythia bootstrap patched on CPU: reference_neck={ref} "
      f"fraction_match_reference={frac}  ({cnt}/{n_boot} boots)")
