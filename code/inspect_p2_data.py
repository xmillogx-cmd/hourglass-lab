# -*- coding: utf-8 -*-
"""Осмотр структур p.2 данных перед корреляционным анализом (CPU-only)."""
import json, io, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
BASE = _ROOT
FILES = {
    "exp5": os.path.join(BASE, "data", "exp5_metrics.json"),
    "exp6icci": os.path.join(BASE, "data", "exp6_ic_ci.json"),
    "exp7mr": os.path.join(BASE, "data", "exp7_multisubsample_metrics.json"),
    "zones": os.path.join(BASE, "data", "neck_zone_stats.json"),
    "boot5": os.path.join(BASE, "data", "exp5_neck_bootstrap_metrics.json"),
    "meta": os.path.join(BASE, "data", "models_metadata.json"),
}

def brief(v, depth=0):
    if isinstance(v, dict):
        return {k: brief(x, depth+1) for k, x in list(v.items())[:20]}
    if isinstance(v, (list, tuple)):
        return {"_list_head": [brief(x, depth+1) for x in v[:3]], "_len": len(v)}
    return v

for name, path in FILES.items():
    print("=" * 70)
    print("FILE:", name, os.path.exists(path))
    with io.open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        for k, v in list(d.items())[:12]:
            b = brief(v)
            s = json.dumps(b, ensure_ascii=False)
            print(f"  [{k}] ({type(v).__name__}, len={len(v) if hasattr(v,'__len__') else '-'})")
            print("    " + (s[:600] + ("..." if len(s) > 600 else "")))
