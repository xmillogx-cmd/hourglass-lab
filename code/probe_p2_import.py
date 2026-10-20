"""One-off probe: cross_common imports cleanly after the p.2 MODELS edit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_common import MODELS, MODELS_DIR  # noqa: E402

print("MODELS_DIR:", MODELS_DIR)
for m in MODELS:
    print(f"{m['key']:<16} {str(m['dtype']):<18} {str(m['exp6_lengths']):<14} dir_exists={os.path.isdir(m['path'])}")
