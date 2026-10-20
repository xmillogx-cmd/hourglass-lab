"""Verify the 2026-09-21 four-bucket reorg (code/data/docs/infographic):
py_compile every .py in the tree and module-load the key experiment scripts
(top-level code only — every main() is guarded by __name__) to prove tvu_common /
cross_common resolution still works after all scripts were flattened into code/.
CPU-only, no model loading, safe to run anytime."""

import importlib.util
import os
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))   # code/ (depth 1 under the project root)
TVU = HERE
while not os.path.isfile(os.path.join(TVU, "tvu_common.py")):
    TVU = os.path.dirname(TVU)                       # aiir-hourglass/ (marker walk up 1 level)

MOVED = [
    "run_final_summary.py",
    "run_exp1_gradients.py",
    "exp_gemma_gradient_perturbation.py",
    "run_exp2_pr_profile.py",
    "run_exp3_spectral_asymmetry.py",
    "exp_pr_cross_corpus.py",
    "run_exp4_pressure_of_mass.py",
    "run_e4b_dc_energy.py",
    "run_e4e_needle.py",
    "run_e4f_position_scan.py",
    "exp_multi_level_retrieval_probe.py",
    "exp_attractor_subspace_perturbation.py",
    "run_exp9_hidden_perturbation.py",
]

# 1) compile-check the whole tree (skipping __pycache__)
bad_compile = []
n_files = 0
for root, _dirs, files in os.walk(TVU):
    if os.path.basename(root) == "__pycache__":
        continue
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        n_files += 1
        p = os.path.join(root, f)
        try:
            py_compile.compile(p, doraise=True)
        except Exception as e:
            bad_compile.append((p, str(e)))

# 2) module-load the key scripts (top-level only; main() is guarded).
#    MOVED entries are flat basenames in code/.
def load_rel(rel):
    p = os.path.join(TVU, "code", rel)
    name = "m_" + os.path.splitext(os.path.basename(p))[0]
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

bad_load = []
for rel in MOVED:
    try:
        load_rel(rel)
    except Exception as e:
        bad_load.append((rel, repr(e)))

print(f"compile check: {n_files} .py files -> "
      + ("ALL OK" if not bad_compile else f"{len(bad_compile)} FAILURES"))
for p, err in bad_compile:
    print("  COMPILE FAIL:", p, "->", err)
print(f"module load:   {len(MOVED)} key scripts -> "
      + ("ALL OK (tvu_common / cross_common resolve)" if not bad_load else f"{len(bad_load)} FAILURES"))
for rel, err in bad_load:
    print("  LOAD FAIL:", rel, "->", err)

if bad_compile or bad_load:
    raise SystemExit(1)
print("\nREORG VERIFICATION PASSED (four-bucket layout)")
