"""Final step — merge data/data/expN_metrics.json into the single deliverable
data/empirical_validation_summary.json (5th result file, per TZ §6).

Adds: the summary table from the TZ (PR layer 0 / mid / last + Anisotropy
Score for Tied and Untied) and a compact data-facts block per hypothesis
(no interpretation beyond what the numbers state).
"""

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import tvu_common as C                                           # noqa: E402


def load(name):
    path = os.path.join(C.WORK, name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    exp1 = load("exp1_metrics.json")
    exp2 = load("exp2_metrics.json")
    exp3 = load("exp3_metrics.json")
    exp4 = load("exp4_metrics.json")

    summary_table = {}
    for name in (C.NAME_A, C.NAME_B):
        e2 = exp2[name]
        e1 = exp1[name]
        summary_table[name] = {
            "pr_layer0": e2["pr_layer0"],
            "pr_mid": e2["pr_mid"],
            "pr_last": e2["pr_last"],
            "anisotropy_raw": e1["anisotropy_raw"],
            "anisotropy_centered": e1["anisotropy_centered"],
        }

    facts = {
        "H1_gradient_interference": {
            C.NAME_A: {
                "cos(G_in, G_out)": exp1[C.NAME_A]["cos_gin_vs_gout"],
                "norm_ratio_out_over_in": exp1[C.NAME_A]["norm_ratio_gout_over_gin"],
                "cos(G_total, G_in)": exp1[C.NAME_A]["cos_gtotal_vs_gin"],
                "cos(G_total, G_out)": exp1[C.NAME_A]["cos_gtotal_vs_gout"],
                "emb_update_vs_qkv0_effect_proxy": exp1[C.NAME_A]["cos_embupdate_vs_qkv0_effect_proxy"],
            },
            C.NAME_B: {
                "cos(G_emb, G_lm)": exp1[C.NAME_B]["cos_gemb_vs_glm"],
                "norm_ratio_lm_over_emb": exp1[C.NAME_B]["norm_ratio_glm_over_gemb"],
                "emb_update_vs_qkv0_effect_proxy": exp1[C.NAME_B]["cos_embupdate_vs_qkv0_effect_proxy"],
            },
        },
        "H2_hourglass_PR_profile": {
            name: {
                "neck_level": exp2[name]["neck_level"],
                "neck_pr_centered": exp2[name]["neck_pr_centered"],
                "pr_layer0": exp2[name]["pr_layer0"],
                "pr_mid": exp2[name]["pr_mid"],
                "pr_last": exp2[name]["pr_last"],
            } for name in (C.NAME_A, C.NAME_B)
        },
        "H3_spectral_asymmetry_at_neck": {
            name: {
                "weights_k95_of_dim": f"{exp3[name]['W_stats']['k95']}/{min(exp3[name]['weights_shape'])}",
                "weights_top10_energy": exp3[name]["W_stats"]["top10_energy"],
                "activations_k95": exp3[name]["X_stats"]["k95"],
                "activations_top10_energy": exp3[name]["X_stats"]["top10_energy"],
            } for name in (C.NAME_A, C.NAME_B)
        },
        "H4_pressure_of_mass": {
            "lengths": exp4["lengths"],
            "raw_cos_to_com_mean": exp4["raw_cos_to_com_mean"],
            "centered_pairwise_abs_cos": exp4["centered_pairwise_abs_cos"],
        },
    }

    payload = {
        "task": ("Empirical validation of the unified geometric theory — "
                 "tied vs untied control pair (TZ: 4 experiments)"),
        "models": {
            C.NAME_A: {"path": exp2[C.NAME_A]["model_path"], "role": "TIED (Model A)",
                       "hidden_size": exp2[C.NAME_A]["hidden_size"],
                       "num_hidden_layers": exp2[C.NAME_A]["num_hidden_layers"]},
            C.NAME_B: {"path": exp2[C.NAME_B]["model_path"], "role": "UNTIED (Model B)",
                       "device_note": "all compute on physical GPU 1 (cuda:1), fp32"},
        },
        "context": ("diverse real-doc corpus (README/PROJECT/FINAL_REPORT/report.md), "
                    f"truncated to {C.N_TOKENS} tokens per model's own tokenizer; "
                    "exp4 uses prefixes 100/1000/4096 of the same corpus"),
        "summary_table": summary_table,
        "hypothesis_facts": facts,
        "exp1_gradient_interference_full": exp1,
        "exp2_pr_profile_full": exp2,
        "exp3_spectral_asymmetry_full": exp3,
        "exp4_pressure_of_mass_full": exp4,
    }

    out = os.path.join(C.RESULTS, "empirical_validation_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"summary JSON written: {out}")


if __name__ == "__main__":
    main()
