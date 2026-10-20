"""exp_multi_level_retrieval_probe.py — Qwen2.5-0.5B multi-level needle retrieval probe.

Protocol (per TZ spec):
    Haystack L = 2048 tokens from local DOC_FILES chunks (technical prose as filler). Three
    needle levels:
        L1 exact      : "The secret verification code is KZVX-9281." -> completion of
                        "The secret verification code is"
        L2 semantic   : rephrased fact ("the client's unique identifier ... value CODE") ->
                        completion of the rephrased continuation asking for that code
        L3 relational : 2-hop chain (User Alpha -> terminal TERM; terminal TERM stores CODE) ->
                        completion "The verification code stored on User Alpha's terminal is"
    Needle inserted at positions p in {0.1, 0.5, 0.9} of the haystack; 3 reps per cell
    (distinct seeded code values + filler rotation offsets — E4f convention).

Metric: lift = logit(target) - mean(logits(alt set)) at the next-token position after the probe
tail; target = first BPE piece of the code, alts = analog codes of the same form whose first BPE
piece differs (E4e convention), fp32 logits. Grouped bar chart: levels x positions.

Documented deviation from spec — the base model has no instruction-following QA, so all three
levels are implemented as PATTERN-COMPLETION probes with rephrased continuation templates
(identical fact, different surface form per level) rather than question answering; L2's spec'd
question "What is the verification code for the client?" becomes a fill-in continuation that
elicits the same target token. This measures retrieval of a stored fact through paraphrase, not
instruction compliance — recorded in JSON notes.

No training; inference only, bf16 on GPU (pick_gpu gate). Local sources only (DOC_FILES filler).
Artifacts: data/data/multi_level_retrieval_probe_metrics.json + data/multi_level_retrieval_probe.png
"""
from __future__ import annotations

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # g3_long_context_h4/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                  # for `import tvu_common`

import tvu_common as C                                   # DOC_FILES paths, make_diverse_tokens  noqa: E402

PARENT = os.path.dirname(TVU)                         # host tree (here cmq_experiment/) with models/
WORK = os.path.join(TVU, "data")
RESULTS = os.path.join(TVU, "data")
MODEL_LOCAL = os.path.join(C.MODELS_DIR, "qwen2.5-0.5b")

HAYSTACK_L = 2048
POSITIONS = (0.1, 0.5, 0.9)
N_REPS = 3
N_ALTS = 5
BASE_CODE = "KZVX-9281"        # L1/L2 code base value
BASE_TERM = "KZVX-T045"       # L3 terminal id base value
BASE_CODE3 = "9281-B7QW"      # L3 stored-code base value
MIN_FREE_GIB = 8.0
WORK_JSON = os.path.join(WORK, "multi_level_retrieval_probe_metrics.json")
PNG_PATH = os.path.join(RESULTS, "multi_level_retrieval_probe.png")

log = logging.getLogger("retrieval_probe")


def pick_gpu() -> tuple[torch.device, float]:
    for idx in (1, 0):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            continue
        free_gib = free_bytes / (1024 ** 3)
        if free_gib >= MIN_FREE_GIB:
            print(f"[device] GPU {idx}: {free_gib:.1f} GiB free -> selected")
            return torch.device("cuda", idx), free_gib
    for idx in (0, 1):
        try:
            free_bytes, _total = torch.cuda.mem_get_info(idx)
            print(f"[device] GPU {idx}: only {free_bytes / (1024 ** 3):.1f} GiB free")
        except Exception as e:
            print(f"[device] GPU {idx}: not available ({e})")
    raise SystemExit("ABORT: neither GPU has >= 8 GiB free VRAM. Close LM Studio (tray!) and rerun.")


# ---------- seeded code/alt generation (E4e/E4f conventions) ---------------------------------

def mutate_string(s: str, rng: np.random.Generator) -> str:
    """One-character mutation preserving form: letters stay letters, digits stay digits."""
    chars = list(s)
    mutable = [i for i, ch in enumerate(chars) if ch.isalnum()]
    i = int(rng.choice(mutable))
    ch = chars[i]
    pool = "ABCDEFGHJKLMNPQRSTUVWXYZ" if ch.isalpha() else "0123456789"
    while True:
        new_ch = str(rng.choice(list(pool)))
        if new_ch != ch:
            break
    chars[i] = new_ch
    return "".join(chars)


def make_alts(code: str, tok, rng: np.random.Generator) -> list[str]:
    """N_ALTS analog codes of the same form whose first BPE piece differs from the target's."""
    target_tk = tok.convert_ids_to_tokens(tok.encode(code, add_special_tokens=False)[0])
    alts: list[str] = []
    seen: set[str] = {code}
    for _ in range(64):                       # bounded retry loop
        cand = mutate_string(code, rng)
        if cand in seen:
            continue
        tk = tok.convert_ids_to_tokens(tok.encode(cand, add_special_tokens=False)[0])
        if tk == target_tk or any(
                tok.convert_ids_to_tokens(tok.encode(a, add_special_tokens=False)[0]) == tk
                for a in alts):
            continue
        seen.add(cand)
        alts.append(cand)
        if len(alts) == N_ALTS:
            break
    return alts


# ---------- probe levels --------------------------------------------------------------------

def level_specs(rep: int) -> dict:
    """Seeded per-rep values + templates for the three levels (seed = 2026 + rep*17).
    L3 mutates both terminal id and stored code."""
    r = np.random.default_rng(2026 + rep * 17)
    code_l12 = mutate_string(BASE_CODE, r) if rep else BASE_CODE
    term = mutate_string(BASE_TERM, r) if rep else BASE_TERM
    code_l3 = mutate_string(BASE_CODE3, r) if rep else BASE_CODE3

    return {
        "L1_exact": {
            "needle": f"The secret verification code is {code_l12}.",
            "tail": "The secret verification code is",
            "target_code": code_l12,
        },
        "L2_semantic": {
            "needle": (f"The client's unique identifier has been assigned the value {code_l12}."),
            "tail": ("Therefore, the verification code for the client — its unique identifier — is"),
            "target_code": code_l12,
        },
        "L3_relational_2hop": {
            "needle": (f"User Alpha operates terminal {term}. Terminal {term} stores "
                       f"the verification code {code_l3}."),
            "tail": "The verification code stored on User Alpha's terminal is",
            "target_code": code_l3,
        },
    }


def build_sequence(filler_ids: list[int], off: int, spec: dict, p: float, tok) -> tuple[list[int], int]:
    """filler[:needle_start] + needle + filler[...] up to HAYSTACK_L - len(tail), then tail.
    Returns (token ids, needle_start)."""
    tail_ids = tok(spec["tail"], add_special_tokens=False)["input_ids"]
    body_len = HAYSTACK_L - len(tail_ids)
    needle_ids = tok(spec["needle"], add_special_tokens=False)["input_ids"]
    needle_start = int(round(p * (body_len - len(needle_ids))))
    pre = filler_ids[off: off + needle_start]
    post_need = body_len - needle_start - len(needle_ids)
    post = filler_ids[(off + needle_start + 791): (off + needle_start + 791) + post_need]
    # rotate the suffix source window so pre/post are non-overlapping chunks of the same filler
    ids = pre + needle_ids + post + tail_ids
    return ids[:HAYSTACK_L], needle_start


def measure_lift(model: torch.nn.Module, tok, ids: list[int], spec: dict, alts: list[str]) -> float:
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0, -1].float().cpu()   # next-token after tail
    target_id = tok.encode(spec["target_code"], add_special_tokens=False)[0]
    alt_ids = [tok.encode(a, add_special_tokens=False)[0] for a in alts]
    lift = float(logits[target_id].item()) - sum(float(logits[a].item()) for a in alt_ids) / len(alt_ids)
    return lift


# ---------- driver ----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    device, free_gib = pick_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL, dtype=torch.bfloat16).to(device)
    model.eval()

    log.info("building filler from local DOC_FILES")
    texts = []
    for f in C.DOC_FILES:                       # [path] under cmq_experiment ROOT (tvu_common)
        with open(f, encoding="utf-8") as fh:
            texts.append(fh.read())
    filler_ids = tok("\n".join(texts), add_special_tokens=False)["input_ids"]

    levels_out: dict = {}
    level_names = ("L1_exact", "L2_semantic", "L3_relational_2hop")
    for p in POSITIONS:
        for rep in range(N_REPS):
            specs = level_specs(rep)                             # seeded per-rep values + templates
            off = rep * 1571                                     # filler rotation offset (E4f)
            for lname in level_names:
                spec = specs[lname]
                rng = np.random.default_rng(900 + list(level_names).index(lname) * 31 + rep * 7)
                alts = make_alts(spec["target_code"], tok, rng)
                ids, needle_start = build_sequence(filler_ids, off, spec, p, tok)
                lift = measure_lift(model, tok, ids, spec, alts)
                key = f"p={p:g}"
                cell = levels_out.setdefault(lname, {}).setdefault(key, {"reps": []})
                cell["reps"].append({
                    "rep": rep,
                    "needle_start": needle_start,
                    "target_code": spec["target_code"],
                    "alt_codes": alts,
                    "lift": round(lift, 4),
                })
                log.info("cell %s p=%.1f rep=%d: lift = %.3f", lname, p, rep, lift)

    # per-cell mean over reps (for the chart + summary)
    for lname in level_names:
        for key, cell in levels_out[lname].items():
            lifts = [r["lift"] for r in cell["reps"]]
            cell["mean_lift"] = round(sum(lifts) / len(lifts), 4)

    result = {
        "model": "qwen2.5-0.5b",
        "path": MODEL_LOCAL,
        "device": str(device),
        "gpu_free_gib_at_start": round(free_gib, 1),
        "haystack_len": HAYSTACK_L,
        "positions": list(POSITIONS),
        "n_reps_per_cell": N_REPS,
        "filler_source": "local DOC_FILES (cmq_experiment README/PROJECT/FINAL_REPORT_v4/report.md)",
        "lift_definition": ("logit(target first BPE piece) - mean(logits of alt set); alts = analog codes "
                            "of the same form with distinct first pieces (E4e), fp32 logits, measured at "
                            "the next-token position after the probe tail"),
        "levels": levels_out,
        "notes": [
            ("Base model has no instruction QA: all three levels are pattern-completion probes with "
             "rephrased continuation templates (same fact, different surface form), not question answering."),
            ("L2 semantic level replaces the spec'd question 'What is the verification code for the client?' "
             "with a fill-in continuation eliciting the same target token; this measures retrieval through "
             "paraphrase, not instruction compliance."),
        ],
    }

    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("saved -> %s", WORK_JSON)

    # ---------- grouped bar chart: levels x positions (mean lift over reps) ----------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors_p = {0.1: "tab:blue", 0.5: "tab:orange", 0.9: "tab:red"}
    width = 0.25
    for gi, lname in enumerate(level_names):
        for pi, p in enumerate(POSITIONS):
            key = f"p={p:g}"
            mean_lift = levels_out[lname][key]["mean_lift"]
            ax.bar(gi + (pi - 1) * width, mean_lift, width=width, color=colors_p[p],
                   label=f"p={p:g}" if gi == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(level_names)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("lift = logit(target) - mean(logits(alts)), nats")
    ax.set_title(f"Qwen2.5-0.5B: needle-retrieval lift by level and haystack position "
                 f"(L={HAYSTACK_L}, {N_REPS} reps/cell, bf16 {device})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    log.info("plot saved -> %s", PNG_PATH)

    print("\n--- multi-level retrieval probe (qwen2.5-0.5b) ---")
    for lname in level_names:
        row = []
        for p in POSITIONS:
            m = levels_out[lname][f"p={p:g}"]["mean_lift"]
            row.append(f"p={p:g}: {m:+.3f}")
        print(f"  {lname:<20} " + "   ".join(row))


if __name__ == "__main__":
    main()
