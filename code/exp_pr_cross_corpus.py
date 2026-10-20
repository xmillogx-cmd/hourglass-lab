"""exp_pr_cross_corpus.py — Qwen2.5-0.5B participation-ratio profile across 5 local corpora.

Protocol (per TZ spec):
    Five distinct corpora, ~2048 tokens each: markdown/technical, code, narrative prose,
    dialogue (synthetic), shuffled token noise. For each corpus a no_grad forward with
    output_hidden_states=True; per level the hidden states are centered over tokens and SVD'd;
    participation ratio PR = (sum(sigma^2))^2 / sum((sigma^2)^2) — i.e. the spec's
    (sum(lambda_i))^2 / sum(lambda_i^2) with lambda_i = sigma_i^2, identical to Exp5's metric.
    Single plot: 5 lines of PR vs layer index + JSON per corpus with the neck layer (argmin PR).

Corpus sources — LOCAL ONLY (no downloads):
    markdown/technical : tvu_common.DOC_FILES (README.md, PROJECT.md, FINAL_REPORT_v4.md,
                         report.md under the host tree) via make_diverse_tokens — same text
                         as Exp9/tied-vs-untied runs.
    code               : concatenated .py sources of aiir-hourglass/ + cross_model/ (sorted).
    narrative          : draft_source.txt (the full Russian CHMC article), resolved via
                         $CHMC_ARTICLE / argv[1] / host-tree copy in results_v6/article/.
                         CAVEAT: it is Russian ML-domain prose, not English general narrative;
                         recorded in JSON as a caveat.
    dialogue           : synthetic multi-turn Q/A template conversation (fixed seed 0) — allowed
                         by the spec for this corpus.
    shuffled noise     : token-shuffle of the narrative corpus ids (fixed seed 1).

No training; inference only, bf16 on GPU (pick_gpu gate, >= 8 GiB free required).
Artifacts: data/data/pr_cross_corpus_metrics.json + data/pr_cross_corpus.png
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # g2_geometry_hourglass/
_TV_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
sys.path.insert(0, TVU)                                  # for `import tvu_common`

import tvu_common as C                                   # make_diverse_tokens, DOC_FILES  noqa: E402

PARENT = os.path.dirname(TVU)                         # host tree (here cmq_experiment/) with models/
WORK = os.path.join(TVU, "data")
RESULTS = os.path.join(TVU, "data")
MODEL_LOCAL = os.path.join(C.MODELS_DIR, "qwen2.5-0.5b")

N_TOKENS = 2048
MIN_FREE_GIB = 8.0
WORK_JSON = os.path.join(WORK, "pr_cross_corpus_metrics.json")
PNG_PATH = os.path.join(RESULTS, "pr_cross_corpus.png")

log = logging.getLogger("pr_cross_corpus")


def pick_gpu() -> tuple[torch.device, float]:
    """Prefer cuda:1; fall back to cuda:0 if it lacks enough free VRAM."""
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


# ---------- corpus builders: each returns (token_ids list[int], source_note str) ----------------

def corpus_markdown(tok) -> tuple[list, str]:
    # make_diverse_tokens returns shape [1, N] — flatten to a flat int list before slicing,
    # otherwise [:N_TOKENS] slices the outer dim and yields one nested element (tokens_used=1)
    ids = C.make_diverse_tokens(tok, n_target=N_TOKENS).cpu().flatten().tolist()[:N_TOKENS]
    note = ("technical markdown: host-tree DOC_FILES "
            "(README.md, PROJECT.md, FINAL_REPORT_v4.md, report.md) via tvu_common.make_diverse_tokens")
    return ids, note


def corpus_code(tok) -> tuple[list, str]:
    files = []
    for _root, _dirs, _names in os.walk(os.path.join(TVU, "code")):
        files.extend(os.path.join(_root, n) for n in _names if n.endswith(".py"))
    files = sorted(files)
    texts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            texts.append(fh.read())
    ids = tok("\n\n".join(texts), add_special_tokens=False)["input_ids"][:N_TOKENS]
    note = f"python source of {len(files)} local experiment scripts (aiir-hourglass + cross_model), sorted by path"
    return list(ids), note


def _narrative_src() -> str:
    """Full Russian CHMC article (draft_source.txt): $CHMC_ARTICLE or argv[1] wins,
    then host-tree candidates. No hardcoded absolute path."""
    cands = [
        os.environ.get("CHMC_ARTICLE"),
        sys.argv[1] if len(sys.argv) > 1 else None,
        os.path.join(PARENT, "results_v6", "article", "draft_source.txt"),   # previous research copy in the host tree
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit(
        "narrative corpus (draft_source.txt) not found. Set CHMC_ARTICLE or pass its path as argv[1].")


def corpus_narrative(tok) -> tuple[list, str]:
    src = _narrative_src()
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    ids = tok(text, add_special_tokens=False)["input_ids"][:N_TOKENS]
    note = ("narrative prose: the full Russian CHMC article (draft_source.txt). CAVEAT: Russian "
            "ML-domain prose, not English general narrative — language/domain confound applies")
    return list(ids), note


def corpus_dialogue(tok) -> tuple[list, str]:
    rng = random.Random(0)   # fixed seed per spec (synthetic allowed for this corpus)
    qs = [
        "Why does quantizing the weights hurt more than quantizing the activations?",
        "What is the difference between GPTQ and scalar INT4 quantization?",
        "How do you compare two quantized models fairly?",
        "What does a low-rank hidden-state profile tell you about a layer?",
        "Why can't a low-rank delta fix post-quantization drift?",
        "What is the role of dampening in Hessian-based quantization?",
        "How do bit budgets affect perplexity comparisons?",
        "What does anisotropy of embeddings mean for compression?",
    ]
    as_ = [
        "Weights are dense and full-rank, so data-adaptive error compensation matters more than geometry.",
        "GPTQ compensates errors column by column on the calibration inputs; scalar INT4 just rounds each weight.",
        "Equal bit budget per weight first — otherwise any comparison is fiction.",
        "A low participation ratio means most energy sits in a few directions — that layer is a bottleneck.",
        "The drift spectrum spreads over hundreds of directions, so rank-30 captures at best part of it.",
        "With little calibration the Hessian estimate is noisy and inverting it amplifies noise; dampening tames that.",
        "Perplexity ratios are only meaningful when both methods spend the same bits per weight.",
        "Representations concentrate in a narrow cone, which data-side compression can exploit but parameter-side cannot.",
    ]
    parts: list[str] = []
    i = 0
    ids_all: list[int] = []
    while len(ids_all) < N_TOKENS:
        q, a = qs[i % len(qs)], as_[(i * 3 + 1) % len(as_)]
        turn = f"Q{i}: {q}\nA{i}: {a}"
        ids_all += tok(turn, add_special_tokens=False)["input_ids"]
        parts.append(turn)
        i += 1
    note = "synthetic multi-turn Q/A dialogue on quantization topics (fixed seed 0; spec allows synthetic here)"
    return ids_all[:N_TOKENS], note


def corpus_noise(tok) -> tuple[list, str]:
    base_ids, _ = corpus_narrative(tok)          # same source text as the narrative arm
    perm = np.random.default_rng(1).permutation(len(base_ids))   # fixed seed per spec
    ids = [base_ids[int(i)] for i in perm[:N_TOKENS]]
    note = "shuffled token noise: token-shuffle of the narrative corpus (fixed seed 1), same vocabulary statistics"
    return ids, note


# ---------- measurement ----------------------------------------------------------------------

def participation_ratio(sigma: np.ndarray) -> float:
    """PR = (sum sigma^2)^2 / sum((sigma^2)^2) — spec's (sum lambda_i)^2/sum(lambda_i^2) with
    lambda_i = sigma_i^2; identical to Exp5's tvu_common.participation_ratio."""
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def pr_profile(model: torch.nn.Module, ids: list[int], device: torch.device) -> tuple[list[float], int]:
    """Centered-PR per hidden-state level; returns (pr_list over levels 0..L, neck layer index)."""
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hs = model(input_ids=input_ids, output_hidden_states=True).hidden_states

    prs: list[float] = []
    for h in hs:                                   # [1, T, D] bf16 on GPU
        Hc = h.detach().float().cpu()
        Hc -= Hc.mean(dim=1, keepdim=True)         # center over tokens (project convention)
        sigma = torch.linalg.svdvals(Hc).numpy()   # singular values; PR needs only the spectrum
        prs.append(participation_ratio(sigma))

    mins = prs[1:]                        # decoder levels only; skip level 0 (embeddings)
    neck_layer = int(np.argmin(mins))    # 0-based decoder block index (Exp5 convention)
    return [round(p, 4) for p in prs], neck_layer


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

    builders = [
        ("markdown_technical", corpus_markdown),
        ("code", corpus_code),
        ("narrative_prose", corpus_narrative),
        ("dialogue_synthetic", corpus_dialogue),
        ("shuffled_noise", corpus_noise),
    ]

    corpora: dict = {}
    for name, build in builders:
        log.info("corpus %s: building + forward", name)
        ids, note = build(tok)
        prs, neck = pr_profile(model, ids, device)
        corpora[name] = {
            "source_note": note,
            "tokens_used": len(ids),
            "pr_by_layer": prs,
            "neck_layer": neck,          # 0-based decoder block index; tuple level = neck + 1
            "neck_pr": prs[neck + 1],
        }
        log.info("corpus %s: PR profile done (neck L%d, PR %.3f)", name, neck, prs[neck + 1])

    n_levels = len(next(iter(corpora.values()))["pr_by_layer"])
    result = {
        "model": "qwen2.5-0.5b",
        "path": MODEL_LOCAL,
        "device": str(device),
        "gpu_free_gib_at_start": round(free_gib, 1),
        "n_levels_reported": n_levels,
        "pr_formula": "(sum sigma^2)^2 / sum((sigma^2)^2) on centered hidden states — spec's (sum lambda_i)^2/sum(lambda_i^2) with lambda_i=sigma_i^2; Exp5 convention",
        "corpora": corpora,
    }

    with open(WORK_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    log.info("saved -> %s", WORK_JSON)

    # ---------- single plot: 5 lines of PR vs level ----------
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for (name, _), c in zip(builders, colors):
        prs = corpora[name]["pr_by_layer"]
        ax.plot(range(n_levels), prs, color=c, lw=1.6, label=name)
    ax.set_xlabel("hidden-state level (0 = embeddings, 1..L = decoder layers)")
    ax.set_ylabel("participation ratio of centered hidden states")
    ax.set_title(f"Qwen2.5-0.5B: PR per layer across 5 local corpora ({N_TOKENS} tokens each, bf16 {device})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    log.info("plot saved -> %s", PNG_PATH)

    print("\n--- PR cross-corpus summary (qwen2.5-0.5b) ---")
    for name in corpora:
        c = corpora[name]
        print(f"  {name:<20} neck L{c['neck_layer']:>3}  PR={c['neck_pr']:.3f}")


if __name__ == "__main__":
    main()
