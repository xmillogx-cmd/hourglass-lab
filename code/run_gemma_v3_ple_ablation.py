"""V3 ablation — scale-stripped embedding PLUS zeroed PLE token-identity table.

Follow-up to run_gemma_functional_impact.py (Part B): the stripped ratio was 5.84,
i.e. a residual structural input dominance survived removing the main embedding's
sqrt(hidden) gain. The prime suspect is the SECOND scaled channel — PLE
(embed_tokens_per_layer), a [vocab=262144 x 35*256=8960] table with scale
sqrt(256)=16 feeding every decoder layer (token-identity component of
project_per_layer_inputs; the context-projection component is RMSNorm'd and not a
scaled channel, so zeroing the identity table leaves it intact — no NaN risk).

This script recomputes G_in'/G_out' on the SAME bf16 GPU protocol as Part B:
  v2 (control)     — plain nn.Embedding sharing W, PLE untouched   -> expect ~5.84
  v3 (ablation)    — plain nn.Embedding sharing W + PLE identity table zeroed in-place

Decision rule for the paper:
  ratio(v3) <= 2        -> the second scaled channel explains most of the residual
                           input dominance; inversion = artifact of scaled channels.
  ratio(v3) ~ ratio(v2) -> residual is NOT PLE-driven; attribute to stack depth /
                           softcap dynamics (structural).

PLE weight backup lives on CPU RAM (~4.7 GB bf16); no extra GPU memory beyond Part B's
peak, so the same >= 14 GiB free-VRAM requirement applies. Writes
data/data/gemma_v3_ple_ablation_metrics.json (separate file: a rerun of the main bat
rewrites gemma_functional_impact_metrics.json wholesale and would wipe merged fields).

Run (из cross_model/ или из корня проекта): <python с transformers>=5> [cross_model\]run_gemma_v3_ple_ablation.py
(или `cross_model\run_gemma_v3_ple_ablation.bat`)
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

import torch
from cross_common import GEMMA_LOCAL  # noqa: E402
import tvu_common as C                # noqa: E402
from run_exp7_gradient_interference import g_to_np, norm_np, cos_np  # noqa: E402
from run_gemma_functional_impact import (  # noqa: E402
    N_TOKENS, pick_gpu, get_text_stack, compute_gin_gout, clear_grads,
)

WORK_JSON = os.path.join(C.WORK, "gemma_v3_ple_ablation_metrics.json")


def _restore_ple(ple_mod, backup_cpu_bf16, device):
    """Chunked restore of the zeroed PLE table.

    A full-size `backup.to(device)` copy needs another ~4.38 GiB of VRAM on top of
    the resident model (~9.6 GB) — that did not fit (OOM). Chunked copies keep the
    peak extra allocation at one 8192-row slice (~0.15 GB bf16); verification is a
    chunked equality check on CPU, so no full-size GPU temp either way."""
    w = ple_mod.weight.data
    n, step = backup_cpu_bf16.shape[0], 8192
    for i in range(0, n, step):
        j = min(i + step, n)
        chunk = backup_cpu_bf16[i:j].to(device)
        with torch.no_grad():
            w[i:j].copy_(chunk)
        del chunk
    for i in range(0, n, 65536):
        j = min(i + 65536, n)
        if not torch.equal(w[i:j].cpu(), backup_cpu_bf16[i:j]):
            raise RuntimeError("PLE restore failed")


def main():
    device, free_gib = pick_gpu()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL, dtype=torch.bfloat16).to(device)
    model.eval()

    tsm = get_text_stack(model)
    import torch.nn as nn
    emb_mod = tsm.embed_tokens
    ple_mod = getattr(tsm, "embed_tokens_per_layer", None)
    if ple_mod is None:
        raise SystemExit("ABORT: no embed_tokens_per_layer on the text stack — nothing to ablate.")

    ids = C.make_diverse_tokens(tok, n_target=N_TOKENS).to(device)
    orig_emb = tsm.embed_tokens
    plain = nn.Embedding(emb_mod.num_embeddings, emb_mod.embedding_dim,
                         padding_idx=emb_mod.padding_idx)
    plain.weight = emb_mod.weight  # share the Parameter — same W, no sqrt(hidden) gain

    results = {}
    if os.path.exists(WORK_JSON):
        with open(WORK_JSON, encoding="utf-8") as fh:
            results.update(json.load(fh))

    def save():
        with open(WORK_JSON, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"  (saved -> {WORK_JSON})")

    try:
        # ---------- v2 control: scale stripped, PLE intact ----------
        tsm.embed_tokens = plain
        print("[v3] variant v2: scale stripped, PLE identity intact (control)")
        res_v2 = compute_gin_gout(model, ids, device)
        n_in2, n_out2 = norm_np(res_v2["g_in"]), norm_np(res_v2["g_out"])

        # ---------- v3 ablation: + PLE token-identity table zeroed ----------
        ple_backup = ple_mod.weight.data.detach().cpu()  # ~4.7 GB bf16 in RAM
        tsm.embed_tokens = plain
        print("[v3] variant v3: scale stripped AND PLE identity zeroed")
        with torch.no_grad():
            ple_mod.weight.data.zero_()
        try:
            res_v3 = compute_gin_gout(model, ids, device)
        finally:
            # free G_out-phase leftovers before the PLE restore — a full-size
            # .to(device) temp (4.38 GiB) did not fit alongside them (OOM):
            gc.collect()
            torch.cuda.empty_cache()
            _restore_ple(ple_mod, ple_backup, device)
            clear_grads(model)
        del ple_backup
        gc.collect()

        n_in3, n_out3 = norm_np(res_v3["g_in"]), norm_np(res_v3["g_out"])
        r2, r3 = n_in2 / n_out2, n_in3 / n_out3

        if r3 <= 2.0:
            decision = ("The second scaled channel (PLE token identity) explains most of the "
                        f"residual input dominance: stripped ratio {r2:.2f} -> {r3:.2f} with PLE off.")
        elif r3 <= r2 * 1.15:
            decision = (f"Residual is NOT PLE-driven: stripped ratio {r2:.2f} vs {r3:.2f} with "
                        "PLE identity zeroed — attribute to stack depth / softcap dynamics.")
        else:
            decision = (f"Intermediate: PLE off changes the stripped ratio from {r2:.2f} to "
                        f"{r3:.2f} — both scaled channels and structure contribute.")

        results["gemma-4-e2b"] = {
            "path": GEMMA_LOCAL,
            "gpu_used": str(device),
            "gpu_free_gib_at_start": round(free_gib, 1),
            "protocol": ("same as Part B of run_gemma_functional_impact.py (bf16, exp7 G_in/G_out); "
                         "v3 additionally zeros the PLE token-identity table in-place "
                         "(backup on CPU RAM, restored after)"),
            "v2_scale_stripped_ple_intact": {
                "norm_gin": round(n_in2, 3), "norm_gout": round(n_out2, 3),
                "ratio_gin_over_gout": round(r2, 3),
                "cos_gin_vs_gout": round(cos_np(res_v2["g_in"].ravel(), res_v2["g_out"].ravel()), 4),
            },
            "v3_scale_stripped_ple_identity_zeroed": {
                "norm_gin": round(n_in3, 3), "norm_gout": round(n_out3, 3),
                "ratio_gin_over_gout": round(r3, 3),
                "cos_gin_vs_gout": round(cos_np(res_v3["g_in"].ravel(), res_v3["g_out"].ravel()), 4),
            },
            "ple_facts": {
                "weight_shape": list(ple_mod.weight.shape),
                "params_millions": round(ple_mod.weight.numel() / 1e6, 1),
                "embed_scale": float(ple_mod.scalar_embed_scale) if hasattr(ple_mod, "scalar_embed_scale") else None,
            },
            "decision_rule_result": decision,
        }
    finally:
        tsm.embed_tokens = orig_emb
        clear_grads(model)
        del tok, model
        gc.collect()
        torch.cuda.empty_cache()
    save()

    g = results["gemma-4-e2b"]
    print("\n--- v3 ablation summary (gemma-4-e2b) ---")
    print(f"  v2 stripped (PLE intact)   ratio = {g['v2_scale_stripped_ple_intact']['ratio_gin_over_gout']}")
    print(f"  v3 stripped (PLE zeroed)   ratio = {g['v3_scale_stripped_ple_identity_zeroed']['ratio_gin_over_gout']}")
    print(f"  => {g['decision_rule_result']}")


if __name__ == "__main__":
    main()
