"""One-shot probe (v2): resolve text-stack attribute paths inside
Gemma4ForConditionalGeneration AND validate that the built-in CE loss backprops
into embed_tokens with CORRECT Parameter targets (v1 bug: used a module instead
of its .weight, so no param had requires_grad=True and out.loss was detached).

Run under .venv-gemma4 python, physical GPU 1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # cross_model/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_ROOT, "tvu_common.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import torch

from cross_common import GEMMA_LOCAL  # noqa: E402
import tvu_common as C  # noqa: E402


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEMMA_LOCAL)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL, dtype=torch.bfloat16).to(C.DEVICE)
    model.eval()
    print("top type:", type(model).__name__)

    print("--- embed_tokens / lm_head locations ---")
    wte_mod = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Embedding) and name.endswith("embed_tokens"):
            wte_mod = (name, m)
        if name.endswith("lm_head") and isinstance(m, torch.nn.Linear):
            print("lmh:", name, type(m).__name__)
    print("wte:", wte_mod[0], type(wte_mod[1]).__name__)

    lmh = None
    for name, m in model.named_modules():
        if name.endswith("lm_head") and isinstance(m, torch.nn.Linear):
            lmh = (name, m)
    same = (wte_mod[1].weight.data_ptr() == lmh[1].weight.data_ptr())
    print("tied (same data_ptr):", same)

    print("--- layer-0 self_attn (text path only) ---")
    l0 = None
    for name, m in model.named_modules():
        if name.endswith("layers.0.self_attn") and "vision" not in name:
            l0 = (name, m)
            break
    print("l0:", l0[0] if l0 else "NOT FOUND")
    if l0 is not None:
        print("  children:", sorted(n for n, _ in l0[1].named_children()))

    # --- validate CE loss + backward with CORRECT Parameter targets (the exact
    #     exp7 pipeline): wte.weight + layer-0 q/k/v proj weights ---
    ids = torch.randint(0, 1000, (1, 32), device=C.DEVICE)
    targets = {wte_mod[1].weight}
    for pname in ("q_proj", "k_proj", "v_proj"):
        targets.add(getattr(l0[1], pname).weight)
    for p in model.parameters():
        p.requires_grad_(p in targets)

    out = model(input_ids=ids, labels=ids)
    print("loss:", None if out.loss is None else float(out.loss))
    print("out.loss.requires_grad:", bool(getattr(out.loss, "requires_grad", False)))
    print("out.logits.requires_grad:", bool(getattr(out.logits, "requires_grad", False)))

    ok = True
    try:
        if out.loss is not None and out.loss.requires_grad:
            out.loss.backward()
        else:
            # fallback exactly as exp7 would do for a composite wrapper
            from torch.nn import functional as F
            vocab = out.logits.size(-1)
            ce = F.cross_entropy(
                out.logits[:, :-1, :].reshape(-1, vocab), ids[:, 1:].reshape(-1))
            ce.backward()
        g = wte_mod[1].weight.grad
        print("wte.grad:", None if g is None else (tuple(g.shape), str(g.dtype)))
        for pname in ("q_proj", "k_proj", "v_proj"):
            gg = getattr(l0[1], pname).weight.grad
            print(f"  {pname}.grad:", None if gg is None else tuple(gg.shape))
    except Exception as e:
        ok = False
        print("BACKWARD FAILED:", type(e).__name__, str(e)[:300])

    for p in model.parameters():
        p.requires_grad_(True)

    import gc
    del tok, model
    gc.collect()
    torch.cuda.empty_cache()
    print("PROBE_DONE" if ok else "PROBE_FAILED")


if __name__ == "__main__":
    main()
