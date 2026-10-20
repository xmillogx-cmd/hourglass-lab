"""Exp 7 — Cross-model gradient interference on the shared matrix (GPU 1, physical).

The ORIGINAL exp1 protocol (direct G_in / G_out decomposition) generalized from the
two-model control pair to all six models of cross_common.MODELS:

TIED model:
  G_total — one forward+backward with labels (both paths into wte), out.loss.backward().
  G_in    — same forward+backward, lm_head temporarily replaced by a FROZEN clone
            of W in the model's own dtype: only the input-path gradient flows into wte.
  G_out   — no-grad base hidden states + fresh Parameter initialized to a copy of
            W as unembedding; CE loss (HF shift convention) -> output-path-only grad.
  By backprop linearity through the shared tensor, G_total = G_in + G_out holds
  exactly; cosines characterize the interference geometry.

UNTIED model (tinyllama): single backward; wte.grad = pure input path,
lm_head.grad = output matrix; baseline separation metrics.

Plus the TZ activation-level proxy ("или через прокси"): per context token i, cosine
of how layer-0 q/k/v states are perturbed by the embedding update (W_q dE_i) vs by
the layer-0 weight updates (dQ e_i), concatenated over all 128 tokens.

PROTOCOL UNIFORMITY: probe v2 confirmed that even the gemma composite wrapper's
built-in CE loss is differentiable (out.loss.requires_grad == True with correct
Parameter targets — the earlier "detached" conclusion was a probe bug), so
G_total/G_in use out.loss.backward() for ALL six models, exactly like the original
exp1. The only architecture-specific handling is module resolution: gemma-4-e2b
= Gemma4ForConditionalGeneration keeps its text stack under model.language_model
with embed_tokens/layers as DIRECT children (no inner .model), and its vision
tower also has encoder.layers — layer-0 resolution anchors on the text tower.

tie_word_embeddings per key is read from data/models_metadata.json (Step 0).

Anisotropy values are NOT recomputed: read from data/data/exp5_metrics.json (same
protocol, seed 42), computed fresh only if that file lacks the model.

Per-model dtype from cross_common.MODELS; merge pattern + `only` CLI filter like
exp5/exp6 (a later per-model rerun overwrites just its key in the work JSON).

Writes data/data/exp7_metrics.json incrementally.
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

import numpy as np
import torch
from torch.nn import functional as F

from cross_common import MODELS  # noqa: E402
import tvu_common as C           # noqa: E402

N_GRAD_TOKENS = 128  # original exp1 protocol (autograd retention is the cost driver)


def tie_flags():
    """tie_word_embeddings per model key from the Step-0 metadata."""
    meta_path = os.path.join(C.BASE, "data", "models_metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return {k: bool(v["tie_word_embeddings"]) for k, v in meta["models"].items()}


# ---------- architecture resolution (standard causal LM + composite wrappers) ----

def text_stack_module(model):
    """The module whose forward(input_ids=...) yields .last_hidden_state and holds
    the transformer layers as a direct `.layers` child.

    Standard causal LMs: model.model. Composite multimodal wrappers (gemma-4-e2b =
    Gemma4ForConditionalGeneration): the language tower under `language_model`,
    whose embed_tokens/layers are DIRECT children (no inner .model). The vision
    tower also has encoder.layers — it is never returned here. GPTNeoX names its
    stack gpt_neox with embed_in — resolved by a generic direct-child scan."""
    for attr in ("language_model", "text_model"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
            return inner          # wrapper with an inner text stack (Llama-style)
        if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
            return m              # direct text model (gemma3/4 style)
    for _name, m in model.named_children():   # GPTNeoX: gpt_neox + embed_in
        emb = getattr(m, "embed_tokens", None) or getattr(m, "embed_in", None)
        if hasattr(m, "layers") and isinstance(emb, torch.nn.Embedding):
            return m
    return model.model


def get_wte_weight(model):
    """Text input-embedding weight, naming-agnostic (transformers' own resolver;
    covers GPTNeoX's embed_in and multimodal wrappers alike)."""
    try:
        return model.get_input_embeddings().weight
    except AttributeError:
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Embedding) and name.endswith("embed_tokens") \
                    and "vision" not in name:
                return m.weight
        raise AttributeError(f"no text embed_tokens found on {type(model).__name__}")


def get_lm_head(model):
    """(owner_module, lm_head_linear). Head naming: lm_head (Llama/Qwen/Phi/Gemma),
    embed_out or output_projection (GPTNeoX across transformers versions)."""
    for attr in ("lm_head", "embed_out", "output_projection"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Linear):
            return model, m
    head_suffixes = ("lm_head", "embed_out", "output_projection")
    for name, m in model.named_modules():
        if any(name.endswith(s) for s in head_suffixes) and isinstance(m, torch.nn.Linear):
            cur = model
            for p in name.split(".")[:-1]:
                nxt = getattr(cur, p, None)
                if not hasattr(nxt, "lm_head"):
                    break
                cur = nxt
            return (cur if hasattr(cur, "lm_head") else model), m
    raise AttributeError(f"no lm_head Linear found on {type(model).__name__}")


def get_l0_attn(model):
    tsm = text_stack_module(model)
    try:
        return tsm.layers[0].self_attn
    except (AttributeError, IndexError):
        for name, m in model.named_modules():
            if "vision" not in name \
                    and (name.endswith("layers.0.self_attn")
                         or name.endswith("layers.0.attention")):   # GPTNeoX
                return m
        raise AttributeError(f"no text layer-0 self_attn found on {type(model).__name__}")


# ---------- gradient math helpers (fp32/fp64 numpy, no forced upcast) ----------

def g_to_np(g):
    """Grad tensor -> CPU numpy. float64 for small matrices (matches original exp1
    precision), float32 when the matrix is big enough that fp64 would be a RAM
    problem (> 200M elements)."""
    if g.numel() > 200_000_000:
        return g.detach().float().cpu().numpy()
    return g.detach().double().cpu().numpy()


def cos_np(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.dtype == np.float32:  # upcast the dot/norm only for big fp32 matrices
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(float(a @ b) / (da * db))


def norm_np(x):
    x = np.asarray(x).ravel()
    if x.dtype == np.float32:
        return float(np.linalg.norm(x.astype(np.float64)))
    return float(np.linalg.norm(x))


# q/k/v projection naming differs across architectures and transformers versions:
# Llama/Qwen/Phi/Gemma use separate q_proj/k_proj/v_proj; GPTNeoX used separate
# query/key/value in older releases and a single FUSED query_key_value Linear
# (weight [3H, H], rows split q|k|v) in newer ones. Resolution is alias-based,
# with fused-matrix row slicing as the fallback.
QKV_ALIASES = ({"q_proj", "query"}, {"k_proj", "key"}, {"v_proj", "value"})


def _attn_linear(attn_mod, aliases):
    for a in aliases:
        m = getattr(attn_mod, a, None)
        if isinstance(m, torch.nn.Linear):
            return m
    return None


def l0_qkv_weight_grads(l0attn):
    """Three (W, G) pairs — q/k/v projection matrices and their gradients, fp32.
    Separate Linears give their own weight/grad; a fused query_key_value Linear is
    sliced by hidden-size rows in the same q|k|v order its forward uses."""
    seps = [_attn_linear(l0attn, al) for al in QKV_ALIASES]
    if all(s is not None for s in seps):
        return [(s.weight.detach().float(), s.weight.grad.detach().float()) for s in seps]
    fused = getattr(l0attn, "query_key_value", None)
    if isinstance(fused, torch.nn.Linear):
        h = fused.weight.size(1)
        w = fused.weight.detach().float()
        g = fused.weight.grad.detach().float()
        return [(w[i * h:(i + 1) * h], g[i * h:(i + 1) * h]) for i in range(3)]
    raise AttributeError("no q/k/v projections found on " + type(l0attn).__name__)


def l0_qkv_targets(l0attn):
    """Weight Parameters that receive layer-0 q/k/v gradients (autograd targets)."""
    seps = [_attn_linear(l0attn, al) for al in QKV_ALIASES]
    if all(s is not None for s in seps):
        return [s.weight for s in seps]
    fused = getattr(l0attn, "query_key_value", None)
    if isinstance(fused, torch.nn.Linear):
        return [fused.weight]
    raise AttributeError("no q/k/v projections found on " + type(l0attn).__name__)


# ---------- the two protocol arms ----------------------------------------------

def qkv0_effect_cos(model, l0attn, emb_grad_np, input_ids):
    """TZ activation-level proxy ('или через прокси'): cosine between the layer-0
    q/k/v perturbation caused by the embedding update (W_q dE_i) vs by the layer-0
    weight updates (dQ e_i), concatenated over all context tokens."""
    E_cpu = C._input_embedding_weight(model)  # [vocab, hidden] fp32
    rows = input_ids.squeeze(0).cpu().numpy()
    e_rows = torch.from_numpy(E_cpu[rows].astype(np.float32)).to(C.DEVICE)
    gE_rows = torch.from_numpy(emb_grad_np[rows].astype(np.float32)).to(C.DEVICE)

    parts_a, parts_b = [], []
    for W, Gw in l0_qkv_weight_grads(l0attn):
        a = gE_rows @ W.T     # dE_i then W_q -> perturbation from emb update
        b = e_rows @ Gw.T     # e_i then dQ  -> perturbation from qkv update
        parts_a.append(a.reshape(-1))
        parts_b.append(b.reshape(-1))
    va = torch.cat(parts_a).cpu().numpy()
    vb = torch.cat(parts_b).cpu().numpy()
    return cos_np(va, vb)


def set_targets(model, targets):
    for p in model.parameters():
        p.requires_grad_(p in targets)


def clear_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def restore_grads(model):
    for p in model.parameters():
        p.requires_grad_(True)


def manual_ce_loss(logits, input_ids):
    """HF shift convention CE from raw logits (G_out phase only — identical to the
    original exp1's output-path construction)."""
    vocab = logits.size(-1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.reshape(-1, vocab), shift_labels.reshape(-1))


def run_tied(model, input_ids):
    """Direct G_in / G_out decomposition on the shared matrix W (tied models)."""
    import torch.nn as nn

    wte = get_wte_weight(model)
    l0attn = get_l0_attn(model)
    lm_owner, lmh = get_lm_head(model)
    hidden_size = int(lmh.in_features)
    vocab = int(lmh.out_features)

    targets = {wte}
    targets.update(l0_qkv_targets(l0attn))

    # --- G_total: both paths at once (normal forward+backward with labels)
    set_targets(model, targets)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_total = g_to_np(wte.grad)

    # --- G_in: output path frozen (lm_head swapped for a frozen clone of W)
    clear_grads(model)
    orig_lm = lm_owner.lm_head
    frozen = nn.Linear(hidden_size, vocab, bias=False).to(C.DEVICE).to(wte.dtype)
    with torch.no_grad():
        frozen.weight.copy_(wte.data)
    for p in frozen.parameters():
        p.requires_grad_(False)
    lm_owner.lm_head = frozen
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_in = g_to_np(wte.grad)
    cos_emb_vs_qkv0 = qkv0_effect_cos(model, l0attn, g_in, input_ids)  # while l0 grads alive
    lm_owner.lm_head = orig_lm

    # --- G_out: input path inert (no-grad hidden states + fresh Parameter W')
    clear_grads(model)
    with torch.no_grad():
        base_hidden = text_stack_module(model)(input_ids=input_ids).last_hidden_state
    w_out = nn.Parameter(wte.data.clone().to(C.DEVICE))
    logits = F.linear(base_hidden, w_out)
    manual_ce_loss(logits, input_ids).backward()
    g_out = g_to_np(w_out.grad)

    restore_grads(model)
    return {
        "protocol": "tied G_in/G_out decomposition",
        "cos_gin_vs_gout": cos_np(g_in.ravel(), g_out.ravel()),
        "norm_ratio_gout_over_gin": norm_np(g_out) / norm_np(g_in),
        "norm_gin": norm_np(g_in),
        "norm_gout": norm_np(g_out),
        "cos_gtotal_vs_gin": cos_np(g_total.ravel(), g_in.ravel()),
        "cos_gtotal_vs_gout": cos_np(g_total.ravel(), g_out.ravel()),
        "cos_embupdate_vs_qkv0_effect_proxy": cos_emb_vs_qkv0,
    }


def run_untied(model, input_ids):
    """Separate-matrix baseline (untied models)."""
    wte = get_wte_weight(model)
    lmh = get_lm_head(model)[1].weight
    l0attn = get_l0_attn(model)

    targets = {wte, lmh}
    targets.update(l0_qkv_targets(l0attn))

    set_targets(model, targets)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    g_emb = g_to_np(wte.grad)
    g_lm = g_to_np(lmh.grad)
    cos_emb_vs_qkv0 = qkv0_effect_cos(model, l0attn, g_emb, input_ids)

    restore_grads(model)
    return {
        "protocol": "untied baseline (separate matrices)",
        "cos_gemb_vs_glm": cos_np(g_emb.ravel(), g_lm.ravel()),
        "norm_ratio_glm_over_gemb": norm_np(g_lm) / norm_np(g_emb),
        "norm_gemb": norm_np(g_emb),
        "norm_glm": norm_np(g_lm),
        "cos_embupdate_vs_qkv0_effect_proxy": cos_emb_vs_qkv0,
    }


# ---------- driver ----------------------------------------------------------------

def load_with_dtype(path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(C.DEVICE)
    model.eval()
    return tokenizer, model


def main():
    # optional CLI filter: python run_exp7_gradient_interference.py qwen3-4b[,...]
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

    tie = tie_flags()

    exp5_path = os.path.join(C.WORK, "exp5_metrics.json")
    exp5 = {}
    if os.path.exists(exp5_path):
        with open(exp5_path, encoding="utf-8") as fh:
            exp5 = json.load(fh)

    work_json = os.path.join(C.WORK, "exp7_metrics.json")
    results = {}
    if os.path.exists(work_json):   # merge: a later per-model run must not wipe earlier ones
        with open(work_json, encoding="utf-8") as fh:
            results.update(json.load(fh))

    for m in MODELS:
        key, path, dtype = m["key"], m["path"], m["dtype"]
        if only is not None and key not in only:
            continue
        print(f"=== Exp7 {key}  dtype={str(dtype)}  tied={bool(tie.get(key))}")
        tok = mod = None
        try:
            tok, mod = load_with_dtype(path, dtype)
            ids = C.make_diverse_tokens(tok, n_target=N_GRAD_TOKENS)

            if tie.get(key):
                r = run_tied(mod, ids)
            else:
                r = run_untied(mod, ids)

            e5 = exp5.get(key, {})
            if "aniso_raw" in e5:   # same protocol/seed as original exp1 — reuse Exp 5 values
                r["anisotropy_raw"] = e5["aniso_raw"]
                r["anisotropy_centered"] = e5["aniso_centered"]
            else:
                raw, cen = C.embedding_anisotropy(mod)
                r["anisotropy_raw"] = round(float(raw), 4)
                r["anisotropy_centered"] = round(float(cen), 4)

            for k in list(r.keys()):
                if isinstance(r[k], float):
                    r[k] = round(r[k], 4)
            results[key] = {"path": path, "dtype": str(dtype), **r}
            head_cos = r.get("cos_gin_vs_gout", r.get("cos_gemb_vs_glm"))
            print(f"  cos(shared/split grads) = {head_cos:+.4f}")
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

    print("\n--- Exp7 summary ---")
    for key, r in results.items():
        if "status" in r:
            print(f"{key:<16} SKIPPED: {r['status']}")
        elif r.get("protocol", "").startswith("tied"):
            ratio = r.get("norm_ratio_gout_over_gin")
            print(f"{key:<16} TIED   cos(G_in,G_out)={r['cos_gin_vs_gout']:+.4f}  "
                  f"||G_out||/||G_in||={ratio:.3f}")
        else:
            ratio = r.get("norm_ratio_glm_over_gemb")
            print(f"{key:<16} UNTIED cos(G_emb,G_lm)={r['cos_gemb_vs_glm']:+.4f}  "
                  f"||G_lm||/||G_emb||={ratio:.3f}")


if __name__ == "__main__":
    main()
