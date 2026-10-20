"""aiir-hourglass — shared helpers for the empirical validation of the unified
geometric theory (weight tying -> gradient interference -> anisotropy cone ->
low-dim internal attractor -> weights/activations spectral asymmetry -> pressure
of mass).

Control pair (flags verified in each model's config.json):
    Model A (TIED)   = Qwen2.5-0.5B  (tie_word_embeddings=True)
    Model B (UNTIED) = TinyLlama-1.1B (tie_word_embeddings=False)

All compute runs on physical GPU 1 via torch device cuda:1 (GPU 0 is reserved
for the user's manual work — never touch it). fp32, local models only, no HF
downloads.

Results deliverables are EXACTLY 5 files in data/:
    01_gradient_interference.png
    02_pr_profile.png
    03_spectral_asymmetry.png
    04_pressure_of_mass.png
    empirical_validation_summary.json   (written once at the end, merging data/*)

Per-experiment raw numbers land in data/expN_metrics.json and are merged into
the single summary JSON by run_final_summary.py.

Path parameterization (no absolute paths anywhere): the model root is $MODELS_DIR,
falling back to <host-tree>/models next to this project; the diverse-doc corpus
is $CORPUS_DOCS_DIR, falling back to the host tree's docs and then to this
project's own README/CONCLUSIONS/SUMMARY/PAPER.
"""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))   # aiir-hourglass/
PARENT = os.path.dirname(BASE)                      # host tree (here cmq_experiment/) with models/ and the corpus docs
DATA = os.path.join(BASE, "data")
WORK = DATA      # raw per-experiment metrics + working figures (alias of data/)
LOGS = DATA      # run logs from .bat redirects (alias of data/)
RESULTS = DATA   # working figure PNGs land here (alias of data/)
os.makedirs(DATA, exist_ok=True)

import numpy as np
import torch

DEVICE = torch.device("cuda", 1)  # physical GPU 1 (user directive; no env vars)


def _resolve_models_dir():
    """Model root: $MODELS_DIR wins; otherwise the sibling <host>/models folder."""
    env = os.environ.get("MODELS_DIR")
    if env:
        return os.path.abspath(env)
    cand = os.path.join(PARENT, "models")
    if os.path.isdir(cand):
        return cand
    raise RuntimeError(
        "model directory not found (looked in %s). Set the MODELS_DIR environment "
        "variable to a folder containing the model subfolders." % cand)


def _resolve_corpus_docs():
    """Diverse-doc corpus for make_diverse_tokens: $CORPUS_DOCS_DIR wins; otherwise
    the host tree's docs (the previous research's README/PROJECT/FINAL_REPORT_v4/report.md);
    portable fallback — this project's own docs."""
    names = ("README.md", "PROJECT.md", "FINAL_REPORT_v4.md", "report.md")
    env = os.environ.get("CORPUS_DOCS_DIR")
    if env:
        d = os.path.abspath(env)
        return [os.path.join(d, n) for n in names]
    if all(os.path.isfile(os.path.join(PARENT, n)) for n in names):
        return [os.path.join(PARENT, n) for n in names]
    local_docs = ("README.md", "CONCLUSIONS.md", "SUMMARY.md")
    local_paper = ("PAPER_v1.md",)
    files = [os.path.join(BASE, d, n) for d in ("docs", "paper")
             for n in local_docs + local_paper if os.path.isfile(os.path.join(BASE, d, n))]
    if not files:
        raise RuntimeError("no corpus docs found; set CORPUS_DOCS_DIR to a folder with them")
    return files


MODELS_DIR = _resolve_models_dir()

MODEL_A_PATH = os.path.join(MODELS_DIR, "qwen2.5-0.5b")    # TIED   — Model A
MODEL_B_PATH = os.path.join(MODELS_DIR, "tinyllama-1.1b")  # UNTIED — Model B
NAME_A = "tied_qwen2_5_0_5b"
NAME_B = "untied_tinylama_1_1b"
N_TOKENS = 2048

DOC_FILES = _resolve_corpus_docs()


def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def free_model(tokenizer, model):
    import gc

    del tokenizer, model
    gc.collect()
    torch.cuda.empty_cache()


def make_diverse_tokens(tokenizer, n_target=N_TOKENS):
    """Same diverse real-doc corpus as hourlass_anatomy v1-v4 (README/PROJECT/
    FINAL_REPORT/report.md), tokenized by the given tokenizer and truncated to
    n_target tokens. Returns a LongTensor [1, n] on DEVICE."""
    texts = []
    for path in DOC_FILES:
        with open(path, "r", encoding="utf-8") as fh:
            texts.append(fh.read())
    enc = tokenizer(" ".join(texts), add_special_tokens=False)
    ids = enc["input_ids"][:n_target]
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


def k95_of_spectrum(sigma):
    cumvar = np.cumsum((sigma ** 2) / float(np.sum(sigma**2)))
    return int(np.searchsorted(cumvar, 0.95) + 1)


def participation_ratio(sigma):
    s2 = float(np.sum(sigma**2))
    s4 = float(np.sum(sigma**4))
    return (s2**2) / s4


def mean_abs_cos(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = M / (norms + 1e-8)
    cs = Mn @ Mn.T
    np.fill_diagonal(cs, 0.0)
    return float(np.mean(np.abs(cs)))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(float(a @ b) / (da * db))


def norm_of(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))


def save_work_json(name, payload):
    out = os.path.join(WORK, name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"work JSON written: {out}")


def _input_embedding_weight(model):
    """Text input-embedding matrix, robust to wrapper class: transformers' own
    get_input_embeddings() first (covers GPTNeoX's embed_in and multimodal
    wrappers alike); fallback searches named modules for an Embedding whose
    name ends with 'embed_tokens'."""
    try:
        return model.get_input_embeddings().weight.detach().float().cpu().numpy()
    except AttributeError:
        for _name, m in model.named_modules():
            if isinstance(m, torch.nn.Embedding) and _name.endswith("embed_tokens"):
                return m.weight.detach().float().cpu().numpy()
        raise AttributeError(
            "no text embed_tokens module found on " + type(model).__name__)


def embedding_anisotropy(model, n_words=2000, seed=42):
    """v1-style anisotropy of the raw embedding matrix (random word sample)."""
    w = _input_embedding_weight(model)
    rng = np.random.default_rng(seed)
    idx = rng.choice(w.shape[0], min(n_words, w.shape[0]), replace=False)
    M = w[idx]
    return mean_abs_cos(M), mean_abs_cos(M - M.mean(axis=0))


def first_layer_qkv_grads(model):
    """Flattened grads of layer-0 q/k/v proj weights concatenated into one vector."""
    l0attn = model.model.layers[0].self_attn
    parts = []
    for name in ("q_proj", "k_proj", "v_proj"):
        g = getattr(l0attn, name).weight.grad
        parts.append(g.detach().float().cpu().numpy().ravel())
    return np.concatenate(parts)
