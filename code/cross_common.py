"""Shared registry for the cross-model extension (Exp 5 / Exp 6) of aiir-hourglass.

Six local models, all under $MODELS_DIR (default: <host-tree>/models next to this
project). Per-model dtype policy:
fp32 where the fp32 peak fits in the 15.93 GiB VRAM of physical GPU 1; bf16
where it does not (Qwen3-4B fp32 weights alone exceed VRAM — user spec allows
memory-driven adaptation, "если не влезает — пропусти"). Exp6 lengths are capped
by each model's max_position_embeddings and by memory:

    smollm-135m   fp32  [2048]            (max_position_embeddings = 2048)
    qwen2.5-0.5b  fp32  [2048, 4096, 8192]  (proven at L=8192 in E4f)
    tinyllama-1.1b fp32 [2048]            (max_position_embeddings = 2048)
    qwen2.5-3b    bf16  [2048, 4096]      (fp32 weights + full hidden-state capture exceed 15.93 GiB; L=8192 skipped per spec)
    qwen3-4b      bf16  [2048, 4096]      (fp32 weights ~16 GB do not fit; L=8192 skipped per spec)
    gemma-4-e2b   bf16  [2048, 4096]      (base of google/gemma-4-E2B; fp32 weights ~20 GB do not fit;
                                           most layers are sliding-window attention, window 512)
    pythia-2.8b   bf16  [2048]            (EleutherAI/pythia-2.8b — GPTNeoX base, UNTIED; p.2 reviewer
                                           extension; max_position_embeddings = 2048)
    phi-1_5       fp32  [2048]            (microsoft/phi-1_5 — Phi base, UNTIED; p.2 reviewer extension;
                                           max_position_embeddings = 2048)

GEMMA SLOT — RESOLVED: clean unquantized base google/gemma-4-E2B downloaded to
$MODELS_DIR/gemma-4-e2b (user decision). model_type "gemma4", architecture
Gemma4ForConditionalGeneration (multimodal config with text_config: hidden 1536,
35 layers, vocab 262144, max_pos 131072, tie_word_embeddings=true; 7 of 35
layers full attention, the rest sliding window 512). It requires transformers
>= 5.x — on this machine it runs under an isolated venv (.venv-gemma4 next to
the host tree; transformers 5.17.0 with torch/numpy/safetensors installed
locally into the venv, NOT system-site-packages: conda's broken editable
flash_attn install crashes transformers 5.x import-time probes). Its gemma-only
Exp5/Exp6 runs use that python and append into the same data/data JSONs (merge logic).
"""

import os

import torch

BASE = os.path.dirname(os.path.abspath(__file__))   # aiir-hourglass/code
_TV_ROOT = os.path.abspath(os.path.join(BASE, ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
PARENT = os.path.dirname(TVU)                       # host tree (here cmq_experiment/) with models/
RESULTS = os.path.join(TVU, "data")


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


MODELS_DIR = _resolve_models_dir()

GEMMA_LOCAL = os.path.join(MODELS_DIR, "gemma-4-e2b")   # clean unquantized base (see module docstring)

MODELS = [
    {"key": "smollm-135m",   "path": os.path.join(MODELS_DIR, "smollm-135m"),
     "dtype": torch.float32, "exp6_lengths": [2048]},
    {"key": "qwen2.5-0.5b",  "path": os.path.join(MODELS_DIR, "qwen2.5-0.5b"),
     "dtype": torch.float32, "exp6_lengths": [2048, 4096, 8192]},
    {"key": "tinyllama-1.1b","path": os.path.join(MODELS_DIR, "tinyllama-1.1b"),
     "dtype": torch.float32, "exp6_lengths": [2048]},
    {"key": "qwen2.5-3b",    "path": os.path.join(MODELS_DIR, "qwen2.5-3b"),
     "dtype": torch.bfloat16,"exp6_lengths": [2048, 4096]},
    {"key": "qwen3-4b",      "path": os.path.join(MODELS_DIR, "qwen3-4b"),
     "dtype": torch.bfloat16,"exp6_lengths": [2048, 4096]},
    # gemma base: run under a python with transformers >= 5 (here .venv-gemma4); bf16 only
    {"key": "gemma-4-e2b",   "path": GEMMA_LOCAL,
     "dtype": torch.bfloat16,"exp6_lengths": [2048, 4096]},
    # p.2 reviewer extension (2026-09-19): two public UNTIED base models of the same scale class;
    # both max_position_embeddings = 2048 -> Exp6 at L=2048 only
    {"key": "pythia-2.8b",   "path": os.path.join(MODELS_DIR, "pythia-2.8b"),
     "dtype": torch.bfloat16,"exp6_lengths": [2048]},
    {"key": "phi-1_5",       "path": os.path.join(MODELS_DIR, "phi-1_5"),
     "dtype": torch.float32, "exp6_lengths": [2048]},
]
