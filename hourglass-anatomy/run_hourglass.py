"""
Hourglass Anatomy — concept-verification protocol (paper draft v1).

Runs on the LOCAL Qwen2.5-0.5B from ../models/qwen2.5-0.5b, CPU-only, no HF downloads.

Experiment 1: "Narrow Cone"      — SVD spectrum of the embedding matrix + effective
                                   dimensionality (participation ratio).
Experiment 2: "Pressure of Mass" — cosine similarity of last-layer hidden states to the
                                   context center of mass; protocol version (repetitive
                                   text) + control on real diverse text from project docs.
Experiment 3: Anisotropy         — mean pairwise cosine between random word embeddings
                                   (raw and centered variants).

Outputs (results/):
    01_narrow_cone_svd.png
    02_pressure_of_mass.png
    hourglass_summary.json
"""

import gc
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, no plt.show()

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # repo root
MODEL_PATH = os.path.join(os.environ.get("MODELS_DIR") or os.path.join(ROOT, "models"), "qwen2.5-0.5b")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

DOC_FILES = [
    os.path.join(ROOT, "README.md"),
    os.path.join(ROOT, "PROJECT.md"),
    os.path.join(ROOT, "FINAL_REPORT_v4.md"),
    os.path.join(ROOT, "report.md"),
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # fp32 for stable CPU math (weights ship as bf16, which numpy() rejects)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()

    # architecture-agnostic access (gpt2: transformer.wte / qwen: model.embed_tokens)
    if hasattr(model, "transformer"):
        W_emb = model.transformer.wte.weight.detach().float().cpu().numpy()
        hidden_dim = int(model.config.n_embd)
    else:
        W_emb = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        hidden_dim = int(
            getattr(model.config, "hidden_size", None) or model.config.n_embd
        )

    return tokenizer, model, W_emb.astype(np.float64), hidden_dim


def experiment_1(W_emb):
    """Narrow cone: SVD spectrum of centered embedding matrix."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — NARROW CONE (SVD of embeddings)")
    print("=" * 60)

    W_centered = W_emb - np.mean(W_emb, axis=0)

    # top-100 spectrum for the plots (protocol)
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(W_centered)
    eigvals_100 = svd.singular_values_

    # full singular values for exact effective dimensionality / k95 (slow on CPU)
    print("Full SVD of centered embedding matrix... ", end="", flush=True)
    try:
        sigma_full = np.linalg.svd(W_centered.astype(np.float32), compute_uv=False)
        full_svd_ok = True
    except Exception as e:  # fallback to top-100 estimate if LAPACK chokes
        print(f"FAILED ({e}); using top-100 approximation")
        sigma_full = eigvals_100.astype(np.float64)
        full_svd_ok = False
    print("done.")

    s2 = float(np.sum(sigma_full**2))
    s4 = float(np.sum(sigma_full**4))
    d_eff = (s2**2) / s4  # participation ratio
    cumvar = np.cumsum((sigma_full**2) / s2)
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)

    print(f"Vocab x D: {W_emb.shape[0]} x {W_emb.shape[1]}")
    print(f"Singular values (full): n={sigma_full.size}")
    print(f"Participation ratio d_eff = (sum s^2)^2 / sum s^4  ->  {d_eff:.1f}")
    print(f"k95 (components holding 95% of energy)              ->  {k95}")

    # ---- plot ----
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(eigvals_100, marker=".", linestyle="-", color="b")
    plt.title('Спектр Сингулярных Чисел Эмбеддингов\n(Физика "Узкого Конуса")')
    plt.xlabel("Индекс компоненты")
    plt.ylabel("Сингулярное значение (Log Scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    cumvar_100 = np.cumsum(svd.explained_variance_ratio_)
    k95_top = int(np.searchsorted(cumvar_100, 0.95) + 1)
    plt.subplot(1, 2, 2)
    plt.plot(cumvar_100, marker=".", linestyle="-", color="r")
    plt.title("Накопленная Дисперсия (Эффективная размерность)")
    xlabel = "Количество измерений"
    plt.xlabel(xlabel)
    plt.ylabel("% объясненной дисперсии")
    plt.axhline(y=0.95, color="g", linestyle="--", label="95% энергии")
    plt.axvline(
        x=k95_top,
        color="k",
        linestyle=":",
        label=f"{k95_top} изм. (полный SVD: {k95})",
    )
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS, "01_narrow_cone_svd.png")
    plt.savefig(out, dpi=300)
    plt.close()

    return {
        "vocab_size": int(W_emb.shape[0]),
        "hidden_dim": int(W_emb.shape[1]),
        "d_eff_participation_ratio": round(d_eff, 2),
        "k95_full_svd": k95,
        "k95_top100_truncated": k95_top,
        "full_svd_exact": full_svd_ok,
        "sigma_top5": [round(float(x), 3) for x in sigma_full[:5]],
    }


def make_contexts(tokenizer):
    """Protocol context (repetitive) + diverse real-text control of equal length."""
    text_chunk = "The history of mathematics and physics reveals complex patterns. "
    proto_text = text_chunk * 200 + "Suddenly, the true bottleneck was"
    proto_t = torch.tensor(
        [tokenizer.encode(proto_text, add_special_tokens=False)], dtype=torch.long
    )

    docs = []
    for f in DOC_FILES:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fh:
                docs.append(fh.read())
    if not docs:
        raise SystemExit("no corpus docs at repo root (README.md etc.) - add markdown there or edit DOC_FILES")
    diverse_text = "\n".join(docs)
    diverse_ids = tokenizer.encode(diverse_text, add_special_tokens=False)
    n_target = proto_t.shape[1]
    if len(diverse_ids) < n_target:  # recycle text until long enough
        while len(diverse_ids) < n_target:
            diverse_ids += tokenizer.encode(diverse_text, add_special_tokens=False)
    diverse_t = torch.tensor([diverse_ids[:n_target]], dtype=torch.long)

    return proto_t, diverse_t, n_target


def experiment_2(model, tokenizer):
    """Pressure of mass: cosine similarity to the context center of mass."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — PRESSURE OF MASS (hidden states -> mean vector)")
    print("=" * 60)

    proto_t, diverse_t, n_target = make_contexts(tokenizer)

    def hidden_profile(ids_tensor):
        with torch.no_grad():
            outputs = model(input_ids=ids_tensor, output_hidden_states=True)
        hs = outputs.hidden_states[-1].squeeze(0).numpy().astype(np.float64)
        del outputs
        mean_vector = np.mean(hs, axis=0)
        mean_norm = mean_vector / (np.linalg.norm(mean_vector) + 1e-8)
        sims = []
        for h in hs:
            h_norm = h / (np.linalg.norm(h) + 1e-8)
            sims.append(float(np.dot(h_norm, mean_norm)))
        sims = np.array(sims)
        pos = np.arange(len(sims), dtype=np.float64)
        corr = float(np.corrcoef(pos, sims)[0, 1])
        return sims, {
            "n_tokens": int(len(sims)),
            "mean_sim": round(float(np.mean(sims)), 4),
            "std_sim": round(float(np.std(sims)), 4),
            "min_sim": round(float(np.min(sims)), 4),
            "max_sim": round(float(np.max(sims)), 4),
            "pos_vs_sim_correlation": round(corr, 4),
            "first50_mean": round(float(np.mean(sims[:50])), 4),
            "last50_mean": round(float(np.mean(sims[-50:])), 4),
        }

    print(f"Context A (protocol, repetitive text): {n_target} tokens")
    sims_a, stats_a = hidden_profile(proto_t)
    del proto_t
    print(
        f"  mean sim={stats_a['mean_sim']} std={stats_a['std_sim']} "
        f"min={stats_a['min_sim']} max={stats_a['max_sim']} "
        f"corr(pos,sim)={stats_a['pos_vs_sim_correlation']}"
    )

    print(f"Context B (diverse real docs, trimmed to same length): {n_target} tokens")
    sims_b, stats_b = hidden_profile(diverse_t)
    del diverse_t
    print(
        f"  mean sim={stats_b['mean_sim']} std={stats_b['std_sim']} "
        f"min={stats_b['min_sim']} max={stats_b['max_sim']} "
        f"corr(pos,sim)={stats_b['pos_vs_sim_correlation']}"
    )

    # ---- plot ----
    plt.figure(figsize=(12, 5))
    plt.plot(
        sims_a, color="purple", alpha=0.7, label="Протокол (повторяющийся текст)"
    )
    plt.plot(
        sims_b, color="teal", alpha=0.6, linestyle="--",
        label="Реальный разнообразный контекст",
    )
    plt.title(
        "Гравитационное притяжение к Центру Масс\n"
        "(Как масса контекста стирает индивидуальность токена)"
    )
    plt.xlabel("Позиция токена в контексте (Время/Глубина)")
    plt.ylabel("Косинусное сходство с Центром Масс")
    plt.axhline(
        y=stats_a["mean_sim"], color="r", linestyle="--",
        label=f'Среднее давление A ({stats_a["mean_sim"]:.2f})',
    )
    plt.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS, "02_pressure_of_mass.png")
    plt.savefig(out, dpi=300)
    plt.close()

    return {"context_A_repetitive": stats_a, "context_B_diverse_docs": stats_b}


def experiment_3(W_emb):
    """Anisotropy: mean pairwise cosine between random word embeddings."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — ANISOTROPY (random-pair cosines)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    idx = rng.choice(W_emb.shape[0], 2000, replace=False)

    def pair_stats(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        Mn = M / (norms + 1e-8)
        cs = np.dot(Mn[idx], Mn[idx].T)
        np.fill_diagonal(cs, 0.0)
        return float(np.mean(np.abs(cs))), float(np.mean(cs))

    abs_raw, raw = pair_stats(W_emb)
    centered = W_emb - np.mean(W_emb, axis=0)
    abs_cen, cen = pair_stats(centered)

    print(f"Среднее |cos| между случайными словами (raw):      {abs_raw:.4f}")
    print(f"Средний cos  между случайными словами (raw):       {raw:.4f}")
    print(f"Средний cos  между случайными словами (centered):  {cen:.4f}")
    print("Чем ближе |cos| к 1.0, тем сильнее слова 'слиплись' в узкий конус.")

    return {
        "mean_abs_cos_raw": round(abs_raw, 4),
        "mean_cos_raw": round(raw, 4),
        "mean_cos_centered": round(cen, 4),
        "n_pairs_sampled": 2000,
    }


def main():
    print(f"Loading local model: {MODEL_PATH}")
    tokenizer, model, W_emb, hidden_dim = load_model()
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    max_pos = int(getattr(model.config, "max_position_embeddings", 0))
    n_layers = int(
        getattr(model.config, "num_hidden_layers", 0)
        or getattr(model.config, "n_layer", 0)
    )
    print(f"Модель загружена. Скрытая размерность (D): {hidden_dim}")
    print(
        f"vocab={W_emb.shape[0]} layers={n_layers} "
        f"max_pos={max_pos} tie_word_embeddings={tie}"
    )

    summary = {
        "model": "qwen2.5-0.5b (local)",
        "hidden_dim": hidden_dim,
        "tie_word_embeddings": tie,
        "num_hidden_layers": n_layers,
        "max_position_embeddings": max_pos,
    }

    summary["experiment_1_narrow_cone"] = experiment_1(W_emb)
    summary["experiment_2_pressure_of_mass"] = experiment_2(model, tokenizer)
    del model
    gc.collect()
    summary["experiment_3_anisotropy"] = experiment_3(W_emb)

    out = os.path.join(RESULTS, "hourglass_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
