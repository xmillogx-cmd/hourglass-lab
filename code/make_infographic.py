"""Deliverable 3 — INFOGRAPHIC.png: единая карта результатов программы aiir-hourglass.

CPU-only (matplotlib Agg): читает готовые data/data/*.json (Exp7 multi-rep, Exp5) +
data/models_metadata.json, рисует 6 панелей по группам G1–G4 + вердикт-полосу G1–G5.
Не грузит модели и не трогает GPU — можно запускать напрямую любым python с matplotlib:
    python make_infographic.py
(на этой машине — из venv .venv-gemma4 рядом с хост-деревом)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

_TV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # project root via tvu_common.py marker
while not os.path.isfile(os.path.join(_TV_ROOT, "tvu_common.py")):
    _TV_ROOT = os.path.dirname(_TV_ROOT)
TVU = _TV_ROOT
WORK = os.path.join(TVU, "data")


def load(name):
    with open(os.path.join(WORK, name), encoding="utf-8") as fh:
        return json.load(fh)


# --- Данные Exp7/Exp5 (источник: data/data/exp7_multisubsample_metrics.json — multi-rep ×5;
#     gemma-4-e2b — single-run значения из docs/cross_model_summary.md, т.к. multi-rep
#     не поместился на карте 16 GiB (OOM); aniso_raw — data/data/exp5_metrics.json) ---
EXP7_ORDER = ["smollm-135m", "qwen2.5-0.5b", "tinyllama-1.1b", "phi-1_5",
              "pythia-2.8b", "qwen2.5-3b", "qwen3-4b", "gemma-4-e2b"]


def _build_exp7():
    with open(os.path.join(TVU, "data", "models_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)["models"]
    e7 = load("exp7_multisubsample_metrics.json")
    e5 = load("exp5_metrics.json")
    rows = []
    for name in EXP7_ORDER:
        d = e7[name]
        if "ratio_mean" not in d:  # gemma — OOM-скип multi-rep, single-run из cross_model_summary.md
            ratio, cos = 0.021, 0.0229
        else:
            ratio, cos = d["ratio_mean"], d["cos_mean"]
        rows.append((name, meta[name]["params"]["exact"] / 1e9, bool(meta[name]["tie_word_embeddings"]),
                     ratio, cos, e5[name]["aniso_raw"]))
    return rows


# model, params_B, tied, norm_ratio_out_over_in (mean из 5 окон), cos(G_in,G_out) (mean), aniso_raw
EXP7 = _build_exp7()

VERDICTS = [
    "G1 (H1) — подтверждена: выходной путь доминирует ×4.0–×5.2 у 4/5 tied (multi-rep ×5); gemma — инверсия (single-run; ~8× из 47× — артефакт масштаба √h, остаток ≈4.3× структурный); untied ×1.5–×2.6 «доминирование без оппозиции» на всех трёх (p.2)",
    "G2 (H2/H3) — hourglass 5/8: PRc ≈ 1.08–1.17, 36–62% глубины; Pythia-2.8B и Phi-1_5 — ранняя шея при высоком полу PRc 7.3/5.1 без коллапса <2; gemma исключена (sliding window); PR-neck L3 контентонезависим во всех 5 корпусах",
    "G3 (H4) — отрицательна: «давление массы» и lost-in-the-middle ≤8192 нет; устойчивое ограничение только relational binding L3",
    "G4 — аттрактор low-dim реален (98.2% энергии в top-10), но НЕ сжимающий; функциональная чувствительность ≠ геометрический PR-neck",
    "G5 — анизотропия: у всех трёх untied общий пол raw ≈ 0.02, у tied конус 0.064–0.349 (tying); центрированная анизотропия чисто масштабная (внутри tied строго монотонно по hidden_size)",
]


def panel_g1(ax):
    """G1 — Exp7: отношение норм градиентов out/in по 8 моделям (multi-rep ×5)."""
    names = [m[0] for m in EXP7]
    ratios = [m[3] for m in EXP7]
    colors = ["#2b6cb0" if tied else "#8a8f98" for (_, _, tied, *_r) in EXP7]
    bars = ax.bar(range(len(EXP7)), ratios, color=colors, width=0.72)
    # gemma — инверсия: подсветить (single-run; multi-rep OOM-скип)
    gemma_idx = names.index("gemma-4-e2b")
    bars[gemma_idx].set_color("#c0392b")
    for i, (r, c) in enumerate(zip(ratios, [m[4] for m in EXP7])):
        ax.annotate(f"{r:.2f}", (i, r), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"cos {c:+.3f}", (i, max(r * 0.3, 0.014)), ha="center", fontsize=7, color="#444")
    ax.annotate("ИНВЕРСИЯ\n(вход ×48)", xy=(gemma_idx, ratios[gemma_idx]), textcoords="offset points",
                xytext=(-52, 26), fontsize=8, fontweight="bold", color="#c0392b")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 14)
    ax.set_xticks(range(len(EXP7)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.4)
    ax.set_ylabel("‖G_out‖ / ‖G_in‖ (log)")
    ax.set_title("G1 — доминирование путей: отношение норм по 8 моделям "
                 "(Exp7 multi-rep ×5; gemma single-run)", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def panel_g2(ax):
    """G2 — PR cross corpus: 5 кривых neck-профилей."""
    data = load("pr_cross_corpus_metrics.json")["corpora"]
    cmap = plt.get_cmap("tab10")
    for i, (name, d) in enumerate(data.items()):
        prs = d["pr_by_layer"]
        ax.plot(range(len(prs)), prs, color=cmap(i % 10), lw=1.4, label=name)
        ax.scatter([d["neck_layer"]], [d["neck_pr"]], color=cmap(i % 10), s=28, zorder=5)
    ax.axvline(3, color="#c0392b", ls="--", lw=1.0, alpha=0.7)
    ax.text(0.985, 0.55, "neck = L3 во всех корпусах\nPR 1.125–1.174 (контентонезависимо)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, fontweight="bold", color="#c0392b")
    ax.set_yscale("log")   # уровень 0/1 даёт PR ~70–90 (эмбеддинги), без log шкалы всё схлопнулось бы
    ax.set_ylim(1.0, 140)
    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xlabel("уровень (0 = embeddings)")
    ax.set_ylabel("centered PR (log)")
    ax.set_title("G2 — hourglass: PR-профиль по 5 корпусам × 2048 токенов (включая shuffled noise)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5)
    ax.grid(alpha=0.3)


def panel_g3(ax):
    """G3 — multi-level retrieval probe: mean lift по уровням × позициям."""
    from matplotlib.patches import Patch

    data = load("multi_level_retrieval_probe_metrics.json")["levels"]
    positions = ["p=0.1", "p=0.5", "p=0.9"]
    level_meta = [("L1_exact", "#2b6cb0"), ("L2_semantic", "#3aa17e"), ("L3_relational_2hop", "#c0392b")]
    alphas = [0.60, 0.80, 1.0]   # светлее → p=0.1, темнее → p=0.9
    for li, (lvl, color) in enumerate(level_meta):
        lifts = [data[lvl][p]["mean_lift"] for p in positions]
        bars = ax.bar([li + (j - 1) * 0.26 for j in range(3)], lifts, width=0.24, color=color)
        for b, a in zip(bars, alphas):   # alpha-кодирование позиции: светлее → p=0.1
            b.set_alpha(a)
        for b, v in zip(bars, lifts):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7.5)
    ax.annotate("L3 обваливается → ограничение\nrelational binding (не позиция)", xy=(1.95, 1.8),
                xytext=(0.25, 6.3), fontsize=8.5, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xticks(range(3))
    ax.set_xticklabels([lvl for lvl, _ in level_meta], fontsize=8)
    ax.set_xlabel("уровень retrieval-пробы (внутри группы: позиции p=0.1 / 0.5 / 0.9)")
    ax.legend(handles=[Patch(facecolor="#8a8f98", alpha=a, label=p) for a, p in zip(alphas, positions)],
              fontsize=7, loc="upper right")
    ax.set_ylabel("mean lift, nats")
    ax.set_title("G3 — retrieval probe 0.5B: flat по позиции; L3 (2-hop binding) падает", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def panel_g4a(ax):
    """G4a — attractor subspace perturbation: отклонения трёх плеч при eps=0.01."""
    data = load("attractor_subspace_perturbation_metrics.json")
    sweep = [s for s in data["sweep"] if abs(s["epsilon"] - 0.01) < 1e-9]
    order = ["radial", "parallel", "orthogonal"]
    colors = {"radial": "#8a8f98", "parallel": "#c0392b", "orthogonal": "#2b6cb0"}
    vals, amps = [], {}
    for s in sweep:
        dn = data["arms_facts"][s["arm"]]["direction_norm"]
        amps[s["arm"]] = s["per_token_mean_l2"] / (s["epsilon"] * dn)
    xs = [order.index(s["arm"]) for s in sweep]
    vals = [s["per_token_mean_l2"] for s in sweep]
    ax.bar(xs, vals, color=[colors[o] for o in order], width=0.55)
    for s in sweep:
        i = order.index(s["arm"])
        ax.annotate(f"{s['per_token_mean_l2']:.3f}\n×{amps[s['arm']]:.1f} на единицу", (i, s["per_token_mean_l2"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["radial", "parallel\n(в подпространстве)", "orthogonal"], fontsize=8)
    ax.set_ylabel("per-token L2 отклонения (eps=0.01, log)")
    sub = data["subspace"]
    ax.set_title(f"G4a — аттрактор neck-L4: top-10 несёт {sub['energy_frac_top_k']*100:.1f}% энергии "
                 f"(σ1={sub['top_singular_values'][0]:.0f} vs ~74); усиление, а не сжатие",
                 fontsize=9.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def panel_g4b(ax):
    """G4b — Exp9: KL@eps=0.01 по слоям для gemma/qwen/tinyllama."""
    data = load("exp9_hidden_perturbation_metrics.json")
    styles = {"gemma-4-e2b": ("#c0392b", 1.8, "-"), "qwen2.5-0.5b": ("#2b6cb0", 1.1, "--"),
              "tinyllama-1.1b": ("#8a8f98", 1.1, ":")}
    for name in ("gemma-4-e2b", "qwen2.5-0.5b", "tinyllama-1.1b"):
        d = data[name]
        layers = [p["layer"] for p in d["per_layer"]]
        kls = [p["kl_by_eps"]["0.01"] for p in d["per_layer"]]
        c, lw, ls = styles[name]
        ax.plot(layers, kls, color=c, lw=lw, ls=ls, label=name)
    g = data["gemma-4-e2b"]
    nl = g["neck_at_max_eps"]["layer"]
    nk = g["neck_at_max_eps"]["kl"]
    ax.scatter([nl], [nk], color="#c0392b", s=60, zorder=5)
    ax.annotate(f"gemma: пик L{nl} = {nk:.2f} nats\n(геометрический neck — L35)", xy=(nl, nk),
                xytext=(18, -4), fontsize=8.5, fontweight="bold", color="#c0392b")
    ax.annotate("qwen: flat ≤0.002\n(совпадение с PR-neck — случайное)", xy=(12, 0.0015),
                fontsize=7.5, color="#2b6cb0", ha="center")
    ax.annotate("tinyllama: front-loaded L0≈0.34", xy=(2, 0.3), xytext=(9, 0.42), fontsize=7.5,
                color="#8a8f98", arrowprops=dict(arrowstyle="->", color="#8a8f98", lw=0.8))
    ax.set_xlabel("слой")
    ax.set_ylabel("KL @eps=1e-2, nats")
    ax.set_title("G4b — Exp9: функциональная чувствительность ≠ геометрический neck (3 модели)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)


def panel_g5(ax):
    """G5 — анизотропия сырых эмбеддингов vs размер (tied/untied)."""
    label_offsets = {  # в points, чтобы подписи не перекрывались
        "smollm-135m": (-8, 10),
        "qwen2.5-0.5b": (6, 8),
        "tinyllama-1.1b": (4, -16),
        "phi-1_5": (-58, 10),
        "pythia-2.8b": (-6, -16),
        "qwen2.5-3b": (-12, 10),
        "qwen3-4b": (-46, 10),
        "gemma-4-e2b": (8, -14),
    }
    for name, params, tied, _ratio, _cos, aniso in EXP7:
        if tied:
            ax.scatter([params], [aniso], s=130, color="#2b6cb0", zorder=4)
        else:
            ax.scatter([params], [aniso], facecolors="none", edgecolors="#c0392b", linewidths=2.2, s=150, zorder=4)
        dx, dy = label_offsets[name]
        ax.annotate(name + ("" if tied else " [untied]"), (params, aniso), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("параметры, B (log)")
    ax.set_ylabel("raw anisotropy score")
    ax.set_title("G5 — сырая анизотропия эмбеддингов vs размер (8 моделей):\n"
                 "конус у tied (0.064–0.349), общий пол ≈ 0.02 у всех трёх untied",
                 fontsize=9.8, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2b6cb0", label="tied"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec="#c0392b", mew=1.8, label="untied")],
              fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)


def verdict_strip(ax):
    ax.axis("off")
    y = 1.0
    for line in VERDICTS:
        ax.text(0.0, y, "▪", fontsize=12, va="top")
        ax.text(0.015, y, line, fontsize=9.3, va="top", wrap=True)
        y -= 0.185
    ax.set_title("Вердикты по группам", fontsize=10.5, fontweight="bold", loc="left")


def main():
    fig = plt.figure(figsize=(20, 13.5), dpi=140)
    fig.patch.set_facecolor("#fafbfc")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.12, 1.12, 0.62], hspace=0.42, wspace=0.30)

    fig.suptitle("aiir-hourglass — карта результатов программы (эксперименты 2026-09-15 → 2026-09-19)",
                 fontsize=16, fontweight="bold", x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.958, "Контрольная пара: Qwen2.5-0.5B (tied) vs TinyLlama-1.1B (untied); кросс-модельная выборка — 8 моделей "
                          "(5 tied + 3 untied), пять архитектурных семейств; все прогоны локальные, GPU 1", fontsize=9.3, color="#555")

    panel_g1(fig.add_subplot(gs[0, 0]))
    panel_g2(fig.add_subplot(gs[0, 1]))
    panel_g3(fig.add_subplot(gs[0, 2]))
    panel_g4a(fig.add_subplot(gs[1, 0]))
    panel_g4b(fig.add_subplot(gs[1, 1]))
    panel_g5(fig.add_subplot(gs[1, 2]))

    axv = fig.add_subplot(gs[2, :])
    verdict_strip(axv)

    out = os.path.join(TVU, "infographic", "ИНФОГРАФИКА.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"infographic written: {out}")


if __name__ == "__main__":
    main()
