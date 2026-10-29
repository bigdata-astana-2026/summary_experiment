#!/usr/bin/env python3
"""
Этап 8. Рисунки и таблицы для статьи (по готовым результатам этапа 7, GPU не нужен, ≈10 секунд).

Читает ~/summary/data/runs/final/{table_main, table_tests, table_cost, table_by_discipline}.csv
и judge__qwen3.6-35b.jsonl. Пишет в ~/summary/data/runs/paper/:
    fig1_resource_gradient.(png|pdf)   — качество ключевых систем на en / ru / kk (ROUGE-1, смысл, верность)
    fig2_cost_faithfulness.(png|pdf)   — время на статью против верности для вариантов LLM (стрелка: полный -> гибрид)
    fig3_effects.(png|pdf)             — эффекты H1, H1b, H2, H4, H5b с 95% ДИ (forest plot)
    fig4_disciplines.(png|pdf)         — ROUGE-1 по дисциплинам (тепловая карта)
    table_paper_main.csv / .md         — главная таблица: ключевые системы, среднее [95% ДИ]
    table_paper_tests.csv / .md        — проверки гипотез (разница, ДИ, p после Холма)
    error_samples.csv                  — неподтверждённые предложения LLM-гибрида для ручной разметки ошибок
Подписи на рисунках — на английском (статья для Q1/Q2).

Запуск:
    python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path.home() / "summary" / "data"
FIN = DATA / "runs" / "final"
OUT = DATA / "runs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "ru", "kk"]
LANG_NAME = {"en": "English", "ru": "Russian", "kk": "Kazakh"}
GEN = "gemma4:26b"                      # основная генерирующая модель для рисунков
# Палитра проверена на различимость при дальтонизме (порядок фиксирован, не перекрашивается)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})

main = pd.read_csv(FIN / "table_main.csv")
tests = pd.read_csv(FIN / "table_tests.csv")


def best_of(L, group_prefix):
    g = main[(main.lang == L) & main["group"].str.startswith(group_prefix)]
    return None if g.empty else g.sort_values("rouge1", ascending=False).iloc[0]["method"]


# ключевые системы: подпись -> функция выбора метода для языка
SYSTEMS = [
    ("Lead", lambda L: "lead"),
    ("Unsupervised hybrid (extr.)", lambda L: best_of(L, "экстр. без обучения: гибрид")),
    ("Supervised hybrid (extr.)", lambda L: best_of(L, "экстр. с обучением: гибрид")),
    ("LLM, full text", lambda L: f"llm-full|{GEN}"),
    ("LLM, hybrid selection", lambda L: f"llm-hybrid|{GEN}"),
]


def val(L, method, col):
    r = main[(main.lang == L) & (main.method == method)]
    return np.nan if r.empty or col not in r else float(r[col].iloc[0])


def ci(L, method, col):
    r = main[(main.lang == L) & (main.method == method)]
    if r.empty or f"{col}_ci" not in r or pd.isna(r[f"{col}_ci"].iloc[0]):
        return (np.nan, np.nan)
    lo, hi = r[f"{col}_ci"].iloc[0].strip("[]").split(",")
    return float(lo), float(hi)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ---------------- Рисунок 1. Ресурсный градиент ----------------
def fig1():
    metrics = [("rouge1", "ROUGE-1 vs. author abstract"), ("semsim", "Semantic similarity (LaBSE)"),
               ("faith_supported", "Supported sentences (LLM judge)")]
    metrics = [m for m in metrics if m[0] in main]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.2))
    x = np.arange(len(LANGS))
    w = 0.8 / len(SYSTEMS)
    for ax, (col, title) in zip(np.atleast_1d(axes), metrics):
        lo_all = []
        for k, (name, pick) in enumerate(SYSTEMS):
            ys, errs = [], [[], []]
            for L in LANGS:
                m = pick(L)
                y = val(L, m, col)
                lo, hi = ci(L, m, col)
                ys.append(y)
                errs[0].append(y - lo if not np.isnan(lo) else 0)
                errs[1].append(hi - y if not np.isnan(hi) else 0)
            lo_all += ys
            xs = x + (k - (len(SYSTEMS) - 1) / 2) * w
            if col == "rouge1":                             # столбцы — только при оси от нуля
                ax.bar(xs, ys, w * 0.92, color=C[k], label=name,
                       yerr=errs, error_kw={"elinewidth": 0.7, "ecolor": INK2, "capsize": 0})
            else:                                           # узкий диапазон оси -> точки с ДИ, не столбцы
                ax.errorbar(xs, ys, yerr=errs, fmt="o", color=C[k], markersize=6, elinewidth=1.2,
                            markeredgecolor="white", markeredgewidth=1, capsize=0, label=name)
        if col == "rouge1":                                 # потолок Oracle
            for i, L in enumerate(LANGS):
                o = val(L, "oracle", col)
                ax.hlines(o, i - 0.42, i + 0.42, colors=INK, linestyles="--", linewidth=1)
                ax.text(i, o + 0.008, "oracle", ha="center", va="bottom", fontsize=7, color=INK2)
        finite = [v for v in lo_all if not np.isnan(v)]
        if col != "rouge1" and finite:
            ax.set_ylim(min(finite) - 0.05, min(1.0, max(finite) + 0.03))
        ax.set_xticks(x, [LANG_NAME[L] for L in LANGS])
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="lower center",
               ncol=len(SYSTEMS), bbox_to_anchor=(0.5, -0.06), fontsize=8)
    save(fig, "fig1_resource_gradient")


# ---------------- Рисунок 2. Стоимость и верность ----------------
def fig2():
    cost = pd.read_csv(FIN / "table_cost.csv")
    if "faith_supported" not in main:
        print("  fig2 пропущен: нет faith_supported")
        return
    variants = [("full", "Full text", C[2]), ("single", "Single-repr. selection", C[0]),
                ("hybrid", "Hybrid selection", C[1])]
    markers = {"gemma4:26b": "o", "qwen3.6:35b": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, L in zip(axes, LANGS):
        for mdl, mk in markers.items():
            pts = {}
            for v, vname, col in variants:
                meth = f"llm-{v}|{mdl}"
                c = cost[(cost.lang == L) & (cost.method == meth)]
                f = val(L, meth, "faith_supported")
                if c.empty or np.isnan(f):
                    continue
                pts[v] = (float(c["sec_per_doc_median"].iloc[0]), f)
                ax.scatter(*pts[v], s=60, marker=mk, color=col, edgecolor="white", linewidth=1.5, zorder=3)
            if "full" in pts and "hybrid" in pts:
                ax.annotate("", xy=pts["hybrid"], xytext=pts["full"],
                            arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1}, zorder=2)
            if "hybrid" in pts:
                ax.annotate(mdl.split(":")[0], pts["hybrid"], textcoords="offset points",
                            xytext=(6, 6) if mk == "o" else (8, -12), ha="left",
                            fontsize=7, color=INK2)
        ax.set_title(LANG_NAME[L], fontsize=9, color=INK)
        ax.set_xlabel("Median seconds per article")
    axes[0].set_ylabel("Supported sentences (LLM judge)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=n, markersize=7) for _, n, c in variants]
    h += [Line2D([], [], marker=mk, ls="", color=INK2, label=m, markersize=7) for m, mk in markers.items()]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    save(fig, "fig2_cost_faithfulness")


# ---------------- Рисунок 3. Эффекты с ДИ ----------------
COMP = [("H1 гибрид", "Hybrid vs dense (extr.)"), ("H1b", "Hybrid vs sparse (extr.)"),
        ("H2 ", "Supervised vs unsupervised"), (f"H4 LLM: гибрид vs полный текст ({GEN})", "LLM: hybrid vs full text"),
        ("H5b", "LLM-hybrid vs best extractive")]


def fig3():
    mets = [m for m in [("rouge1", "Δ ROUGE-1"), ("bertscore", "Δ BERTScore"), ("faith_supported", "Δ supported share")]
            if m[0] in set(tests.metric)]
    fig, axes = plt.subplots(1, len(mets), figsize=(3.3 * len(mets), 3.4), sharey=True)
    ylabels = [n for _, n in COMP]
    for ax, (m, title) in zip(np.atleast_1d(axes), mets):
        for j, (key, _) in enumerate(COMP):
            for k, L in enumerate(LANGS):
                r = tests[(tests.lang == L) & (tests.metric == m) & tests.comparison.str.startswith(key)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j + (k - 1) * 0.22
                sig = bool(r["significant"])
                ax.hlines(y, r["ci_low"], r["ci_high"], color=C[k], linewidth=1.5)
                ax.scatter(r["diff"], y, s=28, color=C[k] if sig else "white", edgecolor=C[k], linewidth=1.5, zorder=3,
                           label=LANG_NAME[L] if j == 0 else None)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_yticks(range(len(COMP)), ylabels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=C[k], label=LANG_NAME[L], markersize=6) for k, L in enumerate(LANGS)]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.text(0.5, -0.11, "Filled marker: significant after Holm correction (p < 0.05); lines: 95% bootstrap CI",
             ha="center", fontsize=7, color=INK2)
    save(fig, "fig3_effects")


# ---------------- Рисунок 4. Дисциплины ----------------
def fig4():
    f = FIN / "table_by_discipline.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rows = [("Supervised hybrid (extr.)", "экстр. с обучением: гибрид"),
            ("LLM, full text", f"LLM full ({GEN})"), ("LLM, hybrid selection", f"LLM hybrid ({GEN})")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4))
    for ax, L in zip(axes, LANGS):
        g = d[d.lang == L]
        discs = sorted(g["discipline"].dropna().unique())
        M = np.array([[g[(g.group == grp) & (g.discipline == ds)]["rouge1"].mean() for ds in discs]
                      for _, grp in rows])
        im = ax.imshow(M, cmap="Blues", vmin=np.nanmin(M) - 0.05, vmax=np.nanmax(M), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if (M[i, j] - np.nanmin(M)) > 0.6 * (np.nanmax(M) - np.nanmin(M)) else INK)
        ax.set_xticks(range(len(discs)), discs, rotation=20, fontsize=7)
        ax.set_yticks(range(len(rows)), [r for r, _ in rows] if L == "en" else [""] * len(rows), fontsize=7)
        ax.set_title(f"{LANG_NAME[L]} — ROUGE-1", fontsize=9, color=INK)
        ax.grid(False)
    save(fig, "fig4_disciplines")


# ---------------- Таблицы ----------------
def tables():
    cols = [c for c in ["rouge1", "rouge2", "rougeL", "chrf", "bertscore", "semsim", "faith_supported"] if c in main]
    names = SYSTEMS + [("LLM, hybrid selection (qwen)", lambda L: "llm-hybrid|qwen3.6:35b"),
                       ("Oracle (upper bound)", lambda L: "oracle")]
    rows = []
    for L in LANGS:
        for name, pick in names:
            m = pick(L)
            r = {"lang": L, "system": name, "method": m}
            for c in cols:
                v, (lo, hi) = val(L, m, c), ci(L, m, c)
                r[c] = "" if np.isnan(v) else (f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else f"{v:.3f}")
            r["rouge1_of_oracle"] = f"{val(L, m, 'rouge1_of_oracle'):.2f}"
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_paper_main.csv", index=False, encoding="utf-8-sig")
    (OUT / "table_paper_main.md").write_text(to_md(t), encoding="utf-8")

    keep = tests[tests.comparison.str.match(r"H(1|1b|2|4|5b) ") & ~tests.comparison.str.contains("qwen")]
    keep = keep[keep.metric.isin(["rouge1", "bertscore", "semsim", "faith_supported"])].copy()
    keep["result"] = keep.apply(lambda r: f"{r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                                + (" *" if r["significant"] else ""), axis=1)
    piv = keep.pivot_table(index=["comparison", "metric"], columns="lang", values="result", aggfunc="first")
    piv = piv[[L for L in LANGS if L in piv]].reset_index()
    piv.to_csv(OUT / "table_paper_tests.csv", index=False, encoding="utf-8-sig")
    (OUT / "table_paper_tests.md").write_text(to_md(piv) + "\n\n* p < 0.05 after Holm correction; "
                                              "paired bootstrap, 10,000 resamples.\n", encoding="utf-8")
    print("  table_paper_main.csv/.md, table_paper_tests.csv/.md")


def to_md(df):
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


# ---------------- Примеры ошибок для ручного анализа ----------------
def error_samples(n_per_lang=40):
    f = FIN / "judge__qwen3.6-35b.jsonl"
    if not f.exists():
        return
    docs = {}
    for l in open(DATA / "dataset.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["split"] == "test":
            docs[d["id"]] = d["reference"]
    rows = []
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") or r["method"] != f"llm-hybrid|{GEN}":
            continue
        for s, lab in zip(r["sents"], r["labels"]):
            if lab in ("NOT_SUPPORTED", "CONTRADICTED"):
                rows.append({"lang": r["lang"], "id": r["id"], "label": lab, "sentence": s,
                             "reference_abstract": docs.get(r["id"], "")[:600], "error_type": "", "comment": ""})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df = df.groupby("lang", group_keys=False).apply(lambda g: g.sample(min(n_per_lang, len(g)), random_state=0))
    df.to_csv(OUT / "error_samples.csv", index=False, encoding="utf-8-sig")
    print(f"  error_samples.csv ({len(df)} предложений; заполните столбец error_type)")


if __name__ == "__main__":
    print(f"-> {OUT}")
    for fn in (fig1, fig2, fig3, fig4, tables, error_samples):
        try:
            fn()
        except Exception as e:
            print(f"  {fn.__name__}: ошибка {type(e).__name__}: {e}")
