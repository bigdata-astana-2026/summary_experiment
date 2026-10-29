#!/usr/bin/env python3
"""
Этап 7. Итоговая оценка всех методов на test.

Подкоманды:
    python evaluate.py metrics   # ROUGE-1/2/L, chrF++, BERTScore, смысловое сходство с аннотацией   [GPU]
    python evaluate.py faith     # верность тексту статьи (NLI): подтверждается ли каждое предложение   [GPU]
    python evaluate.py stats     # 95% бутстрэп-интервалы, проверка значимости, таблицы для статьи      [CPU]
    python evaluate.py all

Какие методы оцениваются (для каждого языка):
    базовые: Oracle, Lead, Random;
    лучший по val в каждой группе экстрактивных методов
        (без обучения / с обучением × разреженное / плотное / гибридное представление);
    все варианты LLM (полный текст, гибридный отбор → LLM, одиночный отбор → LLM).

Ключевые сравнения (парный бутстрэп, 10 000 повторов, поправка Холма внутри языка):
    H1 гибридное vs одиночное плотное представление (экстрактив с обучением)
    H2 экстрактив с обучением vs без обучения (лучшие гибриды)
    H3 гибридный отбор → LLM vs одиночный отбор → LLM (для каждой модели)
    H4 гибридный отбор → LLM vs полный текст → LLM (для каждой модели)
    H5 лучший LLM vs лучший экстрактив

Пишет в ~/summary/data/runs/final/:
    scores_per_doc.parquet, faith_per_doc.parquet,
    table_main.csv (метод × язык, среднее и 95% ДИ), table_tests.csv (сравнения, p, p_Holm),
    table_by_discipline.csv, table_parallel.csv, table_cost.csv
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import read_jsonl, n_words, split_sents, UnicodeTokenizer

DATA = Path.home() / "summary" / "data"
EXT = DATA / "runs" / "extractive"
LLM = DATA / "runs" / "llm"
OUT = DATA / "runs" / "final"
SEED = 42
N_BOOT = 10_000
METRICS = ["rouge1", "rouge2", "rougeL", "chrf", "bertscore", "semsim"]
DENSE = ("bge-m3", "labse", "e5-large", "qwen3-emb")


# ============================== выбор методов ==============================
def group_of(method):
    if method in ("oracle", "lead", "random"):
        return method
    if method.startswith("llm-"):
        kind, model = method.split("|")
        return f"LLM {kind.replace('llm-', '')} ({model})"
    algo, rep = method.split("|")[:2]
    fam = "экстр. с обучением" if algo in ("logreg", "svm", "lgbm", "mlp") else "экстр. без обучения"
    kind = ("гибрид" if rep.startswith(("hyb", "rrf")) else "позиция" if rep == "pos"
            else "разреж." if rep in ("tfidf", "bm25") else "плотное")
    return f"{fam}: {kind}"


def finalists():
    v = pd.read_csv(EXT / "summary_metrics_val.csv")
    v["key"] = v["rouge1"] + v["rouge2"]
    v["group"] = v["method"].map(group_of)
    best = v.sort_values("key", ascending=False).groupby(["lang", "group"]).head(1)
    sel = {L: set(best.loc[best.lang == L, "method"]) for L in best["lang"].unique()}
    return sel


def load_summaries(sel):
    parts = [pd.read_parquet(p, columns=["id", "lang", "split", "method", "summary"])
             for p in EXT.glob("summaries_*.parquet")]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["split"] == "test"]
    keep = df.apply(lambda r: r["method"].startswith("llm-") or r["method"] in sel.get(r["lang"], ()), axis=1)
    df = df[keep].drop_duplicates(["id", "method"], keep="last").reset_index(drop=True)
    df["group"] = df["method"].map(group_of)
    return df


def load_docs():
    return {d["id"]: d for d in read_jsonl(DATA / "dataset.jsonl") if d["split"] == "test"}


# ============================== 1. метрики относительно аннотации ==============================
def stage_metrics(no_gpu=False):
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import CHRF
    OUT.mkdir(parents=True, exist_ok=True)
    sel = finalists()
    df = load_summaries(sel)
    docs = load_docs()
    refs = [docs[i]["reference"] for i in df["id"]]
    cands = [s if isinstance(s, str) and s.strip() else "." for s in df["summary"]]
    print(f"Резюме для оценки: {len(df):,} ({df['method'].nunique()} методов)")

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=UnicodeTokenizer())
    chrf = CHRF(word_order=2)
    t = time.time()
    r = [sc.score(a, b) for a, b in zip(refs, cands)]
    df["rouge1"] = [x["rouge1"].fmeasure for x in r]
    df["rouge2"] = [x["rouge2"].fmeasure for x in r]
    df["rougeL"] = [x["rougeL"].fmeasure for x in r]
    df["chrf"] = [chrf.sentence_score(b, [a]).score / 100 for a, b in zip(refs, cands)]
    df["len_ratio"] = [n_words(b) / max(n_words(a), 1) for a, b in zip(refs, cands)]
    print(f"ROUGE и chrF++ посчитаны за {time.time() - t:.0f} c")

    if no_gpu:
        df["bertscore"] = np.nan
        df["semsim"] = np.nan
    else:
        import torch
        from bert_score import score as bert_score
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t = time.time()
        _, _, F = bert_score(cands, refs, model_type="xlm-roberta-large", batch_size=64, device=dev, verbose=False)
        df["bertscore"] = F.numpy()
        print(f"BERTScore посчитан за {time.time() - t:.0f} c")
        t = time.time()
        m = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
        ea = m.encode(refs, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        eb = m.encode(cands, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        df["semsim"] = (ea * eb).sum(1)                       # смысловое сходство резюме и аннотации (LaBSE)
        print(f"Смысловое сходство (LaBSE) посчитано за {time.time() - t:.0f} c")
    df.drop(columns=["summary"]).to_parquet(OUT / "scores_per_doc.parquet", index=False)
    print(f"-> {OUT / 'scores_per_doc.parquet'}")


# ============================== 2. верность тексту статьи (NLI) ==============================
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def stage_faith(methods_filter=None, batch_size=64):
    """Для каждого предложения резюме: находим 3 самых похожих предложения статьи (LaBSE) и
    проверяем моделью NLI, следует ли предложение резюме из них. Доля подтверждённых предложений —
    верность тексту статьи. Экстрактивные резюме состоят из предложений статьи и служат проверкой
    калибровки (у них верность должна быть близка к 100%)."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sel = finalists()
    df = load_summaries(sel)
    if methods_filter:
        df = df[df["method"].str.contains(methods_filter)]
    docs = load_docs()
    enc = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=torch.float16).to(dev).eval()
    ent_id = {v.lower(): k for k, v in nli.config.id2label.items()}["entailment"]
    con_id = {v.lower(): k for k, v in nli.config.id2label.items()}["contradiction"]

    # векторы предложений статей (один раз на статью)
    doc_vecs = {}
    print(f"Векторы предложений {len(docs)} тестовых статей ...")
    for i, d in docs.items():
        doc_vecs[i] = enc.encode(d["sents"], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)

    pairs, owners = [], []                                  # (посылка, гипотеза) и к какому резюме относится
    for k, r in df.iterrows():
        d = docs[r["id"]]
        ss = split_sents(r["summary"] or "", r["lang"])[:20]
        if not ss:
            continue
        sv = enc.encode(ss, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        sims = sv @ doc_vecs[r["id"]].T
        for j, s in enumerate(ss):
            top = np.argsort(-sims[j])[:3]
            premise = " ".join(d["sents"][t] for t in sorted(top))
            pairs.append((premise, s))
            owners.append(k)
    print(f"Пар «статья — предложение резюме» для NLI: {len(pairs):,}")
    ent, con = np.zeros(len(pairs), np.float32), np.zeros(len(pairs), np.float32)
    t = time.time()
    with torch.no_grad():
        for b in range(0, len(pairs), batch_size):
            p, h = zip(*pairs[b:b + batch_size])
            x = tok(list(p), list(h), truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(dev)
            pr = torch.softmax(nli(**x).logits.float(), -1).cpu().numpy()
            ent[b:b + len(p)], con[b:b + len(p)] = pr[:, ent_id], pr[:, con_id]
    print(f"NLI посчитан за {time.time() - t:.0f} c")
    res = pd.DataFrame({"row": owners, "ent": ent, "con": con})
    agg = res.groupby("row").agg(faith_entail=("ent", "mean"), faith_supported=("ent", lambda x: (x > 0.5).mean()),
                                 faith_contradicted=("con", lambda x: (x > 0.5).mean()), n_sents=("ent", "size"))
    out = df[["id", "lang", "method", "group"]].join(agg, how="inner")
    out.to_parquet(OUT / "faith_per_doc.parquet", index=False)
    print(out.groupby(["lang", "group"])[["faith_supported", "faith_contradicted"]].mean().round(3).to_string())
    print(f"-> {OUT / 'faith_per_doc.parquet'}")


# ============================== 3. статистика и таблицы ==============================
def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired_boot(a, b, rng, n=N_BOOT):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), min(p, 1.0)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def stage_stats():
    rng = np.random.default_rng(SEED)
    sc = pd.read_parquet(OUT / "scores_per_doc.parquet")
    f = OUT / "faith_per_doc.parquet"
    if f.exists():
        fa = pd.read_parquet(f)[["id", "method", "faith_supported", "faith_contradicted"]]
        sc = sc.merge(fa, on=["id", "method"], how="left")
    metrics = [m for m in METRICS + ["faith_supported"] if m in sc and sc[m].notna().any()]
    meta = pd.read_json(DATA / "dataset.jsonl", lines=True)[["id", "discipline", "parallel"]]
    sc = sc.merge(meta, on="id", how="left")

    # ---- главная таблица: среднее и 95% ДИ ----
    rows = []
    for (L, method), g in sc.groupby(["lang", "method"]):
        row = {"lang": L, "method": method, "group": g["group"].iloc[0], "n": len(g)}
        for m in metrics:
            x = g[m].dropna()
            if len(x):
                lo, hi = boot_ci(x, rng, 2000)
                row[m], row[m + "_ci"] = x.mean(), f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    main = pd.DataFrame(rows)
    oracle = main[main.method == "oracle"].set_index("lang")["rouge1"]
    main["rouge1_of_oracle"] = main["rouge1"] / main["lang"].map(oracle)
    main.sort_values(["lang", "rouge1"], ascending=[True, False]).round(4) \
        .to_csv(OUT / "table_main.csv", index=False, encoding="utf-8-sig")

    # ---- ключевые сравнения ----
    def best(L, group_prefix):
        g = main[(main.lang == L) & main["group"].str.startswith(group_prefix)]
        return None if g.empty else g.sort_values("rouge1", ascending=False).iloc[0]["method"]

    tests = []
    for L in sorted(sc["lang"].unique()):
        cmp = [("H1 гибрид vs плотное (экстр. с обучением)", best(L, "экстр. с обучением: гибрид"), best(L, "экстр. с обучением: плотное")),
               ("H1b гибрид vs разреж. (экстр. с обучением)", best(L, "экстр. с обучением: гибрид"), best(L, "экстр. с обучением: разреж.")),
               ("H2 с обучением vs без обучения (гибриды)", best(L, "экстр. с обучением: гибрид"), best(L, "экстр. без обучения: гибрид"))]
        llm_models = sorted({m.split("|")[1] for m in sc.loc[sc.lang == L, "method"] if m.startswith("llm-")})
        for mdl in llm_models:
            cmp.append((f"H3 LLM: гибрид vs одиночный отбор ({mdl})", f"llm-hybrid|{mdl}", f"llm-single|{mdl}"))
            cmp.append((f"H4 LLM: гибрид vs полный текст ({mdl})", f"llm-hybrid|{mdl}", f"llm-full|{mdl}"))
        best_llm = main[(main.lang == L) & main.method.str.startswith("llm-")].sort_values("rouge1", ascending=False)
        if len(best_llm):
            cmp.append(("H5 лучший LLM vs лучший экстрактив", best_llm.iloc[0]["method"], best(L, "экстр. с обучением: гибрид")))
        # H5b: рекомендуемая система (гибридный отбор -> LLM) против лучшего экстрактива — одна и та же
        # конфигурация на всех языках, а не победитель по ROUGE (на ru им оказался полный текст)
        rec = main[(main.lang == L) & main.method.str.startswith("llm-hybrid|")].sort_values("rouge1", ascending=False)
        if len(rec):
            cmp.append(("H5b LLM-гибрид vs лучший экстрактив", rec.iloc[0]["method"], best(L, "экстр. с обучением: гибрид")))
        piv = sc[sc.lang == L].pivot_table(index="id", columns="method", values=metrics)
        lang_tests = []
        for name, a, b in cmp:
            if a is None or b is None or (metrics[0], a) not in piv or (metrics[0], b) not in piv:
                continue
            for m in metrics:
                pair = piv[[(m, a), (m, b)]].dropna()
                if len(pair) < 10:
                    continue
                diff, lo, hi, p = paired_boot(pair[(m, a)], pair[(m, b)], rng)
                lang_tests.append({"lang": L, "comparison": name, "A": a, "B": b, "metric": m, "n": len(pair),
                                   "mean_A": pair[(m, a)].mean(), "mean_B": pair[(m, b)].mean(),
                                   "diff": diff, "ci_low": lo, "ci_high": hi, "p": p})
        if lang_tests:
            adj = holm([t["p"] for t in lang_tests])
            for t, pa in zip(lang_tests, adj):
                t["p_holm"] = pa
                t["significant"] = pa < 0.05
            tests += lang_tests
    tests = pd.DataFrame(tests)
    tests.round(4).to_csv(OUT / "table_tests.csv", index=False, encoding="utf-8-sig")

    # ---- по дисциплинам и на параллельной подвыборке ----
    keyg = main.sort_values("rouge1", ascending=False).groupby(["lang", "group"]).head(1)
    key_methods = set(keyg["method"])
    sub = sc[sc.method.isin(key_methods)]
    sub.groupby(["lang", "group", "discipline"])[["rouge1", "chrf"] + (["bertscore"] if "bertscore" in metrics else [])] \
        .mean().round(4).reset_index().to_csv(OUT / "table_by_discipline.csv", index=False, encoding="utf-8-sig")
    sub[sub["parallel"] == True].groupby(["lang", "group"])[metrics].mean().round(4).reset_index() \
        .to_csv(OUT / "table_parallel.csv", index=False, encoding="utf-8-sig")

    # ---- стоимость ----
    cost = []
    for f_ in LLM.glob("*__*.jsonl"):
        r = pd.DataFrame([json.loads(l) for l in open(f_, encoding="utf-8")])
        r = r[r["error"].isna() & (r["split"] == "test")]
        for L, g in r.groupby("lang"):
            cost.append({"lang": L, "method": g["method"].iloc[0], "sec_per_doc_median": g["seconds"].median(),
                         "input_words_median": g["input_words"].median(), "prompt_tokens_median": g["prompt_tokens"].median(),
                         "output_tokens_median": g["output_tokens"].median()})
    us = EXT / "summaries_unsup.parquet"
    if us.exists():
        u = pd.read_parquet(us, columns=["lang", "method", "ms"])
        for (L, m), g in u.groupby(["lang", "method"]):
            if m in key_methods:
                cost.append({"lang": L, "method": m, "sec_per_doc_median": g["ms"].median() / 1000})
    pd.DataFrame(cost).round(4).to_csv(OUT / "table_cost.csv", index=False, encoding="utf-8-sig")

    # ---- вывод ----
    pd.set_option("display.width", 250)
    show = [m for m in ["rouge1", "rouge2", "rougeL", "chrf", "bertscore", "semsim", "faith_supported"] if m in main]
    for L in sorted(main["lang"].unique()):
        print(f"\n===== {L}: все методы (test), среднее; ДИ — в table_main.csv =====")
        t = main[main.lang == L].sort_values("rouge1", ascending=False)
        print(t[["group", "method"] + show + ["rouge1_of_oracle"]].round(3).to_string(index=False))
    print("\n===== Ключевые сравнения (ROUGE-1, BERTScore, верность): разница A − B, p после поправки Холма =====")
    t = tests[tests.metric.isin(["rouge1", "bertscore", "faith_supported"])]
    for _, r in t.iterrows():
        star = "  *" if r["significant"] else ""
        print(f"{r['lang']}  {r['comparison']:50s} {r['metric']:15s} {r['diff']:+.4f}  "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p_Holm={r['p_holm']:.4f}{star}")
    print(f"\nТаблицы: {OUT}  (* — значимо при p_Holm < 0.05)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["metrics", "faith", "stats", "all"])
    ap.add_argument("--no-gpu", action="store_true", help="только ROUGE и chrF (без BERTScore и смыслового сходства)")
    ap.add_argument("--faith-methods", default=None, help="регулярное выражение: проверять верность только этих методов")
    a = ap.parse_args()
    stages = ["metrics", "faith", "stats"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"\n######## {s} ########")
        {"metrics": lambda: stage_metrics(a.no_gpu), "faith": lambda: stage_faith(a.faith_methods),
         "stats": stage_stats}[s]()
