#!/usr/bin/env python3
"""
Этапы 5–6. Генерация резюме LLM: абстрактивный и гибридный методы.

Варианты входа для LLM (--variants):
    full     — полный текст статьи                                  (абстрактивный метод)
    hybrid   — предложения, отобранные лучшим ГИБРИДНЫМ методом этапа 4  (гибридный метод: отбор → генерация)
    single   — предложения, отобранные лучшим ОДИНОЧНЫМ плотным методом  (контроль: вклад гибридного представления)
    lead     — первые предложения статьи                              (контроль: нужен ли умный отбор вообще)
    oracle   — Oracle-предложения (используют аннотацию!)             (только верхняя граница, не метод)
Лучшие методы отбора выбираются автоматически по val для каждого языка (summary_metrics_val.csv).

Модели — через Ollama (адрес из OLLAMA_HOST). Запросы идут параллельно (--parallel), чтобы GPU
не простаивал между статьями; результаты дописываются по мере готовности, поэтому прерванный
запуск продолжается с места остановки.

Подкоманды:
    python generate.py run   --models gemma4:26b qwen3.6:35b --variants full hybrid single
    python generate.py score            # ROUGE/chrF на test + сравнение с лучшими экстрактивными методами

Пишет в ~/summary/data/runs/llm/:
    <модель>__<вариант>.jsonl   — резюме, время, токены, ошибки
    selection.json              — какие методы отбора взяты для hybrid/single по каждому языку
    llm_scores_test.csv, comparison_test.csv
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import (PROMPTS, LABELS, CFG, clean_llm_output, _ollama_chat, truncate_input, n_words,
                   read_jsonl, wrong_language, UnicodeTokenizer)

DATA = Path.home() / "summary" / "data"
EXT = DATA / "runs" / "extractive"
OUT = DATA / "runs" / "llm"
TOPK = 25                      # сколько отобранных предложений подавать LLM
LOCK = threading.Lock()


def load_docs(split):
    return [{k: d[k] for k in ("id", "lang", "split", "sents", "reference", "text")}
            for d in read_jsonl(DATA / "dataset.jsonl") if d["split"] == split]


def pick_selectors():
    """Лучший гибридный и лучший одиночный плотный метод отбора на val для каждого языка."""
    v = pd.read_csv(EXT / "summary_metrics_val.csv")
    v = v[v["family"] == "sup"].copy()
    v["key"] = v["rouge1"] + v["rouge2"]
    rep = v["method"].str.split("|").str[1]
    dense = ("bge-m3", "labse", "e5-large", "qwen3-emb")
    sel = {}
    for L in v["lang"].unique():
        vl = v[v["lang"] == L]
        hyb = vl[rep.loc[vl.index].str.startswith("hyb")].sort_values("key", ascending=False).iloc[0]["method"]
        sgl = vl[rep.loc[vl.index].isin(dense)].sort_values("key", ascending=False).iloc[0]["method"]
        sel[L] = {"hybrid": hyb, "single": sgl}
    return sel


def load_rankings(methods):
    parts = [pd.read_parquet(p, columns=["id", "method", "ranking", "split"]) for p in EXT.glob("summaries_*.parquet")]
    df = pd.concat(parts)
    df = df[df["method"].isin(methods)]
    return {(r.id, r.method): list(r.ranking) for r in df.itertuples()}


def build_input(doc, variant, sel, ranks):
    sents = doc["sents"]
    if variant == "full":
        return doc["text"], "full"
    if variant == "lead":
        idx = list(range(min(TOPK, len(sents))))
    elif variant == "oracle":
        idx = ranks.get((doc["id"], "oracle"), [])[:TOPK]
    else:
        idx = ranks.get((doc["id"], sel[doc["lang"]][variant]), [])[:TOPK]
    idx = sorted(int(i) for i in idx if int(i) < len(sents))       # в исходном порядке текста
    return " ".join(sents[i] for i in idx), "extracted"


def one_request(model, doc, variant, sel, ranks):
    src, label_kind = build_input(doc, variant, sel, ranks)
    src, cut = truncate_input(src, CFG["llm_max_input_words"][doc["lang"]])
    n = n_words(doc["reference"])
    prompt = PROMPTS[doc["lang"]].format(n=n, lo=int(0.85 * n), label=LABELS[label_kind][doc["lang"]], text=src)
    num_predict = int(n * 4) + 300 + (2048 if CFG["llm_think"].get(model) else 0)
    t = time.time()
    try:
        resp = _ollama_chat(model, prompt, num_predict)
        summary, err = clean_llm_output(resp.get("message", {}).get("content", "")), None
    except Exception as e:
        resp, summary, err = {}, "", str(e)[:300]
    pt = resp.get("prompt_eval_count")
    return {"id": doc["id"], "lang": doc["lang"], "split": doc["split"], "model": model, "variant": variant,
            "method": f"llm-{variant}|{model}", "summary": summary, "seconds": round(time.time() - t, 2),
            "input_words": n_words(src), "input_truncated": cut, "prompt_tokens": pt,
            "output_tokens": resp.get("eval_count"), "finish_reason": resp.get("done_reason"),
            "context_overflow": bool(pt and pt >= CFG["llm_num_ctx"] - 8), "error": err}


def stage_run(models, variants, split, parallel, langs, limit):
    OUT.mkdir(parents=True, exist_ok=True)
    docs = [d for d in load_docs(split) if d["lang"] in langs]
    if limit:
        docs = [d for L in langs for d in [x for x in docs if x["lang"] == L][:limit]]
    sel = pick_selectors()
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Методы отбора для гибридного варианта (выбраны по val):")
    for L, s in sel.items():
        print(f"  {L}: hybrid = {s['hybrid']};  single = {s['single']}")
    ranks = load_rankings({m for s in sel.values() for m in s.values()} | {"oracle"})

    for model in models:
        print(f"\n=== {model}: прогрев (загрузка в GPU) ...")
        t0 = time.time()
        try:
            _ollama_chat(model, "Hi", 1)
        except Exception as e:
            sys.exit(f"Модель {model} недоступна в Ollama ({CFG['ollama_url']}): {e}")
        print(f"    загружена за {time.time() - t0:.0f} c")
        for variant in variants:
            safe = re.sub(r"[^\w.-]+", "-", model)
            f = OUT / f"{safe}__{variant}.jsonl"
            done = set()
            if f.exists():
                done = {json.loads(l)["id"] for l in open(f, encoding="utf-8") if l.strip() and not json.loads(l).get("error")}
            todo = [d for d in docs if d["id"] not in done]
            print(f"--- {model} | {variant}: готово {len(done)}, осталось {len(todo)}, параллельных запросов {parallel}")
            if not todo:
                continue
            t_start, n_ok, n_err = time.time(), 0, 0
            with ThreadPoolExecutor(parallel) as ex, open(f, "a", encoding="utf-8") as out:
                futs = [ex.submit(one_request, model, d, variant, sel, ranks) for d in todo]
                for k, fu in enumerate(as_completed(futs), 1):
                    r = fu.result()
                    with LOCK:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out.flush()
                    n_ok += r["error"] is None
                    n_err += r["error"] is not None
                    if k % 50 == 0 or k == len(todo):
                        el = time.time() - t_start
                        print(f"    {k}/{len(todo)}  {el / k:.2f} c/статью (с учётом параллельности), ошибок {n_err}", flush=True)
            wall = time.time() - t_start
            print(f"    итого {wall / 60:.1f} мин; пропускная способность {len(todo) / wall * 60:.1f} статей/мин")


def stage_score(split):
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import CHRF
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=UnicodeTokenizer())
    chrf = CHRF(word_order=2)
    refs = {d["id"]: d["reference"] for d in read_jsonl(DATA / "dataset.jsonl") if d["split"] == split}
    rows = []
    for f in sorted(OUT.glob("*__*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in refs and not r.get("error"):
                rows.append(r)
    if not rows:
        sys.exit("Нет результатов генерации.")
    df = pd.DataFrame(rows).drop_duplicates(["id", "method"], keep="last")
    s = [sc.score(refs[i], t or "") for i, t in zip(df["id"], df["summary"])]
    df["rouge1"] = [x["rouge1"].fmeasure for x in s]
    df["rouge2"] = [x["rouge2"].fmeasure for x in s]
    df["rougeL"] = [x["rougeL"].fmeasure for x in s]
    df["chrf"] = [chrf.sentence_score(t or "", [refs[i]]).score / 100 for i, t in zip(df["id"], df["summary"])]
    df["len_ratio"] = [n_words(t or "") / max(n_words(refs[i]), 1) for i, t in zip(df["id"], df["summary"])]
    df["wrong_lang"] = [wrong_language(t or "", L) for t, L in zip(df["summary"], df["lang"])]
    df.to_parquet(OUT / f"scores_per_doc_{split}.parquet", index=False)
    df.assign(family="llm")[["id", "lang", "split", "family", "method", "summary"]] \
        .to_parquet(EXT / "summaries_llm.parquet", index=False)        # для общей оценки на этапе 7
    agg = df.groupby(["lang", "method"]).agg(
        n=("id", "size"), rouge1=("rouge1", "mean"), rouge2=("rouge2", "mean"), rougeL=("rougeL", "mean"),
        chrf=("chrf", "mean"), len_ratio=("len_ratio", "mean"), wrong_lang=("wrong_lang", "mean"),
        sec_per_doc=("seconds", "median"), out_tokens=("output_tokens", "median")).round(4).reset_index()
    agg.to_csv(OUT / f"llm_scores_{split}.csv", index=False, encoding="utf-8-sig")

    # сравнение с лучшими экстрактивными методами (по val) на том же test
    t = pd.read_csv(EXT / f"summary_metrics_{split}.csv")
    v = pd.read_csv(EXT / "summary_metrics_val.csv")
    v["key"] = v["rouge1"] + v["rouge2"]
    comp = []
    for L in agg["lang"].unique():
        for fam, name in (("baseline", "oracle"), ("baseline", "lead")):
            r = t[(t.lang == L) & (t.method == name)].iloc[0]
            comp.append({"lang": L, "group": name, "method": name, **r[["rouge1", "rouge2", "rougeL", "chrf"]].to_dict()})
        for fam, label in (("unsup", "экстрактив без обучения (лучший)"), ("sup", "экстрактив с обучением (лучший)")):
            best = v[(v.lang == L) & (v.family == fam)].sort_values("key", ascending=False).iloc[0]["method"]
            r = t[(t.lang == L) & (t.method == best)].iloc[0]
            comp.append({"lang": L, "group": label, "method": best, **r[["rouge1", "rouge2", "rougeL", "chrf"]].to_dict()})
        for _, r in agg[agg.lang == L].iterrows():
            comp.append({"lang": L, "group": "LLM: " + r["method"].split("|")[0].replace("llm-", ""),
                         "method": r["method"], **r[["rouge1", "rouge2", "rougeL", "chrf"]].to_dict()})
    comp = pd.DataFrame(comp).round(3)
    comp.to_csv(OUT / f"comparison_{split}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    for L in comp["lang"].unique():
        print(f"\n===== {L}: экстрактивные против LLM (test) =====")
        print(comp[comp.lang == L].sort_values("rouge1", ascending=False).drop(columns="lang").to_string(index=False))
    print("\nСлужебное (длина резюме / доля не на том языке / секунд на статью):")
    print(agg[["lang", "method", "n", "len_ratio", "wrong_lang", "sec_per_doc"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score"])
    ap.add_argument("--models", nargs="*", default=["gemma4:26b", "qwen3.6:35b"])
    ap.add_argument("--variants", nargs="*", default=["full", "hybrid", "single"],
                    choices=["full", "hybrid", "single", "lead", "oracle"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--langs", nargs="*", default=["en", "ru", "kk"])
    ap.add_argument("--limit", type=int, default=None, help="тест: столько статей на язык")
    a = ap.parse_args()
    print(f"Ollama: {CFG['ollama_url']}")
    if a.stage == "run":
        stage_run(a.models, a.variants, a.split, a.parallel, a.langs, a.limit)
    else:
        stage_score(a.split)
