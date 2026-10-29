#!/usr/bin/env python3
"""
Этап 7b. Верность резюме тексту статьи: LLM-судья вместо NLI.

Почему: проверка diag_faith.py показала, что модель NLI (mDeBERTa-xnli) ненадёжна на наших
научных текстах. Даже когда посылка и гипотеза — одно и то же предложение, она подтверждает
только 78–84%, а на 3 предложениях статьи — 58% (у Oracle должно быть почти 100%).

Как работает судья:
    для каждого предложения резюме находим 5 самых похожих предложений статьи (LaBSE),
    объединяем их в «выдержки», и локальная LLM размечает каждое предложение резюме:
    SUPPORTED / NOT_SUPPORTED / CONTRADICTED. Один запрос — одно резюме целиком.

Контроль качества судьи (обязательно для статьи):
    ctrl-pos  — резюме Oracle (предложения из статьи)              -> должно быть ≈ 95–100% SUPPORTED;
    ctrl-swap — предложения Oracle ДРУГОЙ статьи того же языка       -> должно быть ≈ 0–5% SUPPORTED;
    ctrl-num  — предложения Oracle с изменёнными числами             -> судья должен их ловить.
    Второй судья (другая модель) на подвыборке -> согласие (каппа Коэна) и проверка
    «самопредпочтения» (не завышает ли судья оценки резюме своей же модели).

Запуск (Ollama на 11435 должен работать):
    export OLLAMA_HOST=127.0.0.1:11435
    python judge.py run --judge qwen3.6:35b --controls              # всё (≈1,5–3 ч)
    python judge.py run --judge gemma4:26b --limit-docs 60          # второй судья, подвыборка (≈20–30 мин)
    python judge.py report                                          # таблицы + faith_per_doc.parquet
    python evaluate.py stats                                        # статистика уже с верностью судьи
"""
import argparse
import json
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pilot import CFG, split_sents, _ollama_headers
from evaluate import OUT, finalists, load_summaries, load_docs

TOP_K = 5
MAX_EVID = 60            # не больше 60 предложений-выдержек на запрос
MAX_SUM_SENTS = 20
CTRL_DOCS = 150          # статей на язык для контрольных наборов
LABELS = ("SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED")

PROMPT = """You are a strict fact-checker for scientific article summaries.
Below are EXCERPTS from a scientific article (in its original language) and numbered SENTENCES of a summary of that article.
For each summary sentence decide:
- SUPPORTED: every factual claim in the sentence is stated in the excerpts or follows directly from them. Paraphrase, translation-like rewording and generalization are fine.
- NOT_SUPPORTED: the sentence contains at least one specific claim (a number, result, method, object, name, conclusion) that the excerpts do not state.
- CONTRADICTED: the sentence conflicts with the excerpts (wrong number, opposite result, etc.).
Judge only against the excerpts, not your own knowledge.

EXCERPTS:
{evidence}

SUMMARY SENTENCES:
{sentences}

Answer with JSON only, one key per summary sentence, e.g. {{"S1": "SUPPORTED", "S2": "NOT_SUPPORTED"}}."""

_lock = threading.Lock()


def ollama_json(model, prompt, num_predict):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "think": False, "format": "json",
            "options": {"temperature": 0, "seed": 42, "num_ctx": CFG["llm_num_ctx"], "num_predict": num_predict}}

    def post(b):
        req = urllib.request.Request(CFG["ollama_url"] + "/api/chat", data=json.dumps(b).encode("utf-8"),
                                     headers=_ollama_headers())
        with urllib.request.urlopen(req, timeout=CFG["llm_timeout_s"]) as r:
            return json.loads(r.read())
    try:
        return post(body)
    except urllib.error.HTTPError as e:
        if "think" in e.read().decode("utf-8", "ignore").lower():
            body.pop("think")
            return post(body)
        raise


def parse_labels(text, n):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {k: v for k, v in re.findall(r'"?(S\d+)"?\s*:\s*"?([A-Z_]+)', text)}
    out = []
    for i in range(1, n + 1):
        v = str(d.get(f"S{i}", d.get(str(i), ""))).upper().replace(" ", "_")
        out.append(v if v in LABELS else None)
    return out


def perturb_numbers(s, rng):
    """Меняет каждое число в предложении на другое (контроль ctrl-num)."""
    def rep(m):
        x = m.group(0)
        if "." in x or "," in x:
            sep = "." if "." in x else ","
            a, b = x.split(sep, 1)
            return f"{int(a) + rng.integers(1, 9)}{sep}{b}"
        v = int(x)
        if 1900 <= v <= 2030:
            return str(v + int(rng.choice([-7, -5, -3, 3, 5])))
        if v < 10:
            return str(v + int(rng.integers(1, 9)))
        return str(int(v * rng.choice([0.5, 1.5, 2, 3])))
    return re.sub(r"\d+(?:[.,]\d+)?", rep, s)


def build_items(args, docs, enc):
    sel = finalists()
    df = load_summaries(sel)
    if args.methods:
        df = df[df["method"].str.contains(args.methods)]
    rng = np.random.default_rng(42)
    if args.limit_docs:
        keep = set()
        for L, g in df.groupby("lang"):
            ids = sorted(g["id"].unique())
            keep |= set(rng.choice(ids, min(args.limit_docs, len(ids)), replace=False))
        df = df[df["id"].isin(keep)]
    items = [(r["id"], r["lang"], r["method"], split_sents(r["summary"] or "", r["lang"])[:MAX_SUM_SENTS])
             for _, r in df.iterrows()]

    if args.controls:                                      # контрольные наборы на подвыборке статей
        orc = df[df["method"] == "oracle"] if "oracle" in set(df["method"]) else \
            load_summaries(sel).query("method == 'oracle'")
        for L, g in orc.groupby("lang"):
            g = g.sample(min(CTRL_DOCS, len(g)), random_state=1)
            sents = {r["id"]: split_sents(r["summary"] or "", L)[:MAX_SUM_SENTS] for _, r in g.iterrows()}
            ids = list(sents)
            for k, i in enumerate(ids):
                other = ids[(k + 1 + int(rng.integers(0, len(ids) - 1))) % len(ids)]
                if other == i:
                    other = ids[(k + 1) % len(ids)]
                items.append((i, L, "ctrl-swap", sents[other]))
                numbered = [perturb_numbers(s, rng) for s in sents[i] if re.search(r"\d", s)]
                numbered = [s for s in numbered if s not in sents[i]]
                if numbered:
                    items.append((i, L, "ctrl-num", numbered))
    items = [it for it in items if it[3]]
    return items


def main_run(args):
    import torch
    from sentence_transformers import SentenceTransformer
    try:
        with urllib.request.urlopen(urllib.request.Request(CFG["ollama_url"] + "/api/tags",
                                                           headers=_ollama_headers()), timeout=10) as r:
            avail = {m["name"] for m in json.loads(r.read())["models"]}
    except Exception as e:
        sys.exit(f"Ollama не отвечает по адресу {CFG['ollama_url']}: {e}\nexport OLLAMA_HOST=127.0.0.1:11435")
    if args.judge not in avail:
        sys.exit(f"Нет модели {args.judge}. Доступны: {sorted(avail)}")

    OUT.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "-", args.judge)
    out_f = OUT / f"judge__{safe}.jsonl"
    done = set()
    if out_f.exists():
        for l in open(out_f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("error") is None:
                done.add((r["id"], r["method"]))

    docs = load_docs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer("sentence-transformers/LaBSE", device=dev)
    items = [it for it in build_items(args, docs, enc) if (it[0], it[2]) not in done]
    print(f"Судья {args.judge}: к проверке {len(items):,} резюме (уже готово {len(done):,})")
    if not items:
        return

    print("Готовлю выдержки из статей ...")
    need = sorted({it[0] for it in items})
    dv = {i: enc.encode(docs[i]["sents"], batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
          for i in need}
    jobs = []
    for doc_id, L, method, ss in items:
        sv = enc.encode(ss, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        sims = sv @ dv[doc_id].T
        ev = []
        for j in range(len(ss)):
            for t in np.argsort(-sims[j])[:TOP_K]:
                if t not in ev:
                    ev.append(int(t))
        ev = sorted(ev[:MAX_EVID])
        evidence = "\n".join(f"[E{k + 1}] {docs[doc_id]['sents'][t]}" for k, t in enumerate(ev))
        sentences = "\n".join(f"S{k + 1}: {s}" for k, s in enumerate(ss))
        jobs.append((doc_id, L, method, ss, PROMPT.format(evidence=evidence, sentences=sentences)))
    del enc
    torch.cuda.empty_cache()

    t0 = time.time()
    ollama_json(args.judge, "Hi", 1)                       # прогрев
    print(f"Модель загружена за {time.time() - t0:.0f} c; запросов параллельно: {args.parallel}")

    def work(job):
        doc_id, L, method, ss, prompt = job
        t = time.time()
        try:
            resp = ollama_json(args.judge, prompt, 15 * len(ss) + 60)
            labels = parse_labels(resp.get("message", {}).get("content", ""), len(ss))
            err = None if any(labels) else "unparsed"
        except Exception as e:
            labels, err = [None] * len(ss), str(e)[:300]
        return {"id": doc_id, "lang": L, "method": method, "judge": args.judge, "sents": ss,
                "labels": labels, "seconds": round(time.time() - t, 2), "error": err}

    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as ex, open(out_f, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            n_ok += r["error"] is None
            n_err += r["error"] is not None
            with _lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            if k % 200 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k:,}/{len(futs):,}  ошибок {n_err}  прошло {el / 60:.0f} мин, "
                      f"осталось ≈{el / k * (len(futs) - k) / 60:.0f} мин", flush=True)
    print(f"-> {out_f}")


def load_judge(f):
    rows = []
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") is not None:
            continue
        for k, (s, lab) in enumerate(zip(r["sents"], r["labels"])):
            if lab is not None:
                rows.append({"id": r["id"], "lang": r["lang"], "method": r["method"], "judge": r["judge"],
                             "k": k, "label": lab})
    return pd.DataFrame(rows)


def main_report(args):
    from sklearn.metrics import cohen_kappa_score
    from evaluate import group_of
    files = sorted(OUT.glob("judge__*.jsonl"))
    if not files:
        sys.exit("Нет файлов judge__*.jsonl — сначала python judge.py run")
    J = pd.concat([load_judge(f) for f in files], ignore_index=True)
    main_judge = args.main or J.groupby("judge").size().idxmax()
    print(f"Основной судья: {main_judge}   (размечено предложений: "
          + ", ".join(f"{j}: {n:,}" for j, n in J.groupby('judge').size().items()) + ")")
    M = J[J.judge == main_judge]

    # ---- 1. контроль качества судьи ----
    ctrl = M[M.method.isin(["oracle", "ctrl-swap", "ctrl-num"])]
    tab = ctrl.assign(sup=ctrl.label.eq("SUPPORTED"), con=ctrl.label.eq("CONTRADICTED")) \
        .groupby(["lang", "method"]).agg(sentences=("sup", "size"), supported=("sup", "mean"),
                                         contradicted=("con", "mean")).round(3)
    tab.to_csv(OUT / "judge_controls.csv", encoding="utf-8-sig")
    print("\n=== Контроль судьи (oracle — должно быть высоко; ctrl-swap и ctrl-num — низко) ===")
    print(tab.to_string())

    # ---- 2. верность по резюме -> faith_per_doc.parquet (для evaluate.py stats) ----
    S = M[~M.method.str.startswith("ctrl-")]
    agg = S.assign(sup=S.label.eq("SUPPORTED"), con=S.label.eq("CONTRADICTED")) \
        .groupby(["id", "lang", "method"]).agg(faith_supported=("sup", "mean"),
                                               faith_contradicted=("con", "mean"),
                                               n_sents=("sup", "size")).reset_index()
    agg["group"] = agg["method"].map(group_of)
    nli = OUT / "faith_per_doc.parquet"
    nli_bak = OUT / "faith_nli_per_doc.parquet"
    if nli.exists() and not nli_bak.exists():
        shutil.copy(nli, nli_bak)                          # результат NLI сохраняем отдельно
    agg.to_parquet(nli, index=False)
    print("\n=== Верность (доля подтверждённых предложений), судья", main_judge, "===")
    print(agg.groupby(["lang", "group"])[["faith_supported", "faith_contradicted"]].mean().round(3)
          .unstack(0).to_string())
    print(f"-> {nli}  (результат NLI сохранён в {nli_bak.name})")

    # ---- 3. согласие двух судей и самопредпочтение ----
    judges = sorted(J.judge.unique())
    if len(judges) >= 2:
        other = [j for j in judges if j != main_judge][0]
        P = J[J.judge.isin([main_judge, other])].pivot_table(index=["id", "method", "k", "lang"],
                                                             columns="judge", values="label", aggfunc="first").dropna()
        P = P.reset_index()
        rows = []
        for L, g in list(P.groupby("lang")) + [("all", P)]:
            a = g[main_judge].eq("SUPPORTED")
            b = g[other].eq("SUPPORTED")
            rows.append({"lang": L, "sentences": len(g), "agreement": (a == b).mean(),
                         "kappa": cohen_kappa_score(a, b)})
        ag = pd.DataFrame(rows).round(3)
        ag.to_csv(OUT / "judge_agreement.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== Согласие судей {main_judge} и {other} (SUPPORTED или нет) ===")
        print(ag.to_string(index=False))

        L_ = P[P.method.str.startswith("llm-")].copy()
        L_["gen"] = L_.method.str.split("|").str[1]
        sp = L_.groupby("gen")[[main_judge, other]].agg(lambda x: x.eq("SUPPORTED").mean()).round(3)
        sp.to_csv(OUT / "judge_self_preference.csv", encoding="utf-8-sig")
        print("\n=== Самопредпочтение: доля SUPPORTED по модели-генератору (строки) и судье (столбцы) ===")
        print(sp.to_string())
        print("Если каждый судья заметно выше оценивает «свою» модель — есть самопредпочтение; сообщаем в статье.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--judge", default="qwen3.6:35b")
    r.add_argument("--methods", default=None, help="регулярное выражение по имени метода")
    r.add_argument("--limit-docs", type=int, default=None, help="столько статей на язык (подвыборка)")
    r.add_argument("--controls", action="store_true", help="добавить контрольные наборы ctrl-swap / ctrl-num")
    r.add_argument("--parallel", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--main", default=None, help="основной судья (по умолчанию — с наибольшей разметкой)")
    a = ap.parse_args()
    main_run(a) if a.cmd == "run" else main_report(a)
