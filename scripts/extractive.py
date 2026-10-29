#!/usr/bin/env python3
"""
Этап 4. Экстрактивная суммаризация: представление текста × метод отбора предложений.

Подкоманды (запускаются по порядку):
    python extractive.py labels   # Oracle-метки предложений + базовые методы (Lead, Random, Oracle)   [CPU]
    python extractive.py unsup    # методы без обучения: TextRank / Centroid / MMR × все представления   [CPU]
    python extractive.py sup      # ML/DL-классификаторы отбора × все представления                     [CPU + GPU]
    python extractive.py report   # ROUGE-1/2/L и chrF++ для всех методов, таблицы val/test              [CPU]
    python extractive.py all      # всё подряд

Представления предложений:
    одиночные:  TF-IDF, BM25, BGE-M3, LaBSE, e5-large, Qwen3-Embedding
    гибридные:  взвешенная сумма TF-IDF + плотное (w = 0.25 / 0.5 / 0.75, выбирается на val),
                слияние рангов TF-IDF и плотного (Reciprocal Rank Fusion),
                для классификаторов — конкатенация TF-IDF(SVD) ⊕ плотное ⊕ позиционные признаки
Классификаторы: логистическая регрессия, линейный SVM, LightGBM (градиентный бустинг), MLP на GPU.

Метрики:
    уровень предложений (только классификаторы): precision, recall, F1, accuracy, ROC-AUC
        (метка предложения = входит ли оно в Oracle-набор, ближайший к авторской аннотации;
         порог выбирается на val и применяется к test)
    уровень резюме: ROUGE-1/2/L и chrF++ относительно авторской аннотации

Длина каждого резюме ограничивается длиной авторской аннотации (одинаково для всех методов).

Пишет в ~/summary/data/runs/extractive/:
    oracle_labels.parquet, summaries_*.parquet, sentence_metrics.csv,
    summary_metrics_val.csv, summary_metrics_test.csv, best_by_lang.csv
"""
import argparse
import math
import warnings
import os
import pickle
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import tokenize, n_words, read_jsonl, UnicodeTokenizer

DATA = Path.home() / "summary" / "data"
EMB = DATA / "emb"
OUT = DATA / "runs" / "extractive"
DENSE = ["bge-m3", "labse", "e5-large", "qwen3-emb"]
WEIGHTS = [0.25, 0.5, 0.75]          # вес TF-IDF в гибриде (остальное — плотное представление)
MMR_LAMBDA = 0.7
TOPK_RANK = 40                       # сколько лучших предложений сохранять для гибридного этапа 6
SEED = 42
WORKERS = max(1, (os.cpu_count() or 4) - 2)
G = {}                               # общие данные для процессов (наследуются при fork)


# ============================== общие функции ==============================
def load_dataset(langs, max_docs=None):
    docs = []
    per = Counter()
    for d in read_jsonl(DATA / "dataset.jsonl"):
        if d["lang"] not in langs:
            continue
        key = (d["lang"], d["split"])
        per[key] += 1
        if max_docs and per[key] > max_docs:
            continue
        docs.append({k: d[k] for k in ("id", "lang", "split", "sents", "reference")})
    return docs


def budget_of(doc):
    return n_words(doc["reference"])


def truncate_words(text, budget):
    out, n = [], 0
    for tok in text.split():
        out.append(tok)
        n += len(tokenize(tok))
        if n >= budget:
            break
    return " ".join(out)


def summary_from_order(sents, order, budget):
    """Берёт предложения в порядке order, пока не набран бюджет; выводит в порядке текста."""
    chosen, words = [], 0
    for i in order:
        chosen.append(int(i))
        words += n_words(sents[i])
        if words >= budget:
            break
    return truncate_words(" ".join(sents[i] for i in sorted(chosen)), budget)


def ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else Counter()


def rouge_n_f(cand, ref):
    overlap = sum((cand & ref).values())
    if not overlap:
        return 0.0
    p, r = overlap / max(sum(cand.values()), 1), overlap / max(sum(ref.values()), 1)
    return 2 * p * r / (p + r)


def pagerank(S, d=0.85, iters=60):
    n = len(S)
    if n == 0:
        return np.zeros(0)
    rows = S.sum(1, keepdims=True)
    M = np.where(rows > 0, S / np.maximum(rows, 1e-12), 1.0 / n)
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        p = (1 - d) / n + d * (M.T @ p)
    return p


def minmax(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def mmr_order(r, S, budget_sents, lam=MMR_LAMBDA):
    n = len(r)
    chosen, cand = [], set(range(n))
    while cand and len(chosen) < budget_sents:
        best, best_v = None, -1e9
        for i in cand:
            red = max((S[i, j] for j in chosen), default=0.0)
            v = lam * r[i] - (1 - lam) * red
            if v > best_v:
                best, best_v = i, v
        chosen.append(best)
        cand.remove(best)
    return chosen


def rrf(*score_lists, k=60):
    total = np.zeros(len(score_lists[0]))
    for s in score_lists:
        ranks = np.empty(len(s))
        ranks[np.argsort(-s)] = np.arange(len(s))
        total += 1.0 / (k + ranks + 1)
    return total


def doc_rows(doc_id):
    start, n = G["offsets"][doc_id]
    return start, n


# ============================== 1. Oracle-метки и базовые методы ==============================
def oracle_for_doc(doc):
    sents, ref = doc["sents"], doc["reference"]
    rt = tokenize(ref)
    r1, r2 = ngrams(rt, 1), ngrams(rt, 2)
    st = [tokenize(s) for s in sents]
    s1 = [ngrams(t, 1) for t in st]
    s2 = [ngrams(t, 2) for t in st]
    budget = budget_of(doc)
    chosen, c1, c2, best, words = [], Counter(), Counter(), 0.0, 0
    while words < 1.2 * budget:
        cand_best, cand_i = best, None
        for i in range(len(sents)):
            if i in chosen:
                continue
            v = rouge_n_f(c1 + s1[i], r1) + rouge_n_f(c2 + s2[i], r2)
            if v > cand_best:
                cand_best, cand_i = v, i
        if cand_i is None:
            break
        chosen.append(cand_i)
        c1 += s1[cand_i]
        c2 += s2[cand_i]
        best = cand_best
        words += len(st[cand_i])
    labels = [(doc["id"], i, int(i in chosen)) for i in range(len(sents))]
    base = {"id": doc["id"], "lang": doc["lang"], "split": doc["split"]}
    rnd = random.Random(f"{SEED}-{doc['id']}")
    rnd_order = list(range(len(sents)))
    rnd.shuffle(rnd_order)
    rows = [
        {**base, "method": "oracle", "family": "baseline", "summary": summary_from_order(sents, chosen, budget),
         "ranking": chosen[:TOPK_RANK]},
        {**base, "method": "lead", "family": "baseline",
         "summary": summary_from_order(sents, range(len(sents)), budget), "ranking": list(range(min(TOPK_RANK, len(sents))))},
        {**base, "method": "random", "family": "baseline", "summary": summary_from_order(sents, rnd_order, budget),
         "ranking": rnd_order[:TOPK_RANK]},
    ]
    return labels, rows


def stage_labels(langs, max_docs):
    docs = load_dataset(langs, max_docs)
    print(f"Oracle-метки для {len(docs)} статей, процессов: {WORKERS} ...")
    t = time.time()
    with Pool(WORKERS) as pool:
        res = pool.map(oracle_for_doc, docs, chunksize=8)
    labels = pd.DataFrame([l for lab, _ in res for l in lab], columns=["id", "sid", "label"])
    labels.to_parquet(OUT / "oracle_labels.parquet", index=False)
    rows = [r for _, rr in res for r in rr if r["split"] in ("val", "test")]
    pd.DataFrame(rows).to_parquet(OUT / "summaries_baseline.parquet", index=False)
    pos = labels.merge(pd.DataFrame([{"id": d["id"], "lang": d["lang"]} for d in docs]), on="id")
    print(f"Готово за {time.time() - t:.0f} c. Доля предложений с меткой 1 (входят в Oracle):")
    print(pos.groupby("lang")["label"].mean().round(3).to_string())


# ============================== 2. Методы без обучения ==============================
def fit_sparse(lang, train_sents):
    """TF-IDF и статистики BM25 по обучающим предложениям языка (одни на все статьи)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=2,
                            max_features=100_000, sublinear_tf=True)
    tfidf.fit(train_sents)
    cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, vocabulary=tfidf.vocabulary_)
    C = cv.transform(train_sents)
    N = C.shape[0]
    df = np.asarray((C > 0).sum(0)).ravel()
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    avglen = float(C.sum(1).mean())
    return {"tfidf": tfidf, "cv": cv, "bm25_idf": idf, "bm25_avglen": avglen}


def bm25_sim(C, idf, avglen, k1=1.2, b=0.75):
    C = C.tocsr().astype(np.float32)
    lens = np.asarray(C.sum(1)).ravel()
    W = C.copy()
    for r in range(C.shape[0]):
        lo, hi = W.indptr[r], W.indptr[r + 1]
        tf = W.data[lo:hi]
        W.data[lo:hi] = idf[W.indices[lo:hi]] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens[r] / avglen))
    Q = (C > 0).astype(np.float32)
    S = (Q @ W.T).toarray()
    S = (S + S.T) / 2
    return S / S.max() if S.max() > 0 else S


def representations(doc):
    """Матрица сходства S (n×n) и релевантность r (n) для каждого представления."""
    sp = G["sparse"][doc["lang"]]
    sents = doc["sents"]
    reps = {}
    X = sp["tfidf"].transform(sents)
    St = (X @ X.T).toarray()
    c = np.asarray(X.mean(0)).ravel()
    rt = (X @ c) / (np.linalg.norm(c) + 1e-12)
    reps["tfidf"] = (St, np.asarray(rt).ravel())
    Sb = bm25_sim(sp["cv"].transform(sents), sp["bm25_idf"], sp["bm25_avglen"])
    reps["bm25"] = (Sb, Sb.mean(1))
    start, n = doc_rows(doc["id"])
    for m in G["dense"]:
        V = np.asarray(G["emb"][m][start:start + n], dtype=np.float32)
        Sd = V @ V.T
        mu = V.mean(0)
        reps[m] = (Sd, V @ (mu / (np.linalg.norm(mu) + 1e-12)))
    return reps


def unsup_for_doc(doc):
    sents, budget = doc["sents"], budget_of(doc)
    n = len(sents)
    base = {"id": doc["id"], "lang": doc["lang"], "split": doc["split"], "family": "unsup"}
    rows = []

    def add(rep, algo, order, w=None, t0=None):
        rows.append({**base, "method": f"{algo}|{rep}" + (f"|w{w}" if w is not None else ""), "rep": rep,
                     "algo": algo, "w": w, "summary": summary_from_order(sents, order, budget),
                     "ranking": [int(i) for i in order[:TOPK_RANK]],
                     "ms": (time.perf_counter() - t0) * 1000 if t0 else None})

    budget_sents = max(1, int(np.ceil(budget / max(np.mean([n_words(s) for s in sents]), 1))) + 2)
    t_rep = time.perf_counter()
    reps = representations(doc)
    rep_ms = (time.perf_counter() - t_rep) * 1000 / max(len(reps), 1)

    def run(rep, S, r, w=None):
        S = np.clip(S, 0, None).copy()
        np.fill_diagonal(S, 0)
        out = {}
        t0 = time.perf_counter()
        sc = pagerank(S)
        out["textrank"] = sc
        add(rep, "textrank", np.argsort(-sc), w, t0)
        t0 = time.perf_counter()
        out["centroid"] = r
        add(rep, "centroid", np.argsort(-r), w, t0)
        t0 = time.perf_counter()
        order = mmr_order(minmax(r), S, budget_sents)
        rest = [i for i in np.argsort(-r) if i not in set(order)]
        add(rep, "mmr", order + rest, w, t0)
        return out

    scores = {rep: run(rep, S, r) for rep, (S, r) in reps.items()}
    St, rt = reps["tfidf"]
    for m in G["dense"]:
        Sd, rd = reps[m]
        for w in WEIGHTS:                                # гибрид: взвешенная сумма нормированных сходств
            Sh = w * minmax(St) + (1 - w) * minmax(Sd)
            rh = w * minmax(rt) + (1 - w) * minmax(rd)
            run(f"hyb-tfidf+{m}", Sh, rh, w)
        for algo in ("textrank", "centroid"):            # гибрид: слияние рангов (RRF)
            t0 = time.perf_counter()
            add(f"rrf-tfidf+{m}", algo, np.argsort(-rrf(scores["tfidf"][algo], scores[m][algo])), None, t0)
    for r_ in rows:
        r_["ms"] = (r_["ms"] or 0) + rep_ms
    return rows


def stage_unsup(langs, max_docs):
    docs = load_dataset(langs, max_docs)
    train_by_lang = {L: [s for d in docs if d["lang"] == L and d["split"] == "train" for s in d["sents"]] for L in langs}
    eval_docs = [d for d in docs if d["split"] in ("val", "test")]
    G["sparse"] = {}
    for L in langs:
        G["sparse"][L] = fit_sparse(L, train_by_lang[L])
        with open(OUT / f"sparse_{L}.pkl", "wb") as f:
            pickle.dump(G["sparse"][L], f)
        print(f"[{L}] TF-IDF/BM25 обучены на {len(train_by_lang[L]):,} предложениях, словарь {len(G['sparse'][L]['tfidf'].vocabulary_):,}")
    load_embeddings()
    print(f"Методы без обучения для {len(eval_docs)} статей (val+test), процессов: {WORKERS} ...")
    t = time.time()
    with Pool(WORKERS) as pool:
        res = pool.map(unsup_for_doc, eval_docs, chunksize=4)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "summaries_unsup.parquet", index=False)
    print(f"Готово за {time.time() - t:.0f} c: {df['method'].nunique()} конфигураций × {len(eval_docs)} статей")


def load_embeddings():
    sents = pd.read_parquet(EMB / "sentences.parquet", columns=["id", "sid"])
    sents["row"] = np.arange(len(sents))
    g = sents.groupby("id", sort=False)["row"].agg(["min", "size"])
    G["offsets"] = {i: (int(a), int(b)) for i, (a, b) in g.iterrows()}
    G["dense"] = [m for m in DENSE if (EMB / m / "sent.npy").exists()]
    G["emb"] = {m: np.load(EMB / m / "sent.npy", mmap_mode="r") for m in G["dense"]}
    print(f"Плотные представления: {G['dense']}")


# ============================== 3. Классификаторы отбора (ML/DL) ==============================
def position_features(docs):
    feats = []
    for d in docs:
        n = len(d["sents"])
        for i, s in enumerate(d["sents"]):
            rel = i / max(n - 1, 1)
            feats.append([rel, rel ** 2, float(i < 3), float(i >= n - 3), math.log1p(n_words(s)), math.log1p(n)])
    return np.asarray(feats, dtype=np.float32)


def centroid_cos(M, doc_index):
    """Косинус предложения к центроиду своей статьи (признак «центральности»)."""
    out = np.zeros(M.shape[0], dtype=np.float32)
    for lo, hi in doc_index:
        V = M[lo:hi]
        mu = V.mean(0)
        out[lo:hi] = V @ (mu / (np.linalg.norm(mu) + 1e-12))
    return out


def train_mlp(Xtr, ytr, Xva, yva, seed=SEED):
    import torch
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32, device=dev), torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xva, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(128, 1)).to(dev)
    pw = torch.tensor((1 - ytr.mean()) / max(ytr.mean(), 1e-6), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best_auc, best_state, bad = -1, None, 0
    for epoch in range(30):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for k in range(0, len(Xt), 2048):
            idx = perm[k:k + 2048]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:                                  # ранняя остановка по val
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32, device=dev)).squeeze(1)).cpu().numpy()
    return predict


def stage_sup(langs, max_docs, models):
    warnings.filterwarnings("ignore")                       # предупреждения LightGBM/sklearn не нужны в выводе
    from scipy import sparse as sps
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        from sklearn.ensemble import HistGradientBoostingClassifier

    docs_all = load_dataset(langs, max_docs)
    labels = pd.read_parquet(OUT / "oracle_labels.parquet")
    lab = {(i, s): l for i, s, l in labels.itertuples(index=False)}
    load_embeddings()
    sent_metrics, summ_rows = [], []

    for L in langs:
        docs = [d for d in docs_all if d["lang"] == L]
        with open(OUT / f"sparse_{L}.pkl", "rb") as f:
            sp = pickle.load(f)
        # выравнивание: строки признаков в порядке (статья, предложение)
        split_arr, y, doc_index, row = [], [], [], 0
        for d in docs:
            n = len(d["sents"])
            doc_index.append((row, row + n))
            row += n
            split_arr += [d["split"]] * n
            y += [lab.get((d["id"], i), 0) for i in range(n)]
        split_arr, y = np.asarray(split_arr), np.asarray(y)
        tr, va, te = split_arr == "train", split_arr == "val", split_arr == "test"
        all_sents = [s for d in docs for s in d["sents"]]
        print(f"\n[{L}] предложений: train {tr.sum():,}, val {va.sum():,}, test {te.sum():,}; доля 1: {y[tr].mean():.3f}")

        # ---- признаки ----
        Xtf = sp["tfidf"].transform(all_sents).tocsr()
        svd = TruncatedSVD(min(256, Xtf.shape[1] - 1), random_state=SEED).fit(Xtf[tr])
        Xsvd = svd.transform(Xtf).astype(np.float32)
        P = position_features(docs)
        dense = {}
        for m in G["dense"]:
            rows_ = np.concatenate([np.arange(G["offsets"][d["id"]][0], G["offsets"][d["id"]][0] + G["offsets"][d["id"]][1])
                                    for d in docs])
            dense[m] = np.asarray(G["emb"][m][rows_], dtype=np.float32)
        cen = {m: centroid_cos(dense[m], doc_index)[:, None] for m in dense}
        cen_tf = centroid_cos(Xsvd, doc_index)[:, None]

        feature_sets = {"pos": ("dense", P), "tfidf": ("sparse", None)}
        for m in dense:
            feature_sets[m] = ("dense", dense[m])
            feature_sets[f"hyb-tfidf+{m}+pos"] = ("dense", np.hstack([Xsvd, dense[m], P, cen[m], cen_tf]))
        if len(dense) > 1:
            feature_sets["hyb-all+pos"] = ("dense", np.hstack([Xsvd] + [dense[m] for m in dense] + [P, cen_tf]
                                                              + [cen[m] for m in dense]))

        for fname, (kind, X) in feature_sets.items():
            for model in models:
                t0 = time.time()
                if kind == "sparse" and model in ("lgbm", "mlp"):
                    Xuse, kind_use = Xsvd, "dense"            # для бустинга и MLP — TF-IDF, сжатый SVD до 256
                elif kind == "sparse":
                    Xuse, kind_use = Xtf, "sparse"
                else:
                    Xuse, kind_use = X, "dense"
                if kind_use == "dense" and model in ("logreg", "svm", "mlp"):
                    sc = StandardScaler().fit(Xuse[tr])
                    Xuse = sc.transform(Xuse).astype(np.float32)
                if model == "logreg":
                    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                             solver="liblinear" if kind_use == "sparse" else "lbfgs")
                    clf.fit(Xuse[tr], y[tr])
                    score = clf.predict_proba(Xuse)[:, 1]
                elif model == "svm":
                    clf = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)
                    clf.fit(Xuse[tr], y[tr])
                    score = clf.decision_function(Xuse)
                elif model == "lgbm":
                    if lgb is not None:
                        # metric="auc": ранняя остановка только по AUC (раньше срабатывала по logloss,
                        # который при взвешивании классов растёт с первой итерации — модель недообучалась)
                        clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                                                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                                                 colsample_bytree=0.5, is_unbalance=True, metric="auc",
                                                 n_jobs=WORKERS, verbose=-1, random_state=SEED)
                        clf.fit(Xuse[tr], y[tr], eval_set=[(Xuse[va], y[va])],
                                callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
                    else:
                        clf = HistGradientBoostingClassifier(max_iter=400, class_weight="balanced",
                                                             random_state=SEED).fit(Xuse[tr], y[tr])
                    score = clf.predict_proba(Xuse)[:, 1]
                elif model == "mlp":
                    pred = train_mlp(Xuse[tr], y[tr].astype(np.float32), Xuse[va], y[va])
                    score = pred(Xuse)
                fit_s = time.time() - t0

                # ---- метрики уровня предложений: порог по F1 на val, применяется к test ----
                qs = np.quantile(score[va], np.linspace(0.5, 0.995, 100))
                thr = max(qs, key=lambda q: f1_score(y[va], score[va] >= q, zero_division=0))
                yp = score[te] >= thr
                method = f"{model}|{fname}"
                sent_metrics.append({
                    "lang": L, "method": method, "model": model, "features": fname, "dim": Xuse.shape[1],
                    "auc_test": roc_auc_score(y[te], score[te]), "auc_val": roc_auc_score(y[va], score[va]),
                    "precision": precision_score(y[te], yp, zero_division=0), "recall": recall_score(y[te], yp, zero_division=0),
                    "f1": f1_score(y[te], yp, zero_division=0), "accuracy": accuracy_score(y[te], yp),
                    "threshold": float(thr), "fit_seconds": round(fit_s, 1)})
                # ---- резюме: лучшие по оценке предложения в пределах бюджета ----
                for d, (lo, hi) in zip(docs, doc_index):
                    if d["split"] not in ("val", "test"):
                        continue
                    order = list(np.argsort(-score[lo:hi]))
                    summ_rows.append({"id": d["id"], "lang": L, "split": d["split"], "family": "sup",
                                      "method": method, "rep": fname, "algo": model, "w": None,
                                      "summary": summary_from_order(d["sents"], order, budget_of(d)),
                                      "ranking": [int(i) for i in order[:TOPK_RANK]]})
                m_ = sent_metrics[-1]
                print(f"  {method:40s} AUC {m_['auc_test']:.3f}  F1 {m_['f1']:.3f}  Acc {m_['accuracy']:.3f}  ({fit_s:.0f} c)")

    # объединяем с уже посчитанным: перезапуск одной модели не стирает результаты остальных
    new_m, new_s = pd.DataFrame(sent_metrics), pd.DataFrame(summ_rows)
    f_m, f_s = OUT / "sentence_metrics.csv", OUT / "summaries_sup.parquet"
    if f_m.exists():
        old_m = pd.read_csv(f_m)
        keep = ~old_m.set_index(["lang", "method"]).index.isin(new_m.set_index(["lang", "method"]).index)
        new_m = pd.concat([old_m[keep], new_m], ignore_index=True)
    if f_s.exists():
        old_s = pd.read_parquet(f_s)
        keep = ~old_s.set_index(["lang", "method"]).index.isin(new_s.set_index(["lang", "method"]).index)
        new_s = pd.concat([old_s[keep], new_s], ignore_index=True)
    new_m.round(4).sort_values(["lang", "method"]).to_csv(f_m, index=False, encoding="utf-8-sig")
    new_s.to_parquet(f_s, index=False)
    print(f"\nМетрики уровня предложений -> {OUT / 'sentence_metrics.csv'}")


# ============================== 4. Оценка резюме ==============================
def score_row(args):
    summary, ref = args
    s = G["rouge"].score(ref, summary or "")
    return (s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeL"].fmeasure,
            G["chrf"].sentence_score(summary or "", [ref]).score / 100)


def stage_report(langs, max_docs):
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import CHRF
    G["rouge"] = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=UnicodeTokenizer())
    G["chrf"] = CHRF(word_order=2)
    refs = {d["id"]: d["reference"] for d in load_dataset(langs, max_docs)}
    parts = [pd.read_parquet(p) for p in sorted(OUT.glob("summaries_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["lang"].isin(langs) & df["id"].isin(refs)]
    print(f"Оцениваю {len(df):,} резюме ({df['method'].nunique()} методов), процессов: {WORKERS} ...")
    with Pool(WORKERS) as pool:
        sc = pool.map(score_row, zip(df["summary"], df["id"].map(refs)), chunksize=200)
    df[["rouge1", "rouge2", "rougeL", "chrf"]] = np.asarray(sc)
    df[["id", "lang", "split", "family", "method", "rouge1", "rouge2", "rougeL", "chrf"]] \
        .to_parquet(OUT / "scores_per_doc.parquet", index=False)

    agg = df.groupby(["lang", "split", "family", "method"])[["rouge1", "rouge2", "rougeL", "chrf"]].mean().reset_index()
    for split in ("val", "test"):
        agg[agg.split == split].round(4).to_csv(OUT / f"summary_metrics_{split}.csv", index=False, encoding="utf-8-sig")

    # лучшая конфигурация по val (R1+R2) в каждой группе; гибрид с весом w выбирается тоже по val
    agg["key"] = agg["rouge1"] + agg["rouge2"]
    val, test = agg[agg.split == "val"], agg[agg.split == "test"].set_index(["lang", "method"])
    best = []
    for L in langs:
        v = val[val.lang == L].sort_values("key", ascending=False)
        print(f"\n===== {L}: 15 лучших методов по val, значения на TEST =====")
        print(f"{'метод':45s} {'R1':>6} {'R2':>6} {'RL':>6} {'chrF':>6}")
        for _, r in v.head(15).iterrows():
            t = test.loc[(L, r["method"])]
            print(f"{r['method']:45s} {t['rouge1']:.3f}  {t['rouge2']:.3f}  {t['rougeL']:.3f}  {t['chrf']:.3f}")
        for fam in ("baseline", "unsup", "sup"):
            for _, r in v[v.family == fam].head(1 if fam != "baseline" else 3).iterrows():
                t = test.loc[(L, r["method"])]
                best.append({"lang": L, "family": fam, "method": r["method"], **t[["rouge1", "rouge2", "rougeL", "chrf"]].to_dict()})
    pd.DataFrame(best).round(4).to_csv(OUT / "best_by_lang.csv", index=False, encoding="utf-8-sig")
    print(f"\nВсе методы: {OUT / 'summary_metrics_test.csv'}\nЛучшие по группам: {OUT / 'best_by_lang.csv'}")


# ============================== main ==============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["labels", "unsup", "sup", "report", "all"])
    ap.add_argument("--langs", nargs="*", default=["en", "ru", "kk"])
    ap.add_argument("--max-docs", type=int, default=None, help="тест: не больше N статей на язык и выборку")
    ap.add_argument("--models", nargs="*", default=["logreg", "svm", "lgbm", "mlp"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = ["labels", "unsup", "sup", "report"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n######## {st} ########")
        if st == "labels":
            stage_labels(args.langs, args.max_docs)
        elif st == "unsup":
            stage_unsup(args.langs, args.max_docs)
        elif st == "sup":
            stage_sup(args.langs, args.max_docs, args.models)
        elif st == "report":
            stage_report(args.langs, args.max_docs)
