#!/usr/bin/env python3
"""
Проверка калибровки NLI (≈1–2 минуты на GPU).

У резюме Oracle все предложения взяты из статьи дословно, поэтому NLI должен
почти всегда отвечать «следует» (entailment). Сейчас выходит 0,58 — значит, оценка
верности ненадёжна. Скрипт ищет причину, сравнивая на 300 предложениях Oracle на каждый язык:
  A) посылка = то же самое предложение (тождество), fp16
  B) тождество, fp32
  C) тождество, bf16
  D) посылка как в evaluate.py (3 похожих предложения), fp32
Если A низко, а B высоко — виновата половинная точность (у DeBERTa-v3 известная проблема с fp16).
Если низко даже B — модель плохо работает на научных текстах / этом языке.
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

from pilot import split_sents
from evaluate import NLI_MODEL, finalists, load_summaries, load_docs

N = 300
dev = "cuda"
rng = np.random.default_rng(0)
docs = load_docs()
df = load_summaries(finalists())
df = df[df["method"] == "oracle"]
tok = AutoTokenizer.from_pretrained(NLI_MODEL)
enc = SentenceTransformer("sentence-transformers/LaBSE", device=dev)

samples = {}
for lang, g in df.groupby("lang"):
    rows = []
    for _, r in g.iterrows():
        d = docs[r["id"]]
        for s in split_sents(r["summary"] or "", lang)[:20]:
            rows.append((r["id"], s))
    idx = rng.choice(len(rows), min(N, len(rows)), replace=False)
    samples[lang] = [rows[i] for i in idx]


def premise_top3(doc_id, s):
    d = docs[doc_id]
    dv = enc.encode(d["sents"], normalize_embeddings=True, convert_to_numpy=True)
    sv = enc.encode([s], normalize_embeddings=True, convert_to_numpy=True)
    top = np.argsort(-(sv @ dv.T)[0])[:3]
    return " ".join(d["sents"][t] for t in sorted(top))


def run(dtype, pairs):
    m = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=dtype).to(dev).eval()
    ent_id = {v.lower(): k for k, v in m.config.id2label.items()}["entailment"]
    out = []
    with torch.no_grad():
        for b in range(0, len(pairs), 32):
            p, h = zip(*pairs[b:b + 32])
            x = tok(list(p), list(h), truncation="longest_first", max_length=512,
                    padding=True, return_tensors="pt").to(dev)
            pr = torch.softmax(m(**x).logits.float(), -1)[:, ent_id].cpu().numpy()
            out.extend(pr)
    out = np.array(out)
    del m
    torch.cuda.empty_cache()
    return f"supported={np.mean(out > 0.5):.3f}  mean_p={np.nanmean(out):.3f}  NaN={np.isnan(out).sum()}"


print("id2label:", AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).config.id2label)
for lang, smp in samples.items():
    ident = [(s, s) for _, s in smp]
    top3 = [(premise_top3(i, s), s) for i, s in smp]
    print(f"\n=== {lang}  ({len(smp)} предложений Oracle) ===")
    print("A тождество fp16 :", run(torch.float16, ident))
    print("B тождество fp32 :", run(torch.float32, ident))
    print("C тождество bf16 :", run(torch.bfloat16, ident))
    print("D top-3     fp32 :", run(torch.float32, top3))
    print("пример:", smp[0][1][:200])
