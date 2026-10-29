#!/usr/bin/env python3
"""
Добавляет в error_samples.csv столбец evidence: 5 предложений статьи, наиболее похожих на
проверяемое предложение (по совпадению слов), чтобы при разметке не искать вручную в PDF.
Уже заполненные error_type / comment сохраняются.

Запуск:
    python add_evidence.py
-> data/runs/paper/error_samples_evidence.csv
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path.home() / "summary" / "data"
F = DATA / "runs" / "paper" / "error_samples.csv"
TOK = re.compile(r"\w+", re.U)


def toks(s):
    return {t.lower() for t in TOK.findall(s) if len(t) > 2 or t.isdigit()}


docs = {}
for l in open(DATA / "dataset.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["split"] == "test":
        docs[d["id"]] = d["sents"]

df = pd.read_csv(F, encoding="utf-8-sig")
ev = []
for _, r in df.iterrows():
    q = toks(r["sentence"])
    sents = docs.get(r["id"], [])
    scored = sorted(((len(q & toks(s)) / (len(q) or 1), i) for i, s in enumerate(sents)), reverse=True)[:5]
    ev.append("\n".join(f"[{i}] {sents[i]}" for _, i in sorted(scored, key=lambda x: x[1])))
df.insert(df.columns.get_loc("sentence") + 1, "evidence", ev)
out = F.with_name("error_samples_evidence.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"-> {out}")
