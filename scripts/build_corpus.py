#!/usr/bin/env python3
"""
Сборка корпуса для суммаризации из TEI-файлов GROBID.

Читает (ничего не меняет в проекте citeverify):
    TEI_DIR/<citing_id>.tei.xml
    INVENTORY (data/inventory.csv): метаданные PDF
Пишет:
    OUT_PATH (~/summary/data/corpus.jsonl): одна строка JSON = одна статья

Поля: citing_id, organization, journal, discipline, year, lang_folder, lang_text,
      title, abstract, text, n_words, n_refs, has_text

Запуск:
    python build_corpus.py            # весь корпус
    python build_corpus.py --limit 50 # быстрый тест на 50 файлах
"""
import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

# ============================ НАСТРОЙКИ ============================
SRC = Path.home() / "verify" / "citeverify"
TEI_DIR = SRC / "data" / "tei"
INVENTORY = SRC / "data" / "inventory.csv"
OUT_PATH = Path.home() / "summary" / "data" / "corpus.jsonl"
MIN_WORDS_HAS_TEXT = 300
# ===================================================================

NS = "{http://www.tei-c.org/ns/1.0}"
SKIP_TAGS = {NS + "formula", NS + "figure", NS + "note", NS + "table"}
WORD_RE = re.compile(r"\w+", re.UNICODE)
KK_LETTERS = set("әғқңөұүһі")


def clean(s):
    s = unicodedata.normalize("NFC", s or "").replace("­", "")
    return re.sub(r"\s+", " ", s).strip()


def text_of(el):
    """Текст элемента без формул, рисунков и сносок (их «хвостовой» текст сохраняется)."""
    parts = [el.text or ""]
    for ch in el:
        if ch.tag not in SKIP_TAGS:
            parts.append(text_of(ch))
        parts.append(ch.tail or "")
    return "".join(parts)


def div_to_lines(div):
    lines = []
    for ch in div:
        tag = ch.tag
        if tag == NS + "head":
            h = clean(text_of(ch))
            n = ch.get("n")
            if h:
                lines.append("## " + (f"{n} " if n else "") + h)
        elif tag == NS + "p":
            t = clean(text_of(ch))
            if t:
                lines.append(t)
        elif tag == NS + "list":
            for item in ch.iter(NS + "item"):
                t = clean(text_of(item))
                if t:
                    lines.append(t)
        elif tag == NS + "div":
            lines += div_to_lines(ch)
    return lines


def guess_lang(text):
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return None
    n = len(letters)
    cyr = sum("Ѐ" <= c <= "ӿ" for c in letters) / n
    lat = sum("a" <= c <= "z" for c in letters) / n
    if lat > cyr:
        return "en"
    return "kk" if sum(c in KK_LETTERS for c in letters) / n >= 0.01 else "ru"


def parse_tei(path):
    cid = path.name.replace(".tei.xml", "")
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return {"citing_id": cid, "parse_error": str(e)[:200]}

    title_el = root.find(f".//{NS}titleStmt/{NS}title")
    title = clean(text_of(title_el)) if title_el is not None else ""

    abs_lines = []
    for ab in root.findall(f".//{NS}profileDesc/{NS}abstract"):
        abs_lines += div_to_lines(ab)
        for p in ab.findall(f"{NS}p"):            # абзацы прямо внутри <abstract>
            t = clean(text_of(p))
            if t and t not in abs_lines:
                abs_lines.append(t)
    abstract = "\n".join(l for l in abs_lines if not l.startswith("## "))

    body = root.find(f".//{NS}text/{NS}body")
    lines = div_to_lines(body) if body is not None else []
    text = "\n".join(lines)
    n_words = len(WORD_RE.findall(text))
    n_refs = len(root.findall(f".//{NS}back//{NS}listBibl/{NS}biblStruct"))

    return {"citing_id": cid, "title": title, "abstract": abstract, "text": text,
            "n_words": n_words, "n_refs": n_refs, "has_text": n_words >= MIN_WORDS_HAS_TEXT,
            "lang_text": guess_lang(text[:20000])}


def pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(TEI_DIR.glob("*.tei.xml"))
    if args.limit:
        files = files[:args.limit]
    print(f"TEI-файлов: {len(files)}")
    with Pool(args.workers) as pool:
        recs = pool.map(parse_tei, files, chunksize=50)
    df = pd.DataFrame(recs)
    if "parse_error" in df.columns:
        bad = df["parse_error"].notna()
        print(f"Не разобрано: {bad.sum()}")
        df = df[~bad].drop(columns="parse_error")

    inv = pd.read_csv(INVENTORY)
    print(f"Колонки inventory.csv: {list(inv.columns)}")
    id_col = pick_col(inv.columns, ["citing_id", "id", "paper_id"])
    if id_col is None:
        raise SystemExit("В inventory.csv не найдена колонка с id статьи (citing_id)")
    inv = inv.rename(columns={id_col: "citing_id"})
    inv["citing_id"] = inv["citing_id"].astype(str)

    meta = pd.DataFrame({"citing_id": inv["citing_id"]})
    for out_name, cands in {
        "organization": ["organization", "org", "организация"],
        "discipline": ["discipline", "дисциплина"],
        "year": ["year", "год"],
        "journal": ["journal", "журнал"],
        "lang_folder": ["lang_folder", "lang", "language", "язык"],
    }.items():
        c = pick_col(inv.columns, cands)
        meta[out_name] = inv[c] if c else None

    # Если языка нет отдельной колонкой — берём из пути: организация/дисциплина/год/журнал/язык/файл.pdf
    if meta["lang_folder"].isna().all():
        path_col = pick_col(inv.columns, ["path", "pdf_path", "file", "filepath"])
        if path_col:
            meta["lang_folder"] = inv[path_col].astype(str).map(lambda p: Path(p).parent.name)
    meta["lang_folder"] = meta["lang_folder"].astype(str).str.strip().str.lower()

    df = meta.drop_duplicates("citing_id").merge(df, on="citing_id", how="inner")
    cols = ["citing_id", "organization", "journal", "discipline", "year", "lang_folder", "lang_text",
            "title", "abstract", "text", "n_words", "n_refs", "has_text"]
    df = df[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)

    print(f"\nСохранено {len(df)} статей -> {OUT_PATH}")
    print("\nСтатей по языку (папка) и has_text:")
    print(pd.crosstab(df["lang_folder"], df["has_text"]).to_string())
    print("\nСтатей без аннотации в TEI:", (df["abstract"].str.len() == 0).sum())
    mism = (df["lang_text"].notna() & (df["lang_text"] != df["lang_folder"])).sum()
    print("Язык по тексту не совпадает с папкой:", mism)
    print("\nСредняя длина текста (слов) по языкам:")
    print(df[df["has_text"]].groupby("lang_folder")["n_words"].median().to_string())


if __name__ == "__main__":
    main()
