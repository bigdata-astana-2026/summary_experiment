#!/usr/bin/env python3
"""
Этап 1. Извлечение аннотаций на всех языках прямо из PDF.

Зачем: GROBID обычно сохраняет одну (чаще английскую) аннотацию, а русские и казахские
аннотации теряются. В казахстанских журналах они есть в самом PDF: в начале статьи
или в конце, после списка литературы.

Читает (ничего не меняет):
    ~/summary/data/corpus.jsonl                       — корпус (citing_id, lang_folder)
    ~/verify/citeverify/data/inventory.csv            — пути к PDF
    ~/KZ_Journals/...                                 — сами PDF
Пишет:
    ~/summary/data/pdf_abstracts.jsonl                — найденные аннотации по языкам
    ~/summary/data/pdf_abstracts_stats.txt            — сводка

Запуск:
    python extract_pdf_abstracts.py --limit 30     # тест на 30 PDF
    python extract_pdf_abstracts.py                # все PDF (несколько минут)
"""
import argparse
import os
import json
import re
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HOME = Path.home()
CORPUS = HOME / "summary" / "data" / "corpus.jsonl"
INVENTORY = HOME / "verify" / "citeverify" / "data" / "inventory.csv"
PDF_ROOT = HOME / "KZ_Journals"
OUT = HOME / "summary" / "data" / "pdf_abstracts.jsonl"
STATS = HOME / "summary" / "data" / "pdf_abstracts_stats.txt"

MIN_WORDS, MAX_WORDS = 10, 600
WORD_RE = re.compile(r"\w+", re.UNICODE)
KK_LETTERS = set("әғқңөұүһі")

# Метка аннотации в начале строки: «Аңдатпа.», «Аннотация:», «Abstract —», «Түйіндеме», «Резюме», «Summary»
# В части журналов в слове «Аннотация» есть латинские буквы-двойники (A, o, a, e, c, p…),
# поэтому каждую букву метки ищем и в кириллическом, и в латинском написании.
_TWINS = {"а": "аa", "о": "оo", "е": "еe", "с": "сc", "р": "рp", "х": "хx", "у": "уy", "к": "кk", "т": "тt",
          "н": "нh", "м": "мm", "в": "вb", "і": "іi"}
_TWINS.update({lat: cyr + lat for cyr, pair in list(_TWINS.items()) for lat in pair[1:]})


def _twin_regex(word):
    return "".join(f"[{_TWINS[c]}]" if c in _TWINS else re.escape(c) for c in word)


_LABEL_WORDS = ["аңдатпа", "андатпа", "аннотация", "түйіндеме", "резюме", "abstract", "annotation", "summary"]
_LABEL_ALT = "|".join(_twin_regex(w) for w in _LABEL_WORDS)
# метка в начале строки; после неё знак препинания или заглавная буква («Abstract This…», «AbstractHippotherapy»),
# но не строчная («abstract algebra» — это обычный текст)
LABEL_RE = re.compile(
    r"(?im)^[ \t]*(" + _LABEL_ALT + r")(?:[ \t]*[.:\-–—][ \t]*|\s*(?=(?-i:[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ])))")
# та же метка в любом месте — чтобы отрезать заголовок и авторов, оказавшихся перед ней
LABEL_ANY_RE = re.compile(
    r"(?i)(?<![^\W\d_])(" + _LABEL_ALT + r")(?:[ \t]*[.:\-–—][ \t]*|\s*(?=(?-i:[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ])))")


def clean_segment(seg):
    """Убирает метку и всё, что стоит перед ней в первой части текста (авторы, номера сносок)."""
    m = LABEL_ANY_RE.search(seg)
    if m and m.start() < 0.4 * len(seg):
        seg = seg[m.end():]
    return seg.lstrip(" .:-–—").strip()
# Где аннотация заканчивается
STOP_RE = re.compile(
    r"(?im)(ключевые\s+слова|түйін(ді)?\s+сөздер|тірек\s+сөздер|key\s*words|^[ \t]*(введение|кіріспе|introduction)\b"
    r"|^[ \t]*(список\s+литературы|литература|әдебиеттер(\s+тізімі)?|references|пайдаланылған\s+әдебиеттер)\b)")


def normalize(text):
    text = unicodedata.normalize("NFC", text or "").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # перенос слова в конце строки
    text = re.sub(r"[ \t ]+", " ", text)
    return text


def lang_of(s):
    letters = [c for c in s.lower() if c.isalpha()]
    if not letters:
        return None
    n = len(letters)
    cyr = sum("Ѐ" <= c <= "ӿ" for c in letters) / n
    lat = sum("a" <= c <= "z" for c in letters) / n
    if max(cyr, lat) < 0.7:
        return None                                        # формулы, мусор, смешанный текст
    if lat > cyr:
        return "en"
    return "kk" if sum(c in KK_LETTERS for c in letters) / n >= 0.01 else "ru"


def word_like_share(text):
    toks = text.split()
    if not toks:
        return 0.0
    good = sum(bool(re.match(r"^[^\W\d_]{2,}$", t.strip(".,;:!?()[]«»\"'").replace("-", ""))) for t in toks)
    return good / len(toks)


KW_RE = re.compile(r"(?i)(ключевые\s+слова|түйін(ді)?\s+сөздер|тірек\s+сөздер|key\s*words|index\s+terms)\s*[:.\-–—]")
_UP = "A-ZА-ЯЁӘҒҚҢӨҰҮҺІ"
_LOW = "a-zа-яёәғқңөұүһі"
INITIALS_RE = re.compile(r"\b[" + _UP + r"]\.\s?(?:[" + _UP + r"]\.\s?)?[" + _UP + "][" + _LOW + "]+")
AFFIL_RE = re.compile(r"(университет|university|institute|институт|академия|academy|кафедра|department|"
                      r"laboratory|лаборатория|kazakhstan|казахстан|қазақстан|e-?mail|@|orcid|doi|удк|ӘОЖ|ҒТАХР|мрнти)", re.I)


def upper_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def is_front_matter(block):
    """Заголовок капсом, авторы, аффилиации, e-mail, УДК/DOI — не аннотация."""
    words = len(WORD_RE.findall(block))
    return (upper_share(block) > 0.6
            or len(INITIALS_RE.findall(block)) >= 2
            or (bool(AFFIL_RE.search(block)) and words <= 40)
            or words < 8)


def abstract_before_keywords(blocks, i, kw_start):
    """Абзац(ы) непосредственно перед «Ключевые слова:» в блоке i (без метки «Аннотация»)."""
    parts = []
    head = blocks[i][:kw_start].strip()
    if head and not is_front_matter(head):
        parts.append(head)
    j = i - 1
    while j >= 0 and sum(len(p.split()) for p in parts) < MAX_WORDS:
        b = blocks[j].strip()
        if not b or is_front_matter(b) or KW_RE.search(b) or LABEL_RE.search(b):
            break
        parts.insert(0, b)
        j -= 1
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def extract_from_pdf(task):
    cid, pdf_path = task
    rec = {"citing_id": cid, "pdf_found": False, "error": None, "abstracts": {}, "n_candidates": 0}
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # старые версии PyMuPDF
        doc = fitz.open(pdf_path)
        blocks = [normalize(b[4]) for page in doc for b in page.get_text("blocks") if b[6] == 0]
        doc.close()
        text = "\n".join(blocks)
        rec["pdf_found"] = True
    except Exception as e:
        rec["error"] = str(e)[:200]
        return rec

    labels = list(LABEL_RE.finditer(text))
    rec["n_candidates"] = len(labels)
    found = {}
    for i, m in enumerate(labels):
        start = m.end()
        end = labels[i + 1].start() if i + 1 < len(labels) else len(text)
        stop = STOP_RE.search(text, start, end)
        if stop:
            end = stop.start()
        seg = clean_segment(re.sub(r"\s+", " ", text[start:end]).strip())
        words = seg.split()
        if len(words) > MAX_WORDS:
            seg = " ".join(words[:MAX_WORDS])
        n = len(WORD_RE.findall(seg))
        if n < MIN_WORDS or word_like_share(seg) < 0.5:
            continue
        lang = lang_of(seg)
        if lang is None:
            continue
        # если на одном языке найдено несколько — оставляем самую длинную
        if lang not in found or n > len(WORD_RE.findall(found[lang]["text"])):
            found[lang] = {"text": seg, "label": m.group(1), "pos": round(start / max(len(text), 1), 3)}
    for f in found.values():
        f["method"] = "label"

    # Запасной способ: аннотация без метки — абзац прямо перед «Ключевые слова / Түйін сөздер / Keywords»
    n_kw = 0
    for i, b in enumerate(blocks):
        for m in KW_RE.finditer(b):
            n_kw += 1
            seg = clean_segment(abstract_before_keywords(blocks, i, m.start()))
            words = seg.split()
            if len(words) > MAX_WORDS:
                seg = " ".join(words[:MAX_WORDS])
            n = len(WORD_RE.findall(seg))
            if n < MIN_WORDS or word_like_share(seg) < 0.5:
                continue
            lang = lang_of(seg)
            if lang is None or lang in found:          # найденное по метке надёжнее, его не заменяем
                continue
            found[lang] = {"text": seg, "label": "", "method": "before_keywords",
                           "pos": round(sum(len(x) for x in blocks[:i]) / max(len(text), 1), 3)}
    rec["n_keywords"] = n_kw
    for f in found.values():
        # аннотация с маленькой буквы — скорее всего, потеряно начало (перенос между колонками/страницами)
        f["truncated_start"] = bool(f["text"]) and f["text"][0].islower()
    rec["abstracts"] = found
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    # PDF разбираются на CPU: по умолчанию все ядра, кроме двух (оставляем системе и другим задачам)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    corpus = pd.read_json(CORPUS, lines=True)[["citing_id", "lang_folder", "has_text"]]
    corpus["citing_id"] = corpus["citing_id"].astype(str)
    inv = pd.read_csv(INVENTORY)
    inv["citing_id"] = inv["citing_id"].astype(str)
    if "needs_ocr" in inv.columns:
        n_ocr = inv["needs_ocr"].fillna(False).astype(bool).sum()
        print(f"PDF, помеченных как сканы (needs_ocr): {n_ocr} — текста в них нет, будут пропущены")
        inv = inv[~inv["needs_ocr"].fillna(False).astype(bool)]

    def resolve(row):
        for col in ("path", "rel_path"):
            if col in row and isinstance(row[col], str):
                p = Path(row[col])
                for cand in (p, PDF_ROOT / p, PDF_ROOT / p.name):
                    if cand.exists():
                        return str(cand)
        return None

    inv["pdf"] = inv.apply(resolve, axis=1)
    df = corpus.merge(inv[["citing_id", "pdf"]], on="citing_id", how="left")
    df = df[df["lang_folder"].isin(["en", "ru", "kk"]) & df["has_text"].astype(bool)]
    missing = df["pdf"].isna().sum()
    print(f"Статей: {len(df)}; PDF не найден по пути: {missing}")
    if missing == len(df):
        raise SystemExit("Ни один PDF не найден. Пришлите: head -3 ~/verify/citeverify/data/inventory.csv")
    df = df[df["pdf"].notna()]
    if args.limit:
        df = df.groupby("lang_folder").head(args.limit // 3 + 1)   # поровну на каждый язык

    tasks = list(zip(df["citing_id"], df["pdf"]))
    print(f"Обрабатываю PDF: {len(tasks)}, процессов: {args.workers}")
    with Pool(args.workers) as pool:
        recs = pool.map(extract_from_pdf, tasks, chunksize=20)

    lang_by_id = dict(zip(df["citing_id"], df["lang_folder"]))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            r["lang_folder"] = lang_by_id[r["citing_id"]]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    res = pd.DataFrame(recs)
    res["lang_folder"] = res["citing_id"].map(lang_by_id)
    res["own"] = [lang in a for lang, a in zip(res["lang_folder"], res["abstracts"])]
    res["langs"] = res["abstracts"].map(lambda a: "+".join(sorted(a)) or "-")
    lines = [f"Обработано PDF: {len(res)}; ошибок чтения: {res['error'].notna().sum()}",
             "", "Найдена аннотация на языке статьи:"]
    for lang in ("en", "ru", "kk"):
        m = res["lang_folder"] == lang
        if m.sum():
            lines.append(f"  {lang}: {res.loc[m, 'own'].sum()} из {m.sum()} ({res.loc[m, 'own'].mean():.0%})")
    lines += ["", "Какие языки аннотаций найдены в PDF:", res["langs"].value_counts().head(10).to_string()]
    meth = pd.Series([v["method"] for a in res["abstracts"] for v in a.values()]).value_counts()
    lines += ["", "Способ извлечения (все найденные аннотации):", meth.to_string()]
    trunc = sum(v["truncated_start"] for a in res["abstracts"] for v in a.values())
    lines.append(f"Из них с обрезанным началом (начинаются с маленькой буквы): {trunc}")
    tri = res["abstracts"].map(lambda a: {"en", "ru", "kk"} <= set(a)).sum()
    lines.append(f"\nСтатей с аннотациями на всех трёх языках: {tri}")
    report = "\n".join(lines)
    STATS.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nРезультат -> {OUT}\nСводка -> {STATS}")


if __name__ == "__main__":
    main()
