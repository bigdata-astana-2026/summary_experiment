#!/usr/bin/env python3
"""
Этап 2. Итоговый датасет и фиксированное разбиение train / val / test.

Объединяет два источника эталонов:
    - аннотации из GROBID (data/corpus.jsonl, поле abstract + разделы-аннотации в тексте);
    - аннотации, извлечённые из PDF на этапе 1 (data/pdf_abstracts.jsonl).
Чистит основной текст от любых аннотаций (на всех языках) и ключевых слов,
применяет фильтры качества с журналом отсева и делает стратифицированное разбиение.

Пишет (в ~/summary/data/):
    dataset.jsonl            — итоговый датасет (одна строка = одна статья)
    splits.csv               — id, язык, выборка (публикуется вместе с кодом)
    filter_stats.csv         — сколько статей осталось после каждого фильтра
    dataset_stats.csv        — статистика по языкам и выборкам (таблица для раздела Dataset)
    extraction_agreement.txt — насколько совпадают аннотации из GROBID и из PDF (проверка извлечения)
    reference_samples.txt    — случайные эталоны для ручной проверки

Запуск (из папки ~/summary, нужен pilot.py рядом):
    python make_dataset.py
"""
import json
import os
import random
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from pilot import (normalize, strip_abstract_sections, split_multilang_abstract, split_sents,
                   tokenize, n_words, word_like_share, ngram_leak, KEYWORDS_RE)

DATA = Path.home() / "summary" / "data"
CORPUS = DATA / "corpus.jsonl"
PDF_ABS = DATA / "pdf_abstracts.jsonl"
SEED = 42

MIN_TEXT_WORDS = 800
MAX_TEXT_WORDS = 20000         # длиннее — сборник, а не статья
MIN_REF_WORDS = 30
MIN_WORD_LIKE = 0.5            # ниже — сломанная кодировка PDF
MIN_COMPRESSION = 3            # текст хотя бы в 3 раза длиннее эталона
MAX_LEAK = 0.2                 # доля 8-грамм эталона, найденных в тексте после очистки
LANG_CONFIDENCE = 0.75         # язык текста берётся вместо папки, если в нём уверены (>= 75% служебных слов)
TEST_MAX, VAL_MAX = 500, 250   # верхние границы размеров выборок на язык

KW_LINE_RE = re.compile(r"^\s*(ключевые\s+слова|түйін(ді)?\s+сөздер|тірек\s+сөздер|key\s*words)\s*[:.\-–—]", re.I)
KW_ANY_RE = re.compile(r"(ключевые\s+слова|түйін(ді)?\s+сөздер|тірек\s+сөздер|key\s*words)\s*[:.\-–—]", re.I)
# служебные разделы журнала: «ARTICLE INFO», «Информация о статье», «Мақала туралы ақпарат»
INFO_HEAD_RE = re.compile(r"^#+\s*(article\s+info|информация\s+о\s+статье|мақала\s+туралы\s+ақпарат)", re.I)
# раздел литературы, который GROBID иногда оставляет в основном тексте
REF_HEAD_RE = re.compile(r"^#+\s*(список\s+(использованн\w+\s+)?литератур\w*|литература|references|bibliography|"
                         r"(пайдаланылған\s+)?әдебиеттер(\s+тізімі)?)\b", re.I)


KK_FUNC = set("және бұл үшін мен деп болып бойынша арқылы осы ол оның сондай сияқты кезінде болады "
              "болса екі бір олар біз еді жатыр туралы қатар немесе емес мұнда онда тиіс ретінде".split())
RU_FUNC = set("и в на что для по это как не с из при к от или также так который которые является "
              "можно было были быть если где этого этом между".split())


def detect_lang(text):
    """Язык очищенного текста.
    en/кириллица — по доле латинских и кириллических букв;
    kk/ru — по служебным словам («және, бұл, үшін…» против «и, в, на, что…»): на них не влияют
    формулы, заимствованные термины и отдельные казахские названия в русском тексте."""
    sample = text[:60000]
    letters = [c for c in sample.lower() if c.isalpha()]
    if len(letters) < 200:
        return None, 0.0
    lat = sum("a" <= c <= "z" for c in letters)
    cyr = sum("\u0400" <= c <= "\u04ff" for c in letters)
    if lat > cyr:                                   # уверенность — доля латиницы среди лат.+кир. букв
        return "en", lat / max(lat + cyr, 1)          # (формулы и греческие буквы на неё не влияют)
    words = re.findall(r"[^\W\d_]+", sample.lower())
    kk = sum(w in KK_FUNC for w in words)
    ru = sum(w in RU_FUNC for w in words)
    if kk + ru == 0:
        return None, 0.0
    return ("kk", kk / (kk + ru)) if kk > ru else ("ru", ru / (kk + ru))


# строки с датами поступления: «Получено 17.05.2026», «Received …», «Қабылданды …»
DATES_RE = re.compile(r"(получено|поступила|принято|received|accepted|revised|түсті|қабылданды)\b[^\n]{0,40}\d{4}", re.I)


def unigram_f1(a, b):
    """Совпадение двух текстов по словам (F1) — для сравнения версий аннотации из GROBID и PDF."""
    ta, tb = Counter(tokenize(a)), Counter(tokenize(b))
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(tb.values()), common / sum(ta.values())
    return 2 * p * r / (p + r)


def ngram_set(text, n=8):
    t = tokenize(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def remove_abstract_paragraphs(text, abstracts):
    """Удаляет из текста абзацы, совпадающие с любой известной аннотацией (на любом языке),
    и строки с ключевыми словами. Нужно, потому что многие аннотации стоят в тексте без метки."""
    abs_ngrams = set()
    for a in abstracts:
        abs_ngrams |= ngram_set(a)
    out, removed, skip = [], 0, False
    for par in text.split("\n"):
        s = par.strip()
        if s.startswith("#"):
            # служебный раздел журнала или список литературы — пропускаем целиком до следующего заголовка
            skip = bool(INFO_HEAD_RE.match(s) or REF_HEAD_RE.match(s))
            removed += skip
            if skip:
                continue
        if skip:
            continue
        if s and not s.startswith("#"):
            short = n_words(s) <= 80
            if KW_LINE_RE.match(s) or (short and (KW_ANY_RE.search(s) or DATES_RE.search(s))):
                removed += 1
                continue
            g = ngram_set(s)
            if g and abs_ngrams and len(g & abs_ngrams) / len(g) >= 0.5:
                removed += 1
                continue
        out.append(par)
    return "\n".join(out), removed


def process(rec):
    """Обработка одной статьи: эталоны по языкам, очистка текста, предложения, признаки для фильтров."""
    lang = rec["lang"]
    text = normalize(rec["text"] or "")
    text, n_sec, harvested = strip_abstract_sections(text)

    # --- эталоны из GROBID (заголовок TEI + разделы-аннотации из текста) ---
    tei_all = "\n".join(x for x in (normalize(rec.get("abstract") or ""), harvested) if x)
    tei = split_multilang_abstract(tei_all) if tei_all else {}

    # --- эталоны из PDF ---
    pdf = rec.get("pdf_abstracts")
    pdf = pdf if isinstance(pdf, dict) else {}       # у 82 статей PDF не найден

    by_lang, sources = {}, {}
    for L in ("en", "ru", "kk"):
        p, t = pdf.get(L), tei.get(L, "")
        p_ok = p is not None and n_words(p["text"]) >= MIN_REF_WORDS and not p.get("truncated_start")
        if p_ok:                                  # PDF-версия чище: без заголовков и переносов GROBID
            by_lang[L], sources[L] = p["text"], "pdf_" + p["method"]
        elif n_words(t) >= MIN_REF_WORDS:
            by_lang[L], sources[L] = t, "grobid"
        elif p is not None and n_words(p["text"]) >= MIN_REF_WORDS:
            by_lang[L], sources[L] = p["text"], "pdf_truncated"

    # согласие GROBID и PDF для языка статьи (проверка качества извлечения)
    agree = None
    if pdf.get(lang) and n_words(tei.get(lang, "")) >= MIN_REF_WORDS:
        agree = unigram_f1(tei[lang], pdf[lang]["text"])

    # --- чистим текст от всех найденных аннотаций и ключевых слов ---
    all_abs = [v for v in by_lang.values()] + [v for v in tei.values()] + [v["text"] for v in pdf.values()]
    text, n_par = remove_abstract_paragraphs(text, all_abs)

    # --- язык статьи определяется её текстом, а не папкой архива ---
    # (в папке «kk» встречаются русские статьи, в папке «ru» — английские)
    folder_lang = lang
    text_lang, text_lang_share = detect_lang(text)
    if text_lang in ("en", "ru", "kk") and text_lang_share >= LANG_CONFIDENCE:
        lang = text_lang

    ref = by_lang.get(lang, "")
    sents = split_sents(text, lang)
    return {
        "id": rec["citing_id"], "lang": lang, "lang_folder": folder_lang,
        "discipline": rec.get("discipline"), "journal": rec.get("journal"),
        "year": rec.get("year"), "title": rec.get("title"), "lang_text": text_lang,
        "lang_text_share": text_lang_share, "lang_text_grobid": rec.get("lang_text"),
        "text": text, "sents": sents, "reference": ref, "ref_source": sources.get(lang),
        "abstracts": by_lang, "abstract_sources": sources,
        "parallel": all(L in by_lang for L in ("en", "ru", "kk")),
        "n_removed_abstract_parts": n_sec + n_par, "grobid_pdf_agreement": agree,
        "text_words": n_words(text), "ref_words": n_words(ref),
        "text_word_like": word_like_share(text[:20000]), "ref_word_like": word_like_share(ref) if ref else 0.0,
        "leak": ngram_leak(ref, text) if ref else 0.0,
        "dup_key": " ".join(tokenize(text)[:300]),
    }


def sizes(n):
    test = min(TEST_MAX, max(round(0.2 * n), min(300, round(0.3 * n))))
    val = min(VAL_MAX, max(100, round(0.1 * n)))
    return test, val


def stratified_split(df):
    from sklearn.model_selection import train_test_split
    parts = []
    for lang, g in df.groupby("lang"):
        test_n, val_n = sizes(len(g))
        strat = g["discipline"].fillna("other")
        strat = strat.where(strat.map(strat.value_counts()) >= 10, "other")
        rest, test = train_test_split(g, test_size=test_n, stratify=strat, random_state=SEED)
        strat_r = strat.loc[rest.index]
        train, val = train_test_split(rest, test_size=val_n, stratify=strat_r, random_state=SEED)
        parts += [test.assign(split="test"), val.assign(split="val"), train.assign(split="train")]
    return pd.concat(parts)


def main():
    corpus = pd.read_json(CORPUS, lines=True)
    corpus["citing_id"] = corpus["citing_id"].astype(str)
    corpus = corpus[corpus["lang_folder"].isin(["en", "ru", "kk"]) & corpus["has_text"].astype(bool)].copy()
    corpus["lang"] = corpus["lang_folder"]
    pdf = {}
    for line in open(PDF_ABS, encoding="utf-8"):
        r = json.loads(line)
        pdf[str(r["citing_id"])] = r["abstracts"]
    corpus["pdf_abstracts"] = corpus["citing_id"].map(pdf)
    print(f"Статей с текстом: {len(corpus)}; с результатом этапа 1: {corpus['pdf_abstracts'].notna().sum()}")

    recs = corpus.to_dict("records")
    workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Обрабатываю статьи, процессов: {workers} (1–3 минуты)")
    with Pool(workers) as pool:
        rows = pool.map(process, recs, chunksize=20)
    df = pd.DataFrame(rows)

    # ---------------- проверка извлечения: GROBID против PDF ----------------
    ag = df["grobid_pdf_agreement"].dropna()
    agree_txt = (f"Статей, где аннотация на языке статьи есть и в GROBID, и в PDF: {len(ag)}\n"
                 f"Совпадение по словам (F1): медиана {ag.median():.3f}, среднее {ag.mean():.3f}\n"
                 f"Доля пар с F1 >= 0.8: {(ag >= 0.8).mean():.1%}; с F1 < 0.5: {(ag < 0.5).mean():.1%}\n"
                 + "\nПо языкам (медиана F1):\n" + df.dropna(subset=["grobid_pdf_agreement"])
                 .groupby("lang")["grobid_pdf_agreement"].median().round(3).to_string())
    (DATA / "extraction_agreement.txt").write_text(agree_txt, encoding="utf-8")
    print("\n" + agree_txt)

    # ---------------- фильтры качества ----------------
    # оставляем, если язык текста совпадает с итоговым языком статьи:
    # уверенно определённый язык уже записан в lang; неуверенный оставляем, только если он совпал с папкой
    lang_ok = df["lang_text"].notna() & (df["lang_text"] == df["lang"])
    print("\nЯзык статьи: папка (строки) -> язык текста (столбцы):\n"
          + pd.crosstab(df["lang_folder"], df["lang_text"].fillna("?")).to_string())
    moved = df[df["lang"] != df["lang_folder"]]
    print(f"Статей, которым язык переопределён по тексту: {len(moved)}")
    steps = [
        (f"текст >= {MIN_TEXT_WORDS} слов", df["text_words"] >= MIN_TEXT_WORDS),
        (f"текст <= {MAX_TEXT_WORDS} слов (не сборник)", df["text_words"] <= MAX_TEXT_WORDS),
        (f"эталон на языке статьи >= {MIN_REF_WORDS} слов", df["ref_words"] >= MIN_REF_WORDS),
        ("эталон не обрезан в начале", df["ref_source"] != "pdf_truncated"),
        ("текст без сбоев кодировки", df["text_word_like"] >= MIN_WORD_LIKE),
        ("эталон без сбоев кодировки", df["ref_word_like"] >= MIN_WORD_LIKE),
        ("язык текста определён и согласован", lang_ok),
        (f"текст >= {MIN_COMPRESSION}x длиннее эталона", df["text_words"] >= MIN_COMPRESSION * df["ref_words"]),
        ("не дубликат", ~df["dup_key"].duplicated(keep=False)),
        (f"эталон не скопирован в текст (утечка <= {MAX_LEAK})", df["leak"] <= MAX_LEAK),
    ]
    mask = pd.Series(True, index=df.index)
    table = [("исходно (есть текст), язык по тексту", df["lang"].value_counts())]
    for name, cond in steps:
        mask &= cond
        table.append((name, df.loc[mask, "lang"].value_counts()))
    fstats = pd.DataFrame({k: v for k, v in table}).T.reindex(columns=["en", "ru", "kk"]).fillna(0).astype(int)
    fstats.index.name = "шаг фильтрации"
    fstats.to_csv(DATA / "filter_stats.csv", encoding="utf-8-sig")
    print("\nСколько статей остаётся после каждого фильтра:\n" + fstats.to_string())

    clean = df[mask].copy()

    # ---------------- разбиение ----------------
    clean = stratified_split(clean)
    clean["fresh_2025plus"] = pd.to_numeric(clean["year"], errors="coerce") >= 2025
    clean[["id", "lang", "lang_folder", "discipline", "journal", "year", "split", "parallel", "ref_source"]] \
        .sort_values(["lang", "split", "id"]).to_csv(DATA / "splits.csv", index=False, encoding="utf-8-sig")

    cols = ["id", "lang", "lang_folder", "split", "discipline", "journal", "year", "title", "text", "sents", "reference",
            "ref_source", "abstracts", "abstract_sources", "parallel", "fresh_2025plus", "text_words", "ref_words"]
    clean[cols].to_json(DATA / "dataset.jsonl", orient="records", lines=True, force_ascii=False)

    # ---------------- статистика для раздела Dataset ----------------
    clean["compression"] = clean["text_words"] / clean["ref_words"]
    clean["n_sents"] = clean["sents"].map(len)
    st = clean.groupby(["lang", "split"]).agg(
        статей=("id", "size"), слов_в_тексте_медиана=("text_words", "median"),
        слов_в_эталоне_медиана=("ref_words", "median"), предложений_медиана=("n_sents", "median"),
        сжатие_медиана=("compression", "median"), параллельных=("parallel", "sum")).round(1)
    st.to_csv(DATA / "dataset_stats.csv", encoding="utf-8-sig")
    print("\nИтоговый датасет по языкам и выборкам:\n" + st.to_string())
    print("\nИсточник эталона:\n" + pd.crosstab(clean["lang"], clean["ref_source"]).to_string())
    print("\nДисциплины в тесте:\n" + pd.crosstab(clean.loc[clean.split == "test", "lang"],
                                                  clean.loc[clean.split == "test", "discipline"]).to_string())

    # ---------------- образцы для ручной проверки ----------------
    rnd = random.Random(SEED)
    lines = []
    for (lang, src), g in clean.groupby(["lang", "ref_source"]):
        for _, x in g.sample(min(2, len(g)), random_state=SEED).iterrows():
            lines.append(f"=== {x.id} | {lang} | источник: {src} | {x.journal}\nЭТАЛОН: {x.reference[:500]}\n"
                         f"НАЧАЛО ТЕКСТА: {x.text[:300]}\n")
    (DATA / "reference_samples.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nДатасет -> {DATA / 'dataset.jsonl'}\nРазбиение -> {DATA / 'splits.csv'}\n"
          f"Образцы для проверки -> {DATA / 'reference_samples.txt'}")


if __name__ == "__main__":
    main()
