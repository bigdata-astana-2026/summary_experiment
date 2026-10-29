#!/usr/bin/env python3
"""
Пилот: суммаризация научных статей EN/RU/KZ на 10 статьях (4 EN, 3 RU, 3 KK).

Стадии запускаются по отдельности, чтобы не держать GPU занятым дольше нужного
(параллельно идёт другой эксперимент):

    python pilot.py prepare   # выбор 10 статей + проверки данных (CPU)
    python pilot.py extract   # Lead-k, Oracle, TextRank/TF-IDF, BGE-M3, гибрид (GPU немного)
    python pilot.py llm       # LLM zero-shot через Ollama: полный текст и гибридная выжимка
    python pilot.py llm --models gemma4:26b   # только одна модель
    python pilot.py eval      # ROUGE, chrF++, BERTScore, проверка языка, таблицы (GPU немного)

Все результаты сохраняются в папку CFG["out_dir"].
"""
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ НАСТРОЙКИ ============================
# Поменяйте под свой датасет: путь и названия колонок.
CFG = {
    "data_path": str(Path.home() / "summary" / "data" / "corpus.jsonl"),   # собирается build_corpus.py
    "col_id": "citing_id",
    "col_lang": "lang_folder",          # язык по папке (надёжнее, чем lang_text)
    "col_lang_text": "lang_text",       # язык по тексту — только для проверки расхождений
    "col_text": "text",                 # основной текст без списка литературы
    "col_abstract": "abstract",         # аннотация (часто сразу на трёх языках)
    "meta_cols": ["journal", "discipline", "year", "organization", "title"],
    "only_has_text": True,              # брать только статьи с has_text = true
    # Аннотация на трёх языках в одном поле: оставить только часть на языке статьи
    "split_multilang_abstract": True,
    # Удалять из текста разделы с заголовками «## Аннотация / Түйіндеме / Abstract / Keywords ...»
    "strip_abstract_sections": True,
    # Если аннотации лежат в разных колонках (abstract_en / abstract_ru / abstract_kk),
    # укажите их здесь: тогда эталоном берётся колонка языка статьи.
    "abstract_cols_by_lang": None,      # например {"en": "abstract_en", "ru": "abstract_ru", "kk": "abstract_kk"}
    "lang_map": {
        "en": "en", "eng": "en", "english": "en", "англ": "en",
        "ru": "ru", "rus": "ru", "russian": "ru", "рус": "ru",
        "kk": "kk", "kz": "kk", "kaz": "kk", "kazakh": "kk", "қаз": "kk", "каз": "kk",
    },
    "n_per_lang": {"en": 4, "ru": 3, "kk": 3},
    "seed": 42,
    "min_text_words": 800,
    "min_abs_words": 30,
    "max_text_words": 20000,            # длиннее — это сборник или книга, а не статья
    "min_word_like_share": 0.5,         # ниже — сломанная кодировка PDF
    "min_compression": 3,               # текст должен быть хотя бы в 3 раза длиннее аннотации
    "max_abstract_leak": 0.2,           # доля 8-грамм аннотации, найденных в тексте
    "out_dir": "pilot_out",

    # Экстрактивная часть
    "dense_model": "BAAI/bge-m3",
    "hybrid_w_sparse": 0.5,             # вес TF-IDF в гибриде (остальное — BGE-M3)
    "extract_for_llm_sents": 25,        # сколько предложений подавать LLM в гибриде

    # LLM через Ollama (уже запущен на сервере, те же модели, что в config.yaml проекта citeverify)
    # Адрес Ollama: берётся из OLLAMA_HOST (например 127.0.0.1:11435), иначе стандартный порт
    "ollama_url": ("http://" + os.environ["OLLAMA_HOST"].replace("http://", "")
                   if os.environ.get("OLLAMA_HOST") else "http://localhost:11434"),
    "llm_models": ["qwen3.6:35b", "gemma4:26b", "gpt-oss:20b"],
    "llm_num_ctx": 32768,               # окно контекста; по умолчанию Ollama берёт мало и молча обрезает вход
    # Ограничение входа в словах (у казахского больше токенов на слово, поэтому лимит ниже)
    "llm_max_input_words": {"en": 12000, "ru": 9000, "kk": 7000},
    # Режим «размышления»: выключен; gpt-oss полностью не отключается, ставим минимальный
    "llm_think": {"gpt-oss:20b": "low"},
    "llm_timeout_s": 1800,

    # Оценка
    "bertscore_model": "xlm-roberta-large",
}
# ===================================================================

OUT = Path(CFG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
F_DATA = OUT / "pilot_data.jsonl"
F_CHECKS = OUT / "data_checks.csv"
F_SUMM = OUT / "summaries.jsonl"
F_LLM_IN = OUT / "llm_extracted_inputs.jsonl"
F_METRICS = OUT / "metrics_per_doc.csv"
F_TABLE = OUT / "metrics_lang_x_method.csv"
F_REVIEW = OUT / "review_sheet.csv"

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ----------------------------- утилиты -----------------------------
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
KK_LETTERS = set("әғқңөұүһі")


def tokenize(text):
    """Токенизатор для ROUGE: работает с кириллицей (стандартный в rouge-score её выкидывает)."""
    return TOKEN_RE.findall(text.lower())


class UnicodeTokenizer:
    def tokenize(self, text):
        return tokenize(text)


def n_words(text):
    return len(tokenize(text))


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")          # мягкий перенос
    # GROBID склеивает предложения: «mathematics.GFT», «древу».Так» -> вставляем пробел после точки
    text = re.sub(r"([.!?;])(?=[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ«\"(])", r"\1 ", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_any(path):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p)
    if suf == ".jsonl":
        return pd.read_json(p, lines=True)
    if suf == ".json":
        return pd.read_json(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf == ".parquet":
        return pd.read_parquet(p)
    sys.exit(f"Неизвестный формат файла: {suf}")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_summaries(rows):
    """Добавляет резюме, заменяя старые записи того же (id, method)."""
    old = read_jsonl(F_SUMM) if F_SUMM.exists() else []
    new_keys = {(r["id"], r["method"]) for r in rows}
    kept = [r for r in old if (r["id"], r["method"]) not in new_keys]
    write_jsonl(F_SUMM, kept + rows)


# ------------------------ разбиение на предложения ------------------------
try:
    from razdel import sentenize as _razdel_sentenize
except ImportError:
    _razdel_sentenize = None

_SENT_FALLBACK = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ0-9«\"(\[])")


def _split_paragraph(par, lang):
    sents = None
    if lang in ("ru", "kk") and _razdel_sentenize is not None:
        sents = [s.text for s in _razdel_sentenize(par)]
    elif lang == "en":
        try:
            import nltk
            sents = nltk.sent_tokenize(par, language="english")
        except Exception:
            sents = None
    if sents is None:
        sents = _SENT_FALLBACK.split(par)
    return sents


def alpha_ratio(s):
    """Доля букв среди непробельных символов: низкая у плохо распознанных формул."""
    chars = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def split_sents(text, lang, min_alpha=0.6):
    """Абзацы через \\n, строки-заголовки «## ...» пропускаются, мусор от формул отфильтровывается."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s+", " ", line)
        for s in _split_paragraph(line, lang):
            s = s.strip()
            if n_words(s) >= 4 and alpha_ratio(s) >= min_alpha:
                out.append(s)
    return out


# ---------------------- аннотации на нескольких языках ----------------------
ABS_SECTION_RE = re.compile(
    r"^#+\s*(аннотация|аңдатпа|андатпа|түйіндеме|түйін|резюме|abstract|summary|resume|"
    r"ключевые слова|түйін сөздер|түйінді сөздер|тірек сөздер|keywords|key words)\b", re.I)
ABS_LABEL_RE = re.compile(
    r"^\s*(аннотация|аңдатпа|андатпа|түйіндеме|резюме|abstract|summary)\s*[.:\-–—]?\s*", re.I)
KEYWORDS_RE = re.compile(
    r"(ключевые слова|түйін сөздер|түйінді сөздер|тірек сөздер|keywords|key words)\s*[:.\-–—].*", re.I)


ABS_INLINE_RE = re.compile(
    r"^\s*(аннотация|аңдатпа|андатпа|түйіндеме|резюме|abstract|summary)\s*[.:\-–—]\s+\S", re.I)
KW_HEAD_RE = re.compile(r"слова|сөздер|keywords|key words", re.I)


def strip_abstract_sections(text):
    """Вырезает из текста аннотации, попавшие в основной текст, и сохраняет их.
    Ловит разделы с заголовком «## Аннотация / Түйіндеме / Abstract ...» и абзацы,
    начинающиеся с «Аңдатпа. / Аннотация: / Abstract. ...».
    Возвращает (текст, число вырезанных фрагментов, найденные аннотации)."""
    out, harvested, removed = [], [], 0
    skip, harvest = False, False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("#"):
            skip = bool(ABS_SECTION_RE.match(s))
            if skip:
                removed += 1
                harvest = not KW_HEAD_RE.search(s)
                continue
        if skip:
            if harvest and s:
                harvested.append(s)
            continue
        if ABS_INLINE_RE.match(s) and n_words(s) >= 20:
            harvested.append(s)
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), removed, "\n".join(harvested)


def segment_lang(s):
    cyr, lat, kk = script_stats(s)
    if lat > cyr:
        return "en"
    return "kk" if kk >= 0.01 else "ru"


_UP = "A-ZА-ЯЁӘҒҚҢӨҰҮҺІ"
_LOW = "a-zа-яёәғқңөұүһі"
LABEL_ANY_RE = re.compile(
    r"\b(аннотация|аңдатпа|андатпа|түйіндеме|резюме|abstract|summary)\b\s*[.:\-–—]?\s*(?=[" + _UP + "])", re.I)
AFFIL_RE = re.compile(r"(университет|university|кафедра|department of|институт|institute)", re.I)
EMAIL_RE = re.compile(r"(e-?mail|@)", re.I)
INITIALS_RE = re.compile(r"\b[" + _UP + r"]\.\s?(?:[" + _UP + r"]\.\s?)?[" + _UP + "][" + _LOW + "]+")
WORDLIKE_RE = re.compile(r"^[^\W\d_]{2,}$")


def upper_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def word_like_share(text):
    """Доля «настоящих» слов: у текста со сломанной кодировкой («2 342 ÿ 4ÿ 678») она низкая."""
    toks = text.split()
    if not toks:
        return 0.0
    good = sum(bool(WORDLIKE_RE.match(t.strip(".,;:!?()[]«»\"'").replace("-", ""))) for t in toks)
    return good / len(toks)


def is_front_matter(s):
    """Заголовок капсом, список авторов или аффилиация — не часть аннотации."""
    return (upper_share(s) > 0.6
            or bool(EMAIL_RE.search(s))
            # «Казахский национальный университет им. аль-Фараби, Алматы, Казахстан» — короткая строка без точки
            or (bool(AFFIL_RE.search(s)) and n_words(s) <= 15 and not s.rstrip().endswith((".", "!", "?")))
            or len(INITIALS_RE.findall(s)) >= 2)


def split_multilang_abstract(abstract):
    """Делит аннотацию на части по языкам. Возвращает {язык: текст}.
    Отрезает всё до метки «Аннотация/Аңдатпа/Abstract» (заголовок, авторы) и выбрасывает
    предложения-заголовки, списки авторов и аффилиации."""
    parts = {}
    for par in abstract.split("\n"):
        par = par.strip()
        m = LABEL_ANY_RE.search(par)
        if m and m.start() < 0.6 * len(par):
            par = par[m.end():]
        par = KEYWORDS_RE.sub("", ABS_LABEL_RE.sub("", par)).strip()
        if not par:
            continue
        for s in _split_paragraph(re.sub(r"\s+", " ", par), "ru"):
            s = s.strip()
            m = LABEL_ANY_RE.search(s)
            if m and m.start() < 0.6 * len(s):
                s = s[m.end():]
            s = KEYWORDS_RE.sub("", ABS_LABEL_RE.sub("", s)).strip()
            if n_words(s) < 3 or is_front_matter(s):
                continue
            parts.setdefault(segment_lang(s), []).append(s)
    return {k: " ".join(v) for k, v in parts.items()}


# ------------------------------ проверки ------------------------------
def ngram_leak(abstract, text, n=8):
    """Доля n-грамм аннотации, найденных в тексте (утечка аннотации во вход)."""
    a, t = tokenize(abstract), tokenize(text)
    a_ng = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    if not a_ng:
        return 0.0
    t_ng = {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}
    return len(a_ng & t_ng) / len(a_ng)


MIXED_WORD = re.compile(r"\b(?=\w*[a-zA-Z])(?=\w*[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ])\w+\b")


def script_stats(text):
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0, 0.0, 0.0
    n = len(letters)
    cyr = sum("\u0400" <= c <= "\u04ff" for c in letters) / n
    lat = sum("a" <= c <= "z" for c in letters) / n
    kk = sum(c in KK_LETTERS for c in letters) / n
    return cyr, lat, kk


def wrong_language(text, lang):
    """Грубая проверка: ловит явные ошибки (резюме на другом языке), допускает латиницу в формулах и терминах."""
    cyr, lat, kk = script_stats(text)
    if lang == "en":
        return lat < 0.6
    if lang == "ru":
        return cyr < 0.5 or kk > 0.02
    if lang == "kk":
        return cyr < 0.5 or kk < 0.005
    return False


# =============================== PREPARE ===============================
def stage_prepare():
    df = read_any(CFG["data_path"])
    print(f"Загружено строк: {len(df)}; колонки: {list(df.columns)}")

    df["_lang"] = df[CFG["col_lang"]].astype(str).str.strip().str.lower().map(CFG["lang_map"])
    unknown = df["_lang"].isna().sum()
    if unknown:
        print(f"ВНИМАНИЕ: {unknown} строк с нераспознанным языком; значения: "
              f"{df.loc[df['_lang'].isna(), CFG['col_lang']].unique()[:10]}")
    df = df[df["_lang"].notna()].copy()
    if CFG.get("only_has_text") and "has_text" in df.columns:
        df = df[df["has_text"].astype(bool)].copy()

    df["_text"] = df[CFG["col_text"]].map(normalize)
    df["_removed_sections"] = 0
    df["_abs_harvested"] = ""
    if CFG.get("strip_abstract_sections"):
        res = df["_text"].map(strip_abstract_sections)
        df["_text"] = res.map(lambda x: x[0])
        df["_removed_sections"] = res.map(lambda x: x[1])
        df["_abs_harvested"] = res.map(lambda x: x[2])

    # Пул аннотаций = аннотация из заголовка TEI + аннотации, найденные в основном тексте
    abs_header = df[CFG["col_abstract"]].map(normalize) if CFG["col_abstract"] in df.columns else ""
    df["_abs_all"] = [("\n".join(x for x in (a, h) if x)) for a, h in zip(abs_header, df["_abs_harvested"])]
    if CFG["abstract_cols_by_lang"]:
        cols = CFG["abstract_cols_by_lang"]
        df["_abs"] = [normalize(r[cols[r["_lang"]]]) for _, r in df.iterrows()]
        df["_abs_langs"] = ""
    elif CFG.get("split_multilang_abstract"):
        parts = df["_abs_all"].map(split_multilang_abstract)
        df["_abs"] = [p.get(l, "") for p, l in zip(parts, df["_lang"])]
        df["_abs_langs"] = parts.map(lambda p: "+".join(sorted(p)))
    else:
        df["_abs"] = df["_abs_all"]
        df["_abs_langs"] = ""

    print("\nЯзыки найденных аннотаций (заголовок TEI + основной текст):")
    print(df["_abs_langs"].value_counts().head(10).to_string())
    print(f"Статей, где аннотации найдены в основном тексте и вырезаны: {(df['_removed_sections'] > 0).sum()}")

    aw = df["_abs"].map(n_words)
    print("\nЕсть ли эталон на языке статьи (>= min_abs_words слов):")
    for lang in ("en", "ru", "kk"):
        m = df["_lang"] == lang
        ok_n = (aw[m] >= CFG["min_abs_words"]).sum()
        print(f"  {lang}: {ok_n} из {m.sum()} ({ok_n / max(m.sum(), 1):.0%})")
    tri = df["_abs_langs"].str.contains("en") & df["_abs_langs"].str.contains("ru") & df["_abs_langs"].str.contains("kk")
    print(f"Статей с аннотациями на всех трёх языках (параллельная подвыборка): {tri.sum()}")
    if CFG.get("col_lang_text") in df.columns:
        lt = df[CFG["col_lang_text"]].astype(str).str.lower().map(CFG["lang_map"])
        print(f"Расхождение lang_folder и lang_text: {(lt.notna() & (lt != df['_lang'])).sum()} статей")

    df["_tw"] = df["_text"].map(n_words)
    df["_aw"] = df["_abs"].map(n_words)

    # ---------------- фильтры качества (по шагам, со статистикой) ----------------
    if CFG.get("col_lang_text") in df.columns:
        lt = df[CFG["col_lang_text"]].astype(str).str.lower().map(CFG["lang_map"])
        lang_ok = lt == df["_lang"]
    else:
        lang_ok = pd.Series(True, index=df.index)
    dup_key = df["_text"].map(lambda t: " ".join(tokenize(t)[:300]))
    thr = CFG["min_word_like_share"]
    steps = [
        (f"текст >= {CFG['min_text_words']} слов", df["_tw"] >= CFG["min_text_words"]),
        (f"текст <= {CFG['max_text_words']} слов (не сборник)", df["_tw"] <= CFG["max_text_words"]),
        (f"эталон >= {CFG['min_abs_words']} слов", df["_aw"] >= CFG["min_abs_words"]),
        ("текст не «мусор» (кодировка PDF)", df["_text"].map(word_like_share) >= thr),
        ("эталон не «мусор»", df["_abs"].map(word_like_share) >= thr),
        ("язык текста = язык папки", lang_ok),
        (f"текст >= {CFG['min_compression']}× длиннее эталона", df["_tw"] >= CFG["min_compression"] * df["_aw"]),
        ("не дубликат другой статьи", ~dup_key.duplicated(keep=False)),
    ]
    mask = pd.Series(True, index=df.index)
    table = [("исходно (has_text)", df["_lang"].value_counts())]
    for name, cond in steps:
        mask &= cond
        table.append((name, df.loc[mask, "_lang"].value_counts()))
    leak = pd.Series(0.0, index=df.index)
    leak[mask] = [ngram_leak(a, t) for a, t in zip(df.loc[mask, "_abs"], df.loc[mask, "_text"])]
    mask &= leak <= CFG["max_abstract_leak"]
    table.append((f"эталон не скопирован в текст (утечка <= {CFG['max_abstract_leak']})",
                  df.loc[mask, "_lang"].value_counts()))

    stats = pd.DataFrame({name: vc for name, vc in table}).T.reindex(columns=["en", "ru", "kk"]).fillna(0).astype(int)
    stats.index.name = "шаг фильтрации"
    stats.to_csv(OUT / "filter_stats.csv", encoding="utf-8-sig")
    print("\nСколько статей остаётся после каждого фильтра:")
    print(stats.to_string())

    ok = df[mask]
    cols_keep = [c for c in ["discipline", "journal", "year"] if c in ok.columns]
    clean = ok[[CFG["col_id"]] + cols_keep].copy()
    clean.insert(1, "lang", ok["_lang"])
    clean["text_words"] = ok["_tw"]
    clean["abstract_words"] = ok["_aw"]
    clean["abstract_langs"] = ok["_abs_langs"]
    clean.to_csv(OUT / "clean_ids.csv", index=False, encoding="utf-8-sig")
    print(f"\nСписок прошедших фильтры статей -> {OUT / 'clean_ids.csv'}")
    tri_ok = ok["_abs_langs"].str.contains("en") & ok["_abs_langs"].str.contains("ru") & ok["_abs_langs"].str.contains("kk")
    print(f"Из них с аннотациями на трёх языках: {tri_ok.sum()}")

    # Выбор: равномерно по дисциплинам и длине, чтобы попали разные области и короткие/длинные статьи
    picked = []
    sort_cols = (["discipline", "_tw"] if "discipline" in ok.columns else ["_tw"])
    for lang, n in CFG["n_per_lang"].items():
        sub = ok[ok["_lang"] == lang].sort_values(sort_cols).reset_index(drop=True)
        if len(sub) < n:
            sys.exit(f"Мало статей для языка {lang}: {len(sub)}")
        idx = np.linspace(0, len(sub) - 1, n).round().astype(int)
        picked.append(sub.iloc[idx])
    sel = pd.concat(picked, ignore_index=True)

    rows, checks = [], []
    for _, r in sel.iterrows():
        sents = split_sents(r["_text"], r["_lang"])
        leak = ngram_leak(r["_abs"], r["_text"])
        leak_all = ngram_leak(r["_abs_all"], r["_text"])
        cyr, lat, kk = script_stats(r["_text"])
        mixed = len(MIXED_WORD.findall(r["_text"]))
        refs_tail = bool(re.search(r"(список литературы|references|әдебиеттер)", r["_text"][-5000:], re.I))
        meta = {c: (None if pd.isna(r[c]) else r[c]) for c in CFG.get("meta_cols", []) if c in sel.columns}
        meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}
        rows.append({"id": str(r[CFG["col_id"]]), "lang": r["_lang"], **meta, "text": r["_text"],
                     "abstract": r["_abs"], "abstract_all": r["_abs_all"],
                     "abstract_langs": r["_abs_langs"], "sents": sents})
        checks.append({
            "id": str(r[CFG["col_id"]]), "lang": r["_lang"],
            "discipline": meta.get("discipline"), "journal": meta.get("journal"),
            "text_words": r["_tw"], "abstract_words": r["_aw"],
            "abstract_langs": r["_abs_langs"], "removed_abs_sections": r["_removed_sections"],
            "abstract_leak_all_langs_8gram": round(leak_all, 3), "n_sents": len(sents),
            "compression": round(r["_tw"] / max(r["_aw"], 1), 1),
            "abstract_leak_8gram": round(leak, 3),
            "share_cyr": round(cyr, 3), "share_lat": round(lat, 3), "share_kk_letters": round(kk, 4),
            "mixed_script_words": mixed, "references_in_tail": refs_tail,
            "first_sentence": sents[0][:150] if sents else "",
        })
    write_jsonl(F_DATA, rows)
    pd.DataFrame(checks).to_csv(F_CHECKS, index=False, encoding="utf-8-sig")

    print(f"\nВыбрано {len(rows)} статей -> {F_DATA}")
    print(f"Проверки -> {F_CHECKS}")
    print("\nЧто проверить в data_checks.csv:")
    print("  abstract_leak_8gram > 0.2  -> аннотация, скорее всего, осталась в тексте")
    print("  abstract_leak_all_langs_8gram > 0.2 -> в тексте осталась аннотация на другом языке")
    print("  abstract_langs             -> какие языки нашлись в поле abstract (kk+ru+en = трёхъязычная)")
    print("  mixed_script_words > 0     -> латиница внутри кириллических слов (подмена казахских букв)")
    print("  references_in_tail = True  -> список литературы не удалён")
    print("  share_kk_letters < 0.01 у KK -> казахские буквы потеряны при парсинге")
    print("\nТакже откройте 2-3 текста и проверьте разбиение на предложения вручную.")


# =============================== EXTRACT ===============================
def rouge_scorer():
    from rouge_score import rouge_scorer as rs
    return rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=UnicodeTokenizer())


def truncate_words(text, budget):
    """Обрезает резюме до budget слов (стандарт при сравнении экстрактивных методов по ROUGE)."""
    out, n = [], 0
    for tok in text.split():
        out.append(tok)
        n += len(tokenize(tok))
        if n >= budget:
            break
    return " ".join(out)


def select_by_scores(sents, scores, budget):
    order = np.argsort(-np.asarray(scores))
    chosen, words = [], 0
    for i in order:
        chosen.append(int(i))
        words += n_words(sents[i])
        if words >= budget:
            break
    return truncate_words(" ".join(sents[i] for i in sorted(chosen)), budget)


def lead_k(sents, budget):
    out, words = [], 0
    for s in sents:
        out.append(s)
        words += n_words(s)
        if words >= budget:
            break
    return truncate_words(" ".join(out), budget)


def oracle(sents, abstract, budget, scorer):
    chosen, best = [], 0.0
    while sum(n_words(sents[i]) for i in chosen) < 1.5 * budget:
        cand_best, cand_i = best, None
        for i in range(len(sents)):
            if i in chosen:
                continue
            summ = " ".join(sents[j] for j in sorted(chosen + [i]))
            sc = scorer.score(abstract, summ)
            val = sc["rouge1"].fmeasure + sc["rouge2"].fmeasure
            if val > cand_best:
                cand_best, cand_i = val, i
        if cand_i is None:
            break
        chosen.append(cand_i)
        best = cand_best
    return truncate_words(" ".join(sents[i] for i in sorted(chosen)), budget)


def textrank(sim):
    import networkx as nx
    S = np.array(sim, dtype=float)
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    try:
        pr = nx.pagerank(nx.from_numpy_array(S), weight="weight", max_iter=500)
        return np.array([pr[i] for i in range(len(S))])
    except Exception:
        return S.sum(axis=1)


def tfidf_sim(sents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    X = vec.fit_transform(sents)
    return cosine_similarity(X)


def stage_extract():
    from sentence_transformers import SentenceTransformer
    docs = read_jsonl(F_DATA)
    scorer = rouge_scorer()
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"BGE-M3 на устройстве: {device}" + ("  (GPU недоступен — считаем на CPU, для 10 статей это нормально)" if device == "cpu" else ""))
    model = SentenceTransformer(CFG["dense_model"], device=device)
    w = CFG["hybrid_w_sparse"]

    rows, llm_inputs = [], []
    for d in docs:
        sents, budget = d["sents"], n_words(d["abstract"])
        base = {"id": d["id"], "lang": d["lang"]}

        t = time.time(); s = lead_k(sents, budget)
        rows.append({**base, "method": "lead_k", "summary": s, "time_s": time.time() - t})

        t = time.time(); s = oracle(sents, d["abstract"], budget, scorer)
        rows.append({**base, "method": "oracle", "summary": s, "time_s": time.time() - t})

        t = time.time(); S_sp = tfidf_sim(sents); sc_sp = textrank(S_sp)
        s = select_by_scores(sents, sc_sp, budget)
        rows.append({**base, "method": "textrank_tfidf", "summary": s, "time_s": time.time() - t})

        t = time.time()
        emb = model.encode(sents, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        c = emb.mean(axis=0); c /= np.linalg.norm(c)
        s = select_by_scores(sents, emb @ c, budget)
        t_dense = time.time() - t
        rows.append({**base, "method": "centroid_bgem3", "summary": s, "time_s": t_dense})

        t = time.time()
        S_hyb = w * S_sp + (1 - w) * (emb @ emb.T)
        sc_h = textrank(S_hyb)
        s = select_by_scores(sents, sc_h, budget)
        rows.append({**base, "method": f"textrank_hybrid_w{w}", "summary": s,
                     "time_s": time.time() - t + t_dense})

        k = CFG["extract_for_llm_sents"]
        top = sorted(np.argsort(-sc_h)[:k].tolist())
        llm_inputs.append({**base, "extracted": " ".join(sents[i] for i in top)})
        print(f"  {d['id']} ({d['lang']}): {len(sents)} предложений, бюджет {budget} слов")

    append_summaries(rows)
    write_jsonl(F_LLM_IN, llm_inputs)
    print(f"\nЭкстрактивные резюме -> {F_SUMM}")


# ================================= LLM =================================
PROMPTS = {
    "en": ("You are an expert in scientific writing. Write an abstract of the following scientific article "
           "in English. The abstract must be about {n} words (no fewer than {lo}), a single paragraph, and must "
           "state the aim, methods, main results and novelty of the study. Use only information from the article; "
           "do not add facts that are not in it. Output only the text of the abstract: no title, no heading, "
           "no introductory phrase, no markdown.\n\n"
           "{label}:\n{text}"),
    "ru": ("Вы эксперт в научном письме. Напишите аннотацию к следующей научной статье на русском языке. "
           "Объём около {n} слов (не меньше {lo}), один абзац; укажите цель, методы, основные результаты и новизну "
           "исследования. Используйте только информацию из статьи, не добавляйте фактов, которых в ней нет. "
           "Выведите только текст аннотации: без заголовка, без вводной фразы и без разметки.\n\n{label}:\n{text}"),
    "kk": ("Сіз ғылыми мәтін жазу саласының маманысыз. Төмендегі ғылыми мақалаға қазақ тілінде аңдатпа жазыңыз. "
           "Көлемі шамамен {n} сөз (кемінде {lo} сөз), бір абзац; зерттеудің мақсатын, әдістерін, негізгі "
           "нәтижелерін және жаңалығын көрсетіңіз. Тек мақаладағы ақпаратты пайдаланыңыз, онда жоқ деректерді "
           "қоспаңыз. Тек аңдатпа мәтінін ғана жазыңыз: тақырыпсыз, кіріспе сөйлемсіз және белгілеусіз."
           "\n\n{label}:\n{text}"),
}
LABELS = {
    "full": {"en": "Article", "ru": "Статья", "kk": "Мақала"},
    "extracted": {"en": "Key sentences of the article", "ru": "Ключевые предложения статьи",
                  "kk": "Мақаланың негізгі сөйлемдері"},
}


LABEL_ONLY_RE = re.compile(r"^(аннотация|аңдатпа|андатпа|түйіндеме|резюме|abstract|summary)\s*[.:]?$", re.I)


def clean_llm_output(text):
    """Убирает скрытые рассуждения, markdown, заголовок «Аңдатпа» и вводные фразы вроде «Бұл мақаланың аңдатпасы:»."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)      # заголовки markdown
    text = text.replace("**", "").replace("__", "").replace("`", "")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    while lines and n_words(lines[0]) <= 8 and (lines[0].endswith(":") or LABEL_ONLY_RE.match(lines[0])):
        lines.pop(0)                                          # «Аңдатпа», «Here is the abstract:»
    text = " ".join(lines)
    text = ABS_LABEL_RE.sub("", text)                         # «Аннотация. Текст…» в начале строки
    return text.strip()


def _ollama_headers():
    """Если задана переменная OLLAMA_API_KEY, передаём её как Bearer-токен (сам токен в коде не хранится)."""
    import os
    h = {"Content-Type": "application/json"}
    key = os.environ.get("OLLAMA_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _ollama_chat(model, prompt, num_predict):
    import urllib.error
    import urllib.request
    think = CFG["llm_think"].get(model, False)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "think": think,
            "options": {"temperature": 0, "seed": CFG["seed"], "num_ctx": CFG["llm_num_ctx"],
                        "num_predict": num_predict}}

    def post(b):
        req = urllib.request.Request(CFG["ollama_url"] + "/api/chat", data=json.dumps(b).encode("utf-8"),
                                     headers=_ollama_headers())
        with urllib.request.urlopen(req, timeout=CFG["llm_timeout_s"]) as r:
            return json.loads(r.read())

    try:
        return post(body)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")
        if "think" in msg.lower():              # модель не поддерживает параметр think
            body.pop("think")
            return post(body)
        raise RuntimeError(f"Ollama {e.code}: {msg[:300]}")


def truncate_input(text, max_words):
    words = text.split()
    return (" ".join(words[:max_words]), True) if len(words) > max_words else (text, False)


def stage_llm(models=None):
    import urllib.request
    try:
        req = urllib.request.Request(CFG["ollama_url"] + "/api/tags", headers=_ollama_headers())
        with urllib.request.urlopen(req, timeout=10) as r:
            available = {m["name"] for m in json.loads(r.read())["models"]}
    except Exception as e:
        sys.exit(f"Ollama не отвечает по адресу {CFG['ollama_url']}: {e}")
    models = models or CFG["llm_models"]
    missing = [m for m in models if m not in available]
    if missing:
        sys.exit(f"В Ollama нет моделей {missing}. Доступны: {sorted(available)}")

    docs = read_jsonl(F_DATA)
    extracted = {r["id"]: r["extracted"] for r in read_jsonl(F_LLM_IN)}

    for model in models:
        short = re.sub(r"[^\w.-]+", "-", model)
        # Прогрев: первая загрузка модели в GPU занимает до пары минут и не должна попадать во время метода
        t0 = time.time()
        try:
            _ollama_chat(model, "Hi", 1)
        except Exception as e:
            print(f"  [{model}] прогрев не удался: {e}")
        print(f"  [{model}] модель загружена за {time.time() - t0:.0f} c (во время методов не входит)")
        for variant in ("full", "extracted"):
            rows = []
            for d in docs:
                src = d["text"] if variant == "full" else extracted[d["id"]]
                src, cut = truncate_input(src, CFG["llm_max_input_words"][d["lang"]])
                n = n_words(d["abstract"])
                prompt = PROMPTS[d["lang"]].format(n=n, lo=int(0.85 * n), label=LABELS[variant][d["lang"]], text=src)
                num_predict = int(n * 4) + 300 + (2048 if CFG["llm_think"].get(model) else 0)
                t = time.time()
                try:
                    resp = _ollama_chat(model, prompt, num_predict)
                    summary = clean_llm_output(resp.get("message", {}).get("content", ""))
                    err = None
                except Exception as e:
                    resp, summary, err = {}, "", str(e)[:300]
                dt = time.time() - t
                prompt_tok = resp.get("prompt_eval_count")
                rows.append({"id": d["id"], "lang": d["lang"], "method": f"llm_{variant}_{short}",
                             "summary": summary, "time_s": dt, "input_truncated_words": cut,
                             "prompt_tokens": prompt_tok, "output_tokens": resp.get("eval_count"),
                             "context_overflow": bool(prompt_tok and prompt_tok >= CFG["llm_num_ctx"] - 8),
                             "finish_reason": resp.get("done_reason"), "error": err})
                status = "ОШИБКА " + err if err else f"{n_words(summary)} слов, {prompt_tok} ток. входа"
                print(f"  [{model} | {variant}] {d['id']} ({d['lang']}): {dt:.0f} c, {status}")
            append_summaries(rows)
    print(f"\nLLM-резюме -> {F_SUMM}")
    print("Проверьте: context_overflow=True значит, что вход не поместился в окно и Ollama его обрезал.")


# ================================= EVAL =================================
def stage_eval():
    from sacrebleu.metrics import CHRF
    from bert_score import score as bert_score

    docs = {d["id"]: d for d in read_jsonl(F_DATA)}
    summ = read_jsonl(F_SUMM)
    scorer = rouge_scorer()
    chrf = CHRF(word_order=2)

    cands = [s["summary"] or "." for s in summ]
    refs = [docs[s["id"]]["abstract"] for s in summ]
    _, _, F = bert_score(cands, refs, model_type=CFG["bertscore_model"], batch_size=16, verbose=False)

    rows = []
    for s, bs in zip(summ, F.tolist()):
        ref = docs[s["id"]]["abstract"]
        r = scorer.score(ref, s["summary"])
        rows.append({
            "id": s["id"], "lang": s["lang"], "method": s["method"],
            "rouge1": r["rouge1"].fmeasure, "rouge2": r["rouge2"].fmeasure, "rougeL": r["rougeL"].fmeasure,
            "chrf++": chrf.sentence_score(s["summary"], [ref]).score / 100,
            "bertscore_f1": bs,
            "len_ratio": n_words(s["summary"]) / max(n_words(ref), 1),
            "wrong_language": wrong_language(s["summary"], s["lang"]),
            "time_s": s.get("time_s"),
        })
    m = pd.DataFrame(rows)
    m.to_csv(F_METRICS, index=False, encoding="utf-8-sig")

    metric_cols = ["rouge1", "rouge2", "rougeL", "chrf++", "bertscore_f1", "len_ratio", "time_s"]
    table = m.groupby(["lang", "method"])[metric_cols].mean().round(3)
    table["wrong_language_n"] = m.groupby(["lang", "method"])["wrong_language"].sum()
    table.to_csv(F_TABLE, encoding="utf-8-sig")

    review = pd.DataFrame([{"id": s["id"], "lang": s["lang"], "method": s["method"],
                            "abstract": docs[s["id"]]["abstract"], "summary": s["summary"],
                            "оценка_смысл_1_5": "", "выдуманные_факты_да_нет": "", "комментарий": ""}
                           for s in summ]).sort_values(["lang", "id", "method"])
    review.to_csv(F_REVIEW, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(table.to_string())
    print(f"\nПо статьям -> {F_METRICS}\nСводная -> {F_TABLE}\nДля ручного просмотра -> {F_REVIEW}")
    print("\nНапоминание: 10 статей — проверка пайплайна, не основа для выводов.")


# ================================= main =================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "extract", "llm", "eval"])
    ap.add_argument("--models", nargs="*", default=None, help="модели Ollama для стадии llm (по умолчанию все из CFG)")
    args = ap.parse_args()
    if args.stage == "llm":
        stage_llm(args.models)
    else:
        {"prepare": stage_prepare, "extract": stage_extract, "eval": stage_eval}[args.stage]()
