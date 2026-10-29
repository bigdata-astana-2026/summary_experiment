# Data

Full texts of the articles are **not** distributed: they are copyrighted by the journals.

This folder contains:

| File | Content |
| --- | --- |
| `splits.csv` | article id, language, discipline, journal, year, split (train / val / test) — the fixed split used in the paper |
| `summaries/` | generated summaries of the test articles (LLM and extractive) |
| `judge/` | sentence-level faithfulness labels of the LLM judges (`judge__qwen3.6-35b.jsonl`, `judge__gemma4-26b.jsonl`) |
| `results/` | all result tables (`table_main.csv`, `table_tests.csv`, `sentence_metrics.csv`, judge controls) |

The PDFs can be downloaded from the journal websites; `scripts/build_corpus.py` and
`scripts/extract_pdf_abstracts.py` rebuild the corpus from them.
