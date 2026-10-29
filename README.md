# Hybrid extract-then-generate summarization of scientific articles in English, Russian and Kazakh

Code and data for the paper *«Hybrid representations and extract-then-generate summarization of
English, Russian and Kazakh scientific articles: quality, faithfulness and cost of local LLMs»* (under review).

**Corpus:** 5,669 articles from Kazakhstani journals in physics, mathematics and computer science
(EN 3,273 · RU 1,518 · KK 878) with author abstracts as references.

**Compared methods**

- sentence representations: sparse (TF-IDF, BM25), dense (BGE-M3, LaBSE, multilingual-E5-large,
  Qwen3-Embedding-0.6B) and hybrid (weighted sum, RRF, concatenation + position features);
- extractive summarizers: TextRank, centroid, MMR (unsupervised) and LogReg / SVM / LightGBM / MLP
  trained on greedy-oracle labels;
- local LLMs via Ollama (gemma4:26b, qwen3.6:35b): full text vs. 25 sentences selected by the
  hybrid or single-representation classifier (extract → generate);
- evaluation: ROUGE (Unicode tokenizer), chrF++, BERTScore, LaBSE similarity, faithfulness by an
  LLM judge validated with control sets and a second judge; paired bootstrap with Holm correction.

## Pipeline

| Stage | Script | Output |
| --- | --- | --- |
| 0 | `scripts/build_corpus.py` | GROBID TEI → `data/corpus.jsonl` |
| 1 | `scripts/extract_pdf_abstracts.py` | abstracts in all languages from the PDFs |
| 2 | `scripts/make_dataset.py` | cleaning, language check, filters, fixed split |
| 3 | `scripts/embed.py` | dense sentence embeddings on GPU |
| 4 | `scripts/extractive.py all` | oracle labels, unsupervised and supervised extractive summaries |
| 5–6 | `scripts/generate.py run` / `score` | LLM summaries (full / hybrid / single input) |
| 7 | `scripts/evaluate.py metrics` | ROUGE, chrF++, BERTScore, semantic similarity |
| 7b | `scripts/judge.py run --judge qwen3.6:35b --controls`, `report` | LLM-judge faithfulness + controls |
| 7c | `scripts/evaluate.py stats` | tables with 95% CI and paired bootstrap tests |
| 8 | `scripts/make_figures.py` | figures and paper tables |

`tools/diag_faith.py` reproduces the NLI calibration check; `tools/add_evidence.py` prepares the
manual error-analysis sheet. Prompts are in `scripts/pilot.py` (`PROMPTS`) and `scripts/judge.py` (`PROMPT`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# local LLMs (no paid API): https://ollama.com
ollama pull gemma4:26b && ollama pull qwen3.6:35b
export OLLAMA_HOST=127.0.0.1:11434
```

Experiments were run on 2× NVIDIA H200; a single 48 GB GPU is enough for gemma4:26b.

## Data and copyright

Full texts are not included (see `data/README.md`). We release article identifiers and metadata,
the split, generated summaries, judge labels and all result tables.

## Citation

```
@article{TODO,
  title  = {Hybrid representations and extract-then-generate summarization of English, Russian and Kazakh scientific articles},
  author = {TODO},
  year   = {2026}
}
```

## License

Code: MIT. Generated summaries and annotations: CC BY 4.0.
