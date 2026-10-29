#!/usr/bin/env python3
"""
Этап 3. Плотные векторы предложений (и эталонных аннотаций) на GPU.

Один раз считает эмбеддинги всех предложений всех статей несколькими многоязычными моделями
и сохраняет их на диск; все следующие этапы используют готовые векторы.
Разреженные представления (TF-IDF, BM25) считаются на CPU в этапе 4 — их кэшировать не нужно.

Максимальная загрузка H200:
    - модели распределяются по всем доступным GPU и считаются параллельно (отдельный процесс на GPU);
    - половинная точность (fp16), большие батчи, предложения сортируются по длине (меньше паддинга);
    - перед стартом скрипт смотрит свободную память каждой GPU и не берёт карты, где её мало
      (сервер общий — чужие задачи не трогаем).

Пишет (в ~/summary/data/emb/):
    sentences.parquet         — индекс: строка матрицы -> (id статьи, язык, выборка, номер предложения, текст)
    refs.parquet              — индекс эталонных аннотаций (на всех языках статьи)
    <модель>/sent.npy         — матрица векторов предложений (float16, нормированные)
    <модель>/ref.npy          — векторы аннотаций
    <модель>/info.json        — модель, размерность, время, устройство

Запуск:
    python embed.py                       # все модели, все свободные GPU
    python embed.py --models bge-m3 labse # только выбранные модели
    python embed.py --limit 50            # быстрый тест на 50 статьях каждого языка
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path.home() / "summary" / "data"
DATASET = DATA / "dataset.jsonl"
EMB = DATA / "emb"

# Многоязычные модели (все поддерживают казахский). prefix — служебная приставка, которую требует модель.
MODELS = {
    "bge-m3":     {"hf": "BAAI/bge-m3",                         "prefix": "",        "max_len": 512},
    "labse":      {"hf": "sentence-transformers/LaBSE",         "prefix": "",        "max_len": 256},
    "e5-large":   {"hf": "intfloat/multilingual-e5-large",      "prefix": "query: ", "max_len": 512},
    "qwen3-emb":  {"hf": "Qwen/Qwen3-Embedding-0.6B",           "prefix": "",        "max_len": 512},
}
MIN_FREE_GB = 12          # GPU, на которой свободно меньше, не используем


def gpu_free_memory():
    """Свободная память каждой GPU в ГБ.
    Сначала через PyTorch (работает, даже когда nvidia-smi сломан: «Driver/library version mismatch»),
    затем через nvidia-smi как запасной вариант."""
    try:
        import torch
        if torch.cuda.is_available():
            res = {}
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                res[i] = free / 1024 ** 3
            return res
    except Exception as e:
        print(f"PyTorch не смог опросить GPU ({str(e)[:150]})")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return {int(i): int(m) / 1024 for i, m in (l.split(",") for l in out.strip().splitlines())}
    except Exception:
        print("GPU не найдены ни через PyTorch, ни через nvidia-smi — считаю на CPU")
        return {}


def build_index(limit=None):
    """Индекс всех предложений и эталонов (один на все модели)."""
    sent_rows, ref_rows = [], []
    per_lang = {}
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if limit:
                per_lang[d["lang"]] = per_lang.get(d["lang"], 0) + 1
                if per_lang[d["lang"]] > limit:
                    continue
            for i, s in enumerate(d["sents"]):
                sent_rows.append((d["id"], d["lang"], d["split"], i, s))
            for L, a in d["abstracts"].items():
                ref_rows.append((d["id"], d["lang"], d["split"], L, a))
    sents = pd.DataFrame(sent_rows, columns=["id", "lang", "split", "sid", "text"])
    refs = pd.DataFrame(ref_rows, columns=["id", "lang", "split", "ref_lang", "text"])
    return sents, refs


def encode_worker(model_keys, gpu, batch_size, limit_tag):
    """Считает эмбеддинги для списка моделей на одной GPU (запускается отдельным процессом)."""
    import torch
    from sentence_transformers import SentenceTransformer

    sents = pd.read_parquet(EMB / f"sentences{limit_tag}.parquet")
    refs = pd.read_parquet(EMB / f"refs{limit_tag}.parquet")
    device = "cuda:0" if gpu is not None else "cpu"      # внутри процесса видна только своя GPU
    for key in model_keys:
        cfg = MODELS[key]
        out = EMB / key
        out.mkdir(parents=True, exist_ok=True)
        if (out / f"sent{limit_tag}.npy").exists():
            print(f"[{key}] уже посчитано — пропускаю (удалите папку {out}, чтобы пересчитать)", flush=True)
            continue
        t0 = time.time()
        try:
            model = SentenceTransformer(cfg["hf"], device=device,
                                        model_kwargs={"torch_dtype": torch.float16} if gpu is not None else {})
        except Exception as e:
            print(f"[{key}] не удалось загрузить {cfg['hf']}: {str(e)[:200]} — пропускаю", flush=True)
            continue
        model.max_seq_length = cfg["max_len"]

        def enc(texts):
            texts = [cfg["prefix"] + t for t in texts]
            order = np.argsort([len(t) for t in texts])          # сортировка по длине — меньше паддинга
            vecs = model.encode([texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True)
            res = np.empty_like(vecs)
            res[order] = vecs
            return res.astype(np.float16)

        sv = enc(sents["text"].tolist())
        rv = enc(refs["text"].tolist())
        np.save(out / f"sent{limit_tag}.npy", sv)
        np.save(out / f"ref{limit_tag}.npy", rv)
        dt = time.time() - t0
        info = {"model": cfg["hf"], "dim": int(sv.shape[1]), "n_sents": len(sv), "n_refs": len(rv),
                "seconds": round(dt, 1), "sents_per_second": round(len(sv) / dt, 1),
                "gpu": gpu, "gpu_name": torch.cuda.get_device_name(0) if gpu is not None else "cpu",
                "dtype": "float16", "batch_size": batch_size}
        (out / f"info{limit_tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"[{key}] готово: {len(sv)} предложений за {dt:.0f} c ({info['sents_per_second']} предл./с) "
              f"на GPU {gpu}", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=None, help="тест: столько статей на язык")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpus", default=None, help="например 0,1; по умолчанию все GPU со свободной памятью")
    ap.add_argument("--worker", nargs=2, metavar=("MODELS", "GPU"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    limit_tag = f"_test{args.limit}" if args.limit else ""

    if args.worker:                                      # режим дочернего процесса
        gpu = None if args.worker[1] == "cpu" else int(args.worker[1])
        encode_worker(args.worker[0].split(","), gpu, args.batch_size, limit_tag)
        return

    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        sys.exit(f"Неизвестные модели: {unknown}. Доступны: {list(MODELS)}")

    EMB.mkdir(parents=True, exist_ok=True)
    sents, refs = build_index(args.limit)
    sents.to_parquet(EMB / f"sentences{limit_tag}.parquet", index=False)
    refs.to_parquet(EMB / f"refs{limit_tag}.parquet", index=False)
    print(f"Предложений: {len(sents):,}; эталонных аннотаций: {len(refs):,}")
    print(sents.groupby("lang").size().rename("предложений по языкам").to_string())

    free = gpu_free_memory()
    for g, gb in free.items():
        print(f"GPU {g}: свободно {gb:.0f} ГБ" + ("" if gb >= MIN_FREE_GB else " — мало, не используем"))
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else [g for g, gb in free.items() if gb >= MIN_FREE_GB]
    gpus = gpus or ["cpu"]

    # распределяем модели по GPU по кругу и запускаем по процессу на каждую GPU
    plan = {g: [] for g in gpus}
    for i, m in enumerate(args.models):
        plan[gpus[i % len(gpus)]].append(m)
    procs = []
    for g, ms in plan.items():
        if not ms:
            continue
        env = dict(os.environ)
        if g != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = str(g)
        print(f"GPU {g}: модели {ms}")
        cmd = [sys.executable, __file__, "--worker", ",".join(ms), str(g), "--batch-size", str(args.batch_size)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]

    print("\nИтог:")
    for m in args.models:
        f = EMB / m / f"info{limit_tag}.json"
        if f.exists():
            i = json.loads(f.read_text())
            print(f"  {m:10s} размерность {i['dim']:5d}  {i['sents_per_second']:>9} предл./с  GPU {i['gpu']}")
        else:
            print(f"  {m:10s} НЕ посчитано (см. сообщения выше)")
    if any(codes):
        print("Есть ошибки в дочерних процессах — см. вывод выше.")


if __name__ == "__main__":
    main()
