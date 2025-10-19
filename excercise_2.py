#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and evaluate the MSH-WSD corpus.


Usage
  python msh_loader_and_eval.py --msh_dir "C:\\path\\to\\MSHCorpus"

  # with MeSH-Lesk (if you have CUI → definition text):
  python msh_loader_and_eval.py --msh_dir "C:\\path\\to\\MSHCorpus" --cui_gloss_json mesh_gloss.json
"""
import argparse
import os
import re
import csv
import io
import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import random

CLASS_RE = re.compile(r"^M\d+$")
# ----------------------- ARFF reader (no external deps) -----------------------
def read_arff_instances(path: Path):
    """
    ARFF reader for MSH-WSD:
    - Skips header until @data
    - Parses rows with csv (handles quotes/commas)
    - Returns (text, gold_label) where gold_label is 'M#'
    - Works for files with an extra PMID column
    """
    raw = path.read_text(encoding="latin-1", errors="ignore")
    parts = re.split(r"(?i)\n@data\s*\n", raw, maxsplit=1)
    if len(parts) < 2:
        return
    data_part = parts[1]
    reader = csv.reader(io.StringIO(data_part))
    for row in reader:
        if not row:
            continue
        row = [c.strip() for c in row if c is not None]
        gold = None
        for c in reversed(row):
            if CLASS_RE.match(c):
                gold = c
                break
        if gold is None:
            continue
        candidates = [c for c in row if c != gold]
        if not candidates:
            continue
        text = max(candidates, key=len)
        yield text, gold


# ----------------------- benchmark_mesh.txt loader ----------------------------
def load_benchmark(benchmark_path: Path):
    """
    Returns:
        term -> [CUI1, CUI2, ...]  (M1 aligns to index 0)
    """
    mapping = {}
    with open(benchmark_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            term, cuis = parts[0], parts[1:]
            mapping[term] = cuis
    return mapping


# ----------------------- name normalization & indexing ------------------------
_NORM_SPACE_RE = re.compile(r"[\s_\-]+", flags=re.UNICODE)

def _normalize_name(s: str) -> str:
    """
    Normalizes filenames and benchmark terms to a comparable key.
    """
    s = s.lower().strip()
    s = s.replace("_pmids_tagged", "")
    s = re.sub(r"\.arff$", "", s)
    s = _NORM_SPACE_RE.sub(" ", s)
    return s.strip()

def index_arff_files(root_dir: Path):
    """
    Returns: dict normalized_name -> Path
    """
    idx = {}
    for p in root_dir.glob("*.arff"):
        idx[_normalize_name(p.name)] = p
        idx[_normalize_name(p.name).replace(" ", "_")] = p
        idx[_normalize_name(p.name).replace(" ", "-")] = p
    return idx


# ----------------------- helpers ------------------------
ACRONYM_RE = re.compile(r"^[A-Z0-9\-]{2,6}$")

def is_acronym(term: str) -> bool:
    return bool(ACRONYM_RE.match(term))

def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def lesk_score(ctx_tokens, gloss_tokens, stop=None):
    if stop is None:
        stop = {
            "the","a","an","and","or","to","of","in","on","for","at","by","is","are","was","were",
            "be","been","with","without","from","that","this","these","those"
        }
    ctx = {t for t in ctx_tokens if t not in stop}
    gls = {t for t in gloss_tokens if t not in stop}
    return len(ctx & gls)


# ----------------------- evaluation ------------------------
def evaluate(term2cuis, term_instances, acr_terms, reg_terms, predictors):
    """
    predictors: list of (name, fn)
      fn can be:
        - per-instance: fn._per_instance = True, signature (term, text, gold) -> "M#"
        - per-term:     fn._per_instance = False (default), signature (term, rows) -> "M#"
    """
    for name, predict_fn in predictors:
        correct = 0
        total = 0
        per_term_acc = {}

        for term, rows in term_instances.items():
            if getattr(predict_fn, "_per_instance", False):
                preds = [predict_fn(term, text, gold) for (text, gold) in rows]
            else:
                clazz = predict_fn(term, rows)
                preds = [clazz] * len(rows)

            c = sum(1 for (p, (_, g)) in zip(preds, rows) if p == g)
            per_term_acc[term] = (c, len(rows))
            correct += c
            total += len(rows)

        def macro(terms):
            accs = []
            for t in terms:
                c, n = per_term_acc.get(t, (0, 0))
                if n > 0:
                    accs.append(c / n)
            return sum(accs) / len(accs) if accs else 0.0

        micro = (correct / total) if total else 0.0
        mac_all = macro(list(term_instances.keys()))
        mac_acr = macro(acr_terms)
        mac_reg = macro(reg_terms)

        print(f"\n[{name}]")
        print(f"  Micro accuracy (all instances): {micro:.3f}  ({correct}/{total})")
        print(f"  Macro accuracy (by term):       {mac_all:.3f}")
        print(f"    ├─ Acronyms: {mac_acr:.3f}")
        print(f"    └─ Regular : {mac_reg:.3f}")


# ----------------------- main ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msh_dir", required=True, help="Path to unzipped MSHCorpus folder")
    parser.add_argument("--cui_gloss_json", default=None, help="Optional JSON: {CUI: 'definition text'} for MeSH-Lesk")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    random.seed(args.seed)

    root = Path(args.msh_dir)
    bench_path = root / "benchmark_mesh.txt"
    if not bench_path.exists():
        sys.exit(f"benchmark_mesh.txt not found in: {root}\n"
                 f"Tip: it should live alongside the *.arff files.")

    term2cuis = load_benchmark(bench_path)
    arff_idx = index_arff_files(root)
    term_instances = {}
    total_inst = 0
    missing = []

    for term in term2cuis:
        norm_term = _normalize_name(term)
        path = (
            arff_idx.get(norm_term)
            or arff_idx.get(norm_term.replace(" ", "_"))
            or arff_idx.get(norm_term.replace(" ", "-"))
        )
        if path is None:
            for key, pth in arff_idx.items():
                if key.startswith(norm_term):
                    path = pth
                    break

        if path is None or not path.exists():
            missing.append(term)
            continue

        rows = list(read_arff_instances(path))
        if not rows:
            missing.append(term)
            continue

        print(f"[match] {term}  →  {path.name}")
        term_instances[term] = rows
        total_inst += len(rows)

    print(f"\n[i] Terms in benchmark: {len(term2cuis)}")
    print(f"[i] Terms with ARFF found: {len(term_instances)} | Missing: {len(missing)}")
    print(f"[i] Total instances (all found terms): {total_inst}")

    acr = [t for t in term_instances if is_acronym(t)]
    reg = [t for t in term_instances if not is_acronym(t)]
    random.seed(42)
    keep = set(random.sample(acr, 10) + random.sample(reg, 10))
    term_instances = {t:v for t,v in term_instances.items() if t in keep}
    print(f"[i] Acronym terms: {len(acr)} | Regular terms: {len(reg)}")

    # --------------------- predictors ---------------------
    def rand_pred(term, text=None, gold=None):
        k = len(term2cuis[term])
        return f"M{random.randint(1, k)}"
    rand_pred._per_instance = True

    def majority_pred(term, rows):
        cnt = Counter(g for _, g in rows)
        return max(cnt, key=cnt.get)

    def firstsense_pred(term, rows):
        return "M1"

    predictors = [
        ("Random", rand_pred),
        ("Majority per term", majority_pred),
        ("First-sense (M1)", firstsense_pred),
    ]

    if args.cui_gloss_json:
        try:
            cui2gloss = json.loads(Path(args.cui_gloss_json).read_text(encoding="utf-8"))
        except Exception as e:
            sys.exit(f"Failed to read cui_gloss_json: {e}")

        gloss_tokens = {cui: tokenize(gloss) for cui, gloss in cui2gloss.items()}

        def lesk_pred(term, text, gold=None):
            ctx = tokenize(text)
            cuis = term2cuis[term]
            best_label, best_score = None, -1
            for i, cui in enumerate(cuis, start=1):
                gl = gloss_tokens.get(cui, [])
                score = lesk_score(ctx, gl)
                if score > best_score:
                    best_score = score
                    best_label = f"M{i}"
            return best_label or "M1"

        lesk_pred._per_instance = True
        predictors.append(("MeSH-Lesk (CUI gloss overlap)", lesk_pred))

    evaluate(term2cuis, term_instances, acr, reg, predictors)

    if missing:
        print("\n[warn] Some terms had no matching ARFF file.")
        print("       Check unusual punctuation or rename files to '<term>_pmids_tagged.arff'.")


inp = "C:/Users/epeli/Downloads/umls_sample.json"
out = "mesh_gloss.json"

with io.open(inp, "r", encoding="utf-8") as f:
    data = json.load(f)

cui2gloss = {}
for term, cuimap in data.items():
    for cui, texts in cuimap.items():
        cui2gloss[cui] = " ".join(t.strip() for t in texts if t and t.strip())

with io.open(out, "w", encoding="utf-8") as f:
    json.dump(cui2gloss, f, ensure_ascii=False, indent=2)
    

print(f"Wrote {out} with {len(cui2gloss)} CUIs.")


if __name__ == "__main__":
    main()
