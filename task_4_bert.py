#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BioBERT WSD on MSH-WSD: sentence-context embedding vs synset-gloss embedding.
Outputs: eval_ex4/results_BioBERT.csv and predictions_BioBERT.jsonl

Usage:
  python biobert_eval_msh.py --msh_dir "C:\\...\\MSHCorpus" --cui_gloss_json "C:\\...\\mesh_gloss_filled_subset.json" [--subset_10_10] [--model dmis-lab/biobert-v1.1]
"""

import argparse, json, io, re, random, csv, sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# NLTK
import nltk
for res in ("punkt_tab","punkt","wordnet","omw-1.4"):
    try: nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        try: nltk.data.find(f"corpora/{res}")
        except LookupError: nltk.download(res, quiet=True)
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

# ---- helpers reused from your pipeline ----
_NORM_SPACE_RE = re.compile(r"[\s_\-]+", flags=re.UNICODE)
CLASS_RE = re.compile(r"^M\d+$")
ACRONYM_RE = re.compile(r"^[A-Z0-9\-]{2,6}$")

def is_acronym(term: str) -> bool:
    return bool(ACRONYM_RE.match(term))

def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("_pmids_tagged", "")
    s = re.sub(r"\.arff$", "", s)
    s = _NORM_SPACE_RE.sub(" ", s)
    return s.strip()

def index_arff_files(root_dir: Path):
    idx = {}
    for p in root_dir.glob("*.arff"):
        key = _normalize_name(p.name)
        idx[key] = p
        idx[key.replace(" ", "_")] = p
        idx[key.replace(" ", "-")] = p
    return idx

def read_arff_instances(path: Path):
    raw = path.read_text(encoding="latin-1", errors="ignore")
    parts = re.split(r"(?i)\n@data\s*\n", raw, maxsplit=1)
    if len(parts) < 2: return
    data_part = parts[1]
    reader = csv.reader(io.StringIO(data_part))
    for row in reader:
        if not row: continue
        row = [c.strip() for c in row if c is not None]
        gold = None
        for c in reversed(row):
            if CLASS_RE.match(c):
                gold = c; break
        if gold is None: continue
        candidates = [c for c in row if c != gold]
        if not candidates: continue
        text = max(candidates, key=len)
        yield text, gold

def load_benchmark(benchmark_path: Path):
    mapping = {}
    with open(benchmark_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts: continue
            term, cuis = parts[0], parts[1:]
            mapping[term] = cuis
    return mapping

# --- Lesk-style gloss overlap for synset → M# mapping ---
STOP = {"the","a","an","and","or","to","of","in","on","for","at","by","is","are","was","were","be","been",
        "with","without","from","that","this","these","those","as","it","its","into"}
def toks(s: str): return [w for w in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if w not in STOP]
def overlap(a_tokens, b_tokens): return len(set(a_tokens) & set(b_tokens))

def synset_text(s):
    parts = []
    lemmas = [l.replace("_"," ") for l in s.lemma_names()]
    if lemmas: parts.append("; ".join(lemmas))
    parts.append(s.definition())
    ex = s.examples()
    if ex: parts.append(" ".join(ex))
    return " ".join(parts)

def synset_to_Mlabel(syn, cuis_for_term, cui2gloss):
    if syn is None: return "M1"
    syn_tokens = toks(synset_text(syn))
    best_i, best_score = 0, -1
    for i, cui in enumerate(cuis_for_term):
        gl = cui2gloss.get(cui, "")
        sc = overlap(syn_tokens, toks(gl))
        if sc > best_score:
            best_score = sc; best_i = i
    return f"M{best_i+1}"

# --- BioBERT embedding utilities ---
def load_model(model_name="dmis-lab/biobert-v1.1", fp16=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device)
    if fp16 and device.startswith("cuda"): mdl.half()
    mdl.eval()
    return tok, mdl, device

def mean_pool_hidden(enc_out, attention_mask):
    last_hidden = enc_out.last_hidden_state       # [B, T, H]
    mask = attention_mask.unsqueeze(-1).float()   # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)      # [B, H]
    counts = mask.sum(dim=1).clamp(min=1.0)       # [B, 1]
    return (summed / counts)                      # [B, H]

def embed_text(text, tok, mdl, device, max_length=256):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=False)
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids","attention_mask")}
        out = mdl(**enc)
        vec = mean_pool_hidden(out, enc["attention_mask"])[0].detach().cpu().numpy()
    return vec.reshape(1, -1)

def find_token_span(sentence, target):
    m = re.search(rf"\b{re.escape(target)}\b", sentence, flags=re.IGNORECASE)
    if not m: return None, None
    return m.start(), m.end()

def embed_span_in_context(sentence, span, tok, mdl, device, max_length=256):
    with torch.no_grad():
        enc = tok(sentence, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
        input_ids = enc["input_ids"].to(device); attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0].tolist()
        out = mdl(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state[0]
        if span == (None, None):
            pooled = mean_pool_hidden(out, attention_mask)[0].detach().cpu().numpy()
            return pooled.reshape(1,-1)
        s_char, e_char = span
        token_idxs = [i for i,(s,e) in enumerate(offsets) if not (e <= s_char or s >= e_char)]
        token_idxs = [i for i in token_idxs if offsets[i] != (0,0)]
        if not token_idxs:
            pooled = mean_pool_hidden(out, attention_mask)[0].detach().cpu().numpy()
            return pooled.reshape(1,-1)
        span_emb = last_hidden[token_idxs].mean(dim=0).detach().cpu().numpy()
        return span_emb.reshape(1,-1)

def wn_pos_from_flag(pos_flag): return {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}.get((pos_flag or "n").lower(), wn.NOUN)

def biobert_wsd_sentence(sentence, target, tok, mdl, device, pos="n", topk=1):
    wn_pos = wn_pos_from_flag(pos)
    cands = wn.synsets(target, pos=wn_pos)
    if not cands: return None
    span = find_token_span(sentence, target)
    emb_ctx = embed_span_in_context(sentence, span, tok, mdl, device)
    best = None
    for s in cands:
        gloss = synset_text(s)
        vec = embed_text(gloss, tok, mdl, device)
        score = float(cosine_similarity(vec, emb_ctx)[0,0])
        if best is None or score > best[0]:
            best = (score, s)
    return best[1] if best else None

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msh_dir", required=True, help="Folder with *_pmids_tagged.arff and benchmark_mesh.txt")
    ap.add_argument("--cui_gloss_json", required=True, help="JSON mapping {CUI: 'definition ...'}")
    ap.add_argument("--subset_10_10", action="store_true", help="Evaluate random 10 acronyms + 10 regular terms")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--model", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--pos", default="n", choices=["n","v","a","r"])
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.msh_dir)
    bench_path = root / "benchmark_mesh.txt"
    if not bench_path.exists():
        sys.exit(f"benchmark_mesh.txt not found in {root}")

    term2cuis = load_benchmark(bench_path)
    cui2gloss = json.loads(Path(args.cui_gloss_json).read_text(encoding="utf-8"))

    idx = index_arff_files(root)
    term_instances = {}
    total = 0
    for term in term2cuis:
        path = idx.get(_normalize_name(term)) or idx.get(_normalize_name(term).replace(" ","_")) or idx.get(_normalize_name(term).replace(" ","-"))
        if not path or not path.exists(): continue
        rows = list(read_arff_instances(path))
        if not rows: continue
        term_instances[term] = rows
        total += len(rows)

    acr_terms = [t for t in term_instances if is_acronym(t)]
    reg_terms = [t for t in term_instances if not is_acronym(t)]

    print(f"[i] Terms in benchmark: {len(term2cuis)}")
    print(f"[i] Terms with ARFF found: {len(term_instances)}")
    print(f"[i] Total instances (all found terms): {total}")
    print(f"[i] Acronym terms: {len(acr_terms)} | Regular terms: {len(reg_terms)}")

    if args.subset_10_10:
        k_acr = min(10, len(acr_terms))
        k_reg = min(10, len(reg_terms))
        keep = set(random.sample(acr_terms, k_acr) + random.sample(reg_terms, k_reg))
        term_instances = {t:v for t,v in term_instances.items() if t in keep}
        acr_terms = [t for t in term_instances if is_acronym(t)]
        reg_terms = [t for t in term_instances if not is_acronym(t)]
        print(f"[i] Using subset → Acronyms: {len(acr_terms)} | Regular: {len(reg_terms)}")

    tok, mdl, device = load_model(args.model, fp16=args.fp16)
    print(f"[i] Device: {device} | Model: {args.model}")

    # Eval
    out_dir = root / "eval_ex4"
    out_dir.mkdir(exist_ok=True, parents=True)
    pred_path = out_dir / "predictions_BioBERT.jsonl"
    per_term = {}
    correct = 0; total_inst = 0

    with open(pred_path, "w", encoding="utf-8") as fpred:
        for term, rows in term_instances.items():
            cuis = term2cuis[term]
            term_correct = 0
            for text, gold in rows:
                syn = biobert_wsd_sentence(text, term, tok, mdl, device, pos=args.pos)
                pred = synset_to_Mlabel(syn, cuis, cui2gloss)
                ok = int(pred == gold)
                term_correct += ok
                short = (text[:160] + "…") if len(text) > 160 else text
                fpred.write(json.dumps({"term":term,"gold":gold,"pred":pred,"ok":ok,"text":short}, ensure_ascii=False) + "\n")
            per_term[term] = (term_correct, len(rows))
            correct += term_correct
            total_inst += len(rows)

    def macro(terms):
        accs = []
        for t in terms:
            c, n = per_term.get(t, (0,0))
            if n > 0: accs.append(c/n)
        return sum(accs)/len(accs) if accs else 0.0

    micro = correct / total_inst if total_inst else 0.0
    mac_all = macro(list(term_instances.keys()))
    mac_acr = macro(acr_terms)
    mac_reg = macro(reg_terms)

    csv_path = out_dir / "results_BioBERT.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["term","is_acronym","correct","total","accuracy"])
        for t,(c,n) in sorted(per_term.items()):
            w.writerow([t, int(is_acronym(t)), c, n, f"{(c/n if n else 0):.4f}"])

    print("\n[BioBERT (context→gloss cosine)]")
    print(f"  Micro accuracy: {micro:.3f}  ({correct}/{total_inst})")
    print(f"  Macro accuracy (by term): {mac_all:.3f}")
    print(f"    ├─ Acronyms: {mac_acr:.3f}")
    print(f"    └─ Regular : {mac_reg:.3f}")
    print(f"  ↳ wrote {csv_path.name} and {pred_path.name}")

if __name__ == "__main__":
    main()
