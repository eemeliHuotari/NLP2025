"""
Run multiple pywsd algorithms over MSH-WSD and report accuracy

Usage:
  python task_4.py --msh_dir "C:\\...\\MSHCorpus" --cui_gloss_json "C:\\...\\mesh_gloss.json"

(Optionally limit to a 10+10 subset)
  python task_4.py --msh_dir ... --cui_gloss_json ... --subset_10_10
"""

import argparse, sys, re, json, random, csv, io
from pathlib import Path
from collections import Counter

# -------- NLTK/pywsd imports & downloads ----------
import nltk

nltk.download("punkt_tab")
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("wordnet_ic", quiet=True)

from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.tokenize import word_tokenize

from pywsd.lesk import original_lesk, adapted_lesk, simple_lesk, cosine_lesk
from pywsd.similarity import max_similarity
from pywsd.baseline import max_lemma_count

# -------- ARFF reader (handles extra PMID column) ----------
import csv as _csv

CLASS_RE = re.compile(r"^M\d+$")


def read_arff_instances(path: Path):
    raw = path.read_text(encoding="latin-1", errors="ignore")
    parts = re.split(r"(?i)\n@data\s*\n", raw, maxsplit=1)
    if len(parts) < 2:
        return
    data_part = parts[1]
    reader = _csv.reader(io.StringIO(data_part))
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


# -------- benchmark loader ----------
def load_benchmark(benchmark_path: Path):
    mapping = {}
    with open(benchmark_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            term, cuis = parts[0], parts[1:]
            mapping[term] = cuis
    return mapping


# -------- file name normalization / indexing ----------
_NORM_SPACE_RE = re.compile(r"[\s_\-]+", flags=re.UNICODE)


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


# -------- helpers ----------
ACRONYM_RE = re.compile(r"^[A-Z0-9\-]{2,6}$")


def is_acronym(term: str) -> bool:
    return bool(ACRONYM_RE.match(term))


STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "without",
    "from",
    "that",
    "this",
    "these",
    "those",
    "as",
    "it",
    "its",
    "into",
}


def toks(s: str):
    return [w for w in re.findall(r"[A-Za-z0-9]+", s.lower()) if w not in STOP]


def overlap(a_tokens, b_tokens):
    return len(set(a_tokens) & set(b_tokens))


# -------- synset → M# mapping via gloss overlap ----------
def synset_to_Mlabel(syn, cuis_for_term, cui2gloss):
    """
    Build a small text bag for the synset and choose the candidate CUI whose gloss
    overlaps the most. If gloss missing, treat as empty; tie-break to earliest CUI (M1).
    """
    if syn is None:
        return "M1"
    syn_text = " ".join(
        [syn.definition()] + syn.examples() + [" ".join(syn.lemma_names())]
    )
    syn_tokens = toks(syn_text)
    best_i = 0
    best_score = -1
    for i, cui in enumerate(cuis_for_term):
        gl = cui2gloss.get(cui, "")
        sc = overlap(syn_tokens, toks(gl))
        if sc > best_score:
            best_score = sc
            best_i = i
    return f"M{best_i+1}"


# -------- algorithm wrappers (WordNet-based) ----------
def _call_lesk(fn, sent, term, want_pos=True):
    """
    Robust caller for pywsd lesk functions across versions:
    - try tokens + pos
    - then tokens (no pos)
    - then raw string + pos
    - then raw string (no pos)
    Returns a synset or None.
    """
    tokens = word_tokenize(sent)
    if want_pos:
        try:
            return fn(tokens, term, pos="n")
        except TypeError:
            pass
        except AttributeError:
            pass
    try:
        return fn(tokens, term)
    except (TypeError, AttributeError):
        pass
    if want_pos:
        try:
            return fn(sent, term, pos="n")
        except TypeError:
            pass
        except AttributeError:
            pass
    try:
        return fn(sent, term)
    except Exception:
        return None


def run_original_lesk(term, sent):
    return _call_lesk(original_lesk, sent, term, want_pos=True)


def run_adapted_lesk(term, sent):
    return _call_lesk(adapted_lesk, sent, term, want_pos=True)


def run_simple_lesk(term, sent):
    return _call_lesk(simple_lesk, sent, term, want_pos=True)


def run_cosine_lesk(term, sent):
    return _call_lesk(cosine_lesk, sent, term, want_pos=False)


# similarity (max over candidate synsets) — returns **a synset**
def run_maxsim(term, sent, measure):
    try:
        return max_similarity(sent, term, measure, pos="n")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--msh_dir",
        required=True,
        help="Folder with *_pmids_tagged.arff and benchmark_mesh.txt",
    )
    ap.add_argument(
        "--cui_gloss_json",
        required=True,
        help="JSON mapping {CUI: 'definition text ...'}",
    )
    ap.add_argument(
        "--subset_10_10",
        action="store_true",
        help="Evaluate on a random 10 acronyms + 10 regular terms",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.msh_dir)
    bench_path = root / "benchmark_mesh.txt"
    if not bench_path.exists():
        sys.exit(f"benchmark_mesh.txt not found in {root}")

    # load benchmark and glosses
    term2cuis = load_benchmark(bench_path)
    cui2gloss = json.loads(Path(args.cui_gloss_json).read_text(encoding="utf-8"))

    # index ARFFs and collect instances
    idx = index_arff_files(root)
    term_instances = {}
    total = 0
    for term in term2cuis:
        path = (
            idx.get(_normalize_name(term))
            or idx.get(_normalize_name(term).replace(" ", "_"))
            or idx.get(_normalize_name(term).replace(" ", "-"))
        )
        if not path or not path.exists():
            continue
        rows = list(read_arff_instances(path))
        if not rows:
            continue
        print(f"[match] {term} → {path.name}")
        term_instances[term] = rows
        total += len(rows)

    print(f"\n[i] Terms in benchmark: {len(term2cuis)}")
    print(f"[i] Terms with ARFF found: {len(term_instances)}")
    print(f"[i] Total instances (all found terms): {total}")

    acr_terms = [t for t in term_instances if is_acronym(t)]
    reg_terms = [t for t in term_instances if not is_acronym(t)]
    print(f"[i] Acronym terms: {len(acr_terms)} | Regular terms: {len(reg_terms)}")

    # optional 10+10 subset (sampling by term)
    if args.subset_10_10:
        k_acr = min(10, len(acr_terms))
        k_reg = min(10, len(reg_terms))
        keep = set(random.sample(acr_terms, k_acr) + random.sample(reg_terms, k_reg))
        term_instances = {t: v for t, v in term_instances.items() if t in keep}
        acr_terms = [t for t in term_instances if is_acronym(t)]
        reg_terms = [t for t in term_instances if not is_acronym(t)]
        print(
            f"[i] Using subset → Acronyms: {len(acr_terms)} | Regular: {len(reg_terms)}"
        )

    # IC file for res/jcn/lin
    try:
        brown_ic = wordnet_ic.ic("ic-brown.dat")
    except Exception:
        brown_ic = None

    # define algorithm runners that return predicted M# label
    def algo_predictors():
        algos = []

        def wrap(name, fn_synset):
            def _predict(term, text):
                syn = fn_synset(term, text)
                return synset_to_Mlabel(syn, term2cuis[term], cui2gloss)

            return (name, _predict)

        algos.append(wrap("Original Lesk", run_original_lesk))
        algos.append(wrap("Adapted Lesk", run_adapted_lesk))
        algos.append(wrap("Simple Lesk", run_simple_lesk))
        algos.append(wrap("Cosine Lesk", run_cosine_lesk))
        algos.append(
            wrap("MaxSim wup", lambda term, text: run_maxsim(term, text, "wup"))
        )
        algos.append(
            wrap("MaxSim lch", lambda term, text: run_maxsim(term, text, "lch"))
        )
        algos.append(
            wrap("MaxSim path", lambda term, text: run_maxsim(term, text, "path"))
        )
        if brown_ic is not None:
            algos.append(
                wrap(
                    "MaxSim res",
                    lambda term, text: run_maxsim(term, text, "res"),
                )
            )
            algos.append(
                wrap(
                    "MaxSim jcn",
                    lambda term, text: run_maxsim(term, text, "jcn"),
                )
            )
            algos.append(
                wrap(
                    "MaxSim lin",
                    lambda term, text: run_maxsim(term, text, "lin"),
                )
            )

        def run_max_lemma_count(term):
            try:
                from pywsd.baseline import max_lemma_count as _mlc

                return _mlc(term)
            except ValueError:
                pass
            except Exception:
                pass
            syns_n = wn.synsets(term, pos="n")
            if syns_n:
                return syns_n[0]
            syns_any = wn.synsets(term)
            if syns_any:
                return syns_any[0]
            return None

        algos.append(wrap("Highest Lemma Count", run_max_lemma_count))
        return algos

    predictors = algo_predictors()

    # --- Generate template for missing glosses in this subset ---
    missing_map = {}
    for t in term_instances:
        for cui in term2cuis[t]:
            if cui not in cui2gloss or not cui2gloss[cui].strip():
                missing_map[cui] = ""

    if missing_map:
        tmpl_path = Path(args.msh_dir) / "mesh_gloss_missing_template.json"
        with open(tmpl_path, "w", encoding="utf-8") as f:
            json.dump(missing_map, f, ensure_ascii=False, indent=2)
        print(
            f"[cov] Wrote template with {len(missing_map)} missing CUIs → {tmpl_path}"
        )

    # evaluate each algorithm
    out_dir = root / "eval_ex4"
    out_dir.mkdir(exist_ok=True, parents=True)

    def evaluate_algo(name, predict_fn):
        correct = 0
        total_inst = 0
        per_term = {}
        pred_path = out_dir / f"predictions_{re.sub('[^A-Za-z0-9]+','_',name)}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as fpred:
            for term, rows in term_instances.items():
                cuis = term2cuis[term]
                term_correct = 0
                for text, gold in rows:
                    pred = predict_fn(term, text)
                    ok = int(pred == gold)
                    term_correct += ok
                    short = (text[:160] + "…") if len(text) > 160 else text
                    fpred.write(
                        json.dumps(
                            {
                                "term": term,
                                "gold": gold,
                                "pred": pred,
                                "ok": ok,
                                "text": short,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                per_term[term] = (term_correct, len(rows))
                correct += term_correct
                total_inst += len(rows)

        def macro(terms):
            accs = []
            for t in terms:
                c, n = per_term.get(t, (0, 0))
                if n > 0:
                    accs.append(c / n)
            return sum(accs) / len(accs) if accs else 0.0

        micro = correct / total_inst if total_inst else 0.0
        mac_all = macro(list(term_instances.keys()))
        mac_acr = macro(acr_terms)
        mac_reg = macro(reg_terms)

        csv_path = out_dir / f"results_{re.sub('[^A-Za-z0-9]+','_',name)}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["term", "is_acronym", "correct", "total", "accuracy"])
            for t, (c, n) in sorted(per_term.items()):
                w.writerow([t, int(is_acronym(t)), c, n, f"{(c/n if n else 0):.4f}"])

        print(f"\n[{name}]")
        print(f"  Micro accuracy: {micro:.3f}  ({correct}/{total_inst})")
        print(f"  Macro accuracy (by term): {mac_all:.3f}")
        print(f"    ├─ Acronyms: {mac_acr:.3f}")
        print(f"    └─ Regular : {mac_reg:.3f}")
        print(f"  ↳ wrote {csv_path.name} and {pred_path.name}")

    for name, fn in predictors:
        evaluate_algo(name, fn)


if __name__ == "__main__":
    main()
