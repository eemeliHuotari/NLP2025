"""
Exercise 5:
1) Phonetic distance matrix for a sentence using fuzzy (Soundex/DMetaphone) + edit distance
2) A Lesk-style disambiguator that combines Wu-Palmer semantic similarity and phonetic similarity
   via a convex combination: 0.5 * wup + 0.5 * phonetic_sim (both normalized to [0,1])

Demo uses the sentence:
  "I have been prescribed two important drugs today during my visit to clinic"
Target word: "drug" (or "drugs")

Usage:
  python exc5.py
"""

import re

# --- NLTK setup ---
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt_tab")
for pkg in ["punkt", "wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
WUP_FALLBACK = 0.0

# --- fuzzy (Soundex/DMetaphone) + edit distance helpers ---
import fuzzy


def simple_soundex(word: str) -> str:
    """Tiny built-in Soundex fallback if `fuzzy` is missing."""
    word = re.sub(r"[^A-Za-z]", "", word.upper())
    if not word:
        return ""
    first = word[0]
    codes = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        "L": "4",
        **dict.fromkeys(list("MN"), "5"),
        "R": "6",
    }
    digits = [codes.get(ch, "") for ch in word[1:]]
    out = []
    prev = codes.get(first, "")
    for d in digits:
        if d != prev:
            out.append(d)
        if d != "":
            prev = d
    sdx = first + "".join(out)
    sdx = re.sub(r"[AEIOUYHW]", "", sdx)
    sdx = (sdx + "000")[:4]
    return sdx


def phonetic_codes(word: str):
    """Return a list of phonetic codes (best-first) for a word."""
    w = word.lower()
    if not w:
        return []
    codes = []
    try:
        dmeta = fuzzy.DMetaphone()
        dm = dmeta(w)
        dm = [c.decode("utf-8") for c in dm if c]
        codes.extend(dm)
    except Exception:
        pass
    try:
        sx = fuzzy.Soundex(4)(w)
        codes.append(sx)
    except Exception:
        pass

    seen = set()
    uniq = []
    for c in codes:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq or [""]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (iterative DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def normalized_phonetic_sim(w1: str, w2: str) -> float:
    """
    Phonetic similarity in [0,1], computed as:
      sim = 1 - (min_edit_distance_between_any_code_pair / max_len_of_that_pair)
    If both codes are empty, sim = 0.
    """
    codes1 = phonetic_codes(w1)
    codes2 = phonetic_codes(w2)
    best = 0.0
    for c1 in codes1:
        for c2 in codes2:
            L = max(len(c1), len(c2))
            if L == 0:
                cand = 0.0
            else:
                cand = 1.0 - (edit_distance(c1, c2) / L)
            if cand > best:
                best = cand
    return best


# --- Part 1: Phonetic distance matrix for a sentence ---
def phonetic_distance_matrix(sentence: str):
    """
    Returns (tokens, matrix) where matrix[i][j] = 1 - phonetic_sim(tokens[i], tokens[j]).
    """
    toks = [t for t in word_tokenize(sentence) if re.search(r"[A-Za-z]", t)]
    n = len(toks)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sim = normalized_phonetic_sim(toks[i], toks[j])
            M[i][j] = round(1.0 - sim, 3)
    return toks, M


def pretty_print_matrix(tokens, M):
    width = max(5, max(len(t) for t in tokens) + 1)
    head = " " * (width) + "".join(f"{t:>{width}}" for t in tokens)
    print(head)
    for t, row in zip(tokens, M):
        print(f"{t:>{width}}" + "".join(f"{v:>{width}.3f}" for v in row))


# --- Part 2: Combined Wu-Palmer + Phonetic Lesk scorer ---
def content_words(tokens):
    return [t for t in tokens if re.search(r"[A-Za-z]", t) and t.lower() not in STOP]


def best_context_synsets(tokens):
    """
    For each content word, pick its first noun synset as a rough proxy.
    (This is a lightweight stand-in for full context disambiguation.)
    """
    ctx = {}
    for w in content_words(tokens):
        syns = wn.synsets(w, pos="n")
        if syns:
            ctx[w] = syns[0]
    return ctx


def avg_wup(candidate_syn, ctx_syns):
    if not ctx_syns:
        return 0.0
    vals = []
    for w, s in ctx_syns.items():
        try:
            sim = candidate_syn.wup_similarity(s)
        except Exception:
            sim = None
        vals.append(sim if sim is not None else WUP_FALLBACK)
    return sum(vals) / len(vals)


def avg_phonetic_sim(candidate_syn, ctx_words):
    """
    Compare candidate synset lemma names vs each context word, take the best
    lemma-level match for each context word, then average across context words.
    """
    if not ctx_words:
        return 0.0
    lemmas = set()
    for ln in candidate_syn.lemma_names():
        lemmas.update(re.split(r"[_\s]+", ln))
    if not lemmas:
        return 0.0
    vals = []
    for w in ctx_words:
        best = 0.0
        for ln in lemmas:
            best = max(best, normalized_phonetic_sim(ln, w))
        vals.append(best)
    return sum(vals) / len(vals)


def combined_score(candidate_syn, ctx_words, ctx_syns, alpha=0.5):
    """
    convex combination: alpha * wup + (1-alpha) * phonetic_sim
    alpha=0.5 per assignment spec.
    """
    wup = avg_wup(candidate_syn, ctx_syns)
    phn = avg_phonetic_sim(candidate_syn, ctx_words)
    return alpha * wup + (1.0 - alpha) * phn


def disambiguate_with_combined(target_word: str, sentence: str, alpha=0.5):
    """
    Score all noun senses of target_word by combined similarity against sentence context.
    Returns (best_synset, scored_list)
    """
    tokens = word_tokenize(sentence)
    ctx_words = content_words(tokens)
    cand_syns = wn.synsets(target_word, pos="n") or wn.synsets(
        re.sub(r"s$", "", target_word), pos="n"
    )
    ctx_syns = best_context_synsets(tokens)

    scored = []
    for syn in cand_syns:
        score = combined_score(syn, ctx_words, ctx_syns, alpha=alpha)
        scored.append((score, syn))
    scored.sort(reverse=True, key=lambda x: x[0])

    best = scored[0] if scored else (0.0, None)
    return best, scored


# --- Demo---
if __name__ == "__main__":
    sent = "I have been prescribed two important drugs today during my visit to clinic"

    print("Phonetic distance matrix")
    tokens, M = phonetic_distance_matrix(sent)
    pretty_print_matrix(tokens, M)

    print("Combined Wu–Palmer + Phonetic Lesk")
    target = "drugs"  # plural in sentence; the code will try lemmas as well
    (best_score, best_syn), scored = disambiguate_with_combined(target, sent, alpha=0.5)

    print(f"\nTarget word: {target}")
    if best_syn:
        print(f"Predicted sense: {best_syn.name()}  |  score={best_score:.3f}")
        print(f"Definition   : {best_syn.definition()}")
        # Show a few runners-up
        for s, syn in scored[1:4]:
            print(f"  runner-up  : {syn.name():<30}  score={s:.3f}")
    else:
        print("No noun synsets found")
