import os
import re
import nltk
from nltk.corpus import wordnet as wn, wordnet_ic
from nltk import pos_tag, word_tokenize

# --- Ensure NLTK data ---
NLTK_DIR = os.path.expanduser("~/nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DIR)


def ensure(resource, subdir):
    try:
        nltk.data.find(f"{subdir}/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DIR, quiet=True)


ensure("punkt", "tokenizers")
ensure("averaged_perceptron_tagger", "taggers")
try:
    ensure("averaged_perceptron_tagger_eng", "taggers")
except Exception:
    pass
ensure("wordnet", "corpora")
ensure("omw-1.4", "corpora")
ensure("wordnet_ic", "corpora")
ensure("stopwords", "corpora")

# --- PyWSD imports ---
from pywsd.lesk import original_lesk, adapted_lesk, simple_lesk, cosine_lesk
from pywsd.baseline import random_sense, first_sense, max_lemma_count
from pywsd.similarity import max_similarity


# --- Helpers ----
def synset_str(ss):
    if ss is None:
        return "None"
    return f"{ss.name()} :: {ss.definition()}"


def safe_original_lesk(sentence, target):
    try:
        return original_lesk(sentence, target, pos="n")
    except TypeError:
        return original_lesk(sentence, target)


def safe_adapted_lesk(sentence, target):
    try:
        return adapted_lesk(sentence, target, pos="n")
    except TypeError:
        return adapted_lesk(sentence, target)


def run_path_sim(target, sent, measure, pos="n"):
    try:
        return max_similarity(target, sent, measure, pos=pos)
    except TypeError:
        try:
            return max_similarity(target, sent, measure)
        except Exception:
            return None


def penn2wn(tag):
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("R"):
        return wn.ADV
    return None


def context_synsets(sentence, target_pos):
    toks = word_tokenize(sentence)
    tags = pos_tag(toks)
    syns = []
    for w, t in tags:
        pos = penn2wn(t)
        if pos == target_pos:
            ss = wn.synsets(w, pos=pos)
            if ss:
                syns.extend(ss)
    return syns


def max_pairwise(sim_fn, cands, ctx):
    best_val, best_pair = None, None
    for s1 in cands:
        for s2 in ctx:
            try:
                val = sim_fn(s1, s2)
            except Exception:
                val = None
            if val is not None and (best_val is None or val > best_val):
                best_val, best_pair = val, (s1, s2)
    return best_val, best_pair


def ic_similarity_fallback(target, sentence, which="lin", target_pos="n"):
    pos_map = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}
    pos = pos_map[target_pos]
    target_syns = wn.synsets(target, pos=pos)
    ctx_syns = context_synsets(sentence, pos)
    if not target_syns or not ctx_syns:
        return None, None
    ic = wordnet_ic.ic("ic-brown.dat")
    if which == "lin":
        fn = lambda a, b: a.lin_similarity(b, ic)
    elif which == "res":
        fn = lambda a, b: a.res_similarity(b, ic)
    elif which == "jcn":
        fn = lambda a, b: a.jcn_similarity(b, ic)
    else:
        raise ValueError(which)
    val, pair = max_pairwise(fn, target_syns, ctx_syns)
    best_target = pair[0] if pair else None
    return val, best_target


# --Phonetic + WUP combo --
from nltk.corpus import stopwords

STOP = set(stopwords.words("english"))

try:
    import fuzzy

    _HAVE_FUZZY = True
except Exception:
    _HAVE_FUZZY = False


def _simple_soundex(word: str) -> str:
    w = re.sub(r"[^A-Za-z]", "", word.upper())
    if not w:
        return ""
    first = w[0]
    codes = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        "L": "4",
        **dict.fromkeys(list("MN"), "5"),
        "R": "6",
    }
    digits = [codes.get(ch, "") for ch in w[1:]]
    out = []
    prev = codes.get(first, "")
    for d in digits:
        if d != prev:
            out.append(d)
        if d != "":
            prev = d
    sdx = first + "".join(out)
    sdx = re.sub(r"[AEIOUYHW]", "", sdx)
    return (sdx + "000")[:4]


def _phonetic_codes(word: str):
    w = word.lower()
    if not w:
        return [""]
    codes = []
    if _HAVE_FUZZY:
        try:
            dmeta = fuzzy.DMetaphone()
            dm = dmeta(w)
            dm = [c.decode("utf-8") for c in dm if c]
            codes.extend(dm)
        except Exception:
            pass
        try:
            sx = fuzzy.Soundex(4)(w)
            if sx:
                codes.append(sx)
        except Exception:
            pass
    if not codes:
        codes.append(_simple_soundex(w))
    seen = set()
    uniq = []
    for c in codes:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq or [""]


def _edit_distance(a: str, b: str) -> int:
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


def _phonetic_sim(w1: str, w2: str) -> float:
    best = 0.0
    for c1 in _phonetic_codes(w1):
        for c2 in _phonetic_codes(w2):
            L = max(len(c1), len(c2)) or 1
            cand = 1.0 - (_edit_distance(c1, c2) / L)
            if cand > best:
                best = cand
    return best


def _content_words(tokens):
    return [t for t in tokens if re.search(r"[A-Za-z]", t) and t.lower() not in STOP]


def _best_context_synsets(tokens):
    """Pick first noun synset for each content word as a cheap proxy."""
    ctx = {}
    for w in _content_words(tokens):
        syns = wn.synsets(w, pos="n")
        if syns:
            ctx[w] = syns[0]
    return ctx


def _avg_wup(candidate_syn, ctx_syns):
    if not ctx_syns:
        return 0.0
    vals = []
    for _, s in ctx_syns.items():
        try:
            sim = candidate_syn.wup_similarity(s)
        except Exception:
            sim = None
        vals.append(sim if sim is not None else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def _avg_phonetic(candidate_syn, ctx_words):
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
            best = max(best, _phonetic_sim(ln, w))
        vals.append(best)
    return sum(vals) / len(vals) if vals else 0.0


def wup_phonetic_best_synset(target_word: str, sentence: str, alpha: float = 0.5):
    """
    Return (best_score, best_synset, scored_list) where
      score = alpha * avg_wup + (1-alpha) * avg_phonetic
    """
    tokens = word_tokenize(sentence)
    ctx_words = _content_words(tokens)
    ctx_syns = _best_context_synsets(tokens)
    cand_syns = wn.synsets(target_word, pos="n") or wn.synsets(
        re.sub(r"s$", "", target_word), pos="n"
    )
    scored = []
    for syn in cand_syns:
        wup = _avg_wup(syn, ctx_syns)
        phn = _avg_phonetic(syn, ctx_words)
        sc = alpha * wup + (1.0 - alpha) * phn
        scored.append((sc, wup, phn, syn))
    scored.sort(key=lambda x: x[0], reverse=True)
    return (scored[0] if scored else (0.0, 0.0, 0.0, None)), scored


# ------------- Demo configuration -------------
sent = "The patient was prescribed a new drug at the clinic."
target = "drug"

print("Sentence:", sent)
print("Target  :", target)
print()

# 1) Lesk
print("== Lesk-family WSD ==")
print("Original Lesk               :", synset_str(safe_original_lesk(sent, target)))
print("Adapted/Extended Lesk       :", synset_str(safe_adapted_lesk(sent, target)))
print("Simple Lesk                 :", synset_str(simple_lesk(sent, target, pos="n")))
print(
    "Simple Lesk + hyper/hyponyms:",
    synset_str(simple_lesk(sent, target, pos="n", hyperhypo=True)),
)

try:
    print(
        "Cosine Lesk                 :", synset_str(cosine_lesk(sent, target, pos="n"))
    )
except TypeError:
    print("Cosine Lesk                 :", synset_str(cosine_lesk(sent, target)))

print()

# 2) Path-based semantic similarity
print("== Path-based similarity (max_similarity) ==")
for measure in ["path", "wup", "lch"]:
    val = run_path_sim(target, sent, measure, pos="n")
    print(f"MaxSim ({measure})            :", val)

print()

# 3) IC-based semantic similarity
print("== Information-content similarity ==")


def try_pyswd_ic(measure):
    try:
        return max_similarity(target, sent, measure, pos="n")
    except TypeError:
        try:
            return max_similarity(target, sent, measure, pos="n")
        except TypeError:
            try:
                return max_similarity(target, sent, measure, pos="n")
            except Exception:
                return None
    except Exception:
        return None


for measure in ["res", "jcn", "lin"]:
    val = try_pyswd_ic(measure)
    if val is not None:
        print(f"MaxSim ({measure})            :", val, "(pywsd)")
    else:
        score, best_target = ic_similarity_fallback(
            target, sent, which=measure, target_pos="n"
        )
        print(f"MaxSim ({measure})            :", score, "(fallback via NLTK)")
        if best_target:
            print(
                f"  ↳ best target synset       : {best_target.name()} :: {best_target.definition()}"
            )

print()

# 4) Baselines
print("== Baselines ==")
print("Random Sense                 :", synset_str(random_sense(target, pos="n")))
print("First Sense                  :", synset_str(first_sense(target, pos="n")))

res = max_lemma_count(target)
print("Highest Lemma Count          :", synset_str(res))

# 5) Combined Wu–Palmer + Phonetic
print("\n== WUP + Phonetic (α = 0.5) ==")
(best_sc, best_wup, best_phn, best_syn), scored = wup_phonetic_best_synset(
    target, sent, alpha=0.5
)
print(f"Best synset                  : {synset_str(best_syn)}")
print(
    f"Combined score               : {best_sc:.3f}  (wup={best_wup:.3f}, phonetic={best_phn:.3f})"
)

for sc, wup, phn, ss in scored[1:4]:
    print(f"  runner-up                  : {synset_str(ss)}")
    print(
        f"    scores                   : combined={sc:.3f}, wup={wup:.3f}, phonetic={phn:.3f}"
    )
