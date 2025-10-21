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


# Tokenizer, taggers, WordNet + IC
ensure("punkt", "tokenizers")
# Install both tagger names
ensure("averaged_perceptron_tagger", "taggers")
try:
    ensure("averaged_perceptron_tagger_eng", "taggers")
except Exception:
    pass
ensure("wordnet", "corpora")
ensure("omw-1.4", "corpora")
ensure("wordnet_ic", "corpora")

# --- PyWSD imports ---
from pywsd.lesk import original_lesk, adapted_lesk, simple_lesk, cosine_lesk
from pywsd.baseline import random_sense, first_sense, max_lemma_count
from pywsd.similarity import max_similarity


# ------------- Helpers -------------
def synset_str(ss):
    if ss is None:
        return "None"
    return f"{ss.name()} :: {ss.definition()}"


def safe_adapted_lesk(sentence, target):
    try:
        return adapted_lesk(sentence, target, pos="n")
    except TypeError:
        return adapted_lesk(sentence, target)


def run_path_sim(target, sent, measure, pos):
    try:
        return max_similarity(target, sent, measure, pos=pos)
    except TypeError:
        try:
            return max_similarity(target, sent, measure)
        except Exception:
            return None


# Fallback IC sims
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


def ic_similarity(target, sentence, which="lin", target_pos="n"):
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


# ------------- Demo configuration -------------
sent = "The patient was prescribed a new drug at the clinic."
target = "drug"

print("Sentence:", sent)
print("Target  :", target)
print()

# 1) Lesk family
print("== Lesk-family WSD ==")
print("Original Lesk               :", synset_str(original_lesk(sent, target)))
print("Adapted/Extended Lesk       :", synset_str(safe_adapted_lesk(sent, target)))
print("Simple Lesk                 :", synset_str(simple_lesk(sent, target, pos="n")))
print(
    "Simple Lesk + hyper/hyponyms:",
    synset_str(simple_lesk(sent, target, pos="n", hyperhypo=True)),
)

# Cosine Lesk (vector-based)
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


def try_pyswd_ic(measure, target, sent, pos):
    try:
        return max_similarity(target, sent, measure, pos=pos)
    except Exception:
        return None


for measure in ["res", "jcn", "lin"]:
    val = try_pyswd_ic(measure, target, sent, pos="N")
    if val is not None:
        print(f"MaxSim ({measure})            :", val, "(pywsd)")
    else:
        score, best_target = ic_similarity(target, sent, which=measure, target_pos="n")
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
