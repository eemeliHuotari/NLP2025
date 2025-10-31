import argparse
import re
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

# ---------------------- Model ----------------------

def load_model(model_name="dmis-lab/biobert-v1.1", fp16=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    if fp16 and device.startswith("cuda"):
        model.half()
    model.eval()
    return tokenizer, model, device

# ---------------------- Embedding helpers ----------------------

def mean_pool_hidden(enc_out, attention_mask):
    """
    Mean pool over non-padding tokens.
    """
    last_hidden = enc_out.last_hidden_state  # [B, T, H]
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)     # [B, H]
    counts = mask.sum(dim=1).clamp(min=1.0)      # [B, 1]
    return (summed / counts)                     # [B, H]

def embed_text(text, tokenizer, model, device, max_length=256):
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length, return_offsets_mapping=False)
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids","attention_mask")}
        out = model(**enc)
        vec = mean_pool_hidden(out, enc["attention_mask"])[0].detach().cpu().numpy()
    return vec.reshape(1, -1)

def find_token_span(sentence, target):
    """
    Find a case-insensitive word-boundary match of target in sentence.
    Returns (start_char, end_char) or (None, None) if not found.
    """
    m = re.search(rf"\b{re.escape(target)}\b", sentence, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.start(), m.end()

def embed_span_in_context(sentence, span, tokenizer, model, device, max_length=256):
    """
    Embed the target span *in context* by averaging token vectors that overlap the character span.
    Falls back to mean pool if the span cannot be matched (e.g., due to truncation).
    """
    with torch.no_grad():
        enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"][0].tolist()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state[0]  # [T, H]

        if span == (None, None):
            # fallback: mean pool over sentence
            pooled = mean_pool_hidden(out, attention_mask)[0].detach().cpu().numpy()
            return pooled.reshape(1, -1)

        s_char, e_char = span
        token_idxs = [i for i, (s, e) in enumerate(offsets) if not (e <= s_char or s >= e_char)]
        # drop special tokens (offsets often (0,0))
        token_idxs = [i for i in token_idxs if offsets[i] != (0,0)]
        if not token_idxs:
            pooled = mean_pool_hidden(out, attention_mask)[0].detach().cpu().numpy()
            return pooled.reshape(1, -1)

        span_emb = last_hidden[token_idxs].mean(dim=0).detach().cpu().numpy()
        return span_emb.reshape(1, -1)

# ---------------------- WordNet candidates ----------------------

def wn_pos_from_flag(pos_flag):
    pos_flag = (pos_flag or "n").lower()
    return {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}.get(pos_flag, wn.NOUN)

def synset_text(s):
    """
    Build a rich gloss for a synset: lemma names + definition + examples.
    """
    parts = []
    lemmas = [l.replace("_", " ") for l in s.lemma_names()]
    if lemmas: parts.append("; ".join(lemmas))
    parts.append(s.definition())
    ex = s.examples()
    if ex: parts.append(" ".join(ex))
    return " ".join(parts)

# ---------------------- Core WSD ----------------------

def biobert_wsd_sentence(sentence, target, tokenizer, model, device, pos="n", topk=3):
    wn_pos = wn_pos_from_flag(pos)
    cands = wn.synsets(target, pos=wn_pos)
    if not cands:
        return [], None

    # 1) Embedding of target in context
    span = find_token_span(sentence, target)
    emb_ctx = embed_span_in_context(sentence, span, tokenizer, model, device)

    # 2) Embeddings of candidate gloss texts
    cand_embeds = []
    for s in cands:
        gloss = synset_text(s)
        vec = embed_text(gloss, tokenizer, model, device)
        cand_embeds.append((vec, s, gloss))

    # 3) Rank by cosine similarity
    sims = []
    for vec, s, gloss in cand_embeds:
        score = float(cosine_similarity(vec, emb_ctx)[0,0])
        sims.append((score, s, gloss))
    sims.sort(key=lambda x: x[0], reverse=True)

    best = sims[0] if sims else None
    return sims[:topk], best

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dmis-lab/biobert-v1.1")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA")
    ap.add_argument("--pos", default="n", choices=["n","v","a","r"], help="Target POS for WordNet lookup")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--target", required=True)
    ap.add_argument("--sents", nargs="+", required=True, help="One or more sentences")
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.model, fp16=args.fp16)
    print(f"[i] Device: {device} | Model: {args.model}")

    for sent in args.sents:
        sims, best = biobert_wsd_sentence(sent, args.target, tokenizer, model, device, pos=args.pos, topk=args.topk)
        print("\nSentence:", sent)
        print("Target  :", args.target)
        if not sims:
            print("No candidates found.")
            continue
        print("\nTop candidates (cosine):")
        for sc, s, gloss in sims:
            print(f"  {sc: .4f}  {s.name():<20} — {s.definition()}")
        if best:
            sc, s, gloss = best
            print("\nBest:")
            print(f"  {sc: .4f}  {s.name()} — {s.definition()}")

if __name__ == "__main__":
    main()
