import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize


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


def get_span_embedding(
    text, span_start_char, span_end_char, tokenizer, model, device, max_length=256
):
    """
    Embed in context
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state[0]

    token_idxs = [
        i
        for i, (s, e) in enumerate(offsets)
        if not (e <= span_start_char or s >= span_end_char)
    ]
    if not token_idxs:
        # fallback: pick first non-special token
        token_idxs = [i for i in range(1, len(offsets) - 1)]

    span_emb = last_hidden[token_idxs].mean(dim=0).cpu()
    return np.array(span_emb).reshape(1, -1)


def get_isolated_embed(
    sent, span_start, span_end, tokenizer, model, device, max_length=64
):
    """
    Isolated embed out of context
    """
    word = sent[span_start:span_end]
    enc = tokenizer(
        word,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state[0]

    # pick non-special tokens using offsets (skip tokens with offset (0,0))
    token_idxs = [i for i, (s, e) in enumerate(offsets) if not (s == 0 and e == 0)]
    if not token_idxs:
        token_idxs = [i for i in range(1, last_hidden.size(0) - 1)]

    emb = last_hidden[token_idxs].mean(dim=0).cpu()
    return np.array(emb).reshape(1, -1)


def compare_embeds(target_word: str, sents: list[str], tokenizer, model, device):
    # get the synsets for the target word
    syns = wn.synsets("drug")

    # embed the word in teh synset definitnon
    syn_embeds = []

    for syn in syns:
        if syn:
            definition = syn.definition()
            print(definition)
            start = definition.find(target_word)
            syn_embeds.append(
                [
                    get_span_embedding(
                        syn.definition(),
                        start,
                        start + len(target_word),
                        tokenizer,
                        model,
                        device,
                    ),
                    syn,
                ]
            )

    for sent in sents:
        start = sent.find(target_word)

        emb_in_context = get_span_embedding(
            sent, start, start + len(target_word), tokenizer, model, device
        )

        # Get the highest similarity, and it associated synset
        sims = sorted(
            [(cosine_similarity(emb, emb_in_context), syn) for emb, syn in syn_embeds],
            reverse=True,
        )
        print(sims)
        best_syn = sims[0]

        print("")
        print(f"Original sentence: {sent}")

        print(
            f"Best synset by embedding: {best_syn[0][0][0]:.4f} -> {best_syn[1].name()} : {best_syn[1].definition()}"
        )
        print(
            "------------------------------------------------------------------------------"
        )


def main():
    tokenizer, model, device = load_model()
    sents = [
        "The patient was prescribed a new drug at the clinic.",
        "Someone tried to drug me at the party yesterday.",
        "I picked up a drug from the pharmacy.",
        "Fentanyl is used as a recreational drug.",
    ]
    target_word = "drug"
    compare_embeds(target_word, sents, tokenizer, model, device)


if __name__ == "__main__":
    main()
