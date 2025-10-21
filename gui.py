import string
import tkinter as tk
from tkinter import ttk
import nltk

from pywsd.lesk import original_lesk, adapted_lesk, simple_lesk, cosine_lesk
from pywsd.baseline import random_sense, first_sense, max_lemma_count
from Exercise_3 import run_path_sim, try_pyswd_ic, ic_similarity_fallback

# ensure required NLTK data is present
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

input_sentence = ""
tokens = []
target_word = ""

# prepare stopwords set
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """Map Treebank POS tags to WordNet POS tags for lemmatization."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def format_synset(syn):
    if syn is None:
        return "None"
    try:
        return f"{syn.name()} — {syn.definition()}"
    except Exception:
        return str(syn)


def show_process(original, without_stop_punct, lemmatized):
    process_text.config(state="normal")
    process_text.delete("1.0", "end")
    lines = [
        "Original sentence:",
        original,
        "",
        "After removing stopwords and punctuation:",
        " ".join(without_stop_punct) if without_stop_punct else "(none)",
        "",
        "Lemmatized words (shown in dropdown <-):",
        " ".join(lemmatized) if lemmatized else "(none)",
    ]
    process_text.insert("1.0", "\n".join(lines))
    process_text.config(state="disabled")


def submit_text():
    global input_sentence, tokens, target_word
    input_sentence = sentence_input_field.get("1.0", "end-1c").strip()
    if not input_sentence:
        word_selection_combo["values"] = []
        word_selection_combo.set("")
        target_word = ""
        show_process("", [], [])
        return

    raw_tokens = nltk.tokenize.word_tokenize(input_sentence)

    # remove punctuation tokens and stopwords
    without_stop_punct = [
        token
        for token in raw_tokens
        if (token.lower() not in STOPWORDS and token.lower() not in string.punctuation)
    ]

    # POS tag and lemmatize
    pos_tags = nltk.pos_tag(without_stop_punct)
    lemmatized = [
        LEMMATIZER.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags
    ]

    # set combobox values to lemmatized words
    tokens = lemmatized
    word_selection_combo["values"] = tokens
    word_selection_combo.set("")
    target_word = ""

    # clear previous results (disambiguation results)
    results_text.config(state="normal")
    results_text.delete("1.0", "end")
    results_text.config(state="disabled")

    # show process on the right side
    show_process(input_sentence, without_stop_punct, lemmatized)


def disambiguate():
    if not input_sentence:
        show_message("No sentence submitted.")
        return
    if not target_word:
        show_message("No word selected.")
        return

    lesk_original = original_lesk(input_sentence, target_word)
    lesk_adapted = adapted_lesk(input_sentence, target_word)
    lesk_simple = simple_lesk(input_sentence, target_word)
    lesk_simple_hypo = simple_lesk(input_sentence, target_word, hyperhypo=True)
    lesk_cosine = cosine_lesk(input_sentence, target_word)

    path_sims = []
    for measure in ["path", "wup", "lch"]:
        path_sims.append(run_path_sim(target_word, input_sentence, measure, pos="n"))

    ic_sims = []
    for measure in ["res", "jcn", "lin"]:
        val = try_pyswd_ic(measure, target_word, input_sentence)
        if val is not None:
            ic_sims.append(val)
        else:
            _, best_target = ic_similarity_fallback(
                target_word, input_sentence, which=measure, target_pos="n"
            )
            ic_sims.append(best_target)

    output = [
        f"Target word: {target_word}",
        "Original Lesk:\n" + format_synset(lesk_original),
        "Adapted Lesk:\n" + format_synset(lesk_adapted),
        "Simple Lesk:\n" + format_synset(lesk_simple),
        "Simple Lesk + Hypo:\n" + format_synset(lesk_simple_hypo),
        "Cosine Lesk:\n" + format_synset(lesk_cosine),
        "========================================================================================================================",
        "Path Similarity:\n" + format_synset(path_sims[0]),
        "WUP Similarity:\n" + format_synset(path_sims[1]),
        "LCH Similarity:\n" + format_synset(path_sims[2]),
        "========================================================================================================================",
        "IC RES Similarity:\n" + format_synset(ic_sims[0]),
        "IC JCN Similarity:\n" + format_synset(ic_sims[1]),
        "IC LIN Similarity:\n" + format_synset(ic_sims[2]),
        "========================================================================================================================",
        "Random Sense:\n" + format_synset(random_sense(target_word)),
        "First Sense:\n" + format_synset(first_sense(target_word)),
        "Highest Lemma Count:\n" + format_synset(max_lemma_count(target_word)),
    ]

    results_text.config(state="normal")
    results_text.delete("1.0", "end")
    results_text.insert("1.0", "\n".join(output))
    results_text.config(state="disabled")


def show_message(msg):
    results_text.config(state="normal")
    results_text.delete("1.0", "end")
    results_text.insert("1.0", msg)
    results_text.config(state="disabled")


def on_combo_select(event):
    global target_word
    target_word = word_selection_combo.get()
    disambiguate()


def clear_all():
    sentence_input_field.delete("1.0", "end")
    results_text.config(state="normal")
    results_text.delete("1.0", "end")
    results_text.config(state="disabled")
    process_text.config(state="normal")
    process_text.delete("1.0", "end")
    process_text.config(state="disabled")
    word_selection_combo["values"] = []
    word_selection_combo.set("")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Text Disambiguation")
    # dark purple app background
    root.configure(bg="#2E004F")

    # ttk style so Combobox field is white
    style = ttk.Style()
    try:
        style.theme_use("default")
    except Exception:
        pass
    style.configure(
        "White.TCombobox",
        fieldbackground="white",
        background="white",
        foreground="black",
    )
    style.map("White.TCombobox", fieldbackground=[("readonly", "white")])

    # left: input and result frame
    main_frame = tk.Frame(root, bg=root["bg"])
    main_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

    input_sentence_label = tk.Label(
        main_frame, text="Write a sentence:", fg="white", bg=root["bg"]
    )
    input_sentence_label.grid(row=0, column=0, columnspan=2, sticky="w")

    sentence_input_field = tk.Text(
        main_frame,
        height=6,
        width=120,
        border=1,
        bg="white",
        fg="black",
        insertbackground="black",
    )
    sentence_input_field.grid(row=1, column=0, columnspan=2, sticky="w")

    # input text buttons
    button_frame = tk.Frame(main_frame, bg=root["bg"])
    button_frame.grid(row=2, column=0, sticky="w", pady=(8, 0))

    submit_btn = tk.Button(button_frame, text="Submit", command=submit_text)
    submit_btn.pack(side="left")

    clear_btn = tk.Button(button_frame, text="Clear", command=clear_all)
    clear_btn.pack(side="left", padx=(8, 0))

    # target word selection
    word_selection_frame = tk.Frame(main_frame, bg=root["bg"])
    word_selection_frame.grid(row=3, column=0, sticky="w", pady=(8, 0))

    word_selection_label = tk.Label(
        word_selection_frame,
        text="Choose word to disambiguate: ",
        fg="white",
        bg=root["bg"],
    )
    word_selection_label.pack(side="left")

    word_selection_combo = ttk.Combobox(
        word_selection_frame,
        state="readonly",
        style="White.TCombobox",
    )
    word_selection_combo.bind("<<ComboboxSelected>>", on_combo_select)
    word_selection_combo.pack(side="left", padx=(0, 8))

    # result display
    results_text = tk.Text(
        main_frame,
        height=40,
        width=120,
        state="disabled",
        wrap="word",
        border=1,
        bg="white",
        fg="black",
        insertbackground="black",
    )
    results_text.grid(row=4, column=0, sticky="w", pady=(8, 0))

    # right: process display
    process_frame = tk.Frame(root, bg=root["bg"])
    process_frame.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

    process_label = tk.Label(
        process_frame,
        text="Process",
        fg="white",
        bg=root["bg"],
    )
    process_label.grid(row=0, column=0, columnspan=2, sticky="w")

    process_text = tk.Text(
        process_frame,
        height=51,
        width=80,
        state="disabled",
        wrap="word",
        border=1,
        bg="white",
        fg="black",
        insertbackground="black",
    )
    process_text.grid(row=1, column=0, sticky="w")

    root.bind("<Escape>", lambda event: root.destroy())

    root.mainloop()
