import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize


nltk.download('punkt_tab')
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

sent = "I have been prescribed two important drugs today during my visit to clinic"
tokens = word_tokenize(sent)

syn = lesk(tokens, 'drug', pos='n')   # classic Lesk
print("Predicted synset:", syn, f"→ {syn.definition() if syn else None}")
print("All candidate senses:")
for s in wn.synsets('drug', pos='n'):
    print("-", s.name(), ":", s.definition())
