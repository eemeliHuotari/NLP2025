import argparse, json, io, re
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

CLASS_RE = re.compile(r"^M\d+$")
SPACE_NORM = re.compile(r"[\s_\-]+")
E_TAG_RE = re.compile(r"</?e>")
def _norm(s): return SPACE_NORM.sub(" ", s.lower().strip().replace("_pmids_tagged","")).replace(".arff","").strip()

def read_arff_instances(path: Path):
    raw = path.read_text(encoding="latin-1", errors="ignore")
    parts = re.split(r"(?i)\n@data\s*\n", raw, maxsplit=1)
    if len(parts) < 2: return
    import csv
    rdr = csv.reader(io.StringIO(parts[1]))
    for row in rdr:
        if not row: continue
        row = [c.strip() for c in row if c is not None]
        gold=None
        for c in reversed(row):
            if CLASS_RE.match(c): gold=c; break
        if not gold: continue
        text = max([c for c in row if c!=gold], key=len)
        yield E_TAG_RE.sub("", text).replace('"',"'").strip(), gold

def tfidf_gloss(texts, topk=18):
    if not texts: return ""
    def pre(s):
        s=re.sub(r"[^a-z0-9\s]"," ", s.lower())
        s=re.sub(r"\s+"," ", s).strip()
        toks=[t for t in s.split() if len(t)>1]
        return " ".join(toks)
    vec=TfidfVectorizer(preprocessor=pre, ngram_range=(1,2), min_df=2, max_features=4000, sublinear_tf=True, norm="l2")
    X=vec.fit_transform(texts)
    import numpy as np
    scores=np.asarray(X.mean(axis=0)).ravel()
    feats=vec.get_feature_names_out()
    idx=np.argsort(-scores)
    out=[]; seen=set()
    for i in idx:
        term=feats[i]
        if term in seen or term.isdigit(): continue
        out.append(term); seen.add(term)
        if len(out)>=topk: break
    return "Key topics: " + "; ".join(out) if out else ""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--msh_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--topk", type=int, default=18)
    args=ap.parse_args()

    root=Path(args.msh_dir)
    bench=root/"benchmark_mesh.txt"
    if not bench.exists(): raise SystemExit("benchmark_mesh.txt not found")

    term2ids={}
    for line in bench.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip(): continue
        parts=line.strip().split("\t")
        term, ids = parts[0], parts[1:]
        term2ids[term]=ids

    idx={}
    for p in root.glob("*.arff"):
        k=_norm(p.name)
        idx[k]=p; idx[k.replace(" ","_")]=p; idx[k.replace(" ","-")]=p

    id2gloss={}
    for term, ids in term2ids.items():
        path = idx.get(_norm(term)) or idx.get(_norm(term).replace(" ","_")) or idx.get(_norm(term).replace(" ","-"))
        if not path or not path.exists(): continue
        label2texts=defaultdict(list)
        for text, lab in read_arff_instances(path):
            label2texts[lab].append(text)
        for i, id_ in enumerate(ids, start=1):
            texts = label2texts.get(f"M{i}", [])
            id2gloss[id_] = tfidf_gloss(texts, topk=args.topk)

    Path(args.out_json).write_text(json.dumps(id2gloss, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] Wrote {args.out_json} with {len(id2gloss)} IDs.")

if __name__=="__main__":
    main()
