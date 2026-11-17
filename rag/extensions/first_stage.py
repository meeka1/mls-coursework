"""First-stage alternatives to enc_a dense search: BM25, hybrid and enc_a + enc_b fusion.

Prints, for k = 5, 10, 20, 50, 100: the share of queries with at least one relevant record
in the top k, and Recall@k. No ELM calls. Uses rag/.cache/{split}_transcripts.jsonl and
rag/.cache/{split}_queries_enc_a.npy (created by selective_rerank.py or migration.py).

    python rag/extensions/first_stage.py --split dev
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
import argparse, os
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--split", choices=["practice", "dev"], default="dev")
args = ap.parse_args()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.data import CourseData
import json, collections, time, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
e='data/rag_corpus/'
ids=json.load(open(e+'embeddings/corpus_enc_a_ids.json')); row={r:i for i,r in enumerate(ids)}
A=np.load(e+'embeddings/corpus_enc_a.npy').astype(np.float32)
idb=json.load(open(e+'embeddings/corpus_enc_b_ids.json')); rb={r:i for i,r in enumerate(idb)}
B=np.load(e+'embeddings/corpus_enc_b.npy').astype(np.float32)[[rb[r] for r in ids]]
text={}
for l in open(e+'corpus.jsonl'):
    r=json.loads(l); text[r['record_id']]=r['text']
docs=[text[r] for r in ids]
qr=collections.defaultdict(set)
for l in list(open(e+f'qrels_{args.split}.tsv'))[1:]:
    q,r,s=l.split('\t'); qr[q].add(r)
tr={r['query_id']:r['transcript'] for r in map(json.loads,open(f'.cache/{args.split}_transcripts.jsonl'))}
qids=[it['query_id'] for it in CourseData(e).audio_items(args.split) if it['query_id'] in tr]; qtext=[tr[q] for q in qids]
Q=np.load(f'.cache/{args.split}_queries_enc_a.npy')

# BM25
t0=time.time()
cv=CountVectorizer(lowercase=True, stop_words='english', token_pattern=r"(?u)\b\w+\b")
tf=cv.fit_transform(docs).tocsc().astype(np.float32)
N=tf.shape[0]; df=np.diff(tf.indptr); idf=np.log(1+(N-df+.5)/(df+.5)).astype(np.float32)
dl=np.asarray(tf.sum(1)).ravel(); k1,b=1.2,0.75
tf=tf.tocoo(); denom=tf.data+k1*(1-b+b*dl[tf.row]/dl.mean())
W=sp.csr_matrix((tf.data*(k1+1)/denom*idf[tf.col],(tf.row,tf.col)),shape=tf.shape)
qv=(cv.transform(qtext)>0).astype(np.float32)
S_bm=np.asarray((W@qv.T).todense()); build=time.time()-t0
S_a=A@Q.T
S_b=B@Q.T
def rrf(*Ss,k=60,w=None):
    out=np.zeros_like(Ss[0]); w=w or [1]*len(Ss)
    for S,wi in zip(Ss,w):
        rk=np.argsort(np.argsort(-S,0),0)+1
        out+=wi/(k+rk)
    return out
def z(S): return (S-S.mean(0))/S.std(0)
def report(name,S):
    top=np.argsort(-S,0)[:100]; out=[]
    for k in (5,10,20,50,100):
        anyg=np.mean([bool(qr[q]&{ids[i] for i in top[:k,j]}) for j,q in enumerate(qids)])
        rec=np.mean([len(qr[q]&{ids[i] for i in top[:k,j]})/len(qr[q]) for j,q in enumerate(qids)])
        out.append(f'{anyg:4.0%}/{rec:.2f}')
    print(f'{name:28s}'+'  '.join(out))
print(f'BM25 index build {build:.1f}s, vocab {len(cv.vocabulary_)}, matrix {W.data.nbytes/1e6:.0f} MB')
print(f'{"system":28s}'+'  '.join(f'@{k:<9}' for k in (5,10,20,50,100))+'   (any-gold / Recall)')
report('enc_a (baseline)',S_a)
report('enc_b raw',S_b)
report('BM25',S_bm)
report('RRF enc_a + BM25',rrf(S_a,S_bm))
report('RRF enc_a + enc_b',rrf(S_a,S_b))
report('RRF enc_a + enc_b + BM25',rrf(S_a,S_b,S_bm))
for a in (0.2,0.3,0.5):
    report(f'z-sum enc_a + {a}*BM25',z(S_a)+a*z(S_bm))
report('z-sum enc_a+enc_b+0.3BM25',z(S_a)+z(S_b)+0.3*z(S_bm))
