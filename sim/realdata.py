"""Real-data checks for the survey (added in the V5 revision).

1. trace: replays the public MovieTweetings rating stream (Dooms et al., 2013) as a
   request trace for one edge cache, and compares the reactive and proactive policies
   of RQ1 (LRU, LFU, aged LFU, TinyLFU-style admission; last-epoch, window, EWMA,
   oracle placement once per epoch).
2. sts: measures how cosine similarity of an off-the-shelf sentence representation
   tracks meaning on the STS Benchmark test set, i.e. the precision of semantic reuse
   at a threshold tau, instead of assuming class structure.

Data are not stored in the repository. Download them first:
  curl -sL -o data/ratings.dat https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat
  curl -sL -o data/stsb-en-test.csv https://raw.githubusercontent.com/PhilipMay/stsb-multi-mt/main/data/stsb-en-test.csv
  pip install spacy https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
Usage: python3 realdata.py trace | sts
"""
import collections
import csv
import json
import sys

import numpy as np

DATA = 'data/'


def load_trace():
    items, ts = [], []
    for line in open(DATA + 'ratings.dat'):
        a = line.strip().split('::')
        items.append(a[1])
        ts.append(int(a[3]))
    order = np.argsort(ts, kind='stable')
    ids = {}
    req = np.array([ids.setdefault(items[i], len(ids)) for i in order])
    return req, np.array(ts)[order], len(ids)


def replay(req, ep, F, C, policy, alpha=0.5, W=3):
    """Hit ratio of one cache of size C over the trace; ep[t] is the epoch of request t."""
    hits = 0
    if policy in ('lru', 'lfu', 'lfua', 'tlfu'):
        cache = collections.OrderedDict()
        cnt = collections.Counter()
        cur = ep[0]
        for r, e in zip(req, ep):
            if policy in ('lfua', 'tlfu') and e != cur:
                for _ in range(e - cur):           # halve all counts once per elapsed epoch
                    for k in list(cnt):
                        cnt[k] //= 2
                        if cnt[k] == 0:
                            del cnt[k]
                cur = e
            cnt[r] += 1
            if r in cache:
                hits += 1
                if policy in ('lru', 'tlfu'):
                    cache.move_to_end(r)
                continue
            if len(cache) < C:
                cache[r] = True
                continue
            if policy == 'lru':
                cache.popitem(last=False); cache[r] = True
            elif policy == 'tlfu':
                victim = next(iter(cache))
                if cnt[r] > cnt[victim]:
                    del cache[victim]; cache[r] = True
            else:  # lfu, lfua: replace least counted if the new item is more frequent
                victim = min(cache, key=lambda k: cnt[k])
                if cnt[r] > cnt[victim]:
                    del cache[victim]; cache[r] = True
        return hits / len(req)
    # proactive: place top-C of an estimate at the start of each epoch
    E = ep.max() + 1
    counts = np.zeros((E, F))
    np.add.at(counts, (ep, req), 1)
    est = np.zeros(F)
    hist = []
    for e in range(E):
        if policy == 'oracle':
            score = counts[e]
        elif policy == 'stale':
            score = counts[e - 1] if e else np.zeros(F)
        elif policy == 'window':
            score = sum(hist[-W:]) if hist else np.zeros(F)
        else:  # ewma
            score = est
        top = np.argpartition(-score, C)[:C] if score.any() else np.array([], int)
        hits += counts[e, top].sum()
        hist.append(counts[e])
        est = alpha * counts[e] + (1 - alpha) * est
    return hits / len(req)


def trace():
    req, ts, F = load_trace()
    out = dict(F=F, R=len(req), years=[int(ts[0]), int(ts[-1])], runs=[])
    for days in (1, 7, 28):
        ep = ((ts - ts[0]) // (86400 * days)).astype(int)
        rpi = len(req) / (ep.max() + 1) / F
        for C in (100, 400):
            row = dict(days=days, C=C, req_per_item_epoch=rpi)
            for pol in ('oracle', 'stale', 'window', 'ewma', 'lru', 'lfu', 'lfua', 'tlfu'):
                row[pol] = replay(req, ep, F, C, pol)
            print(row, flush=True)
            out['runs'].append(row)
    json.dump(out, open('res5_trace.json', 'w'), indent=1)


def sts():
    import spacy
    nlp = spacy.load('en_core_web_md')
    rows = list(csv.reader(open(DATA + 'stsb-en-test.csv')))
    s1 = [nlp(r[0]).vector for r in rows]
    s2 = [nlp(r[1]).vector for r in rows]
    gold = np.array([float(r[2]) for r in rows])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0
    sim = np.array([cos(a, b) for a, b in zip(s1, s2)])
    same = gold >= 4.0                                  # "same meaning" in STS-B annotation
    out = dict(n=len(rows), share_same=float(same.mean()),
               spearman=float(np.corrcoef(np.argsort(np.argsort(sim)), np.argsort(np.argsort(gold)))[0, 1]),
               curve=[])
    for tau in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
        sel = sim >= tau
        out['curve'].append(dict(tau=tau, reuse=float(sel.mean()),
                                 precision=float(same[sel].mean()) if sel.any() else None,
                                 recall=float((sel & same).sum() / same.sum())))
    print(json.dumps(out, indent=1))
    json.dump(out, open('res5_sts.json', 'w'), indent=1)


def _pairs():
    """Labelled pairs: STS-B test (same meaning = gold >= 4) and a Quora duplicate-question sample."""
    sts = [(r[0], r[1], float(r[2]) >= 4.0, float(r[2])) for r in csv.reader(open(DATA + 'stsb-en-test.csv'))]
    qqp, seen = [], set()
    for r in csv.reader(open(DATA + 'qqp_dev_sample.tsv'), delimiter='\t'):
        if r[0] == 'id' or len(r) < 6: continue
        if (r[3], r[4]) in seen: continue
        seen.add((r[3], r[4])); qqp.append((r[3], r[4], r[5].strip() == '1', None))
    for r in csv.DictReader(open(DATA + 'quora_first1000.csv')):
        k = (r['question1_en'], r['question2_en'])
        if k in seen: continue
        seen.add(k); qqp.append((k[0], k[1], r['is_duplicate'].strip() == '1', None))
    return dict(stsb=sts, qqp=qqp)


def _auc(score, y):
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    p = y.sum(); n = len(y) - p
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def sts2():
    """Similarity versus meaning for three encoders: averaged spaCy word vectors, all-MiniLM-L6-v2, all-mpnet-base-v2.
    The two sentence encoders are loaded from local copies (env MINILM, MPNET); see README for their source."""
    import os
    data = _pairs()
    encs = {}
    import spacy
    nlp = spacy.load('en_core_web_md')
    encs['spacy-avg'] = lambda xs: np.array([nlp(x).vector for x in xs])
    from sentence_transformers import SentenceTransformer
    for name, env in (('MiniLM-L6', 'MINILM'), ('MPNet-base', 'MPNET')):
        m = SentenceTransformer(os.environ[env], device='cpu')
        encs[name] = (lambda mm: (lambda xs: mm.encode(xs, batch_size=64, normalize_embeddings=True)))(m)
    out = {}
    for dn, pairs in data.items():
        y = np.array([p[2] for p in pairs]); out[dn] = dict(n=len(pairs), share_same=float(y.mean()), enc={})
        for en, f in encs.items():
            a = f([p[0] for p in pairs]); b = f([p[1] for p in pairs])
            na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
            sim = (a * b).sum(1) / np.maximum(na * nb, 1e-12)
            res = dict(auc=_auc(sim, y), curve=[])
            if dn == 'stsb':
                g = np.array([p[3] for p in pairs])
                res['spearman'] = float(np.corrcoef(np.argsort(np.argsort(sim)), np.argsort(np.argsort(g)))[0, 1])
            for tau in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
                sel = sim >= tau
                res['curve'].append(dict(tau=tau, reuse=float(sel.mean()),
                                         precision=float(y[sel].mean()) if sel.any() else None,
                                         recall=float((sel & y).sum() / y.sum())))
            # smallest threshold that reaches 90% precision (a calibrated threshold)
            order = np.argsort(-sim); cum = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
            ok = np.where(cum >= 0.9)[0]
            if len(ok):
                k = ok.max(); res['p90'] = dict(tau=float(sim[order][k]), reuse=float((k + 1) / len(y)),
                                                recall=float(y[order][:k + 1].sum() / y.sum()))
            out[dn]['enc'][en] = res
            print(dn, en, {k: v for k, v in res.items() if k != 'curve'})
    json.dump(out, open('res6_sts.json', 'w'), indent=1)


if __name__ == '__main__':
    {'trace': trace, 'sts': sts, 'sts2': sts2}[sys.argv[1]]()
