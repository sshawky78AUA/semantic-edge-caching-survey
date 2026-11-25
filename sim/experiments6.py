"""UCM control for the V6 revision.
E6r: semantic x coordination 2x2 contrast as a function of r = B*Cb/K, the ratio of the
     distinct items that coordinated BSs can hold to the number of semantic classes.
     Family 'K' varies K at Cb = 20; family 'Cb' varies Cb at K = 80 (B = 4)."""
import json, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments3 import pt
from ucm import workers, res_path


def e6r(pool):
    out = []
    cfgs = [('K', dict(K=k)) for k in (20, 40, 60, 80, 100, 160, 200, 320)] + \
           [('Cb', dict(Cb=c)) for c in (5, 10, 15, 30, 40)]
    for fam, c in cfgs:
        for co, me in itertools.product([0, 1], repeat=2):
            r = pt(pool, dict(est='ewma', coop=bool(co), tau=0.85 if me else 1.0, **c), 'ratio', R=20)
            r['levels'] = [co, me]; r['family'] = fam
            out.append(r)
    return out



def summarize(res, eps=1e-3):
    """Interaction of semantic matching and coordination at each r (supplementary Table S-III).
    I: half the 2x2 contrast of the hit ratio, in percentage points, mean and 95% CI over runs.
    Il: the same contrast on -log(miss ratio), per run (miss ratio floored at eps), then averaged."""
    from ucm import NAMES
    H = NAMES.index('hit')
    g = {}
    for q in res:
        key = (q['family'], q['kw'].get('K', 80), q['kw'].get('Cb', 20))
        g.setdefault(key, {})[tuple(q['levels'])] = np.array(q['raw'])[:, H]
    out = []
    for (fam, K, Cb), h in g.items():
        per = (h[(1, 1)] - h[(0, 1)] - h[(1, 0)] + h[(0, 0)]) / 2 * 100
        L = {k: np.log(np.maximum(1 - v, eps)) for k, v in h.items()}
        il = -(L[(1, 1)] - L[(0, 1)] - L[(1, 0)] + L[(0, 0)]) / 2
        out.append(dict(fam=fam, K=K, Cb=Cb, r=4 * Cb / K, I=per.mean(),
                        ci=1.96 * per.std(ddof=1) / np.sqrt(len(per)), Il=il.mean(),
                        h11=h[(1, 1)].mean(), h01=h[(0, 1)].mean(),
                        h10=h[(1, 0)].mean(), h00=h[(0, 0)].mean()))
    return sorted(out, key=lambda d: (d['fam'], d['r']))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        with ProcessPoolExecutor(workers()) as pool:
            res = globals()[name](pool)
        json.dump(res, open(res_path(f'res6_{name}.json'), 'w'))
        if name == 'e6r':
            json.dump(summarize(res), open(res_path('res6_e6r_summary.json'), 'w'))
        print(name, 'done', len(res), flush=True)
