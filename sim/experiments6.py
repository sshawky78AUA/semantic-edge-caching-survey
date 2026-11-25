"""UCM control for the V6 revision.
E6r: semantic x coordination 2x2 contrast as a function of r = B*Cb/K, the ratio of the
     distinct items that coordinated BSs can hold to the number of semantic classes.
     Family 'K' varies K at Cb = 20; family 'Cb' varies Cb at K = 80 (B = 4)."""
import json, sys, itertools
from concurrent.futures import ProcessPoolExecutor
from experiments3 import pt


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


def e6r_summary(res):
    """Interaction I (points of hit ratio) = half the 2x2 contrast of hit ratio, per seed, with 95% CI;
    Il = the same contrast on -log(1-hit), hit clipped at 0.999. Written to res6_e6r_summary.json (used by figs6.py)."""
    import numpy as np
    from ucm import NAMES
    hi = NAMES.index('hit')
    cells = {}
    for x in res:
        kw = x['kw']
        cells.setdefault((x['family'], kw.get('K', 80), kw.get('Cb', 20)), {})[tuple(x['levels'])] = np.array(x['raw'])[:, hi]
    out = []
    for (fam, K, Cb), h in cells.items():
        I = 100 * ((h[(1, 1)] + h[(0, 0)]) - (h[(0, 1)] + h[(1, 0)])) / 2
        g = {l: -np.log(1 - np.minimum(h[l], 1 - 1e-3)) for l in h}  # hit ratio clipped at 0.999
        Il = ((g[(1, 1)] + g[(0, 0)]) - (g[(0, 1)] + g[(1, 0)])).mean() / 2
        out.append(dict(fam=fam, K=K, Cb=Cb, r=4 * Cb / K, I=float(I.mean()),
                        ci=float(1.96 * I.std(ddof=1) / np.sqrt(len(I))), Il=float(Il),
                        h11=float(h[(1, 1)].mean()), h01=float(h[(0, 1)].mean()),
                        h10=float(h[(1, 0)].mean()), h00=float(h[(0, 0)].mean())))
    return out


if __name__ == '__main__':
    for name in sys.argv[1:]:
        with ProcessPoolExecutor(2) as pool:
            res = globals()[name](pool)
        json.dump(res, open(f'res6_{name}.json', 'w'))
        if name == 'e6r':
            json.dump(e6r_summary(res), open('res6_e6r_summary.json', 'w'))
        print(name, 'done', len(res), flush=True)
