"""UCM experiments for the V5 revision (rejection-risk review).
E5b: the 2^5 factorial with EWMA (not last-epoch counts) as the low level of the time factor,
     so that the prediction effect is measured against the best simple estimator.
E2f: cost-minimizing reward on a 0.05 grid with 30 seeds (base cost model)."""
import json, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments3 import pt


def e5b(pool):
    out = []
    for lv in itertools.product([0, 1], repeat=5):
        t, pe, co, pl, me = lv
        kw = dict(est='ewma' if t == 0 else 'oracle', drift=0.2,
                  p_part=0.3 if pe == 0 else 1.0, coop=bool(co), v=3.0,
                  aware=bool(pl), tau=1.0 if me == 0 else 0.85)
        r = pt(pool, kw, 'factorial_ewma', R=20)
        r['levels'] = list(lv)
        out.append(r)
    return out


def e2f(pool):
    out = []
    for v in (2.0, 10.0):
        for coop in (False, True):
            for p in np.round(np.arange(0, 1.0001, 0.05), 2):
                out.append(pt(pool, dict(p_part=float(p), coop=coop, est='ewma', v=v), 'rho_fine', R=30))
    return out


if __name__ == '__main__':
    for name in sys.argv[1:]:
        with ProcessPoolExecutor(2) as pool:
            res = globals()[name](pool)
        json.dump(res, open(f'res5_{name}.json', 'w'))
        print(name, 'done', len(res), flush=True)
