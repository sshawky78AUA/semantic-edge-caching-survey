"""Further UCM experiments for the V4 revision (second review).
E1c: reactive baselines with aging and admission, under the independent reference model
     and under temporal locality; pooled and per-cell variants.
E1t: estimator crossover versus requests per item per epoch (T).
E1m: bounded-memory knee for a larger catalog.
E3t: D2D under the trajectory-based contact rule versus the mean-speed rule.
E5s: semantic x coordination 2x2 under changed K, Cb, class-correlated and cell-specific popularity.
E5t: the full factorial with the semantic level at tau = 0.9 and 0.8."""
import json, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from experiments3 import pt
from ucm import workers, res_path


def e1c(pool):
    out = []
    pol = [('oracle', 'pooled'), ('ewma', 'pooled'), ('ewma', 'local'), ('lru', 'local'), ('lfu', 'local'),
           ('lfua', 'local'), ('lfua', 'shared'), ('tlfu', 'local'), ('tlfu', 'shared')]
    for loc in [0.0, 0.5]:
        for d in [0, 0.05, 0.1, 0.2, 0.3]:
            for est, info in pol:
                if loc == 0.0 and est in ('lru', 'lfu', 'oracle') :
                    continue          # already in E1p
                out.append(pt(pool, dict(est=est, info=info, drift=d, loc=loc, p_part=0.0, coop=False), 'react2', R=20))
    return out


def e1t(pool):
    out = []
    for T in [10, 40]:
        for d in [0, 0.05, 0.1, 0.2, 0.3]:
            for est in ['oracle', 'stale', 'window', 'ewma']:
                out.append(pt(pool, dict(est=est, drift=d, T=T), 'reqs', R=20))
    return out


def e1m(pool):
    out = []
    for M in [20, 40, 80, 160, 320, 800]:
        out.append(pt(pool, dict(est='mem', alpha=0.5, M=M, drift=0.1, F=800, K=160), 'mem800', R=20))
    out.append(pt(pool, dict(est='ewma', alpha=0.5, drift=0.1, F=800, K=160), 'mem800', R=20))
    return out


def e3t(pool):
    out = []
    for v in [0.0, 1.4, 3, 6, 10, 15, 20, 30]:
        out.append(pt(pool, dict(v=v, aware=True, contact='traj'), 'traj', R=20))
    return out


def e5s(pool):
    out = []
    cfgs = [dict(K=40), dict(K=160), dict(Cb=10), dict(Cb=40), dict(class_corr=True), dict(lam_cell=0.5)]
    for c in cfgs:
        for co, me in itertools.product([0, 1], repeat=2):
            kw = dict(est='ewma', coop=bool(co), tau=0.85 if me else 1.0, **c)
            r = pt(pool, kw, 'sens', R=20)
            r['levels'] = [co, me]
            out.append(r)
    return out


def e5sb(pool):
    """Base case of E5s: the 2x2 semantic x coordination contrast at the defaults."""
    out = []
    for co, me in itertools.product([0, 1], repeat=2):
        r = pt(pool, dict(est='ewma', coop=bool(co), tau=0.85 if me else 1.0), 'sens', R=20)
        r['levels'] = [co, me]
        out.append(r)
    return out


def e5t(pool, tau):
    out = []
    for lv in itertools.product([0, 1], repeat=5):
        t, pe, co, pl, me = lv
        kw = dict(est='stale' if t == 0 else 'oracle', drift=0.2,
                  p_part=0.3 if pe == 0 else 1.0, coop=bool(co), v=3.0,
                  aware=bool(pl), tau=1.0 if me == 0 else tau)
        r = pt(pool, kw, 'factorial', R=20)
        r['levels'] = list(lv)
        out.append(r)
    return out


if __name__ == '__main__':
    name = sys.argv[1]
    with ProcessPoolExecutor(workers()) as pool:
        if name == 'e5t':
            tau = float(sys.argv[2]); res = e5t(pool, tau); name = f'e5t{sys.argv[2]}'
        else:
            res = globals()[name](pool)
    json.dump(res, open(res_path(f'res4_{name}.json' if name != 'e5sb' else 'res4_e5s_base.json'), 'w'))
    print(name, 'done', len(res), flush=True)
