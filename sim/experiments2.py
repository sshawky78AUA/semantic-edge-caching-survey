"""Refined UCM experiments for Section IX (follows SIMULATION_PLAN.md).
30 runs per point; seeds 0..29 reused across policies (common random numbers).
Raw per-run outputs are stored so effects can be estimated with standard errors."""
import json, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ucm import run, NAMES, nruns, workers, res_path



def _job(a):
    seed, kw = a
    return run(seed, **kw).tolist()


def pt(pool, kw, tag=None):
    R = nruns(30)
    raw = np.array(list(pool.map(_job, [(s, kw) for s in range(R)])))
    m = np.nanmean(raw, 0)
    ci = 1.96 * np.nanstd(raw, 0, ddof=1) / np.sqrt(R)
    return dict(kw=kw, tag=tag, m=dict(zip(NAMES, m.tolist())),
                ci=dict(zip(NAMES, ci.tolist())), raw=raw.tolist())


def validate(pool):
    out = []
    for C in [10, 20, 40]:
        out.append(pt(pool, dict(est='oracle', drift=0.0, p_part=0.0, coop=False, Cb=C, Cd=1), 'topC'))
    return out


def e1(pool):
    out = []
    ests = [('oracle', {}), ('stale', {}), ('window', {'W': 3}), ('ewma', {'alpha': 0.5}),
            ('mem', {'alpha': 0.5, 'M': 40})]
    for g in [0.6, 0.8, 1.0]:
        for d in [0, 0.02, 0.05, 0.1, 0.2, 0.3]:
            for name, extra in ests:
                if g != 0.8 and name == 'mem':
                    continue
                out.append(pt(pool, dict(est=name, drift=d, gamma=g, **extra), 'drift'))
    for M in [10, 20, 40, 80, 160, 400]:
        out.append(pt(pool, dict(est='mem', alpha=0.5, M=M, drift=0.1), 'mem'))
    return out


def e1r(pool):
    """Reactive (LRU, LFU) versus proactive BS caches, BS-only network (no D2D, independent BSs)."""
    out = []
    for d in [0, 0.02, 0.05, 0.1, 0.2, 0.3]:
        for name in ['oracle', 'stale', 'ewma', 'lru', 'lfu']:
            out.append(pt(pool, dict(est=name, drift=d, p_part=0.0, coop=False), 'reactive'))
    return out


def e2(pool):
    out = []
    for v in [2.0, 10.0]:
        for coop in [False, True]:
            for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                out.append(pt(pool, dict(p_part=p, coop=coop, est='ewma', v=v), 'rho'))
    return out


def e3(pool):
    out = []
    for v in [0.0, 0.5, 1.4, 3, 6, 10, 15, 20, 30]:
        for aw in [False, 'ref', True]:
            out.append(pt(pool, dict(v=v, aware=aw), 'speed'))
    return out


def e4(pool):
    out = []
    for coop in [False, True]:
        for Cb in [10, 20, 40]:
            for tau in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
                if coop and Cb != 20:
                    continue
                out.append(pt(pool, dict(tau=tau, Cb=Cb, coop=coop), 'tau'))
    return out


def e5(pool):
    out = []
    for lv in itertools.product([0, 1], repeat=5):
        t, pe, co, pl, me = lv
        kw = dict(est='stale' if t == 0 else 'oracle', drift=0.2,
                  p_part=0.3 if pe == 0 else 1.0, coop=bool(co), v=3.0,
                  aware=bool(pl), tau=1.0 if me == 0 else 0.85)
        r = pt(pool, kw, 'factorial')
        r['levels'] = list(lv)
        out.append(r)
    return out


if __name__ == '__main__':
    for name in sys.argv[1:]:
        with ProcessPoolExecutor(workers()) as pool:
            res = globals()[name](pool)
        json.dump(res, open(res_path(f'res2_{name}.json'), 'w'))
        print(name, 'done', len(res), flush=True)
