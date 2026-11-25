"""Additional UCM experiments for the V4 revision (reviewer comments).
E1p: separates pooled information from prediction (pooled/local x proactive/reactive).
E4s: semantic hit accounting (exact, accepted within-class, wrong cross-class) and
     index storage taken from the BS cache.
E2r: robustness of the cost-minimizing reward to the cost distribution, reward scale, and skew.
Seeds 0..R-1 are reused across compared policies (common random numbers)."""
import json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ucm import run, NAMES, nruns, workers, res_path


def _job(a):
    seed, kw = a
    return run(seed, **kw).tolist()


def pt(pool, kw, tag=None, R=30):
    R = nruns(R)
    raw = np.array(list(pool.map(_job, [(s, kw) for s in range(R)])))
    m = np.nanmean(raw, 0)
    ci = 1.96 * np.nanstd(raw, 0, ddof=1) / np.sqrt(R)
    return dict(kw=kw, tag=tag, m=dict(zip(NAMES, m.tolist())),
                ci=dict(zip(NAMES, ci.tolist())), raw=raw.tolist())


def e1p(pool):
    pol = [('oracle', 'pooled'), ('ewma', 'pooled'), ('ewma', 'local'),
           ('lfu', 'shared'), ('lfu', 'local'), ('lru', 'shared'), ('lru', 'local')]
    out = []
    for d in [0, 0.05, 0.1, 0.2, 0.3]:
        for est, info in pol:
            out.append(pt(pool, dict(est=est, info=info, drift=d, p_part=0.0, coop=False), 'pool'))
    return out


def e4s(pool):
    out = []
    for k in [0, 2, 4]:
        for tau in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
            if tau == 1.0 and k > 0:
                continue
            out.append(pt(pool, dict(tau=tau, idx_slots=k), 'sem'))
    return out


def e2r(pool, part):
    var = [dict(cost_dist='beta25'), dict(cost_dist='beta52'), dict(kappa=2.0),
           dict(kappa=8.0), dict(gamma=0.6), dict(gamma=1.0)]
    var = var[part::3]
    out = []
    for vv in var:
        for v in [2.0, 10.0]:
            for coop in [False, True]:
                for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    out.append(pt(pool, dict(p_part=p, coop=coop, est='ewma', v=v, **vv), 'rob', R=20))
    return out


if __name__ == '__main__':
    name = sys.argv[1]
    with ProcessPoolExecutor(workers()) as pool:
        if name == 'e2r':
            part = int(sys.argv[2])
            res = e2r(pool, part)
            name = f'e2r{part}'
        else:
            res = globals()[name](pool)
    json.dump(res, open(res_path(f'res3_{name}.json'), 'w'))
    print(name, 'done', len(res), flush=True)
