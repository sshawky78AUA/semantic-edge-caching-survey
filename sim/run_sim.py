"""Run the UCM with parameters of your choice and print the results.

Examples (from the sim/ folder):
  python3 run_sim.py                                  # defaults, 30 seeds
  python3 run_sim.py v=10 p_part=0.5 coop=1           # change speed, participation, coordination
  python3 run_sim.py est=lru drift=0.2 --seeds 10     # reactive LRU caches, faster drift
  python3 run_sim.py tau=0.85 K=40 --vary Cb=5,10,20,40   # sweep one parameter
  python3 run_sim.py --list                           # all parameters with defaults

Values are parsed as numbers, True/False (also 0/1 for flags), None, or text.
Seeds 0..R-1 are the same ones the paper uses, so a run with the paper's settings reproduces the stored values.
"""
import argparse, ast, json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ucm import run, DEF, NAMES

FLAGS = {'aware', 'coop', 'class_corr'}


def parse(v, key):
    try:
        x = ast.literal_eval(v)
    except Exception:
        return v
    if key in FLAGS:
        return bool(x)
    return x


def _job(a):
    return run(a[0], **a[1]).tolist()


def point(pool, kw, R):
    raw = np.array(list(pool.map(_job, [(s, kw) for s in range(R)])))
    m = np.nanmean(raw, 0)
    ci = 1.96 * np.nanstd(raw, 0, ddof=1) / np.sqrt(R)
    return m, ci


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('params', nargs='*', help='name=value overrides of DEF in ucm.py')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--vary', help='name=v1,v2,... : sweep one parameter')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', help='write all results to this JSON file')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k, v in DEF.items():
            print(f'{k:10s} {v!r}')
        return
    kw = {}
    for p in a.params:
        k, v = p.split('=', 1)
        if k not in DEF:
            sys.exit(f'unknown parameter {k!r}; see --list')
        kw[k] = parse(v, k)
    pts = [kw]
    if a.vary:
        k, vs = a.vary.split('=', 1)
        if k not in DEF:
            sys.exit(f'unknown parameter {k!r}; see --list')
        pts = [dict(kw, **{k: parse(v, k)}) for v in vs.split(',')]
    show = ['hit', 'own', 'd2d', 'bs', 'ibs', 'miss', 'lat', 'fid', 'sp_cost', 'pay']
    out = []
    with ProcessPoolExecutor(a.workers) as pool:
        for q in pts:
            m, ci = point(pool, q, a.seeds)
            out.append(dict(kw=q, m=dict(zip(NAMES, m.tolist())), ci=dict(zip(NAMES, ci.tolist()))))
            lab = ' '.join(f'{k}={q[k]}' for k in (set(q) if not a.vary else {a.vary.split('=')[0]}))
            print(f'[{lab or "defaults"}] ' + '  '.join(f'{n}={m[NAMES.index(n)]:.3f}±{ci[NAMES.index(n)]:.3f}' for n in show), flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
