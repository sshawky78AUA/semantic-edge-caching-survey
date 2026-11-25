"""Run the unified caching model (UCM) with your own parameters.

Examples (run from the sim/ folder):

  python3 run_ucm.py                               # defaults of the paper (Table X), 20 seeds
  python3 run_ucm.py coop=True tau=0.85            # coordinated BSs with semantic matching
  python3 run_ucm.py --sweep v=0,5,10,20,30 aware=True
  python3 run_ucm.py --sweep tau=1,0.95,0.9,0.85,0.8 --sweep coop=False,True --plot tau.pdf
  python3 run_ucm.py --list                        # every parameter with its default

Every parameter of DEF in ucm.py can be set as name=value. Up to two --sweep options vary
parameters over a grid; every grid point is run with the same seeds (common random numbers),
so differences between points are not due to sampling noise in the demand or mobility.
Results are printed as mean and 95% confidence interval over the runs, and can be saved
with --out results.json (all runs) or --csv results.csv (means and intervals).
"""
import argparse
import ast
import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from ucm import DEF, NAMES, run, workers

HELP = {
    'F': 'number of items in the catalog',
    'K': 'number of semantic classes',
    'gamma': 'Zipf exponent of popularity',
    'N': 'number of devices',
    'area': 'side of the square region (m)',
    'r': 'D2D range (m)',
    'v': 'device speed (m/s)',
    'tf': 'time to complete a D2D transfer (s)',
    'dt': 'slot length (s)',
    'Cb': 'BS cache size (items)',
    'Cd': 'device cache size (items)',
    'drift': 'popularity drift per epoch',
    'est': "popularity estimator: oracle, stale, window, ewma, mem, or reactive lru, lfu, lfua, tlfu",
    'alpha': 'EWMA weight of the newest epoch',
    'W': 'window length (epochs) of the window estimator',
    'M': 'counters kept by the memory-bounded estimator (mem)',
    'tau': 'similarity threshold of a semantic hit (1 = exact match only)',
    's': 'spread of item embeddings around their class center',
    'p_part': 'reward rho: devices with private cost <= rho cache for others (1 = all join)',
    'aware': 'mobility-aware device placement (True/False)',
    'coop': 'coordinated BS caches (True/False)',
    'E': 'epochs per run',
    'T': 'slots per epoch',
    'warm': 'warm-up epochs not counted',
    'lat': 'delays of own cache, D2D, BS, other BS, backhaul (ms)',
    'c_bh': 'SP cost of a backhaul fetch',
    'c_ib': 'SP cost of an inter-BS fetch',
    'kappa': 'reward scale',
    'beta': 'probability that a device issues a request in a slot',
    'info': 'pooled, shared, or local estimates at the BSs',
    'cost_dist': 'private participation costs: uniform, beta25, beta52',
    'idx_slots': 'BS cache slots given up to the semantic index',
    'loc': 'probability of re-referencing a recent request (temporal locality)',
    'loc_len': 'length of the recent-request window for loc',
    'lam_cell': 'weight of a cell-specific popularity',
    'class_corr': 'popularity ranks assigned to whole semantic classes',
    'contact': "D2D contact rule: mean (paper) or traj (along trajectories)",
}
SHOW = ['hit', 'own', 'd2d', 'bs', 'ibs', 'miss', 'lat', 'fid', 'sp_cost', 'pay', 'participants',
        'exact', 'sem_ok', 'sem_wrong']
LABEL = {'hit': 'hit ratio', 'own': 'own cache', 'd2d': 'D2D', 'bs': 'local BS', 'ibs': 'other BS',
         'miss': 'backhaul', 'lat': 'latency (ms)', 'fid': 'fidelity', 'sp_cost': 'SP cost',
         'pay': 'rewards paid', 'participants': 'participants', 'exact': 'exact hits',
         'sem_ok': 'same-class hits', 'sem_wrong': 'wrong-class hits'}


def parse_value(text, default):
    if isinstance(default, bool):
        if text.lower() in ('true', '1', 'yes'):
            return True
        if text.lower() in ('false', '0', 'no'):
            return False
        raise ValueError(f'expected True or False, got {text}')
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        val = text
    if isinstance(default, float) and isinstance(val, int) and not isinstance(val, bool):
        val = float(val)
    if text.lower() == 'none':
        val = None
    return val


def parse_assign(item):
    if '=' not in item:
        sys.exit(f'expected name=value, got "{item}"')
    name, text = item.split('=', 1)
    if name not in DEF:
        sys.exit(f'unknown parameter "{name}"; use --list to see all parameters')
    return name, text


def _job(a):
    seed, kw = a
    return run(seed, **kw).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('params', nargs='*', help='name=value overrides of the defaults')
    ap.add_argument('--sweep', action='append', default=[], help='name=v1,v2,... (up to two)')
    ap.add_argument('--runs', type=int, default=20, help='seeded runs per point (default 20)')
    ap.add_argument('--seed0', type=int, default=0, help='first seed (default 0)')
    ap.add_argument('--workers', type=int, default=None, help='worker processes (default: all cores)')
    ap.add_argument('--metrics', default='hit,d2d,bs,miss,lat,sp_cost',
                    help='comma-separated metrics to print, or "all"')
    ap.add_argument('--out', help='save every run to this JSON file')
    ap.add_argument('--csv', help='save means and 95%% intervals to this CSV file')
    ap.add_argument('--plot', help='plot the first metric against the first swept parameter (PDF or PNG)')
    ap.add_argument('--list', action='store_true', help='list every parameter with its default and exit')
    a = ap.parse_args()

    if a.list:
        for k, v in DEF.items():
            print(f'{k:10s} {str(v):24s} {HELP.get(k, "")}')
        return

    base = {}
    for item in a.params:
        name, text = parse_assign(item)
        base[name] = parse_value(text, DEF[name])
    sweeps = []
    for item in a.sweep:
        name, text = parse_assign(item)
        sweeps.append((name, [parse_value(t, DEF[name]) for t in text.split(',')]))
    if len(sweeps) > 2:
        sys.exit('at most two --sweep options')
    metrics = SHOW if a.metrics == 'all' else a.metrics.split(',')
    for m in metrics:
        if m not in NAMES:
            sys.exit(f'unknown metric "{m}"; choose from {", ".join(NAMES)}')

    grid = list(itertools.product(*[vals for _, vals in sweeps])) or [()]
    points = []
    for combo in grid:
        kw = dict(base)
        kw.update({name: val for (name, _), val in zip(sweeps, combo)})
        points.append(kw)

    seeds = range(a.seed0, a.seed0 + a.runs)
    jobs = [(s, kw) for kw in points for s in seeds]
    nw = a.workers or workers()
    print(f'{len(points)} point(s) x {a.runs} runs = {len(jobs)} runs on {nw} worker(s)', file=sys.stderr)
    if nw > 1:
        with ProcessPoolExecutor(nw) as pool:
            out = list(pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * nw))))
    else:
        out = [_job(j) for j in jobs]
    raw = np.array(out).reshape(len(points), a.runs, len(NAMES))

    rows = []
    idx = [NAMES.index(m) for m in metrics]
    head = [name for name, _ in sweeps] + [f'{LABEL.get(m, m)}' for m in metrics]
    print('  '.join(f'{h:>18s}' for h in head))
    for kw, r in zip(points, raw):
        mean = np.nanmean(r, 0)
        ci = 1.96 * np.nanstd(r, 0, ddof=1) / np.sqrt(a.runs) if a.runs > 1 else np.zeros(len(NAMES))
        cells = [str(kw[name]) for name, _ in sweeps] + [f'{mean[i]:.4f} ± {ci[i]:.4f}' for i in idx]
        print('  '.join(f'{c:>18s}' for c in cells))
        rows.append(dict(kw=kw, m=dict(zip(NAMES, mean.tolist())), ci=dict(zip(NAMES, ci.tolist())),
                         raw=r.tolist()))

    if a.out:
        json.dump(dict(defaults=DEF, base=base, sweeps=sweeps, runs=a.runs, seed0=a.seed0, points=rows),
                  open(a.out, 'w'), default=str)
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([name for name, _ in sweeps] + [x for m in NAMES for x in (m, m + '_ci95')])
            for kw, row in zip(points, rows):
                w.writerow([kw[name] for name, _ in sweeps] +
                           [x for m in NAMES for x in (row['m'][m], row['ci'][m])])
    if a.plot:
        if not sweeps:
            sys.exit('--plot needs at least one --sweep')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m = metrics[0]
        xname, xs = sweeps[0]
        fig, ax = plt.subplots(figsize=(4, 3))
        groups = sweeps[1][1] if len(sweeps) > 1 else [None]
        for gv in groups:
            sel = [row for kw, row in zip(points, rows) if gv is None or kw[sweeps[1][0]] == gv]
            y = [row['m'][m] for row in sel]
            e = [row['ci'][m] for row in sel]
            lab = None if gv is None else f'{sweeps[1][0]}={gv}'
            ax.errorbar(range(len(xs)) if not all(isinstance(x, (int, float)) for x in xs) else xs,
                        y, yerr=e, marker='o', capsize=2, label=lab)
        if not all(isinstance(x, (int, float)) for x in xs):
            ax.set_xticks(range(len(xs)), [str(x) for x in xs])
        ax.set_xlabel(xname)
        ax.set_ylabel(LABEL.get(m, m))
        if len(groups) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(a.plot)
        print(f'plot written to {a.plot}', file=sys.stderr)


if __name__ == '__main__':
    main()
