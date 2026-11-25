"""Prints the numbers of the tables of the article and the supplement from the stored results.

Usage (from sim/):  python3 tables.py [trace] [e5] [ratio] [rho] [rho_fine] [encoders]   (all if none given)

  trace     Table XII of the article: MovieTweetings replay (res5_trace.json)
  e5        Main effects and two-way interactions of the factorial E5 (res2_e5.json) and its
            variants E5b, E5t (res5_e5b.json, res4_e5t0.8.json, res4_e5t0.9.json)
  ratio     Table S-III: semantic x coordination interaction versus r = B*Cb/K (res6_e6r_summary.json)
  rho       Table S-II: cost-minimizing reward with bootstrap intervals (res2_e2.json, res3_e2r*.json)
  rho_fine  Section IX: cost-minimizing reward on a grid of step 0.05 (res5_e2f.json)
  encoders  Tables S-IV and S-V: similarity versus meaning on STS-B and Quora (res5_sts.json, res6_sts.json)
"""
import itertools
import json
import os
import sys

import numpy as np

from ucm import NAMES, res_path


def load(name):
    return json.load(open(res_path(name)))


def trace():
    t = load('res5_trace.json')
    pol = [('oracle', 'Epoch oracle'), ('stale', 'Last epoch'), ('ewma', 'EWMA'), ('lru', 'LRU'),
           ('lfu', 'LFU'), ('lfua', 'Aged LFU'), ('tlfu', 'TinyLFU-style')]
    cols = [(7, 100), (7, 400), (28, 100), (28, 400)]
    runs = {(r['days'], r['C']): r for r in t['runs']}
    print(f"Trace: {t['R']:,} requests, {t['F']:,} items")
    print(f"{'Policy':16s}" + ''.join(f'{f"{d}d C={c}":>12s}' for d, c in cols))
    for k, lab in pol:
        print(f'{lab:16s}' + ''.join(f'{runs[c][k]:12.3f}' if c in runs else f'{"-":>12s}' for c in cols))


FAC = ['Prediction', 'Participation', 'Coordination', 'Placement', 'Semantic']


def effects(res, idx, log_miss=False, eps=1e-3):
    L = np.array([q['levels'] for q in res]) * 2 - 1
    Y = np.array([np.array(q['raw'])[:, idx] for q in res])
    if log_miss:
        Y = -np.log(np.maximum(1 - Y, eps))
    out = []
    for k in range(5):
        s = L[:, k]
        d = Y[s > 0].mean(0) - Y[s < 0].mean(0)
        out.append((FAC[k], d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    for a, b in itertools.combinations(range(5), 2):
        s = L[:, a] * L[:, b]
        d = Y[s > 0].mean(0) - Y[s < 0].mean(0)
        out.append((FAC[a] + ' x ' + FAC[b], d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    return out


def e5():
    H = NAMES.index('hit')
    for name, lab in [('res2_e5.json', 'E5 (tau = 0.85, low time level = last epoch)'),
                      ('res5_e5b.json', 'E5b (low time level = EWMA)'),
                      ('res4_e5t0.9.json', 'E5 at tau = 0.9'), ('res4_e5t0.8.json', 'E5 at tau = 0.8')]:
        if not os.path.exists(res_path(name)):
            continue
        res = load(name)
        print(f'\n{lab}: effects in points of hit ratio (interactions are half contrasts I_AB/2)')
        hit = effects(res, H)
        lm = effects(res, H, log_miss=True)
        print(f'{"":34s}{"hit ratio":>18s}{"-log miss":>12s}')
        for (n, m, c), (_, ml, _) in zip(hit, lm):
            print(f'{n:34s}{100 * m:+10.1f} ± {100 * c:4.1f}{ml:+12.2f}')


def ratio():
    s = load('res6_e6r_summary.json')
    print(f'{"Varied":7s}{"K":>5s}{"Cb":>5s}{"r":>6s}{"h00":>7s}{"h01":>7s}{"h10":>7s}{"h11":>7s}'
          f'{"Interaction":>16s}{"Log-miss":>10s}')
    for d in s:
        lm = f"{d['Il']:+.2f}" if d['h11'] < 0.9995 else 'undef.'
        print(f"{d['fam']:7s}{d['K']:5d}{d['Cb']:5d}{d['r']:6.2f}{d['h00']:7.3f}{d['h01']:7.3f}"
              f"{d['h10']:7.3f}{d['h11']:7.3f}{d['I']:+10.2f} ± {d['ci']:.2f}{lm:>10s}")


def boot_rho(res, nb=2000, seed=0):
    """Reward on the grid that minimizes the mean SP cost, and a percentile bootstrap interval
    from resampling the seeds (the same seeds at every grid point)."""
    C = NAMES.index('sp_cost')
    ps = sorted(res, key=lambda q: q['kw']['p_part'])
    grid = np.array([q['kw']['p_part'] for q in ps])
    Y = np.array([np.array(q['raw'])[:, C] for q in ps])
    best = grid[Y.mean(1).argmin()]
    rng = np.random.default_rng(seed)
    n = Y.shape[1]
    b = [grid[Y[:, rng.integers(0, n, n)].mean(1).argmin()] for _ in range(nb)]
    return best, np.percentile(b, 2.5), np.percentile(b, 97.5), n


def rho():
    base = load('res2_e2.json')
    rob = sum((load(f'res3_e2r{p}.json') for p in range(3)), [])
    variants = [('Base (uniform costs)', base, {}), ('Costs Beta(2,5)', rob, dict(cost_dist='beta25')),
                ('Costs Beta(5,2)', rob, dict(cost_dist='beta52')), ('kappa = 2', rob, dict(kappa=2.0)),
                ('kappa = 8', rob, dict(kappa=8.0)), ('gamma = 0.6', rob, dict(gamma=0.6)),
                ('gamma = 1.0', rob, dict(gamma=1.0))]
    print(f'{"Variant":22s}{"v=2 Ind.":>18s}{"v=2 Coord.":>18s}{"v=10 Ind.":>18s}{"v=10 Coord.":>18s}{"seeds":>7s}')
    for lab, res, var in variants:
        cells = []
        for v in (2.0, 10.0):
            for co in (False, True):
                sel = [q for q in res if q['kw'].get('v') == v and q['kw'].get('coop') == co
                       and all(q['kw'].get(k) == x for k, x in var.items())
                       and (var or not any(k in q['kw'] for k in ('cost_dist', 'kappa', 'gamma')))]
                b, lo, hi, n = boot_rho(sel)
                cells.append(f'{b:.1f} [{lo:.1f},{hi:.1f}]')
        print(f'{lab:22s}' + ''.join(f'{c:>18s}' for c in cells) + f'{n:7d}')


def rho_fine():
    """Section IX of the article: reward grid of step 0.05 with 30 seeds (res5_e2f.json)."""
    res = load('res5_e2f.json')
    C = NAMES.index('sp_cost')
    for v in (2.0, 10.0):
        for co in (False, True):
            sel = [q for q in res if q['kw'].get('v') == v and q['kw'].get('coop') == co]
            b, lo, hi, n = boot_rho(sel)
            cost = {q['kw']['p_part']: q['m']['sp_cost'] for q in sel}
            save = 100 * (cost[0] - cost[b]) / cost[0]
            print(f'v={v:4.0f} m/s {"coordinated" if co else "independent":12s} rho* = {b:.2f} '
                  f'[{lo:.2f}, {hi:.2f}]  cost at rho=0: {cost[0]:.3f}, at rho*: {cost[b]:.3f} '
                  f'(saving {save:.1f}%), at rho=1: {cost[1.0]:.3f}  ({n} seeds)')


def encoders():
    s = load('res6_sts.json')
    names = {'spacy-avg': 'Averaged word vectors', 'MiniLM-L6': 'all-MiniLM-L6-v2',
             'MPNet-base': 'all-mpnet-base-v2'}
    taus = [0.8, 0.85, 0.9, 0.95]
    print(f'{"Data":8s}{"Encoder":24s}{"AUC":>6s}{"Spear.":>8s}' + ''.join(f'{f"tau={t}":>13s}' for t in taus)
          + f'{"90% precision: tau / recall":>30s}')
    for data, lab in [('stsb', 'STS-B'), ('qqp', 'Quora')]:
        for k, enc in s[data]['enc'].items():
            curve = {round(c['tau'], 2): c for c in enc['curve']}
            cells = [f"{curve[t]['reuse']:.2f}/{curve[t]['precision']:.2f}" if t in curve else '-' for t in taus]
            p90 = enc.get('p90')
            p = f"{p90['tau']:.3f} / {p90['recall']:.2f}" if p90 and p90['recall'] > 0.005 else 'not reached'
            sp = f"{enc['spearman']:.2f}" if 'spearman' in enc else '--'
            print(f'{lab:8s}{names.get(k, k):24s}{enc["auc"]:6.2f}{sp:>8s}' + ''.join(f'{c:>13s}' for c in cells)
                  + f'{p:>30s}')
    print('\n(cells: share of pairs reused / precision of reuse)')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['trace', 'e5', 'ratio', 'rho', 'rho_fine', 'encoders']
    for t in todo:
        print(f'\n===== {t} =====')
        globals()[t]()
