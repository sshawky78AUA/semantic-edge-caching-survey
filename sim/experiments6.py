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


if __name__ == '__main__':
    for name in sys.argv[1:]:
        with ProcessPoolExecutor(2) as pool:
            res = globals()[name](pool)
        json.dump(res, open(f'res6_{name}.json', 'w'))
        print(name, 'done', len(res), flush=True)
