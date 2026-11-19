"""Unified caching model (UCM): time, people (SP and user cooperation), place, meaning.
Python prototype. A MATLAB port follows once the design is agreed."""
import numpy as np

DEF = dict(F=400, K=80, gamma=0.8, N=80, area=600., r=100., v=2., tf=5., dt=5.,
           Cb=20, Cd=10, drift=0.05, est='ewma', alpha=0.5, W=3, M=None,
           tau=1.0, s=0.4, p_part=1.0, aware=True, coop=False,
           E=20, T=20, warm=3, lat=(1., 5., 20., 40., 100.),
           c_bh=1.0, c_ib=0.3, kappa=4.0, beta=1.0)
# est: 'oracle', 'stale' (last epoch), 'window', 'ewma', 'mem' (EWMA bounded to M counters)
#      proactive estimators, BS caches filled once per epoch from the estimate;
#      'lru', 'lfu' reactive BS caches, updated at every request that reaches the BS.
# beta: probability that a device is active in a slot (at most one request per slot).
REACTIVE = ('lru', 'lfu')
BS = np.array([[200., 200.], [600., 200.], [200., 600.], [600., 600.]])
B = 4
NAMES = ['own', 'd2d', 'bs', 'ibs', 'miss', 'fid', 'lat', 'lat_part', 'lat_non',
         'sp_cost', 'pay', 'hit', 'participants']


def semantic(rng, F, K, s, d=16):
    C = rng.normal(size=(K, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    cls = np.arange(F) % K
    rng.shuffle(cls)
    X = C[cls] + s * rng.normal(size=(F, d)) / np.sqrt(d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.clip(X @ X.T, -1, 1)


def greedy_cover(qh, Cm, n):
    cov = np.zeros(len(qh), bool)
    sel = []
    for _ in range(n):
        g = (qh * (~cov)) @ Cm
        g[sel] = -1
        j = int(np.argmax(g))
        sel.append(j)
        cov |= Cm[j] > 0
    return sel


def place_bs(qh, Cm, p):
    F = len(qh)
    Hb = np.zeros((B, F), bool)
    if not p['coop']:
        Hb[:, greedy_cover(qh, Cm, p['Cb'])] = True
        return Hb
    lat = p['lat']
    L1, L2 = lat[4] - lat[2], lat[4] - lat[3]
    val = np.zeros((B, F))
    used = np.zeros(B, int)
    for _ in range(B * p['Cb']):
        w1 = qh * np.maximum(L1 - val, 0)
        w2 = qh * np.maximum(L2 - val, 0)
        D = w1 + w2.sum(0) - w2
        G = Cm @ D.T
        G[Hb.T] = -1
        G[:, used >= p['Cb']] = -1
        j, b = np.unravel_index(np.argmax(G), G.shape)
        Hb[b, j] = True
        used[b] += 1
        cj = Cm[j] > 0
        for k in range(B):
            val[k, cj] = np.maximum(val[k, cj], L1 if k == b else L2)
    return Hb


def place_dev(rng, qh, covered, part_idx, mhat, p):
    N, F, Cd = p['N'], len(qh), p['Cd']
    Hd = np.zeros((N, F), bool)
    P = len(part_idx)
    if P == 0:
        return Hd
    qr = qh * (~covered)
    if qr.sum() <= 0:
        qr = qh.copy()
    qr = qr / qr.sum()
    if not p['aware']:
        for u in part_idx:
            nz = max(int((qr > 0).sum()), Cd)
            pr = qr + 1e-12
            pr /= pr.sum()
            Hd[u, rng.choice(F, Cd, replace=False, p=pr)] = True
        return Hd
    pc = float(np.clip(mhat / max(P, 1), 1e-3, 0.999))
    n = np.zeros(F)
    free = np.full(P, Cd)
    for _ in range(P * Cd):
        g = qr * pc * (1 - pc) ** n
        g[n >= P] = -1
        j = int(np.argmax(g))
        n[j] += 1
    order = np.argsort(-n)
    for j in order:
        k = int(n[j])
        if k == 0:
            break
        cand = np.flatnonzero(free > 0)
        if len(cand) == 0:
            break
        key = -free[cand] + rng.random(len(cand)) * 0.5
        take = cand[np.argsort(key)[:k]]
        Hd[part_idx[take], j] = True
        free[take] -= 1
    return Hd


def _react(rc, rf, b, i, p):
    """Serve request i at BS b with a reactive cache; return True on a hit."""
    c = rc[b]
    if p['est'] == 'lfu':
        rf[b, i] += 1
    if i in c:
        if p['est'] == 'lru':
            c.remove(i)
            c.append(i)
        return True
    if len(c) < p['Cb']:
        c.append(i)
    elif p['est'] == 'lru':
        c.pop(0)
        c.append(i)
    else:
        k = int(np.argmin(rf[b, c]))
        if rf[b, i] > rf[b, c[k]]:
            c[k] = i
    return False


def run(seed, **kw):
    p = dict(DEF)
    p.update(kw)
    rng = np.random.default_rng(seed)
    F, N, T, E = p['F'], p['N'], p['T'], p['E']
    S = semantic(rng, F, p['K'], p['s'])
    Cm = (S >= p['tau'] - 1e-9).astype(np.float32)
    w = 1.0 / np.arange(1, F + 1) ** p['gamma']
    w /= w.sum()
    rk = rng.permutation(F)
    pos = rng.random((N, 2)) * p['area']
    th = rng.random(N) * 2 * np.pi
    cost = rng.random(N)
    part = cost <= p['p_part'] if p['p_part'] < 1 else np.ones(N, bool)
    part_idx = np.flatnonzero(part)
    BSxy = BS / 800. * p['area']
    hist = []
    sm = np.zeros(F)
    acc = np.zeros(8)  # own d2d bs ibs miss fid_sum lat_sum_part lat_sum_non
    n_part = n_non = 0
    nreq = 0
    L = p['lat']
    reactive = p['est'] in REACTIVE
    rc = [[] for _ in range(B)]          # reactive cache content per BS (LRU: oldest first)
    rf = np.zeros((B, F))                # LFU request counts per BS
    for e in range(E):
        q = w[rk]
        if p['est'] == 'oracle':
            qh = q.copy()
        elif p['est'] in REACTIVE:
            qh = (hist[-1] + 0.1) if hist else np.ones(F)
        elif not hist:
            qh = np.ones(F) / F
        else:
            c = hist[-1]
            if p['est'] == 'stale':
                qh = c + 0.1
            elif p['est'] == 'window':
                qh = sum(hist[-p['W']:]) + 0.1
            else:
                sm = p['alpha'] * c + (1 - p['alpha']) * sm
                if p['est'] == 'mem' and p['M'] is not None:
                    keep = np.argsort(-sm)[:p['M']]
                    m = np.zeros(F, bool)
                    m[keep] = True
                    sm = np.where(m, sm, 0.0)
                qh = sm + 0.1
        qh = qh / qh.sum()
        if p['est'] in REACTIVE:
            Hb = np.zeros((B, F), bool)
            for b in range(B):
                Hb[b, rc[b]] = True
        else:
            Hb = place_bs(qh, Cm, p)
        covered = (Cm @ Hb.any(0).astype(np.float32)) > 0
        D0 = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        A0 = (D0 < p['r']) & ~np.eye(N, dtype=bool)
        vv = 0.5 if p['aware'] == 'ref' else p['v']
        if vv > 0:
            A0 &= (p['r'] - D0) / (4 * vv / np.pi) >= p['tf']
        mh = A0[np.ix_(part_idx, part_idx)].sum(1).mean() if len(part_idx) else 0
        Hd = place_dev(rng, qh, covered, part_idx, mh, p)
        Hdf = Hd.astype(np.float32)
        cnt = np.zeros(F)
        anyb = Hb.any(0)
        for t in range(T):
            req = np.searchsorted(np.cumsum(q), rng.random(N) * q.sum())
            req = np.minimum(req, F - 1)
            act = rng.random(N) < p['beta'] if p['beta'] < 1 else np.ones(N, bool)
            cnt += np.bincount(req[act], minlength=F)
            re = rng.random(N) < 0.2
            th = np.where(re, rng.random(N) * 2 * np.pi, th)
            pos = pos + p['v'] * p['dt'] * np.stack([np.cos(th), np.sin(th)], 1)
            pos = np.where(pos < 0, -pos, pos)
            pos = np.where(pos > p['area'], 2 * p['area'] - pos, pos)
            if reactive and e < p['warm']:
                sw = np.argmin(np.linalg.norm(pos[:, None] - BSxy[None], axis=2), axis=1)
                own_w = (Hd[np.arange(N), req]) & act
                for u in np.flatnonzero(act & ~own_w):
                    _react(rc, rf, sw[u], int(req[u]), p)
            if e < p['warm']:
                continue
            D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
            Af = (D < p['r']) & ~np.eye(N, dtype=bool)
            if p['v'] > 0:
                Af &= (p['r'] - D) / (4 * p['v'] / np.pi) >= p['tf']
            reach = (Af.astype(np.float32) @ Hdf) > 0
            sidx = np.argmin(np.linalg.norm(pos[:, None] - BSxy[None], axis=2), axis=1)
            Mt = Cm[req] > 0
            Sr = S[req]
            own = np.where(Hd & Mt, Sr, -1).max(1)
            d2d = np.where(reach & Mt, Sr, -1).max(1)
            if reactive:
                bs = np.full(N, -1.0)
                for u in np.flatnonzero(act & (own < 0) & (d2d < 0)):
                    if _react(rc, rf, sidx[u], int(req[u]), p):
                        bs[u] = 1.0
                Hb = np.zeros((B, F), bool)
                for b in range(B):
                    Hb[b, rc[b]] = True
                anyb = Hb.any(0)
            else:
                bs = np.where(Hb[sidx] & Mt, Sr, -1).max(1)
            if p['coop']:
                ibs = np.where(anyb[None] & Mt, Sr, -1).max(1)
            else:
                ibs = np.full(N, -1.0)
            tier = np.full(N, 4)
            tier = np.where(ibs >= 0, 3, tier)
            tier = np.where(bs >= 0, 2, tier)
            tier = np.where(d2d >= 0, 1, tier)
            tier = np.where(own >= 0, 0, tier)
            st = np.stack([own, d2d, bs, ibs], 1)
            fid = np.where(tier < 4, st[np.arange(N), np.minimum(tier, 3)], 0)
            lat = np.array([L[0], L[1], L[2], L[3], L[4]])[tier]
            for k in range(5):
                acc[k] += ((tier == k) & act).sum()
            acc[5] += fid[act].sum()
            acc[6] += lat[part & act].sum()
            acc[7] += lat[~part & act].sum()
            n_part += (part & act).sum()
            n_non += (~part & act).sum()
            nreq += act.sum()
        hist.append(cnt)
        drift = np.flatnonzero(rng.random(F) < p['drift'])
        rk[drift] = rng.permutation(rk[drift])
    own, d2d, bs, ibs, miss = acc[:5] / nreq
    hits = nreq - acc[4]
    epochs = E - p['warm']
    pay = p['kappa'] * p['p_part'] * part.sum() / (N * T)
    sp = p['c_bh'] * miss + p['c_ib'] * ibs + pay
    latall = (acc[6] + acc[7]) / nreq
    return np.array([own, d2d, bs, ibs, miss, acc[5] / max(hits, 1), latall,
                     acc[6] / max(n_part, 1), acc[7] / max(n_non, 1) if n_non else np.nan,
                     sp, pay, 1 - miss, part.sum()])


def _job(a):
    seed, kw = a
    return run(seed, **kw)


def avg(kw, R=20, pool=None, base=0):
    jobs = [(base + i, kw) for i in range(R)]
    out = np.array(list(pool.map(_job, jobs)) if pool else [_job(j) for j in jobs])
    m = np.nanmean(out, 0)
    ci = 1.96 * np.nanstd(out, 0, ddof=1) / np.sqrt(R)
    return dict(zip(NAMES, m)), dict(zip(NAMES, ci))


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = run(1)
    print(dict(zip(NAMES, np.round(r, 3))), time.time() - t0)
    for kw in [dict(coop=True), dict(p_part=0.0), dict(tau=0.85), dict(aware=True)]:
        t0 = time.time()
        r = run(1, **kw)
        print(kw, np.round(r[[0, 1, 2, 3, 4, 6]], 3), round(time.time() - t0, 2))
