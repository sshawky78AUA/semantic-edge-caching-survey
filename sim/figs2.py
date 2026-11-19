"""Figures for Section IX from res2_*.json. Vector PDF, IEEE column widths.
Palette: validated categorical slots (fixed order); markers and line styles give a
second channel so identity never rests on color alone (print and CVD readers)."""
import json, sys, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MK = ['o', 's', '^', 'D', 'v', 'P']
LS = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1))]
INK, MUTED, GRID, AXIS = '#1f1f1d', '#5c5c57', '#e1e0d9', '#c3c2b7'
ONE, TWO = 3.5, 7.16
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
    'lines.markersize': 3.5, 'legend.frameon': False, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02})
OUT = '../figs/'


def load(n):
    return json.load(open(f'res2_{n}.json'))


def sel(res, **kw):
    out = [r for r in res if all(r['kw'].get(k) == v for k, v in kw.items())]
    return out


def series(rs, x, y, filt=None):
    rs = sorted(rs, key=lambda r: r['kw'][x])
    xs = np.array([r['kw'][x] for r in rs], float)
    m = np.array([r['m'][y] for r in rs]); c = np.array([r['ci'][y] for r in rs])
    return xs, m, c


def line(ax, x, m, c, i, label, **kw):
    ax.plot(x, m, color=SL[i], marker=MK[i], ls=LS[i], label=label, **kw)
    ax.fill_between(x, m - c, m + c, color=SL[i], alpha=0.15, lw=0)


def tag(ax, t):
    ax.set_title(t, loc='left', fontsize=8, color=INK)


def fig_e1():
    r = load('e1')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.05))
    ests = [('oracle', 'Oracle'), ('ewma', 'EWMA'), ('window', 'Window ($W=3$)'),
            ('stale', 'Last epoch'), ('mem', 'Bounded EWMA ($M=40$)')]
    for i, (e, lab) in enumerate(ests):
        rs = [q for q in r if q['tag'] == 'drift' and q['kw']['est'] == e and q['kw']['gamma'] == 0.8]
        x, m, c = series(rs, 'drift', 'hit')
        line(ax[0], x, m, c, i, lab)
    ax[0].set_xlabel(r'Drift rate $\delta$'); ax[0].set_ylabel('Hit ratio $h$')
    tag(ax[0], r'(a) Hit ratio vs. drift ($\gamma=0.8$)')
    ax[0].set_ylim(0.22, 0.56); ax[0].legend(loc='lower left', ncol=1)
    # (b) regret of EWMA and last epoch vs drift for three gamma, regret = oracle - est
    for i, g in enumerate([0.6, 0.8, 1.0]):
        for e, lsx in [('ewma', '-'), ('stale', '--')]:
            o = sorted([q for q in r if q['tag'] == 'drift' and q['kw']['est'] == 'oracle' and q['kw']['gamma'] == g], key=lambda q: q['kw']['drift'])
            s = sorted([q for q in r if q['tag'] == 'drift' and q['kw']['est'] == e and q['kw']['gamma'] == g], key=lambda q: q['kw']['drift'])
            x = np.array([q['kw']['drift'] for q in o])
            d = np.array(o and [np.array(a['raw'])[:, 11] - np.array(b['raw'])[:, 11] for a, b in zip(o, s)])
            m = d.mean(1); c = 1.96 * d.std(1, ddof=1) / np.sqrt(d.shape[1])
            ax[1].plot(x, m, color=SL[i], marker=MK[i], ls=lsx,
                       label=fr'$\gamma={g}$, ' + ('EWMA' if e == 'ewma' else 'last epoch'))
            ax[1].fill_between(x, m - c, m + c, color=SL[i], alpha=0.15, lw=0)
    ax[1].set_xlabel(r'Drift rate $\delta$'); ax[1].set_ylabel('Regret (oracle $-$ estimator)')
    tag(ax[1], '(b) Regret vs. drift and skew')
    ax[1].legend(loc='upper left', ncol=1)
    rs = [q for q in r if q['tag'] == 'mem']
    x, m, c = series(rs, 'M', 'hit')
    line(ax[2], x, m, c, 4, 'Bounded EWMA')
    for e, i, lab in [('oracle', 0, 'Oracle'), ('ewma', 1, 'EWMA, unbounded')]:
        q = [q for q in r if q['tag'] == 'drift' and q['kw']['est'] == e and q['kw']['gamma'] == 0.8 and q['kw']['drift'] == 0.1][0]
        ax[2].axhline(q['m']['hit'], color=SL[i], ls=LS[i], lw=1, label=lab)
    ax[2].set_xscale('log'); ax[2].set_xlabel('Counters kept $M$ (log scale)')
    ax[2].set_ylabel('Hit ratio $h$'); tag(ax[2], r'(c) Predictor memory ($\delta=0.1$)')
    ax[2].legend(loc='lower right')
    fig.tight_layout(w_pad=1.0); fig.savefig(OUT + 'fig_e1_time.pdf'); plt.close(fig)


def fig_e1r():
    """Reactive (LRU, LFU) versus proactive BS caches in a BS-only network."""
    r = load('e1r')
    fig, ax = plt.subplots(1, 2, figsize=(ONE, 2.3))
    ests = [('oracle', 'Oracle'), ('ewma', 'EWMA'), ('stale', 'Last epoch'),
            ('lru', 'LRU (reactive)'), ('lfu', 'LFU (reactive)')]
    for i, (e, lab) in enumerate(ests):
        x, m, c = series(sel(r, est=e), 'drift', 'hit')
        line(ax[0], x, m, c, i, lab)
    ax[0].set_xlabel(r'Drift rate $\delta$'); ax[0].set_ylabel('Hit ratio $h$')
    tag(ax[0], '(a) Hit ratio, BS only')
    for i, (e, lab) in enumerate(ests):
        x, m, c = series(sel(r, est=e), 'drift', 'lat')
        line(ax[1], x, m, c, i, lab)
    ax[1].set_xlabel(r'Drift rate $\delta$'); ax[1].set_ylabel('Mean latency (ms)')
    tag(ax[1], '(b) Mean latency')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.17, 1, 1)); fig.savefig(OUT + 'fig_e1r_reactive.pdf'); plt.close(fig)


def fig_e2():
    r = load('e2')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.45))
    cfg = [(2.0, False, 'Independent, 2 m/s'), (2.0, True, 'Coordinated, 2 m/s'),
           (10.0, False, 'Independent, 10 m/s'), (10.0, True, 'Coordinated, 10 m/s')]
    for i, (v, co, lab) in enumerate(cfg):
        rs = sel(r, v=v, coop=co)
        x, m, c = series(rs, 'p_part', 'hit'); line(ax[0], x, m, c, i, lab)
        x, m, c = series(rs, 'p_part', 'sp_cost'); line(ax[1], x, m, c, i, lab)
        k = int(np.argmin(m)); ax[1].plot(x[k], m[k], marker='*', ms=8, color=SL[i], mec=INK, mew=0.5, ls='none')
    ax[0].set_xlabel(r'Reward level $\rho$'); ax[0].set_ylabel('Hit ratio $h$')
    tag(ax[0], '(a) Hit ratio vs. reward')
    ax[1].set_xlabel(r'Reward level $\rho$'); ax[1].set_ylabel(r'SP cost $J_{\mathrm{SP}}$ per request')
    tag(ax[1], r'(b) SP cost ($\star$: $\rho^\star$)')
    for i, (co, lab) in enumerate([(False, 'Independent'), (True, 'Coordinated')]):
        rs = [q for q in sel(r, v=2.0, coop=co) if 0 < q['kw']['p_part'] < 1]
        x, m, c = series(rs, 'p_part', 'lat_part'); line(ax[2], x, m, c, 2 * i, lab + ', participants')
        x, m, c = series(rs, 'p_part', 'lat_non'); line(ax[2], x, m, c, 2 * i + 1, lab + ', free riders')
    ax[2].set_xlabel(r'Reward level $\rho$'); ax[2].set_ylabel(r'Mean latency $\bar d$ (ms)')
    tag(ax[2], '(c) Who gains (2 m/s)')
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    h0, l0 = ax[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc='lower center', bbox_to_anchor=(0.35, 0.0), ncol=2, columnspacing=1.0)
    h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h2, l2, loc='lower center', bbox_to_anchor=(0.84, 0.0), ncol=2, columnspacing=0.6)
    fig.savefig(OUT + 'fig_e2_people.pdf'); plt.close(fig)


def fig_e3():
    r = load('e3')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.05))
    pol = [(False, 'Proportional'), ('ref', 'Tuned to 0.5 m/s'), (True, 'Mobility-aware')]
    cut = np.pi * 100 / (4 * 5)
    for i, (aw, lab) in enumerate(pol):
        rs = sel(r, aware=aw)
        for k, y in enumerate(['d2d', 'hit', 'lat']):
            x, m, c = series(rs, 'v', y); line(ax[k], x, m, c, i, lab)
    for a in ax:
        a.axvline(cut, color=MUTED, lw=0.8, ls=':')
        a.set_xlabel('Speed $v$ (m/s)')
    ax[0].text(cut + 0.6, ax[0].get_ylim()[1] * 0.45, 'cutoff\n' + r'$\pi r/(4t_f)$', color=MUTED, fontsize=7)
    ax[0].set_ylabel('D2D share $h_1$'); tag(ax[0], '(a) D2D share vs. speed')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Hit ratio $h$'); tag(ax[1], '(b) Hit ratio vs. speed')
    ax[2].set_ylabel(r'Mean latency $\bar d$ (ms)'); tag(ax[2], '(c) Latency vs. speed')
    fig.tight_layout(w_pad=1.0); fig.savefig(OUT + 'fig_e3_place.pdf'); plt.close(fig)


def fig_e4():
    r = load('e4')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.05))
    cfg = [(10, False, '$C_b=10$'), (20, False, '$C_b=20$'), (40, False, '$C_b=40$'),
           (20, True, '$C_b=20$, coordinated')]
    for i, (cb, co, lab) in enumerate(cfg):
        rs = sel(r, Cb=cb, coop=co)
        x, m, c = series(rs, 'tau', 'hit'); line(ax[0], x, m, c, i, lab)
        x, m, c = series(rs, 'tau', 'lat'); line(ax[2], x, m, c, i, lab)
        rs = sorted(rs, key=lambda q: -q['kw']['tau'])
        f = [q['m']['fid'] for q in rs]; h = [q['m']['hit'] for q in rs]
        ax[1].plot(f, h, color=SL[i], marker=MK[i], ls=LS[i], label=lab)
    ax[0].invert_xaxis(); ax[0].set_xlabel(r'Threshold $\tau$')
    ax[0].set_ylabel('Hit ratio $h$'); tag(ax[0], '(a) Hit ratio vs. threshold')
    ax[0].legend(loc='upper left')
    ax[1].invert_xaxis(); ax[1].set_xlabel(r'Mean fidelity $\bar\phi$ of hits')
    ax[1].set_ylabel('Hit ratio $h$'); tag(ax[1], '(b) Capacity vs. fidelity')
    ax[2].invert_xaxis(); ax[2].set_xlabel(r'Threshold $\tau$')
    ax[2].set_ylabel(r'Mean latency $\bar d$ (ms)'); tag(ax[2], '(c) Latency vs. threshold')
    fig.tight_layout(w_pad=1.0); fig.savefig(OUT + 'fig_e4_meaning.pdf'); plt.close(fig)


FAC = ['Time', 'People', 'SP coord.', 'Place', 'Meaning']


def effects(res, idx):
    """Main effects and two-way interactions with per-seed CIs (common random numbers)."""
    L = np.array([q['levels'] for q in res]) * 2 - 1        # cells x 5, +-1
    Y = np.array([np.array(q['raw'])[:, idx] for q in res])  # cells x runs
    out = []
    for k in range(5):
        s = L[:, k]
        d = Y[s > 0].mean(0) - Y[s < 0].mean(0)
        out.append((FAC[k], d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    for a, b in itertools.combinations(range(5), 2):
        s = L[:, a] * L[:, b]
        d = Y[s > 0].mean(0) - Y[s < 0].mean(0)
        out.append((FAC[a] + r' $\times$ ' + FAC[b], d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))))
    return out


FAC_LONG = ['Prediction (oracle)', 'Participation (100%)', 'SP coordination',
            'Mobility-aware place.', r'Semantic ($\tau$=0.85)']
FAC_S = ['Prediction', 'Participation', 'Coordination', 'Placement', 'Semantic']
FAC_X = ['Pred.', 'Partic.', 'Coord.', 'Place.', 'Sem.']


def cell_means(res, idx, a, b):
    """Mean of metric idx over the other factors for each (a,b) level pair, with per-seed CI."""
    L = np.array([q['levels'] for q in res]); Y = np.array([np.array(q['raw'])[:, idx] for q in res])
    m = np.zeros((2, 2)); c = np.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            d = Y[(L[:, a] == i) & (L[:, b] == j)].mean(0)
            m[i, j] = d.mean(); c[i, j] = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return m, c


def fig_e5():
    r = load('e5')
    fig, axs = plt.subplots(2, 3, figsize=(TWO, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1, 1], hspace=0.62, wspace=0.12))
    axa, axb, axf = axs[0]
    # ---- (a), (b): effects with value labels ----
    for ax, idx, scale, fmt, lab in [(axa, 11, 100, '{:+.1f}', 'Effect on $h$ (points)'),
                                     (axb, 9, 1, '{:+.3f}', r'Effect on $J_{\mathrm{SP}}$ ($-$ = cheaper)')]:
        e = effects(r, idx)
        main = [(FAC_LONG[k], e[k][1]) for k in range(5)]
        inter = []
        for (n, v, _), (p, q) in zip(e[5:], itertools.combinations(range(5), 2)):
            inter.append((FAC_X[p] + r' $\times$ ' + FAC_X[q], v))
        if idx == 11:
            fig_e5.order = sorted(range(10), key=lambda t: -abs(inter[t][1]))[:4]
        rows = main + [inter[t] for t in fig_e5.order]
        y = np.arange(len(rows))[::-1].astype(float); y[5:] -= 0.4
        vals = np.array([t[1] for t in rows]) * scale
        col = [SL[0]] * 5 + [SL[1]] * 4
        ax.barh(y, vals, color=col, height=0.66, edgecolor='white', linewidth=1.0)
        span = np.ptp(np.r_[vals, 0])
        for yy, v in zip(y, vals):
            ax.text(v + (0.015 * span if v >= 0 else -0.015 * span), yy, fmt.format(v).replace('-', '\u2212'),
                    va='center', ha='left' if v >= 0 else 'right', fontsize=6.5, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([t[0] for t in rows])
        ax.axvline(0, color=INK, lw=0.7); ax.set_xlabel(lab); ax.grid(axis='y', visible=False)
        ax.axhline((y[4] + y[5]) / 2, color=AXIS, lw=0.7, ls=':')
        lo, hi = vals.min(), vals.max(); pad = 0.32 * span
        ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
    plt.setp(axb.get_yticklabels(), visible=False); axb.tick_params(axis='y', length=0)
    axa.text(0.99, 0.30, 'main effects', transform=axa.transAxes, ha='right', fontsize=6.5, color=SL[0])
    axa.text(0.99, 0.22, 'interactions (top 4)', transform=axa.transAxes, ha='right', fontsize=6.5, color=SL[1])
    tag(axa, '(a) Hit ratio effects'); tag(axb, '(b) SP cost effects')
    # ---- (c) all 32 designs ----
    ax = axf
    L = np.array([q['levels'] for q in r]); h = np.array([q['m']['hit'] for q in r])
    o = np.argsort(h); grp = L[o, 2] * 2 + L[o, 4]
    names = ['exact', 'semantic', 'exact+coord.', 'sem.+coord.']
    for g in range(4):
        sg = grp == g
        ax.scatter(np.arange(32)[sg], h[o][sg], s=9, color=SL[g], marker=MK[g], label=names[g], zorder=3)
    ax.annotate(f'worst {h.min():.3f}', (0, h.min()), (4, 0.33), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    ax.annotate(f'best {h.max():.3f}', (25, h.max() - 0.02), (6, 0.955), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    ax.set_ylim(0.3, 1.0); ax.set_xticks([0, 15, 31]); ax.set_xticklabels(['1', '16', '32'])
    ax.set_xlabel('Design (sorted by $h$)'); ax.set_ylabel('Hit ratio $h$')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.92), fontsize=6.5, handlelength=0.8, handletextpad=0.3, labelspacing=0.3)
    ax.grid(axis='x', visible=False); tag(ax, '(c) All 32 designs')
    # ---- (d)-(f): interaction plots on h ----
    specs = [(2, 4, 'Coordination', ['off', 'on'], 'Matching', ['exact', 'semantic'], '(d) Reinforce'),
             (2, 1, 'Coordination', ['off', 'on'], 'Participation', ['30%', '100%'], '(e) Substitute'),
             (0, 4, 'Prediction', ['last ep.', 'oracle'], 'Matching', ['exact', 'semantic'], '(f) Easier prediction')]
    for k, (fa, fb, na, la, nb, lb, ttl) in enumerate(specs):
        ax = axs[1, k]
        m, c = cell_means(r, 11, fa, fb)
        for j in (0, 1):
            ax.errorbar([0, 1], m[:, j], yerr=c[:, j], color=MUTED if j == 0 else SL[2], marker=MK[j],
                        ls=LS[j], capsize=1.5, elinewidth=0.6)
        hi = 1 if m[1, 1] >= m[1, 0] else 0
        dy = 0.035 if abs(m[1, 1] - m[1, 0]) < 0.06 else 0.0
        for j in (0, 1):
            ax.text(1.08, m[1, j] + (dy if j == hi else -dy), lb[j], va='center', fontsize=7,
                    color=MUTED if j == 0 else SL[2])
        gap0, gap1 = m[0, 1] - m[0, 0], m[1, 1] - m[1, 0]
        ax.text(0.5, 0.04, f'gap {100*gap0:.1f} $\\to$ {100*gap1:.1f}', transform=ax.transAxes,
                ha='center', fontsize=7, color=MUTED)
        ax.text(0.03, 0.97, nb, transform=ax.transAxes, va='top', fontsize=7, color=MUTED, style='italic')
        ax.set_xticks([0, 1]); ax.set_xticklabels(la); ax.set_xlabel(na); ax.set_xlim(-0.25, 1.75)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel('Hit ratio $h$')
        ax.grid(axis='x', visible=False); tag(ax, ttl)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.95, bottom=0.1, wspace=0.46, hspace=0.62)
    fig.savefig(OUT + 'fig_e5_factorial.pdf'); fig.savefig('/tmp/fig_e5.png', dpi=220); plt.close(fig)


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:]:
        globals()['fig_' + n]()
        print('fig', n)
