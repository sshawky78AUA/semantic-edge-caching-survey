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


LEG = dict(frameon=False, fontsize=6.5, handlelength=1.9, handletextpad=0.4,
           columnspacing=0.9, labelspacing=0.35, borderaxespad=0.0)


def legend_below(fig, axes, handles=None, labels=None, ncol=2, gap=0.012, row=None):
    """One legend style for every figure: frameless, centered under the axes it
    describes, below their tick labels and x label. Call after the layout is fixed."""
    axes = list(np.atleast_1d(axes))
    if handles is None:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    row = axes if row is None else list(np.atleast_1d(row))  # legends in one row share a top edge
    bbs = [inv.transform(a.get_tightbbox(r).get_points()) for a in row]
    pos = [a.get_position() for a in axes]
    x0, x1 = min(p.x0 for p in pos), max(p.x1 for p in pos)
    y = min(b[0, 1] for b in bbs) - gap
    return fig.legend(handles, labels, loc='upper center', bbox_to_anchor=((x0 + x1) / 2, y),
                      ncol=ncol, **LEG)


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
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 1.8))
    ests = [('oracle', 'Oracle'), ('ewma', 'EWMA'), ('window', 'Window ($W=3$)'),
            ('stale', 'Last epoch'), ('mem', 'Bounded EWMA ($M=40$)')]
    for i, (e, lab) in enumerate(ests):
        rs = [q for q in r if q['tag'] == 'drift' and q['kw']['est'] == e and q['kw']['gamma'] == 0.8]
        x, m, c = series(rs, 'drift', 'hit')
        line(ax[0], x, m, c, i, lab)
    ax[0].set_xlabel(r'Drift rate $\delta$'); ax[0].set_ylabel('Hit ratio $h$')
    tag(ax[0], r'(a) Hit ratio vs. drift ($\gamma=0.8$)')
    ax[0].set_ylim(0.22, 0.56)
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
    rs = [q for q in r if q['tag'] == 'mem']
    x, m, c = series(rs, 'M', 'hit')
    line(ax[2], x, m, c, 4, 'Bounded EWMA')
    for e, i, lab in [('oracle', 0, 'Oracle'), ('ewma', 1, 'EWMA, unbounded')]:
        q = [q for q in r if q['tag'] == 'drift' and q['kw']['est'] == e and q['kw']['gamma'] == 0.8 and q['kw']['drift'] == 0.1][0]
        ax[2].axhline(q['m']['hit'], color=SL[i], ls=LS[i], lw=1, label=lab)
    ax[2].set_xscale('log'); ax[2].set_xlabel('Counters kept $M$ (log scale)')
    ax[2].set_ylabel('Hit ratio $h$'); tag(ax[2], r'(c) Predictor memory ($\delta=0.1$)')
    fig.tight_layout(w_pad=1.0)
    # (b): colour and marker give the skew, line style the estimator
    from matplotlib.lines import Line2D
    hb = [Line2D([], [], color=SL[i], marker=MK[i], ls='-') for i in range(3)] + \
         [Line2D([], [], color=MUTED, ls='-'), Line2D([], [], color=MUTED, ls='--')]
    lb = [r'$\gamma=0.6$', r'$\gamma=0.8$', r'$\gamma=1.0$', 'EWMA', 'Last epoch']
    legend_below(fig, ax[0], ncol=3, row=ax)
    k = [0, 3, 1, 4, 2]                    # row 1: skews; row 2: estimators
    legend_below(fig, ax[1], [hb[i] for i in k], [lb[i] for i in k], ncol=3, row=ax)
    legend_below(fig, ax[2], ncol=2, row=ax)
    fig.savefig(OUT + 'fig_e1_time.pdf'); plt.close(fig)


def fig_e1r():
    """Controls for RQ1: proactive placement versus reactive policies, pooled and per-cell
    information, under the independent reference model (a) and temporal locality (b)."""
    r = json.load(open('res3_e1p.json')) + json.load(open('res4_e1c.json'))
    for q in r:
        q['kw'].setdefault('loc', 0.0)
    pol = [(('oracle', 'pooled'), 'Oracle'), (('ewma', 'pooled'), 'EWMA, pooled'),
           (('ewma', 'local'), 'EWMA, per cell'), (('lru', 'local'), 'LRU'),
           (('lfu', 'local'), 'LFU'), (('lfua', 'local'), 'Aged LFU'),
           (('tlfu', 'local'), 'TinyLFU-style')]
    cols = SL + [INK]
    fig, ax = plt.subplots(1, 2, figsize=(ONE, 2.25), sharey=True)
    for k, loc in enumerate([0.0, 0.5]):
        for i, ((e, inf), lab) in enumerate(pol):
            rs = [q for q in r if q['kw']['est'] == e and q['kw']['info'] == inf and q['kw']['loc'] == loc]
            if not rs:
                continue
            x, m, c = series(rs, 'drift', 'hit')
            st = LS[i] if i < len(LS) else (0, (1, 1))
            ax[k].plot(x, m, color=cols[i], marker=MK[i % len(MK)] if i < 6 else 'x', ls=st, label=lab)
            ax[k].fill_between(x, m - c, m + c, color=cols[i], alpha=0.15, lw=0)
        ax[k].set_xlabel(r'Drift rate $\delta$')
    ax[0].set_ylabel('Hit ratio $h$')
    tag(ax[0], '(a) Independent requests'); tag(ax[1], '(b) Temporal locality')
    fig.tight_layout(w_pad=0.6)
    legend_below(fig, ax, ncol=3)
    fig.savefig(OUT + 'fig_e1r_reactive.pdf'); plt.close(fig)


def fig_e2():
    r = load('e2')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 1.95))
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
    fig.tight_layout(w_pad=1.0)
    legend_below(fig, ax[:2], ncol=4, row=ax)
    legend_below(fig, ax[2], ncol=2, row=ax)
    fig.savefig(OUT + 'fig_e2_people.pdf'); plt.close(fig)


def fig_e3():
    r = load('e3')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 1.8))
    pol = [(False, 'Proportional'), ('ref', 'Tuned to 0.5 m/s'), (True, 'Mobility-aware')]
    cut = np.pi * 100 / (4 * 5)
    for i, (aw, lab) in enumerate(pol):
        rs = sel(r, aware=aw)
        for k, y in enumerate(['d2d', 'hit', 'lat']):
            x, m, c = series(rs, 'v', y); line(ax[k], x, m, c, i, lab)
    rt = json.load(open('res4_e3t.json'))
    for k, y in enumerate(['d2d', 'hit', 'lat']):
        x, m, c = series(rt, 'v', y)
        ax[k].plot(x, m, color=INK, marker='x', ls=(0, (1, 1)), label='Mobility-aware, trajectory rule')
        ax[k].fill_between(x, m - c, m + c, color=INK, alpha=0.1, lw=0)
    for a in ax:
        a.axvline(cut, color=MUTED, lw=0.8, ls=':')
        a.set_xlabel('Speed $v$ (m/s)')
    ax[0].text(cut + 0.6, ax[0].get_ylim()[1] * 0.45, 'cutoff\n' + r'$\pi r/(4t_f)$', color=MUTED, fontsize=7)
    ax[0].set_ylabel('D2D share $h_1$'); tag(ax[0], '(a) D2D share vs. speed')
    ax[1].set_ylabel('Hit ratio $h$'); tag(ax[1], '(b) Hit ratio vs. speed')
    ax[2].set_ylabel(r'Mean latency $\bar d$ (ms)'); tag(ax[2], '(c) Latency vs. speed')
    fig.tight_layout(w_pad=1.0); legend_below(fig, ax, ncol=4); fig.savefig(OUT + 'fig_e3_place.pdf'); plt.close(fig)


def fig_e4():
    r = load('e4')
    fig, ax = plt.subplots(1, 3, figsize=(TWO, 1.8))
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
    ax[1].invert_xaxis(); ax[1].set_xlabel(r'Mean fidelity $\bar\phi$ of hits')
    ax[1].set_ylabel('Hit ratio $h$'); tag(ax[1], '(b) Capacity vs. fidelity')
    ax[2].invert_xaxis(); ax[2].set_xlabel(r'Threshold $\tau$')
    ax[2].set_ylabel(r'Mean latency $\bar d$ (ms)'); tag(ax[2], '(c) Latency vs. threshold')
    fig.tight_layout(w_pad=1.0); legend_below(fig, ax, ncol=4); fig.savefig(OUT + 'fig_e4_meaning.pdf'); plt.close(fig)


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
    fig, axs = plt.subplots(2, 3, figsize=(TWO, 3.7), gridspec_kw=dict(width_ratios=[1.3, 1, 1], hspace=0.62, wspace=0.12))
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
    tag(axa, '(a) Hit ratio effects'); tag(axb, '(b) SP cost effects')
    # ---- (c) all 32 designs ----
    ax = axf
    L = np.array([q['levels'] for q in r]); h = np.array([q['m']['hit'] for q in r])
    o = np.argsort(h); grp = L[o, 2] * 2 + L[o, 4]
    names = ['Exact', 'Semantic', 'Exact, coordinated', 'Semantic, coordinated']
    for g in range(4):
        sg = grp == g
        ax.scatter(np.arange(32)[sg], h[o][sg], s=9, color=SL[g], marker=MK[g], label=names[g], zorder=3)

    ax.annotate(f'worst {h.min():.3f}', (0, h.min()), (4, 0.33), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    ax.annotate(f'best {h.max():.3f}', (25, h.max() - 0.02), (6, 0.955), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5))
    ax.set_ylim(0.3, 1.0); ax.set_xticks([0, 15, 31]); ax.set_xticklabels(['1', '16', '32'])
    ax.set_xlabel('Design (sorted by $h$)'); ax.set_ylabel('Hit ratio $h$')
    ax.grid(axis='x', visible=False); tag(ax, '(c) All 32 designs')
    # ---- (d)-(f): interaction plots on h ----
    specs = [(2, 4, 'Coordination', ['off', 'on'], 'Matching', ['exact', 'semantic'], '(d) Reinforce'),
             (2, 1, 'Coordination', ['off', 'on'], 'Participation', ['30%', '100%'], '(e) Substitute'),
             (0, 4, 'Prediction', ['last ep.', 'oracle'], 'Matching', ['exact', 'semantic'], '(f) Easier prediction')]
    for k, (fa, fb, na, la, nb, lb, ttl) in enumerate(specs):
        ax = axs[1, k]
        m, c = cell_means(r, 11, fa, fb)
        for j in (0, 1):
            col = MUTED if j == 0 else SL[2]
            ax.errorbar([0, 1], m[:, j], yerr=c[:, j], color=col, ls='none', capsize=1.5, elinewidth=0.6)
            ax.plot([0, 1], m[:, j], color=col, marker=MK[j], ls=LS[j],
                    label=f'{nb} {lb[j]}' if nb == 'Participation' else f'{lb[j].capitalize()} matching')
        gap0, gap1 = m[0, 1] - m[0, 0], m[1, 1] - m[1, 0]
        ax.text(0.5, 0.04, f'gap {100*gap0:.1f} $\\to$ {100*gap1:.1f}', transform=ax.transAxes,
                ha='center', fontsize=7, color=MUTED)
        ax.set_xticks([0, 1]); ax.set_xticklabels(la); ax.set_xlabel(na); ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel('Hit ratio $h$')
        ax.grid(axis='x', visible=False); tag(ax, ttl)
    fig.subplots_adjust(left=0.2, right=0.995, top=0.95, bottom=0.1, wspace=0.46, hspace=1.0)
    from matplotlib.patches import Patch
    legend_below(fig, [axa, axb], [Patch(color=SL[0]), Patch(color=SL[1])],
                 ['Main effect', 'Two-way interaction (top 4)'], ncol=2, row=axs[0])
    legend_below(fig, axf, ncol=2, row=axs[0])
    for k in range(3):
        legend_below(fig, axs[1, k], ncol=1, row=axs[1])
    fig.savefig(OUT + 'fig_e5_factorial.pdf'); plt.close(fig)


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    for n in sys.argv[1:]:
        globals()['fig_' + n]()
        print('fig', n)
