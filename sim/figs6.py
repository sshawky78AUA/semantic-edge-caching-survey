"""Figure for the V6 illustrations: (a) precision versus reuse share of a fixed similarity threshold
on two labelled sentence-pair sets for three encoders; (b) semantic x coordination interaction versus r = B*Cb/K."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.linewidth':0.6,'lines.linewidth':1.2,'pdf.fonttype':42})
S=json.load(open('res6_sts.json')); E=json.load(open('res6_e6r_summary.json'))
fig,ax=plt.subplots(1,2,figsize=(7.0,2.35))
col={'spacy-avg':'#999999','MiniLM-L6':'#1f77b4','MPNet-base':'#d62728'}
lab={'spacy-avg':'averaged word vectors','MiniLM-L6':'MiniLM sentence encoder','MPNet-base':'MPNet sentence encoder'}
for dn,ls,dl in (('stsb','-','STS-B'),('qqp','--','Quora')):
    for en,v in S[dn]['enc'].items():
        c=[x for x in v['curve'] if x['precision'] is not None]
        ax[0].plot([x['reuse'] for x in c],[x['precision'] for x in c],ls,color=col[en],marker='o',ms=2.2,label=(lab[en] if dn=='stsb' else None))
    ax[0].axhline(S[dn]['share_same'],color='k',lw=0.5,ls=ls)
ax[0].set_xlabel('Share of requests reused'); ax[0].set_ylabel('Share of reuses that are correct')
ax[0].set_xlim(0,1); ax[0].set_ylim(0.14,1.3); ax[0].set_yticks([0.2,0.4,0.6,0.8,1.0]); ax[0].grid(alpha=0.25)
ax[0].text(0.03,0.165,'thin black lines: share of pairs with the same meaning',fontsize=6)
from matplotlib.lines import Line2D
h,_=ax[0].get_legend_handles_labels()
h+= [Line2D([],[],color='k',ls='-',lw=1),Line2D([],[],color='k',ls='--',lw=1)]
ax[0].legend(h,[lab[k] for k in col]+['STS-B','Quora'],fontsize=5.5,loc='upper right',bbox_to_anchor=(1.0,1.0),ncol=1,frameon=True,framealpha=0.95,edgecolor='none')
ax[0].set_title('(a) Similarity threshold on real sentence pairs',fontsize=8)
for fam,m,l in (('K','o','vary classes $K$ ($C_{\\rm b}=20$)'),('Cb','s','vary BS cache $C_{\\rm b}$ ($K=80$)')):
    e=sorted([x for x in E if x['fam']==fam],key=lambda x:x['r'])
    ax[1].errorbar([x['r'] for x in e],[x['I'] for x in e],yerr=[x['ci'] for x in e],marker=m,ms=3,capsize=1.5,label=l)
ax[1].axhline(0,color='k',lw=0.5); ax[1].axvline(1,color='k',lw=0.5,ls=':')
ax[1].set_xscale('log',base=2); ax[1].set_xticks([0.25,0.5,1,2,4]); ax[1].set_xticklabels(['1/4','1/2','1','2','4'])
ax[1].set_xlabel('$r=BC_{\\rm b}/K$ (distinct BS slots per semantic class)'); ax[1].set_ylabel('Interaction, points of hit ratio')
ax[1].grid(alpha=0.25); ax[1].legend(fontsize=6,frameon=False,loc='lower left')
ax[1].set_title('(b) Meaning $\\times$ coordination on the UCM',fontsize=8)
fig.tight_layout(); fig.savefig('../figs/fig_v6_illus.pdf'); print('ok')
