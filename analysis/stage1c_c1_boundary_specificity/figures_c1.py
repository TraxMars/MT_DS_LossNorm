"""Export figures with fixed method order and visible undefined cells."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bridge import ROOT,STAGES,PRIMARY

def main():
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'attribution_by_case.csv');d=d[(d.split=='probe')&(d.support=='B')&(d.dose==.03)&d.objective.isin(PRIMARY)]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,mode,title in zip(axes,('local_GT_gradient','discovery_B_transfer'),('Probe local GT attribution (secondary)','Frozen discovery scores → probe (primary)')):
        p=d[d['mode']==mode].groupby(['stage','objective']).signed_spearman.mean().unstack().reindex(index=STAGES,columns=PRIMARY)
        counts=d[d['mode']==mode].groupby(['stage','objective']).signed_spearman.count().unstack().reindex(index=STAGES,columns=PRIMARY).to_numpy()
        a=p.to_numpy();cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#cfcfcf')
        im=ax.imshow(np.ma.masked_invalid(a),vmin=-1,vmax=1,cmap=cmap)
        ax.set_xticks(range(5),PRIMARY,rotation=35,ha='right');ax.set_yticks(range(5),STAGES);ax.set_title(title,fontsize=11)
        for i in range(5):
            for j in range(5):
                label='NA' if not np.isfinite(a[i,j]) else f'{a[i,j]:.2f}'+('' if counts[i,j]==4 else f'\nn={int(counts[i,j])}')
                ax.text(j,i,label,ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes,label='Mean of available case correlations (n=4 unless shown)',shrink=.8)
    fig.suptitle('Signed rankings alone do not establish useful attribution: D1 Transition direction accuracy = 0%',fontsize=10)
    fig.savefig(folder/'cross_target_attribution.png',dpi=180);fig.savefig(folder/'cross_target_attribution.pdf');plt.close(fig)
    e=pd.read_csv(ROOT/'endpoint_margin_correctness.csv');e=e[(e.control=='true')&(e.margin==1.)]
    case=e.groupby('case')[['satisfied_fraction','satisfied_and_wrong_fraction','wrong_fraction_given_satisfied']].mean()
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(len(case))
    ax.bar(x-.17,case.satisfied_fraction,.32,label='Margin satisfied / all pairs')
    ax.bar(x+.17,case.satisfied_and_wrong_fraction,.32,label='Margin satisfied and ≥1 wrong endpoint / all pairs')
    ax.set_xticks(x,[str(i+1) for i in x]);ax.set_xlabel('L case (two fixed crops averaged)');ax.set_ylabel('Fraction');ax.set_ylim(0,1);ax.legend(fontsize=8)
    fig.savefig(folder/'margin_and_classification.png',dpi=180);fig.savefig(folder/'margin_and_classification.pdf');plt.close(fig)
    print('Exported four standalone figures (PNG/PDF)')

if __name__=='__main__':main()
