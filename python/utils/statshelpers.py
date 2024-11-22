
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def getplot_corrmat_clusters(enrmat,corrfnstr='pearsonr',**kwargs):
    '''enrmat is enrichment-mat like in nregs x nclust'''
    corrfn = getattr(stats,corrfnstr)
    nregs,ncl = enrmat.shape
    corrmat,pmat = np.zeros((ncl,ncl))*np.nan,np.zeros((ncl,ncl))*np.nan
    for nn in np.arange(ncl):
        for mm in np.arange(nn+1,ncl):
            res = corrfn(enrmat[:,nn],enrmat[:,mm])
            pmat[nn,mm] = res.pvalue
            corrmat[nn,mm] = res.correlation

    if not 'sortidx' in kwargs:
        sortidx = np.arange(np.arange(ncl))
    else: sortidx = kwargs['sortidx']
    ticklabs = (np.arange(ncl)+1)[sortidx]
    corrmat2 = np.triu(corrmat) + np.tril(corrmat.T)
    vals = corrmat[np.triu_indices(ncl,1)]
    limval = np.max([np.abs(vals)])
    f,axarr = plt.subplots(1,2,figsize=(4.5,4),gridspec_kw={'width_ratios':[1,0.1]})
    f.subplots_adjust(wspace=0.1,right=0.87)
    ax,cax = axarr
    ax.text(0.99,1.01,'N:%i'%nregs,ha='right',va='bottom',transform=ax.transAxes)
    im = ax.imshow(corrmat2, cmap='coolwarm', aspect='equal', origin='lower',vmin=-limval,vmax=limval)
    for nn in np.arange(ncl):
        for mm in np.arange(nn+1,ncl):
            ax.text(nn,mm,'%1.2f\n%1.2e'%(corrmat[nn,mm],pmat[nn,mm]),ha='center',va='center',fontsize=6)
    f.colorbar(im,cax)
    cax.set_title('corr')
    ax.set_xticks(np.arange(ncl))
    ax.set_yticks(np.arange(ncl))
    tlabs1 = ax.set_xticklabels(ticklabs,fontweight='bold')
    tlabs2 = ax.set_yticklabels(ticklabs,fontweight='bold')
    if 'cdict_clust' in kwargs:
        cdict_clust = kwargs['cdict_clust']
        for ii in np.arange(ncl):
            col = cdict_clust[sortidx[ii]]
            tlabs1[ii].set_color(col)
            tlabs2[ii].set_color(col)

    f.tight_layout()
    return f




