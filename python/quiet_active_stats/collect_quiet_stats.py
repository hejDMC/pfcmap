import h5py
from glob import glob
import os
import numpy as np
from scipy.stats import scoreatpercentile as sap
import matplotlib.pyplot as plt

srcdir = 'ZENODOPATH/quiet_active_detection/quiet_active_stats'
figdir = 'FIGDIR/quiet_active_stats'

if not os.path.isdir(figdir): os.makedirs(figdir)

myfiles = glob(os.path.join(srcdir,'*.h5'))

ddict = {}
for srcfile in myfiles:
    with h5py.File(srcfile,'r') as hand:
        recid = hand.attrs['recid']
        ddict[recid] = {}
        for dsname in  ['recdur','artdur','quietdur','transdur','offdur']:
            ddict[recid][dsname] = float(hand[dsname][()])
        for dsname in ['offtimes','transtimes','quiettimes']:
            ddict[recid][dsname] = hand[dsname][()]

results = {}
for recid,subd in ddict.items():
    dur_analyzed = subd['recdur'] - subd['artdur']
    results[recid] = {}
    results[recid]['frac_quiet'] = subd['quietdur']/dur_analyzed
    results[recid]['frac_off'] = subd['offdur']/dur_analyzed
    results[recid]['frac_trans'] = subd['transdur']/dur_analyzed
    ratio = subd['transdur']/subd['offdur'] if subd['offdur']>0 else np.nan
    results[recid]['trans_off_ratio'] = ratio


percs = [25,50,75]
myvars = ['frac_quiet','frac_off','frac_trans','trans_off_ratio']
for myvar in myvars:
    myvals = np.array([subd[myvar] for subd in results.values() if not np.isnan(subd[myvar])])
    scorevals = [sap(myvals,perc) for perc in percs]
    f,ax = plt.subplots()
    ax.hist(myvals,color='k')
    for pp,[scoreval,linestyle,percentile] in enumerate(zip(scorevals,[':','-',':'],percs)):
        ax.axvline(scoreval,linestyle=linestyle,color='firebrick')
        ax.text(0.95,0.95-pp*0.06,'%ith perc: %1.4f'%(percentile,scoreval),color='firebrick',transform=ax.transAxes,ha='right',va='top')
    ax.set_xlabel(myvar)
    ax.set_ylabel('count')
    f.savefig(os.path.join(figdir,'%s__quietActiveStats.svg'%myvar))
    plt.close(f)
