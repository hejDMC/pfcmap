import sys
import yaml
import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
ncluststr = '8'
cmethod = 'ward'
'''



ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']


enrfile1 = os.path.join(pathdict['statsdict_dir'],'enrichmentdict__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))
enrfile2 = os.path.join(pathdict['statsdict_dir'],'enrichmentdictSpecCtrl__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S

reftags = [ 'refPFC', 'refall']
statstag = 'enr'
statsfn = S.enr_fn

#cluster plotting
rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)


somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/Nc%s/%s/enrichments'%(myrun,kshape[0],kshape[1],ncluststr,cmethod))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(labels)}


pol_dict = {reftag:{} for reftag in reftags}

for efile in [enrfile1,enrfile2]:

    with h5py.File(efile,'r') as hand:
        for reftag in reftags:
            for thiskey in hand[reftag][statstag].keys():
                #print(thiskey)
                mystats = uloader.unpack_statshand(hand[reftag][statstag][thiskey]['src_stats'],remove_srcpath=True)
                X = statsfn(mystats)
                X[np.isnan(X)] = 0
                avg_abs_enr = np.abs(X).mean()
                std_enr = np.var(X)
                clustwise_avg_abs_enr = np.abs(X).mean(axis=0)
                clustwise_std_enr = X.var(axis=0)
                pol_dict[reftag][thiskey] = {'enr_amp':avg_abs_enr,'enr_var':std_enr,'enr_ampC':clustwise_avg_abs_enr,'enr_varC':clustwise_std_enr,'N':len(X)}



label_dict = {'enr_amp':'mean(abs(enr.))',\
              'enr_var': 'var(enr.)',\
              'enr_ampC':'mean(abs(enr.))',\
              'enr_varC': 'var(enr.)'}



for reftag in reftags:
    keys_sorted0 = list(pol_dict[reftag].keys())
    keys_sorted = [key for key in keys_sorted0 if not key.count('just') and not key.count('__')]
    keys_sorted = keys_sorted + [key for key in keys_sorted0 if not key in keys_sorted and key.count('__')]
    keys_sorted = keys_sorted + [key for key in keys_sorted0 if not key in keys_sorted]
    n_keys = len(keys_sorted)


    def make_savename(plotname):
        return '%s/%s__%s/%s__%s_%s_%s' % (reftag, 'polarization', statstag, plotname, reftag, 'polarization', statstag)

    for flav in ['enr_var','enr_amp']:

        heights = np.array([pol_dict[reftag][thiskey][flav] for thiskey in keys_sorted])

        f, ax = plt.subplots(figsize=(4.2, 2.5))
        f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
        ax.barh(np.arange(n_keys),heights,color='k')
        ax.set_yticks(np.arange(n_keys))
        ax.set_yticklabels(keys_sorted)
        ax.set_xlabel(label_dict[flav])
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6,integer=True))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, which='major', color='silver', linestyle='dashed')
        f.tight_layout()
        figsaver(f,make_savename('overall_%s'%flav))


        xvec = np.arange(n_keys)

        val_mat = np.array([pol_dict[reftag][thiskey][flav+'C'] for thiskey in keys_sorted])

        f, ax = plt.subplots(figsize=(4.2, 2.5))
        f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
        for cc in np.arange(ncl):
            ax.plot(val_mat[:, cc], xvec, 'o', mec=cdict_clust[cc], mfc='none', color=cdict_clust[cc], mew=2)
        ax.set_yticks(np.arange(n_keys))
        ax.set_yticklabels(keys_sorted)
        ax.set_xlabel(label_dict[flav])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='major', color='silver', linestyle='-')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6,integer=True))
        f.tight_layout()
        figsaver(f,make_savename('clusterwise_%s'%flav))
