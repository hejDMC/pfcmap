import h5py
import sys
import yaml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import git
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
import pandas as pd

pathpath, myrun, ncluststr, cmethod = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'#
#myrun = 'runIBLPasdMP3_brain_pj'
ncluststr = '8'
cmethod = 'ward'
'''
# clust_of_interest_dict = { 'runC00dMP3_brain':{8:[0,1,2]},\
#                           'runIBLPasdMP3_brain_pj':{8:[0,1,2]},\
#                           'runC00dMI3_brain':{5:[0,1,2]},\
#                           'runIBLPasdMI3_brain_pj':{5:[0,1,2]}}


'''si_thr = 0
res_vec = np.linspace(0.0,2,70)
calgo_fndict = {'leiden': lambda mygraph: mygraph.community_leiden,\
                'louvain': lambda mygraph: mygraph.community_multilevel}
weightfn_dict = {'adj': lambda mygraph: mygraph.es['weight'],\
                'equal':lambda mygraph: len(mygraph.es['weight'])*[1]}'''

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import graphs as gfns

simfn_dict = {'corr': lambda Xmat: np.corrcoef(Xmat),
              'SI': lambda Xmat: gfns.get_similarity_mat(Xmat), \
              'coss': lambda Xmat: cosine_similarity(Xmat)}

sim_mode = 'coss'

'''weight_mode = 'adj'
comalgo = 'leiden'
n_max_mods = 8 #maximum number of modules to consider when sorting vertically
'''

ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

statsfile = os.path.join(pathdict['statsdict_dir'], 'statsdict__%s__ncl%s_%s.h5' % (myrun, ncluststr, cmethod))

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder, myrun)

somfile = uloader.get_somfile(rundict, myrun, savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns', myrun, '%s__kShape%i_%i/clustering/Nc%s/%s/enrichments' % (
myrun, kshape[0], kshape[1], ncluststr, cmethod))



def make_tablepath(nametag):
    return os.path.join(figdir_mother, '%s__%s.xlsx' % (nametag,myrun))


# figsaver


ddict = uloader.extract_clusterlabeldict(somfile, cmethod, get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)  #
cdict_clust = {lab: cmap_clust(norm(lab)) for lab in np.unique(labels)}

# nreps_simshuff = 100
# graphdisp_ew_boost = 3 #just for graph display


# what to sort and not sort in the statsdict according to community identified
# sortkeys = ['avals1','levels','matches','meanshuff','stdshuff','pofs','alabels','commlabels']
# sortkeysT = ['communities']
# sortkeysD = ['A','S']
# nonsortkeys = ['src','qvec','nvec','simshuff_mean','simshuff_std','comm_idx']
# allsortkeys = sortkeys + sortkeysT + sortkeysD + nonsortkeys
reftags = ['refall', 'refPFC']
# statsfns,statstags = [S.enr_fn,S.frac_fn],['enr','frac']
statstag = 'enr'
statsfn = S.enr_fn

repo = git.Repo(search_parent_directories=True)
omnidict = {reftag: {statstag: {} for statstag in [statstag]} for reftag in reftags}
omnidict['methods'] = {'sim_mode': sim_mode,
                       'git_hash': repo.head.object.hexsha}  # ,'comalgo':comalgo,'weight_mode':weight_mode

simfn = simfn_dict[sim_mode]

'''
comfn = calgo_fndict[comalgo]
weightfn = weightfn_dict[weight_mode]
'''

replace_dict = {'|deep': '|d', '|sup': '|s'}


def replace_fn(mystr):
    for oldstr, newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr


# cois = np.array(clust_of_interest_dict[myrun][ncl])


# reftag = 'refall'
with h5py.File(statsfile, 'r') as hand:
    for reftag in reftags:
        ####
        thistag = 'deepCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']  # from this select the deep ones
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if
                                countvec[aa] > S.Nmin_maps and not aval.count('|sup') and not aval == 'na' \
                                and not (S.check_pfc_full(aval) and not aval.count('|deep'))])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}

        ###
        thistag = 'layeredregsCtx'
        statshand = hand['regs'][reftag]['lays']
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array(
            [aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}

        ####
        thistag = 'deepSupCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array(
            [aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps and not aval == 'na' \
             and not (S.check_pfc_full(aval) and not aval.count('|'))])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}

        ####
        thistag = 'deepSupCtx'
        statshand = hand['regs'][reftag]['laydepth']  # from this select the deep ones
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array(
            [aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}

        ####
        thistag = 'justlayers'
        statshand = hand['layers'][reftag]['justlays']
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array(
            [aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps and not aval == 'na'])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}

        ####
        thistag = 'justtasks'
        statshand = hand['tasks'][reftag]['nolays']
        mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps])
        sdict = {key: mystats[key][presel_inds] for key in
                 ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats': sdict}


# reftag = 'refall'
# thistag = 'deepCtx_nolayOtherwise'
# reftag = 'refall'
# thistag = seltags[0]
# srcdict = omnidict[reftag][statstag][thistag]['src_stats']

seltags = [key for key in omnidict[reftag][statstag].keys()]
if myrun.count('IBL'):
    seltags = [stag for stag in seltags if not stag.count('task')]
for reftag in reftags:
    for thistag in seltags:

        print('getting and plotting modules for %s %i - %s %s %s' % (myrun, ncl, reftag, statstag, thistag))
        # reftag, statstag, thistag,statsfn = 'refall','enr','deepCtx_nolayOtherwise',lambda x: S.enr_fn(x)
        ddict = omnidict[reftag][statstag][thistag]


        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s' % (reftag, thistag, statstag, plotname, reftag, thistag, statstag)


        srcdict = ddict['src_stats']

        X = statsfn(srcdict)
        Smat = simfn(X)  # similarity matrix
        # srcdict['S'] = Smat
        plX = np.vstack([vals for vals in X])
        plX[srcdict['levels'] == 0] = 0

        distbaseMat = srcdict['levels']

        srcdict['S'] = simfn(distbaseMat)

        # getting sortinds
        dists = pdist(distbaseMat)
        Z = hac.linkage(dists, method='ward', optimal_ordering=True)  #
        dn = hac.dendrogram(Z, no_plot=True)
        sortinds = np.array(dn['leaves'])

        sorted_labels = srcdict['alabels'][sortinds]
        enrmat = X[sortinds]
        levelmat = srcdict['levels'][sortinds]
        countmat = srcdict['matches'][sortinds]
        countmat_ext = np.vstack([countmat,countmat.sum(axis=0)[None,:]])
        countmat_ext = np.hstack([countmat_ext,countmat_ext.sum(axis=1)[:,None]])

        pofmat = srcdict['pofs'][sortinds]
        pvalmat = np.zeros_like(pofmat)*np.nan
        pvalmat[pofmat<50] = pofmat[pofmat<50]/100.
        pvalmat[pofmat>=50] = (100-pofmat[pofmat>=50])/100.


        outfilename = make_tablepath(make_savename('allstatsEnr'))



        with pd.ExcelWriter(outfilename, engine='openpyxl') as writer:
            for sheetname,mydata in zip(['enrvals', 'pvals', 'counts','pofs','plevels'],[enrmat,pvalmat,countmat_ext,pofmat,levelmat]):
                if sheetname == 'counts':
                    mydict = {cname:{thislab:mydata[jj,cc] for jj,thislab in enumerate(list(sorted_labels)+['SUM'])} \
                              for cc,cname in enumerate([str(cnum+1) for cnum in np.arange(ncl)]+['SUM'])}

                else:
                    mydict = {cc+1:{thislab:mydata[jj,cc] for jj,thislab in enumerate(sorted_labels)} for cc in np.arange(ncl)}
                    #mydict = {thislab:mydata[jj] for jj,thislab in enumerate(sorted_labels)}
                mydf = pd.DataFrame.from_dict(mydict)
                mydf.to_excel(writer, sheet_name=sheetname)






