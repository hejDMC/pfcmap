import h5py

import sys
import yaml
import os
import numpy as np
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import git
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]
plot_just_enrichment = False

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
ncluststr = '8'
cmethod = 'ward'
'''

simfn_dict = {'corr': lambda Xmat:np.corrcoef(Xmat),
              'SI': lambda Xmat: gfns.get_similarity_mat(Xmat),\
              'coss':lambda Xmat: cosine_similarity(Xmat)}

sim_mode = 'coss'

ncl = int(ncluststr)





with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import graphs as gfns
from pfcmap.python.utils import statshelpers as shfns
from pfcmap.python.utils import category_matching as matchfns


statsfile = os.path.join(pathdict['statsdict_dir'],'statsdictSpecCtrl__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
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
#figsaver






ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(labels)}



nreps_simshuff = 100
graphdisp_ew_boost = 3 #just for graph display

plotsel_stats = [ 'z from shuff','perc of score', 'p-levels']

#what to sort and not sort in the statsdict according to community identified
# sortkeys = ['avals1','levels','matches','meanshuff','stdshuff','pofs','alabels','commlabels']
# sortkeysT = ['communities']
# sortkeysD = ['A','S']
# nonsortkeys = ['src','qvec','nvec','simshuff_mean','simshuff_std','comm_idx']
# allsortkeys = sortkeys + sortkeysT + sortkeysD + nonsortkeys

reftags = ['refall','refPFC']
#statsfns,statstags = [S.enr_fn,S.frac_fn],['enr','frac']
statstag = 'enr'
statsfn = S.enr_fn

repo = git.Repo(search_parent_directories=True)
omnidict = {reftag:{statstag:{} for statstag in [statstag]} for reftag in reftags}
omnidict['methods'] = {'sim_mode':sim_mode,'git_hash':repo.head.object.hexsha}#,'comalgo':comalgo,'weight_mode':weight_mode

simfn = simfn_dict[sim_mode]

#cois = np.array(clust_of_interest_dict[myrun][ncl])

#statsdict['tasks'][reftag]['area_ctrl']
#statsdict['recid'][reftag]['area_task_ctrl']
#statsdict['layers'][reftag]['area_task_ctrl']

flavs = ['tasks','layers','recid']
#reftag = 'refall'
with h5py.File(statsfile,'r') as hand:

    for reftag in reftags:

        for flav in flavs:
            statshand0 = hand[flav][reftag]
            for subflav in statshand0.keys():
                thistag = '%s__%s'%(flav,subflav)
                statshand = hand[flav][reftag][subflav]

                mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
                countvec = mystats['matches'][()].sum(axis=1)
                presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps])
                sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
                sdict['alabels'] = np.array([aval for aval in sdict['avals1']])
                sdict['src'] = statshand.name
                omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



height_facs = {'recid':4,'tasks':0.5,'layers':0.4}

seltags = [key for key in omnidict[reftag][statstag].keys()]
if myrun.count('IBL'):
    seltags = [stag for stag in seltags if not stag.count('task')]
for reftag in reftags:
    for thistag in seltags:

        print('getting and plotting modules for %s %i - %s %s %s'%(myrun,ncl,reftag,statstag,thistag))
        #reftag, statstag, thistag,statsfn = 'refall','enr','deepCtx_nolayOtherwise',lambda x: S.enr_fn(x)
        ddict = omnidict[reftag][statstag][thistag]
        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s'%(reftag,thistag,statstag,plotname,reftag,thistag,statstag)
        titlestr = '%s %s'%(myrun,thistag)

        srcdict = ddict['src_stats']

        distbaseMat = srcdict['levels']



        X = statsfn(srcdict)
        plX = np.vstack([vals for vals in X])
        plX[srcdict['levels']==0] = 0


        srcdict['S'] = simfn(distbaseMat)

        #Smat = simfn(X)  # similarity matrix
        #srcdict['S'] = Smat

        dists = pdist(distbaseMat)
        Z = hac.linkage(dists, method='ward',optimal_ordering=True)#
        dn = hac.dendrogram(Z, no_plot=True)
        sortinds = np.array(dn['leaves'])




        #f, axarr = matchfns.plot_all_stats(srcdict, sortinds=sortinds, plotsel=plotsel_stats)
        f,axarr = gfns.plot_all_stats(srcdict, plotsel=plotsel_stats,sortinds=sortinds, zlim=8,zmat_levelbounded=True)

        axarr[0].set_yticks(np.arange(len(X)))
        axarr[0].set_yticklabels(srcdict['alabels'][sortinds])
        axarr[0].set_ylabel('')
        f.suptitle(titlestr)
        mainflav = thistag.split('__')[0]
        if mainflav == 'recid':

            f.set_figwidth(f.get_figwidth()*1.4)

        f.set_figheight(f.get_figheight()*height_facs[mainflav])

        f.tight_layout()
        figsaver(f, make_savename('clustspectAllFlavs'))

        if not plot_just_enrichment:

            for mydata,dtag in zip([srcdict['levels'],plX,X],['plevels','emat_zerod','emat']):

                mydata2 = np.vstack([val for val in mydata])
                mydata2[np.isnan(mydata2)] = 0
                f, axarr = gfns.plot_similarity_mat(simfn(mydata2)[sortinds, :][:, sortinds],  {}, cmap='PiYG_r',
                                                    clab=sim_mode, tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold= lambda val:False,
                                                    regvals_sorted=srcdict['alabels'][sortinds], aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])

                ax = axarr[-1]
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                f.set_figheight(f.get_figheight()*height_facs[mainflav])
                f.set_figwidth(f.get_figwidth()*height_facs[mainflav])
                f.suptitle(titlestr)

                axarr[0].set_yticks(np.arange(len(X)))
                axarr[0].set_yticklabels(srcdict['alabels'][sortinds])
                axarr[0].set_ylabel('')

                axarr[0].set_xticks(np.arange(len(X)))
                axarr[0].set_xticklabels(srcdict['alabels'][sortinds],rotation=90)
                axarr[0].set_xlabel('')
                #axarr[0].set_aspect('equal')
                f.tight_layout()

                figsaver(f, make_savename('similarity_%s'%dtag))



if not plot_just_enrichment:
    uloader.save_dict_to_hdf5(omnidict, statsfile.replace('statsdict','enrichmentdict'))
