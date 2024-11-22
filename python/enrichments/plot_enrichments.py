import h5py
from umap import UMAP
from scipy.spatial import distance_matrix
from scipy.stats import scoreatpercentile as sap
import sys
import yaml
import os
import numpy as np
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from netgraph import Graph
import git
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]
plot_just_enrichment = False

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'#
#myrun = 'runIBLPasdMP3_brain_pj'
ncluststr = '8'
cmethod = 'ward'
'''

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import graphs as gfns
from pfcmap.python.utils import statshelpers as shfns

simfn_dict = {'corr': lambda Xmat:np.corrcoef(Xmat),
              'SI': lambda Xmat: gfns.get_similarity_mat(Xmat),\
              'coss':lambda Xmat: cosine_similarity(Xmat)}

sim_mode = 'coss'

'''weight_mode = 'adj'
comalgo = 'leiden'
n_max_mods = 8 #maximum number of modules to consider when sorting vertically
'''

ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']




statsfile = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

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
#figsaver




ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(labels)}



#nreps_simshuff = 100
#graphdisp_ew_boost = 3 #just for graph display

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

'''
comfn = calgo_fndict[comalgo]
weightfn = weightfn_dict[weight_mode]
'''

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr

#cois = np.array(clust_of_interest_dict[myrun][ncl])



#reftag = 'refall'
with h5py.File(statsfile,'r') as hand:

    for reftag in reftags:


        ####
        thistag = 'deepCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval.count('|sup') and not aval=='na' \
                                and not (S.check_pfc_full(aval) and not aval.count('|deep'))])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ###
        thistag = 'layeredregsCtx'
        statshand = hand['regs'][reftag]['lays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}


        ####
        thistag = 'deepSupCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval=='na' \
                                and not (S.check_pfc_full(aval) and not aval.count('|'))])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'deepSupCtx'
        statshand = hand['regs'][reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'justlayers'
        statshand = hand['layers'][reftag]['justlays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval=='na'])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'justtasks'
        statshand = hand['tasks'][reftag]['nolays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



seltags = [key for key in omnidict[reftag][statstag].keys() ]
if myrun.count('IBL'):
    seltags = [stag for stag in seltags if not stag.count('task')]


if not plot_just_enrichment:
    #make simple correlation plots of the different categories
    for reftag in reftags:
        for thistag in seltags:
            srcdict = omnidict[reftag][statstag][thistag]['src_stats']
            def make_savename(plotname):
                return '%s/%s__%s/%s__%s_%s_%s' % (reftag, thistag, statstag, plotname, reftag, thistag, statstag)

            X = statsfn(srcdict)

            dists = pdist(X.T)
            Z = hac.linkage(dists, method='ward', optimal_ordering=True)  #
            dn = hac.dendrogram(Z, no_plot=True)
            sortidx = np.array(dn['leaves'])

            #sortidx = cfns.get_sortidx_featdist(X.T, linkmethod='ward')
            f = shfns.getplot_corrmat_clusters(X[:,sortidx], corrfnstr='pearsonr',sortidx=sortidx,cdict_clust=cdict_clust)
            f.set_size_inches(f.get_size_inches()*1.3)
            f.suptitle('%s %s'%(myrun,thistag))
            figsaver(f,make_savename('clustcorrs'))


#later when too few sample groups (as in layers/tasks), do a simple hiearchy sort, omit the multiruns, and everything associated with communities
seltags = [key for key in omnidict[reftag][statstag].keys() if not key.count('just')]
nruns = 5
for reftag in reftags:
    for thistag in seltags:

        print('getting and plotting modules for %s %i - %s %s %s'%(myrun,ncl,reftag,statstag,thistag))
        #reftag, statstag, thistag,statsfn = 'refall','enr','deepCtx_nolayOtherwise',lambda x: S.enr_fn(x)
        ddict = omnidict[reftag][statstag][thistag]
        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s'%(reftag,thistag,statstag,plotname,reftag,thistag,statstag)
        titlestr = '%s %s'%(myrun,thistag)

        srcdict = ddict['src_stats']

        X = statsfn(srcdict)
        Smat = simfn(X)#similarity matrix
        plX = np.vstack([vals for vals in X])
        plX[srcdict['levels']==0] = 0

        distbaseMat = srcdict['levels']

        srcdict['S'] = simfn(distbaseMat)

        #getting sortinds
        dists = pdist(distbaseMat)
        Z = hac.linkage(dists, method='ward',optimal_ordering=True)#
        dn = hac.dendrogram(Z, no_plot=True)
        sortinds = np.array(dn['leaves'])

        styled_simp = {'cdictcom':{0:'w'},'labelvec':np.zeros(len(X)).astype(int),'dotcolvec':np.array(['silver']*len(X))}
        f, axarr = gfns.plot_all_stats_with_labels_regs(srcdict, sortinds, styled_simp,
                                                        srcdict['alabels'][sortinds], plotsel=plotsel_stats, \
                                                        check_bold=S.check_pfc_full, aw_frac=1 / 3., labels_on=True, fs=9,zlim=8,zmat_levelbounded=True)#,zmat_zerolims=[-2,2]
        f.suptitle(titlestr)
        figsaver(f, make_savename('clustspectAllFlavs'))

        if not plot_just_enrichment:
            for mydata,dtag in zip([srcdict['levels'],plX,X],['plevels','emat_zerod','emat']):

                f, axarr = gfns.plot_similarity_mat(simfn(mydata)[sortinds, :][:, sortinds], styled_simp, cmap='PiYG_r',
                                                    clab=sim_mode, tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=srcdict['alabels'][sortinds], aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])
                ax = axarr[-1]
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

                f.suptitle(titlestr+' %s'%dtag)
                figsaver(f, make_savename('similarity_%s'%dtag))

                ccmat = np.corrcoef(mydata)


                f, axarr = gfns.plot_similarity_mat(ccmat[sortinds, :][:, sortinds], styled_simp, cmap='PiYG_r',
                                                    clab='CC', tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=srcdict['alabels'][sortinds], aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])
                ax = axarr[-1]
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

                f.suptitle(titlestr+' %s'%dtag)
                figsaver(f, make_savename('corrcoeff_mat_%s'%dtag))



                distmat = distance_matrix(mydata,mydata) #checked distmat[2,1] == np.linalg.norm(X[2]-X[1])
                srcdict['distmat'] = distmat
                for distcmap in ['jet_r','Greys']:
                    if distcmap == 'Greys':
                        vmax,extend = sap(distmat,75),'max'
                    else:
                        vmax,extend = distmat.max(),'neither'
                    f, axarr = gfns.plot_similarity_mat(distmat[sortinds, :][:, sortinds], styled_simp, cmap=distcmap,
                                                        clab='D', tickflav='reg', \
                                                        badcol='silver', fs=9, check_bold=S.check_pfc_full, extend=extend,
                                                        regvals_sorted=srcdict['alabels'][sortinds], aw_frac=1 / 8.,vminmax=[0.,vmax])
                    ax = axarr[-1]

                    f.suptitle(titlestr + ' %s' % dtag)
                    figsaver(f, make_savename('distmat__CMAP%s_%s'%(distcmap,dtag)))


            f,ax = plt.subplots()
            dn = hac.dendrogram(Z,labels=srcdict['alabels'])
            ax.set_ylabel('distance')
            figsaver(f, make_savename('dendrogram'))

            #make a graph out of the Similarity matrix
            ddict['run_layouts'] =  {}
            for rr in np.arange(nruns):
                runtag = 'SWEEP%i' % (rr + 1)


                def make_savename(plotname):
                    return '%s/%s__%s/%s__%s_%s_%s__%s' % (
                    reftag, thistag, statstag, plotname, reftag, thistag, statstag, runtag)

                fumap = UMAP(n_components=2,n_neighbors=6).fit_transform(X)
                ddict['run_layouts'][runtag] = fumap

                for mydata,dtag in zip([srcdict['levels'],plX,X],['plevels','emat_zerod','emat']):

                    G = gfns.calc_graph_from_smat(simfn(mydata),srcdict['alabels'],s_thr=0.1)

                    layout_dict = {alabel:fumap[ii] for ii,alabel in enumerate(srcdict['alabels'])}
                    ews = {(u,v):(weight+1)**4 for u,v,weight in G.edges(data='weight')}

                    if reftag == 'refall':
                        if thistag.count('nolayOtherwise'):
                            reg_colors = {reg: 'silver' if S.check_pfc_full(reg) else S.parent_colors[S.strip_to_area(reg)] for reg in srcdict['alabels']}
                            reg_colors = {reg: 'silver' if not S.check_pfc_full(reg) else S.cdict_pfc[S.strip_to_area(reg)] for reg in srcdict['alabels']}

                        if thistag in ['deepSupCtx','layeredregsCtx']:
                            reg_colors = {reg: 'silver' if not S.check_pfc_full(reg) else S.cdict_pfc[S.strip_to_area(reg)] for reg in srcdict['alabels']}

                    elif reftag == 'refPFC':
                        reg_colors = {reg: S.cdict_pfc[S.strip_to_area(reg)] for reg in srcdict['alabels']}


                    #edge_color_dict = {(src, target): reg_colors[src] for src,target in G.edges()}
                    edge_color_dict = {(src, target): reg_colors[src] if reg_colors[src]!='silver' else reg_colors[target] for src,target in G.edges()}


                    f, ax = mpl.pyplot.subplots(figsize=(5, 5))
                    mygraph = Graph(G,node_label_fontdict=dict(size=9),node_size=18,\
                                    node_layout=layout_dict,edge_width=ews,node_edge_width=0,node_color=reg_colors,node_labels=True,\
                                    edge_color=edge_color_dict,node_alpha=1,edge_layout='bundled')  # edge_layout_kwargs=dict(k=2000),node_labels=True#edge_color_dict

                    figsaver(f, make_savename('graph_%s'%dtag))





seltags2 = [key for key in omnidict[reftag][statstag].keys() if not key in seltags]
if myrun.count('IBL'):
    seltags2 = [stag for stag in seltags2 if not stag.count('task')]
for reftag in reftags:
    for thistag in seltags2:

        print('getting and plotting modules for %s %i - %s %s %s'%(myrun,ncl,reftag,statstag,thistag))
        #reftag, statstag, thistag,statsfn = 'refall','enr','deepCtx_nolayOtherwise',lambda x: S.enr_fn(x)
        ddict = omnidict[reftag][statstag][thistag]
        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s'%(reftag,thistag,statstag,plotname,reftag,thistag,statstag)
        titlestr = '%s %s'%(myrun,thistag)

        srcdict = ddict['src_stats']

        X = statsfn(srcdict)
        plX = np.vstack([vals for vals in X])
        plX[srcdict['levels']==0] = 0
        distbaseMat = srcdict['levels']

        srcdict['S'] = simfn(distbaseMat)

        #Smat = simfn(X)  # similarity matrix
        #srcdict['S'] = Smat

        dists = pdist(X)
        Z = hac.linkage(dists, method='ward',optimal_ordering=True)#
        dn = hac.dendrogram(Z, no_plot=True)
        sortinds = np.array(dn['leaves'])


        sdict['sortinds'] = sortinds


        styled_simp = {'cdictcom':{0:'w'},'labelvec':np.zeros(len(X)).astype(int),'dotcolvec':np.array(['silver']*len(X))}
        f, axarr = gfns.plot_all_stats_with_labels_regs(srcdict, sortinds, styled_simp,
                                                        srcdict['alabels'][sortinds], plotsel=plotsel_stats, \
                                                        check_bold=S.check_pfc_full, aw_frac=1 / 3., labels_on=True, fs=9,zlim=8,zmat_levelbounded=True)#,zmat_zerolims=[-2,2]
        f.suptitle(titlestr)


        #figsaver(f, make_savename('clustspectAllFlavs'))
        if not plot_just_enrichment:



            for mydata,dtag in zip([srcdict['levels'],plX,X],['plevels','emat_zerod','emat']):

                f, axarr = gfns.plot_similarity_mat(simfn(mydata)[sortinds, :][:, sortinds], styled_simp, cmap='PiYG_r',
                                                    clab=sim_mode, tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=srcdict['alabels'][sortinds], aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])
                ax = axarr[-1]
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

                f.suptitle(titlestr+' %s'%dtag)
                figsaver(f, make_savename('similarity_%s'%dtag))

if not plot_just_enrichment:
    uloader.save_dict_to_hdf5(omnidict, statsfile.replace('statsdict','enrichmentdict'))




'''
        sivals = Smat[np.triu_indices_from(Smat, k=1)]

        f,ax = gfns.plot_simval_hist(sivals,si_thr,xtag=sim_mode)
        f.suptitle(titlestr,fontsize=8)
        figsaver(f,make_savename('sivals'))

        A = gfns.get_adjacency_mat(Smat,si_thr,fill_val=np.nan)
        srcdict['A'] = A
        
        ddict['runs_stats'] = {}

        for rr in np.arange(nruns):
            runtag = 'SWEEP%i'%(rr+1)


            def make_savename(plotname):
                return '%s/%s__%s/%s__%s_%s_%s__%s' % (reftag, thistag, statstag, plotname, reftag, thistag, statstag,runtag)


            titlestr = '%s %s %i' % (myrun, thistag,rr+1)


                    ddict['run_stats'] = {}

            sdict = ddict['runs_stats'][runtag]
            qvec,nvec,communities0 = gfns.calc_q_vs_res(A,si_thr,comfn,weightfn,res_vec,with_n=True,with_comm=True)

            sdict['qvec'] = qvec
            sdict['nvec'] = nvec
            ################################
            #same for general regions

            shuff_mat = gfns.get_q_vs_res_shuffled(Smat,si_thr,comfn,weightfn,res_vec,nreps=nreps_simshuff)
            shuff_mean,shuff_std = shuff_mat.mean(axis=1),shuff_mat.std(axis=1)
            sdict['simshuff_mean'] = shuff_mean
            sdict['simshuff_std'] = shuff_std

            idx = np.argmax(qvec-shuff_mat.mean(axis=1))
            res = res_vec[idx]
            sdict['res_idx'] = idx
            sdict['res'] = res



            f,ax = gfns.plot_q_vs_res(qvec,shuff_mat,res_vec,idx,nvec=nvec)#,qvec2=qvec_anat
            #ax.plot(res_vec,qvec_anat,'orange')
            f.suptitle(titlestr)
            figsaver(f, make_savename('q_vs_res'))

            # first within the subcomms: sort labels according to median cluster of interested enrichment
            communities1 = gfns.relabel_clusters_by_enrichment(communities0, X, cois)

            communities = gfns.harmonize_whole_communitymat(communities1, idx)

            sortinds_com = gfns.get_sortinds_community_mat(communities)

            f = gfns.plot_document_community_sorting(communities0, communities1, communities, sortinds_com, idx,
                                                     maxshowind=10)
            figsaver(f, make_savename('community_sorting_doc'))


            sdict['communities'] = communities#the labels are now sorted according to cluster of interest enrichment (cluster 0 == most enriched)
            sdict['sortinds_com'] = sortinds_com
            comm_labels = communities[idx]
            #sorting vertically so that some labels align up to maxind
            n_max_mods_adj = np.min([n_max_mods,communities.max()+1]) # in case n_max_mods> communities available
            maxind = np.where(nvec<=n_max_mods_adj)[0][-1]
            maxind = np.max([idx,maxind]) #in case the chosen idx is bigger
            #
            sdict['commlabels'] = comm_labels

            ulabels = np.unique(comm_labels)

            #just plotting
            if statstag == 'frac':
                zlim = [X.min(),X.max()]
                cmapstr = 'inferno'
                entend = 'neither'
            elif statstag == 'enr':
                zlim = [-5,5]
                cmapstr = S.cmap_div
                extend = 'both'


            if len(np.unique(communities[:maxind]))<= 12:
                cstr_comm = 'Set3'#Set3 has 12 different fields
            else:
                cstr_comm = 'jet'

            f,axarr = gfns.plot_clustspect(X, communities, sortinds_com, idx, res_vec, qvec, shuff_mean, maxind, cmapstr, srcdict['alabels'],
                            cdict_clust, \
                            boldlabfn=S.check_pfc_full, cstr_comm=cstr_comm, zlim=zlim, extend=extend, clab=statstag)
            axarr[1,1].set_title(titlestr.replace(' ','\n'),fontsize=8)
            figsaver(f, make_savename('clustspect'))


            if thistag.count('just'):
                reg_colors = {aval: 'silver' for aval in srcdict['alabels']}
            else:
                reg_colors = {reg: 'silver' if S.check_pfc_full(reg) else S.parent_colors[reg.split('|')[0]] for reg in srcdict['alabels']}
            com_dict = {ll: srcdict['alabels'][comm_labels == ll] for ll in ulabels}

            styled_coms = gfns.get_styledict_rois_and_comms(srcdict['alabels'][sortinds_com], com_dict, reg_colors)#[sortinds_com], cstr_communities='manual'

            cbounds = [communities[:maxind].min(),communities[:maxind].max()]
            norm = mpl.colors.Normalize(vmin=cbounds[0], vmax=cbounds[-1])
            styled_coms['cdictcom'] = {lab: mpl.cm.get_cmap(cstr_comm)(norm(lab)) for lab in ulabels}
            f, axarr = gfns.plot_all_stats_with_labels_regs(srcdict, sortinds_com, styled_coms,
                                                            srcdict['alabels'][sortinds_com], plotsel=plotsel_stats, \
                                                            check_bold=S.check_pfc_full, aw_frac=1 / 3., labels_on=True, fs=9)
            f.suptitle(titlestr)
            figsaver(f, make_savename('clustspectAllFlavs'))

            sdict['styles'] =  styled_coms

            G = gfns.get_igraph_from_adj(A,si_thr)
            graphpropd = gfns.graphprops_to_dict2(G,comm_labels,srcdict['alabels'],reg_colors)#for plotting
            graphpropd['ew'] = {key:val*graphdisp_ew_boost for key,val in graphpropd['ew'].items()}#boost the edge-widht for display
            node_label_dict = {ii:regval for ii,regval in enumerate(srcdict['alabels'])}


            for gg in np.arange(6):
                #np.random.seed(seed)
                f, ax = mpl.pyplot.subplots(figsize=(5, 5))
                mygraph = Graph(G,
                                node_color=graphpropd['nc'], edge_alpha=0.7,
                                node_layout='community', node_layout_kwargs=dict(node_to_community=graphpropd['nodesToComm']),
                                node_edge_width=0, edge_width=graphpropd['ew'], \
                                edge_layout='bundled', ax=ax, edge_color=graphpropd['ec'], node_labels=node_label_dict,
                                node_label_fontdict=dict(size=9),
                                node_size=8)  # edge_layout_kwargs=dict(k=2000),node_labels=True
                f.tight_layout()
                f.suptitle(titlestr)
                figsaver(f, make_savename('graph%i'%gg))

            #plot adjacency matrix
            f, axarr = gfns.plot_similarity_mat(A[sortinds_com, :][:, sortinds_com], styled_coms, cmap='copper_r', clab='adj.',
                                                    tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=srcdict['alabels'][sortinds_com], aw_frac=1 / 8.)

            gfns.plot_community_sqaures_on_mat(axarr[0], styled_coms['cdictcom'], styled_coms['labelvec'], lw=3, bg_on=True,
                                               bg_col='w', bg_fac=1.5)
            f.suptitle(titlestr)
            figsaver(f, make_savename('adjacency'))


            f, axarr = gfns.plot_similarity_mat(Smat[sortinds_com, :][:, sortinds_com], styled_coms, cmap='PiYG_r',
                                                    clab=sim_mode, tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=srcdict['alabels'][sortinds_com], aw_frac=1 / 8.,vminmax=[-1,1])  # diverging maybe
            gfns.plot_community_sqaures_on_mat(axarr[0], styled_coms['cdictcom'], styled_coms['labelvec'], lw=3, bg_on=True,
                                               bg_col='w', bg_fac=1.5)
            ax = axarr[-1]
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

            f.suptitle(titlestr)
            figsaver(f, make_savename('similarity'))



            # assert set(sdict.keys()) == set(allsortkeys),'set of expected and given sortkeys do not match'
            # sdict_sorted = {}
            # for key,val in sdict.items():
            #     if key in sortkeys:
            #         sdict_sorted[key] = val[sortinds_com]
            #     elif key in sortkeysT:
            #         sdict_sorted[key] = val[:,sortinds_com]
            #     elif key in sortkeysD:
            #         sdict_sorted[key] = val[sortinds_com, :][:, sortinds_com]
            #     elif key in nonsortkeys:
            #         sdict_sorted[key] = val
            #     else: assert 0, 'unknown key for sorted statsdict %s'%key

            #omnidict[reftag][statstag][thistag] = sdict_sorted
'''











