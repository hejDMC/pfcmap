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
import pandas as pd

pathpath,myrun,ncluststr,cmethod,roi_tag = sys.argv[1:]
plot_just_enrichment = False

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'#
#myrun = 'runIBLPasdMP3_brain_pj'
ncluststr = '8'
cmethod = 'ward'
roi_tag = 'dataRois'
'''

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import graphs as gfns
from pfcmap.python.utils import tessellation_tools as ttools
from pfcmap.python.utils import clustering as cfns#

simfn_dict = {'corr': lambda Xmat:np.corrcoef(Xmat),
              'SI': lambda Xmat: gfns.get_similarity_mat(Xmat),\
              'coss':lambda Xmat: cosine_similarity(Xmat)}

sim_mode = 'coss'

ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


region_file = os.path.join(pathdict['tesselation_dir'],'flatmap_PFCregions.h5')

if roi_tag == 'dataRois':
    flatmapfile = str(S.roimap_path)
elif roi_tag == 'gaoRois':
    flatmapfile = os.path.join(pathdict['tesselation_dir'],'flatmap_PFCrois.h5')
else:
    assert 0, 'unknown roi_tag: %s'%roi_tag


statsfile = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%(roi_tag,myrun,ncluststr,cmethod))


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/Nc%s/%s/enrichments/rois/%s'%(myrun,kshape[0],kshape[1],ncluststr,cmethod,roi_tag))

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


plotsel_stats = [ 'z from shuff','perc of score', 'p-levels']


reftags = ['refall','refPFC']
#statsfns,statstags = [S.enr_fn,S.frac_fn],['enr','frac']
statstag = 'enr'
statsfn = S.enr_fn

repo = git.Repo(search_parent_directories=True)
omnidict = {reftag:{statstag:{} for statstag in [statstag]} for reftag in reftags}
omnidict['methods'] = {'sim_mode':sim_mode,'git_hash':repo.head.object.hexsha}#,'comalgo':comalgo,'weight_mode':weight_mode

simfn = simfn_dict[sim_mode]

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr

#cois = np.array(clust_of_interest_dict[myrun][ncl])

#LOADING DATA

with h5py.File(statsfile,'r') as hand:

    for reftag in reftags:


        ####
        thistag = 'deepRois'
        statshand = hand[reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



        ####
        thistag = 'nolayRois'
        statshand = hand[reftag]['nolays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval=='0'])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'layeredRois'
        statshand = hand[reftag]['lays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}


        ####
        thistag = 'deepSupRois'
        statshand = hand[reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



seltags = list(omnidict[reftag][statstag].keys())
unique_roiruns = ['deepRois','nolayRois']#




# prepare roi plotting
def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])


with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}


roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}
# check whether rois are sensibly colored

f,ax = plt.subplots()
ttools.colorfill_polygons(ax, polygon_dict, roi_colors, na_col='k', ec='k',
                           show_cmap=False, mylimfn=set_mylim)
figsaver(f,'rois_colored')


nruns = 5
zlim = 8
mapstr_z = 'RdBu_r'
nancol = 'w'
#manual_colors = np.array(['red','deepskyblue','violet','sienna','orange','limegreen','darkviolet','peru','deeppink','darkturquoise','gold','green'])

cfndict = cfns.get_cfn_dict()
clustfn = cfndict[cmethod]
evalfns = cfns.get_evalfn_dict()
clustcmap = 'jet'

for basetag in ['levels','emat_zerod','emat']:

    for reftag in reftags:
        for thistag in seltags:

            print('getting and plotting modules for %s %i - %s %s %s %s'%(myrun,ncl,reftag,statstag,thistag,basetag))
            #reftag, statstag, thistag,statsfn = 'refall','enr','deepCtx_nolayOtherwise',lambda x: S.enr_fn(x)
            ddict = omnidict[reftag][statstag][thistag]

            srcdict = ddict['src_stats']
            X = statsfn(srcdict)




            Smat = simfn(X)#similarity matrix
            #srcdict['S'] = Smat
            plX = np.vstack([vals for vals in X])
            plX[srcdict['levels']==0] = 0

            distbase_dict = {'levels':srcdict['levels'],\
                             'emat_zerod':plX,\
                             'emat':X}

            distbaseMat = distbase_dict[basetag]

            def make_savename(plotname):
                return '%s/%s__%s/%s/%s__%s_%s_%s_%s'%(reftag,thistag,statstag,basetag,plotname,reftag,thistag,statstag,basetag)
            titlestr = '%s %s %s %s'%(myrun,thistag,reftag,basetag)

            srcdict['S'] = simfn(distbaseMat)

            #getting sortinds
            dists = pdist(distbaseMat)
            Z = hac.linkage(dists, method='ward',optimal_ordering=True)#
            dn = hac.dendrogram(Z, no_plot=True)
            sortinds = np.array(dn['leaves'])

            # plot matrix
            styled_simp = {'cdictcom':{0:'w'},'labelvec':np.zeros(len(X)).astype(int),'dotcolvec':np.array(['silver']*len(X))}
            f, axarr = gfns.plot_all_stats_with_labels_regs(srcdict, sortinds, styled_simp,
                                                            srcdict['alabels'][sortinds], plotsel=plotsel_stats, \
                                                            check_bold=S.check_pfc_full, aw_frac=1 / 3., labels_on=True, fs=9,zlim=zlim,zmat_levelbounded=True)#,zmat_zerolims=[-2,2]

            cdict_avals = {roi:roi_colors[roi.split('|')[0]] for roi in srcdict['alabels']}
            labaxarr = gfns.make_clusterlabels_as_bgrnd([axarr[2]], srcdict['alabels'][sortinds], cdict_avals,\
                                                        aw_frac=0.1, output=True,labels_on=True,alpha=1)

            f.suptitle(titlestr)

            figsaver(f, make_savename('clustspectAllFlavs'))

            #prepare clustering
            nclust_max = np.min([8, int(len(X)/2)])
            NCvec = np.arange(2, nclust_max)
            #dendrogram with cutoffs marked
            merge_heights = Z[::-1, 2]
            cutoffs1 = merge_heights[NCvec-1]
            cutoffs2 = merge_heights[NCvec-2]
            cutoffs = (cutoffs1+cutoffs2)/2 #to achieve a nice middle cutoff
            for xtickmode in ['colors','roilabs']:
                f,ax = plt.subplots()
                trans = mpl.transforms.blended_transform_factory(
                    ax.transAxes, ax.transData)
                dn = hac.dendrogram(Z,labels=srcdict['alabels'])
                ax.set_ylabel('distance')
                for nclusters,my_cutoff in zip(NCvec,cutoffs):
                    ax.axhline(my_cutoff,color='grey',linestyle='--')
                    ax.text(1.01,my_cutoff,'%i'%nclusters,transform=trans,ha='left',va='center',color='grey')

                #ax.tick_params(axis='x', pad=30)
                if xtickmode == 'roilabs':
                    labaxarr = gfns.make_clusterlabels_as_bgrnd([ax], srcdict['alabels'][sortinds], cdict_avals,\
                                        aw_frac=0.05, which='x', output=True,labels_on=True,alpha=1)
                figsaver(f, make_savename('dendrogram_%s'%xtickmode))

            nvec_quality = np.arange(1, int(len(X) / 2))
            print (X.shape)
            print(nvec_quality)
            print(distbaseMat.shape)
            # plot thorndike
            f = cfns.plot_thorndike(nvec_quality, distbaseMat, linkage_method=S.cmethod)
            figsaver(f, make_savename('thorndike'))

            evaldict = cfns.eval_asfn_nclust(distbaseMat, nvec_quality[nvec_quality > 1], clustfn, evalfns)
            for randbool, randtag in zip(['False'], ['']):
                flist = cfns.plot_clustquality(evaldict, nvec_quality[nvec_quality > 1], show_rand=randbool)

                for f, evalkey in zip(flist, evaldict.keys()):
                    figsaver(f,make_savename('quality_%s%s' % (evalkey, randtag)))


            if thistag in unique_roiruns:

                # plot the zmat on the flatmap
                mean_shuff, std_shuff = [sdict[key] for key in ['meanshuff', 'stdshuff']]
                zmat = (sdict['matches'] - mean_shuff) / std_shuff
                zmat[sdict['levels'] == 0] = np.nan
                plotdict_z = {roi.split('|')[0]:zmat[rr] for rr,roi in enumerate(sdict['alabels'])}
                cmap_z = ttools.get_scalar_map(mapstr_z,[-zlim,zlim])

                f, axarr = plt.subplots(1, ncl, figsize=(14, 3))
                f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
                for cc, clust in enumerate(np.arange(ncl)):
                    ax = axarr[cc]
                    ax.set_title('cat. %s' % (clust + 1), color=cdict_clust[clust], fontweight='bold')
                    ttools.colorfill_polygons(ax, polygon_dict, plotdict_z, subkey=cc, cmap=cmap_z, clab='', na_col='k',nancol=nancol, ec='k',
                                           show_cmap=cc == 0, mylimfn=set_mylim)
                f.suptitle(titlestr)
                figsaver(f, make_savename('clustspect_on_flatmap'))#should be the same for all the distbaseMats!!!


                if not plot_just_enrichment:
                    labdict = {}
                    #hierarchical clustering with different nclust
                    f, axarr = plt.subplots(1, len(NCvec), figsize=(len(NCvec) * 2 + 2, 3))
                    f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
                    for nn, nclusters in enumerate(NCvec):
                        ax = axarr[nn]
                        ax.set_title('Ncl %s' % (nclusters), fontweight='bold')
                        mylabs = hac.fcluster(Z, nclusters, 'maxclust')
                        mymap = ttools.get_scalar_map(clustcmap,[0,nclusters])

                        plotdict = {roi.split('|')[0]:mylab for roi,mylab in zip(srcdict['alabels'],mylabs)}
                        labdict[nclusters] = {key:[plotval,mymap.to_rgba(plotval)] for key,plotval in plotdict.items()}

                        ttools.colorfill_polygons(ax, polygon_dict, plotdict, cmap=mymap, clab='', na_col='k',nancol=nancol, ec='k',
                                               show_cmap=False, mylimfn=set_mylim)
                    f.suptitle(titlestr)
                    figsaver(f, make_savename('clustered_flatmaps'))
                    labdict_name = make_savename('clustered_flatmaps_labeldict')
                    outfilename = os.path.join(figdir_mother, labdict_name + '__%s.xlsx' % (myrun))

                    with pd.ExcelWriter(outfilename, engine='openpyxl') as writer:
                        for nclusters in NCvec:
                            myD = labdict[nclusters]
                            df = pd.DataFrame.from_dict({key:{'lab':myD[key][0],'color':myD[key][1]} for key in myD.keys()}, \
                                                        orient='index')

                            df.to_excel(writer, sheet_name='ncl%i'%(nclusters))

            if not plot_just_enrichment:


                def labelmaker(axarr):
                    labaxarr = gfns.make_clusterlabels_as_bgrnd([axarr[0]], srcdict['alabels'][sortinds], cdict_avals,\
                                                            aw_frac=0.05, output=True,labels_on=True,alpha=1)
                    labaxarr = gfns.make_clusterlabels_as_bgrnd([axarr[0]], srcdict['alabels'][sortinds], cdict_avals,\
                                                aw_frac=0.05, which='x', output=True,labels_on=True,alpha=1)
                    ax = axarr[0]
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])


                empty_xylabs = ['' for alb in srcdict['alabels']]


                f, axarr = gfns.plot_similarity_mat(simfn(distbaseMat)[sortinds, :][:, sortinds], styled_simp, cmap='PiYG_r',
                                                    clab=sim_mode, tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=empty_xylabs, aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])#srcdict['alabels'][sortinds]
                axarr[-1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))

                labelmaker(axarr)
                f.suptitle(titlestr)
                figsaver(f, make_savename('similarity_%s'))

                ccmat = np.corrcoef(distbaseMat)


                f, axarr = gfns.plot_similarity_mat(ccmat[sortinds, :][:, sortinds], styled_simp, cmap='PiYG_r',
                                                    clab='CC', tickflav='reg', \
                                                    badcol='silver', fs=9, check_bold=S.check_pfc_full,
                                                    regvals_sorted=empty_xylabs, aw_frac=1 / 8.,
                                                    vminmax=[-1, 1])
                axarr[-1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                labelmaker(axarr)


                f.suptitle(titlestr)
                figsaver(f, make_savename('corrcoeff_mat'))



                distmat = distance_matrix(distbaseMat,distbaseMat) #checked distmat[2,1] == np.linalg.norm(X[2]-X[1])
                srcdict['distmat'] = distmat
                for distcmap in ['jet_r','Greys']:
                    if distcmap == 'Greys':
                        vmax,extend = sap(distmat,75),'max'
                    else:
                        vmax,extend = distmat.max(),'neither'
                    f, axarr = gfns.plot_similarity_mat(distmat[sortinds, :][:, sortinds], styled_simp, cmap=distcmap,
                                                        clab='D', tickflav='reg', \
                                                        badcol='silver', fs=9, check_bold=S.check_pfc_full, extend=extend,
                                                        regvals_sorted=empty_xylabs, aw_frac=1 / 8.,vminmax=[0.,vmax])
                    labelmaker(axarr)

                    f.suptitle(titlestr)
                    figsaver(f, make_savename('distmat__CMAP%s'%(distcmap)))


        if not plot_just_enrichment:

            #make a graph out of the Similarity matrix
            ddict['run_layouts'] =  {}
            for rr in np.arange(nruns):

                runtag = 'SWEEP%i' % (rr + 1)


                fumap = UMAP(n_components=2,n_neighbors=6).fit_transform(X)
                ddict['run_layouts'][runtag] = fumap


                G = gfns.calc_graph_from_smat(simfn(distbaseMat),srcdict['alabels'],s_thr=0.1)

                layout_dict = {alabel:fumap[ii] for ii,alabel in enumerate(srcdict['alabels'])}
                ews = {(u,v):(weight+1)**0.5 for u,v,weight in G.edges(data='weight')}

                edge_color_dict = {(src, target): cdict_avals[src] for src,target in G.edges()}


                f, ax = mpl.pyplot.subplots(figsize=(5, 5))
                mygraph = Graph(G,node_label_fontdict=dict(size=9),node_size=12,\
                                node_layout=layout_dict,edge_width=ews,node_edge_width=0,node_color=cdict_avals,node_labels=True,\
                                edge_color=edge_color_dict,node_alpha=1,edge_layout='bundled')  # edge_layout_kwargs=dict(k=2000),node_labels=True#edge_color_dict

                figsaver(f, make_savename('graph_%s'%runtag))
    if not plot_just_enrichment:
        uloader.save_dict_to_hdf5(omnidict, statsfile.replace('statsdict','enrichmentdict__%s_'%basetag))







