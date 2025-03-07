import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import h5py
import matplotlib as mpl
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac


pathpath = 'PATHS/filepaths_IBLTuning.yml'#
myrun =  'runIBLTuning_utypeP'



with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S
from pfcmap_paper.utils import graphs as gfns
from pfcmap_paper.utils import tessellation_tools as ttools
from pfcmap_paper.utils import clustering as cfns

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
statsdict_dir = pathdict['statsdict_dir']#
srcfile = os.path.join(pathdict['statsdict_dir'],'statsdictTuning_rois_iblROIs__%s.h5'%(myrun))

statsdict_rois = uloader.load_dict_from_hdf5(srcfile)

tuning_attr_names = ['stimTuned','choiceTuned','feedbTuned','taskResp']


figdir_base = 'FIGURES/tuning_maps'

figdir_mother = os.path.join(figdir_base,'tuningMaps__%s'%(myrun))

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__tuningMaps_%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr


N_tuningattrs = len(tuning_attr_names)

tattrs_short = [attr.replace('Tuned','').replace('Resp','') for attr in tuning_attr_names]

Nmin_maps = int(S.Nmin_maps)
plotdicts = {}
for attr in tuning_attr_names:
    mystats = statsdict_rois[attr]
    countvec = mystats['matches'].sum(axis=1)
    presel_inds = np.array(
        [aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] >= Nmin_maps  and aval.count('|deep')])#at least 20 units in a roi
    sdict2 = {key: mystats[key][presel_inds] for key in ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
    sdict2['avals2'] = mystats['avals2']
    sdict2['alabels'] = np.array([replace_fn(aval) for aval in sdict2['avals1']])
    plotdicts[attr] = sdict2

zlim = 8
# prepare roi plotting
def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])

mapstr_z = 'RdBu_r'
nancol = 'w'
flatmapfile = str(S.roimap_path)#os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')

with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

cmap_z = ttools.get_scalar_map(mapstr_z, [-zlim, zlim])



def get_enr(sdict2):
    signif_idx = np.where(sdict2['avals2']=='signif')[0]
    mean_shuff, std_shuff = [sdict2[key] for key in ['meanshuff', 'stdshuff']]
    zmat = (sdict2['matches'] - mean_shuff) / std_shuff
    zmat[sdict2['levels'] == 0] = 0
    return zmat[:,signif_idx]

allcounts_signif = np.hstack([plotdicts[attr]['matches'][:,np.where(plotdicts[attr]['avals2']=='signif')[0]][:,0] for attr in tuning_attr_names])
allcounts_ns = np.hstack([plotdicts[attr]['matches'][:,np.where(plotdicts[attr]['avals2']=='ns')[0]][:,0] for attr in tuning_attr_names])
nPerRoi = allcounts_ns+allcounts_signif
cond = (~np.isnan(nPerRoi)) & (nPerRoi>0)
allfracs = allcounts_signif[cond]/(allcounts_ns[cond]+allcounts_signif[cond])
maxfrac = allfracs.max()#wow, nearly 1!
cmap_frac= ttools.get_scalar_map('Reds', [0, 1])
def get_fracs(sdict2):
    signif_temp = sdict2['matches'][:,np.where(sdict2['avals2']=='signif')[0]][:, 0]
    ns_temp = sdict2['matches'][:,np.where(sdict2['avals2']=='ns')[0]][:, 0]
    return signif_temp/(ns_temp+signif_temp)

fnflav_dict = {'escore': {'cmap': cmap_z, 'fn': get_enr,'lab':'E-score'},\
               'frac':{'cmap':cmap_frac,'fn':get_fracs,'lab':'frac.'},\
               'frac_autoscale':{'cmap':cmap_frac,'fn':get_fracs,'lab':'frac.'}}

#flatmaps for escore and fractions
for flav in ['escore','frac','frac_autoscale']:
    f, axarr = plt.subplots(1, N_tuningattrs, figsize=(14, 3))
    f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)

    for aa,attr in enumerate(tuning_attr_names):
        sdict2 = plotdicts[attr]
        dmat = fnflav_dict[flav]['fn'](sdict2)
        plotdict_z = {roi.split('|')[0]: float(dmat[rr]) for rr, roi in enumerate(sdict2['alabels'])}

        ax = axarr[aa]
        ax.set_title(attr, color='k', fontweight='bold')
        if flav == 'frac_autoscale':
            mycmap = ttools.get_scalar_map('Reds', [np.min(list(plotdict_z.values())), np.max(list(plotdict_z.values()))])
            show_cmap = True
        else:
            mycmap = fnflav_dict[flav]['cmap']
            show_cmap = aa == 0
        ttools.colorfill_polygons(ax, polygon_dict, plotdict_z, subkey=aa, cmap=mycmap, clab=fnflav_dict[flav]['lab'], na_col='grey', nancol=nancol,
                                  ec='k',
                                  show_cmap=show_cmap, mylimfn=set_mylim)#
    f.suptitle(myrun)
    figsaver(f,'tuningFlatmaps/tuningFm_%s'%(flav))


##############################
####NOW GATHER ALL TUNINGS IN A COMMON DICTIONARY TO RE-USE ESTABLISHED FUNCTIONS
matkeys = ['levels','matches','meanshuff','pofs','stdshuff']
n_roi_entries = len(statsdict_rois[attr]['matches'])

#set sdict nan where there is too little data in the roi! find general presel inds!
sum_mat = np.vstack([statsdict_rois[attr]['matches'].sum(axis=1) for attr in tuning_attr_names])#should be the same for all...
minnperroi = np.min(sum_mat,axis=0)
deep_cond = np.array([myroiname.count('|deep') for myroiname in  statsdict_rois[attr]['avals1']]).astype(bool)
avail_rois = statsdict_rois[attr]['avals1'][(minnperroi>=Nmin_maps)&(deep_cond)]
avail_inds = np.arange(n_roi_entries)[(minnperroi>=Nmin_maps)&(deep_cond)]
N_avail = len(avail_inds)
sdict = {key:np.zeros((N_avail,N_tuningattrs))*np.nan for key in matkeys}
for aa,attr in enumerate(tuning_attr_names):
    matchdict = statsdict_rois[attr]
    for key in matkeys:
        sdict[key][:,aa] = matchdict[key][avail_inds,matchdict['avals2']=='signif']
sdict['alabels'] = np.array([replace_fn(aval) for aval in matchdict['avals1'][avail_inds]])
sdict['avals2'] = tuning_attr_names
sdict['a1'] = 'cTuning'
sdict['SI'] = 1


#PLOTTING SORTED ENRICHMENT MATRICES
cfndict = cfns.get_cfn_dict()
clustfn = cfndict['ward']
evalfns = cfns.get_evalfn_dict()
region_file = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCregions.h5')
clustcmap_modules = 'jet'
statstag = 'enr'
statsfn = S.enr_fn
simfn = lambda Xmat: cosine_similarity(Xmat)
plotsel_stats = [ 'z from shuff','perc of score', 'p-levels']
roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}

X = statsfn(sdict)

Smat = simfn(X)  # similarity matrix
# srcdict['S'] = Smat
plX = np.vstack([vals for vals in X])
plX[sdict['levels'] == 0] = 0

distbase_dict = {'levels': sdict['levels'], \
                 'emat_zerod': plX, \
                 'emat': X}

for basetag in ['levels','emat_zerod','emat']:
    def make_savename(plotname):
        return 'clusteringTuningMap/%s__%s' %(plotname,basetag)
    #basetag = 'emat_zerod'

    distbaseMat = distbase_dict[basetag]


    titlestr = '%s %s' % (myrun, basetag)

    sdict['S'] = simfn(distbaseMat)

    # getting sortinds
    dists = pdist(distbaseMat)
    Z = hac.linkage(dists, method='ward', optimal_ordering=True)  #
    dn = hac.dendrogram(Z, no_plot=True)
    sortinds = np.array(dn['leaves'])

    # plot matrix
    styled_simp = {'cdictcom': {0: 'w'}, 'labelvec': np.zeros(len(X)).astype(int),
                   'dotcolvec': np.array(['silver'] * len(X))}
    f, axarr = gfns.plot_all_stats_with_labels_regs(sdict, sortinds, styled_simp,
                                                    sdict['alabels'][sortinds], plotsel=plotsel_stats, \
                                                    check_bold=S.check_pfc_full, aw_frac=1 / 3., labels_on=True, fs=9,
                                                    zlim=zlim, zmat_levelbounded=True)  # ,zmat_zerolims=[-2,2]

    cdict_avals = {roi: roi_colors[roi.split('|')[0]] for roi in sdict['alabels']}
    labaxarr = gfns.make_clusterlabels_as_bgrnd([axarr[2]], sdict['alabels'][sortinds], cdict_avals, \
                                                aw_frac=0.1, output=True, labels_on=True, alpha=1)
    for ax in  axarr[::2]:
        ax.set_xticklabels(tattrs_short,rotation=-90)
        ax.set_xlabel('')
    figsaver(f, make_savename('clustspectAllFlavs'))

    #make a module map
    alabels_short = [lab.split('|')[0] for lab in sdict['alabels']]#todo put on top, N.B. this is only suitable when all are deep or superficial
    # prepare clustering
    nclust_max = np.min([8, int(len(X) / 2)])
    NCvec = np.arange(2, nclust_max)
    # dendrogram with cutoffs marked
    merge_heights = Z[::-1, 2]
    cutoffs1 = merge_heights[NCvec - 1]
    cutoffs2 = merge_heights[NCvec - 2]
    cutoffs = (cutoffs1 + cutoffs2) / 2  # to achieve a nice middle cutoff
    for xtickmode in ['colors', 'roilabs']:
        f, ax = plt.subplots()
        trans = mpl.transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        dn = hac.dendrogram(Z, labels=alabels_short)
        ax.set_ylabel('distance')
        for nclusters, my_cutoff in zip(NCvec, cutoffs):
            ax.axhline(my_cutoff, color='grey', linestyle='--')
            ax.text(1.01, my_cutoff, '%i' % nclusters, transform=trans, ha='left', va='center', color='grey')

        # ax.tick_params(axis='x', pad=30)
        if xtickmode == 'roilabs':
            labaxarr = gfns.make_clusterlabels_as_bgrnd([ax], sdict['alabels'], cdict_avals, \
                                                        aw_frac=0.05, which='x', output=True, labels_on=True, alpha=1)
        figsaver(f, make_savename('dendrogram_%s' % xtickmode))



    nvec_quality = np.arange(1, int(len(X) / 2))
    # plot thorndike
    f = cfns.plot_thorndike(nvec_quality, distbaseMat, linkage_method=S.cmethod)
    figsaver(f, make_savename('thorndike'))

    evaldict = cfns.eval_asfn_nclust(distbaseMat, nvec_quality[nvec_quality > 1], clustfn, evalfns)
    for randbool, randtag in zip(['False'], ['']):
        flist = cfns.plot_clustquality(evaldict, nvec_quality[nvec_quality > 1], show_rand=randbool)

        for f, evalkey in zip(flist, evaldict.keys()):
            figsaver(f, make_savename('quality_%s%s' % (evalkey, randtag)))


    labdict = {}
    # hierarchical clustering with different nclust
    f, axarr = plt.subplots(1, len(NCvec), figsize=(len(NCvec) * 2 + 2, 3))
    f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
    for nn, nclusters in enumerate(NCvec):
        ax = axarr[nn]
        ax.set_title('Ncl %s' % (nclusters), fontweight='bold')
        mylabs = hac.fcluster(Z, nclusters, 'maxclust')
        mymap = ttools.get_scalar_map(clustcmap_modules, [0, nclusters])

        plotdict = {roi.split('|')[0]: mylab for roi, mylab in zip(sdict['alabels'], mylabs)}
        labdict[nclusters] = {key: [plotval, mymap.to_rgba(plotval)] for key, plotval in plotdict.items()}

        ttools.colorfill_polygons(ax, polygon_dict, plotdict, cmap=mymap, clab='', na_col='grey', nancol=nancol,
                                  ec='k',
                                  show_cmap=False, mylimfn=set_mylim)
    f.suptitle(titlestr)

    figsaver(f, make_savename('clustered_flatmaps'))
    labdict_name = make_savename('clustered_flatmaps_labeldict')
    outfilename = os.path.join(figdir_mother, labdict_name + '__%s.xlsx' % (myrun))
    
    with pd.ExcelWriter(outfilename, engine='openpyxl') as writer:
        for nclusters in NCvec:
            myD = labdict[nclusters]
            df = pd.DataFrame.from_dict({key: {'lab': myD[key][0], 'color': myD[key][1]} for key in myD.keys()}, \
                                        orient='index')
    
            df.to_excel(writer, sheet_name='ncl%i' % (nclusters))