import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt


pathpath,run1,run2,stag1,stag2,ncluststr1,ncluststr2, cmethod = sys.argv[1:]
stags = [stag1,stag2]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
run1 = 'runC00dMP3_brain'
run2 = 'runCrespZP_brain'
stags = ['spont','resp']
ncluststr1 = '8'
ncluststr2 = '8'
cmethod = 'ward'
'''


signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)
outfile = os.path.join(pathdict['statsdict_dir'],'enrichmentRespVsSpont__%s__vs__%s.h5'%(run1,run2))

sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python.utils import category_matching as matchfns
from pfcmap.python.utils import graphs as gfns


figdir_base = pathdict['figdir_root'] + '/response_vs_spontaneous'

figdir_mother = os.path.join(figdir_base,'%s__vs__%s'%(run2,run1))

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%sVs%s.%s'%(stags[1],stags[0],S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


rundict_folder = os.path.dirname(rundict_path)

unit_dict = {}
for stag,myrun,ncluststr in zip(stags,[run1,run2],[ncluststr1,ncluststr2]):

    rundict = uloader.get_myrun(rundict_folder,myrun)

    metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
    Units_temp,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')))

    somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
    somdict = uloader.load_som(somfile)
    assert (somfeats == somdict['features']).all(), 'mismatching features'
    assert (weightvec == somdict['featureweights']).all(), 'mismatching weights'


    get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])

    # project on map
    refmean, refstd = somdict['refmeanstd']
    featureweights = somdict['featureweights']
    dmat = np.vstack([get_features(elobj) for elobj in Units_temp])
    wmat = uloader.reference_data(dmat, refmean, refstd, featureweights)

    # set BMU for each
    weights = somdict['weights']
    allbmus = somh.get_bmus(wmat, weights)
    for bmu, U in zip(allbmus, Units_temp):
        U.set_feature('bmu_%s'%stag, bmu)

    ncl = int(ncluststr)
    ddict = uloader.extract_clusterlabeldict(somfile, cmethod, get_resorted=True)
    labels = ddict[ncl]

    # categorize units
    for U in Units_temp:
        U.set_feature('clust_%s'%stag, labels[getattr(U,'bmu_%s'%stag)])


    unit_dict[stag] = Units_temp

Units = []
Us1,Us2 = [unit_dict[stag] for stag in stags]
stag_2 = stags[1]

for U1 in Us1:
    U2_matcher = [U2 for U2 in Us2 if U2.id==U1.id]
    if len(U2_matcher)==0:
        pass
    elif len(U2_matcher)==1:
        for feat in ['clust','bmu']:
            fname = '%s_%s'%(feat,stag_2)
            U1.set_feature(fname,getattr(U2_matcher[0],fname))
        Units += [U1]
    else:
        assert 0, 'not <=1 matchers for U %s'%(U1.id)

print('N%s: %i, N%s: %i, Nmatch: %i'%(stags[0],len(Us1),stags[1],len(Us2),len(Units)))
#
uloader.assign_parentregs(Units,pfeatname='area',cond=lambda uobj: S.check_pfc(uobj.area)==False,na_str='na')

Units = [U for U in Units if not U.area=='na']

Units_pfc = np.array([U for U in Units if not U.roi==0 and S.check_pfc(U.area)])
Units_ctx = np.array([U for U in Units if U in Units_pfc or U.area in S.ctx_regions])



attr1 = 'clust_resp'
attr2 = 'clust_spont'
const_attr = ['area','task']
reftags = ['refall','refPFC','refCtx']
statsdict = {reftag:{} for reftag in reftags}

for reftag, Usel in zip(['refall','refPFC','refCtx'],[Units,Units_pfc,Units_ctx]):
    print(reftag)
    deepUs = [U for U in Usel if S.check_layers(U.layer,['5','6'])]
    statsdict[reftag]['allLayers'] = matchfns.get_all_shuffs(Usel,attr1,attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict[reftag]['deepLayers'] = matchfns.get_all_shuffs(deepUs,attr1,attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)


# plot this!
omnidict = {reftag: {}for reftag in reftags}
statsfn = S.enr_fn
plotsel_stats = [ 'z from shuff','perc of score', 'p-levels']

for reftag in reftags:
    for thistag in ['deepLayers','allLayers']:
        #thistag = 'deepLayers'
        mystats = statsdict[reftag][thistag]
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1', 'levels', 'matches', 'meanshuff', 'stdshuff', 'pofs']}
        sdict['alabels'] = np.array([str(val+1) for val in sdict['avals1']])
        omnidict[reftag][thistag] = {'src_stats': sdict}

        X = statsfn(sdict)


        sortinds = np.arange(len(X))
        f,axarr = gfns.plot_all_stats(sdict, plotsel=plotsel_stats,sortinds=sortinds, zlim=8,zmat_levelbounded=True)
        f.set_figheight(2+0.2*len(X))
        f.subplots_adjust(left=0.12)
        axarr[0].set_ylabel('%s cat'%stags[1])
        for ax in axarr[::2]:
            ax.set_yticks(np.arange(len(X)))
            ax.set_yticklabels(np.arange(len(X))+1)
            ax.set_aspect('equal')
            ax.set_xlabel('%s cat'%stags[0])
        for axref,axcol in zip(axarr[::2][:2],axarr[1::2][:2]):
            pos = axref.get_position()
            cpos = axcol.get_position()
            axcol.set_position([pos.x1+0.2*(cpos.x0-pos.x1),pos.y0,cpos.width,pos.height])#[left, bottom, width, height]
        f.suptitle('%s  -   %s  -   %s; N:%i'%(reftag,thistag,' vs '.join(stags),sdict['matches'][()].sum()))
        figsaver(f,'respVsSpont_%s_%s'%(reftag,thistag))

uloader.save_dict_to_hdf5(omnidict, outfile)
