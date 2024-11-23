import sys
import yaml
import os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZP_brain'
cmethod = 'ward'
ncl = 8


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)


somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)


kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/examples/picksPerCat'%(myrun,kshape[0],kshape[1]))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




#reftag,layerlist,datasets,tasks,tsel,statetag,utypes,nnodes = [rundict[key] for key in ['reftag','layerlist','datasets','tasks','tsel','state','utypes','nnodes']]
#wmetrics,imetrics,wmetr_weights,imetr_weights = [rundict[metrtype][mytag] for mytag in ['features','weights'] for metrtype in ['wmetrics','imetrics']]#not strictly necessary: gets called in uloader

metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)

assert (somfeats == somdict['features']).all(),'mismatching features'
assert (weightvec == somdict['featureweights']).all(),'mismatching weights'




get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])


#project on map
refmean,refstd = somdict['refmeanstd']
featureweights = somdict['featureweights']
dmat = np.vstack([get_features(elobj) for elobj in Units])
wmat = uloader.reference_data(dmat,refmean,refstd,featureweights)

#set BMU for each
weights = somdict['weights']
allbmus = somh.get_bmus(wmat,weights)
for bmu,U in zip(allbmus,Units):
    U.set_feature('bmu',bmu)


ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]
#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])


showtints_min = 100 if not rundict['datasets'] == ['IBL_Passive'] else 90
recids_allowed = []
recids_temp = np.unique([[U.recid,U.dataset] for U in Units],axis=0)
for recid,mydataset in recids_temp:
    tintfile = uloader.get_tintfile_rec(recid, mydataset, rundict, metricsextr_path=S.metricsextr_path)
    with h5py.File(tintfile, 'r') as hand: ntints = hand['tints'][()].shape[0]
    if ntints>=showtints_min:
        recids_allowed.append(recid)


src_pool = np.hstack([glob(pathdict['src_dirs']['nwb_dirpatterns'][dset]) for dset in rundict['datasets']])
get_ridx = lambda unit_inds, my_unit: [unit_inds[my_unit], [unit_inds[my_unit - 1] if my_unit > 0 else 0][0]]
borders = np.array([-0.25,0.25])


def get_tintfile(U):
    tintfile = uloader.get_tintfile_rec(recid,U.dataset,rundict,metricsextr_path=S.metricsextr_path)
    return tintfile
getspikes = lambda stimes,tinterval: stimes[(stimes>=tinterval[0]) & (stimes<=tinterval[1])]


reftag = 'refall'
psth_dir = os.path.join(S.metricsextr_path,'psth')
def get_psth(U):
    repl_dict = {'RECID': U.recid, 'mystate': rundict['state'], 'REFTAG': reftag,'__PCAscoresMODE':''}
    psthfile_tag = uloader.replace_by_dict(S.responsefile_pattern, repl_dict)
    psth_file = os.path.join(psth_dir,psthfile_tag)
    with h5py.File(psth_file,'r') as hand:
        huids = hand['uids'][()]
        uidx = int(np.where(huids==U.uid)[0])
        psth = hand['psth'][uidx]
        psth_tvec = hand['psth_tvec'][()]
    pre_bool =psth_tvec<0
    psth_normed = (psth -np.mean(psth[pre_bool])) /np.std(psth[pre_bool])
    return psth_normed,psth_tvec






maxn_per_clust = 20

#myreg = 'PL'
#selfn = regsel_props[myreg]
#usel = [U for U in Units if selfn(U) and U.area==myreg and U.recid in recids_allowed]
#len(usel)

usel_dict = {}
for mycat in np.arange(ncl):
    usel = [U for U in Units if U.clust==mycat and U.recid in recids_allowed and S.check_pfc(U.area)]
    print(len(usel))
    my_n = np.min([maxn_per_clust,len(usel)])
    usel_dict['Cat%s'%(mycat+1)] = np.random.choice(usel,my_n,replace=False)

ufeatdict = {key:{} for key in usel_dict.keys()}





tbound = [-0.25,0.65]
for key in usel_dict.keys():
    for uu,U in enumerate(usel_dict[key]):

        subdict = {'[#]':uu+1,'recid':U.recid,'uid':U.uid,'reg':U.region,'bmu':U.bmu}
        ufeatdict[key][uu+1] = subdict

        srcfile = [fname for fname in src_pool if fname.count(U.recid)][0]
        with h5py.File(srcfile, 'r') as hand:
            utinds = hand['units/spike_times_index'][()]
            r1, r0 = get_ridx(utinds, U.uid)
            spiketimes = hand['units/spike_times'][r0:r1]
        psth_normed, psth_tvec = get_psth(U)

        tintfile = uloader.get_tintfile(U, rundict, metricsextr_path=S.metricsextr_path)

        with h5py.File(tintfile, 'r') as hand:
            tints = hand['tints'][()]
            pre_s = hand['tints'].attrs['pre']
        spikelist = [getspikes(spiketimes, tint) - tint[0] - pre_s for tint in tints]
        rastermat = np.vstack(
            [np.hstack(spikelist), np.hstack([np.ones(len(subl)) + xx for xx, subl in enumerate(spikelist)])])

        f, axarr = plt.subplots(2, 1, figsize=(3, 3), gridspec_kw={'height_ratios': [1, 0.35]}, sharex=True)
        f.subplots_adjust(left=0.23, bottom=0.17, right=0.95, hspace=0.01)
        f.suptitle('[%i] bmu%i %s uid:%i %s' % (uu + 1, U.bmu, U.recid.split('_')[0][:8], U.uid, U.region), fontsize=10)
        ax, ax2 = axarr
        ax.plot(rastermat[0], rastermat[1], 'k.', ms=1)
        ax.set_ylabel('trials')
        ax2.set_xlabel('time [s]')
        ax.set_xlim(tbound)
        for myax in axarr: myax.axvline(0, color='silver', zorder=-10)
        ax2.plot(psth_tvec, psth_normed, 'k')
        ax2.set_ylabel('z-pre')
        ax.set_ylim([0.5, tints.shape[0] + 0.5])
        figsaver(f, '%s/full/uexample_%i_%s' % (key, (uu + 1),key), closeit=False)
        ax.set_ylim([0.5, showtints_min + 0.5])
        figsaver(f, '%s/tint%i/uexample_%i_%s' % (key, showtints_min, (uu + 1),key))


outfile = os.path.join(os.path.join(figdir_mother,'picksPerCat_details__%s.xlsx'%(myrun)))
with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
    for key,mydict in ufeatdict.items():
        df = pd.DataFrame.from_dict(mydict, orient='index')#

        df.to_excel(writer, sheet_name=key)

f,ax = plt.subplots()
for U in usel_dict['Cat1']:
    psthvec,tvec = get_psth(U)
    ax.plot(tvec,psthvec)