import sys
import yaml
import os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd



pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'


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
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/examples/imetrics_targetedPicks'%(myrun,kshape[0],kshape[1]))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


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


#selections
regsel_props = {'PL':lambda U: (U.B_mean<-0.2) & (U.M_mean>-0.1) & (U.rate_mean>0.),\
             'ILA':lambda U:(U.M_mean>0.2) & (U.rate_mean>0),\
             'ACAd':lambda U:(U.rate_mean<0.25) & (True),\
             'ORBl':lambda U: (U.B_mean>0.17) & (U.rate_mean>0.2),\
                'ORBm':lambda U: (True) & (U.rate_mean>0.8)}



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
metricsextr_path = 'ZENODOPATH/preprocessing/metrics_extraction'

def get_tintfile(U):
    tintfile = uloader.get_tintfile_rec(recid,U.dataset,rundict,metricsextr_path=metricsextr_path)
    return tintfile
getspikes = lambda stimes,tinterval: stimes[(stimes>=tinterval[0]) & (stimes<=tinterval[1])]


maxn = 40


usel_dict = {}
for myreg,selfn in regsel_props.items():
    usel = [U for U in Units if selfn(U) and U.area==myreg and U.recid in recids_allowed]
    my_n = np.min([maxn,len(usel)])
    usel_dict[myreg] = np.random.choice(usel,my_n,replace=False)

ufeatdict = {myreg:{} for myreg in regsel_props.keys()}
for reg in regsel_props.keys():
    for uu,U in enumerate(usel_dict[reg]):
        subdict = {'[#]':uu+1,'recid':U.recid,'uid':U.uid,'reg':U.region,'bmu':U.bmu}
        subdict.update({feat.split('_')[0]:getattr(U,feat) for feat in ['rate_mean','B_mean','M_mean']})
        ufeatdict[reg][uu+1] = subdict



        # uu = 10
        # U = Usel[uu]
        srcfile = [fname for fname in src_pool if fname.count(U.recid)][0]
        with h5py.File(srcfile, 'r') as hand:
            utinds = hand['units/spike_times_index'][()]
            r1, r0 = get_ridx(utinds, U.uid)
            spiketimes = hand['units/spike_times'][r0:r1]

        tintfile = get_tintfile(U)

        with h5py.File(tintfile, 'r') as hand:
            tints0 = hand['tints'][()]
        tints = tints0 + borders[None, :]
        spikelist = [getspikes(spiketimes, tint) - tint[0] for tint in tints]
        rastermat = np.vstack(
            [np.hstack(spikelist), np.hstack([np.ones(len(subl)) + xx for xx, subl in enumerate(spikelist)])])
        framedur = float(np.diff(tints0)[0])
        tadder = -framedur if rundict['tsel'] == 'prestim' else 0.
        featstr = ','.join(['%s:%1.1f' % (feat[:1], getattr(U, feat)) for feat in somfeats])
        f, ax = plt.subplots(figsize=(3, 2.5))
        f.subplots_adjust(left=0.23, bottom=0.22, right=0.95)
        f.suptitle('[%i] bmu%i %s u:%i %s (%s)' % (uu + 1, U.bmu, U.recid.split('_')[0][:8], U.uid, U.region, featstr),
                   fontsize=7)
        ax.plot(rastermat[0] + borders[0] + tadder, rastermat[1], 'k.', ms=2)
        ax.set_ylabel('trials')
        ax.set_xlabel('time [s]')
        if rundict['tsel'] == 'prestim':
            ax.set_xlim([borders[0] + tadder, borders[1]])
            ax.axvline(0, color='silver', zorder=-10)
            ax.axvline(tadder, color='silver', zorder=-10)
        elif rundict['tsel'] == 'poststim':
            ax.set_xlim([borders[0], framedur + borders[1]])
            ax.axvline(0, color='silver', zorder=-10)
            ax.axvline(framedur, color='silver', zorder=-10)
        ax.set_ylim([0.5, tints.shape[0] + 0.5])
        # figsaver(f,'uexample_%i'%(uu+1))
        figsaver(f, '%s/full/uexample_%i' % (reg,(uu + 1)),closeit=False)
        ax.set_ylim([0.5, showtints_min + 0.5])
        figsaver(f, '%s/tint%i/uexample_%i' % (reg,showtints_min, (uu+ 1)))



outfile = os.path.join(os.path.join(figdir_mother,'pickedUnitDetails__%s.xlsx'%(myrun)))
with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
    for reglab,mydict in ufeatdict.items():
        df = pd.DataFrame.from_dict(mydict, orient='index')#

        df.to_excel(writer, sheet_name=reglab)

