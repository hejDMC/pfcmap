import numpy as np
import sys
import yaml
import h5py
import os
from glob import glob
import git

varselection = ['M', 'B', 'LvR']
allvarnames = varselection + ['rate']
n_tints_min = 6 #minimum number of tints needed to calculate mean and std
mininput_ifn = 4 #min_nspike_per_trial = mininput_ifn+2#n-spikes = n_intervals +1; if intervals are paired you need +1 to make the pairing; so with 3 mininput you have at least 5 spikes/trial

full_estimate_trials_only = False

tintdur = 3
ttag = 'passive'

pathpath = 'PATHS/general_paths.yml'
srcdir = 'IBLPATH/IBL_Passive'
tintfile_dir = 'ZENODOPATH/preprocessing/timeselections/timeselections_IBL_Passive'
dstpath = 'ZENODOPATH/preprocessing/metrics_extraction/quantities_all_meanvar/IBL_Passive'

with open(pathpath) as yfile: pdict = yaml.safe_load(yfile)

sys.path.append(pdict['code'])
from utils import data_classes as dc
from utils import data_handling as dh
from utils import interval_fns as ifn

fn_dict = ifn.intfn_dict


cfgpath,dspath = dc.get_paths('Passive',pdict)#N.B.: we do not need specific configs for the other types of datasets

filepool = glob(os.path.join(srcdir,'*.nwb'))
nfiles = len(filepool)
for ff,filename in enumerate(filepool):
    aRec = dc.RecPassive(filename,cfgpath,dspath)
    print('EXTRACTING %s %i/%i'%(aRec.id,ff+1,nfiles))
    tintfile = glob(os.path.join(tintfile_dir,'%s*%i*.h5'%(aRec.id,tintdur)))
    assert len(tintfile) == 1, 'not exactly one tintfile %s'%(str(tintfile))
    tintfile = tintfile[0]

    uids = aRec.unit_ids.astype(int)

    unitdict = {unitid:np.unique(dh.grab_spikes_unit(aRec, unitid)) for unitid in uids}


    with h5py.File(tintfile,'r') as hand: tints = hand['tints'][()]

    #now for all units
    datadict = {}
    n_isis = np.zeros((len(uids),len(tints)),dtype='int')
    tintdurs = np.diff(tints)[:,0]
    for uu,uid in enumerate(uids):
        spiketimes = unitdict[uid]
        nspikevec = np.array([len(ifn.getspikes(spiketimes, tint)) for tint in tints])
        isi_list = ifn.collect_intervals_per_tint(tints, spiketimes)

        pairs_list = [np.vstack([isis[:-1], isis[1:]]) for isis in isi_list]
        rates_temp = np.ma.log10(nspikevec/tintdurs)
        subdict = {'rate':rates_temp.filled(np.nan)}
        for varname in varselection:
            subdict[varname] = ifn.calc_quant_segwise(isi_list, pairs_list, varname, fn_dict, mininputsize=mininput_ifn)

        if full_estimate_trials_only:
            # getting all vars nan-ed where one var is nan to ensure same number of trials everywhere
            subdict_stack = np.vstack([subdict[varname] for varname in varselection])
            nancond = np.isnan(subdict_stack.mean(axis=0))
            for varname in varselection + ['rate']:
                subdict[varname][nancond] = np.nan

        datadict[uid] = subdict
        if len(isi_list)>0:
            isis = np.hstack(isi_list)
            nsavail = np.array([len(vec) for vec in isi_list])
            n_isis[uu] = nsavail



    #varnames = ['B','Rho','LvR','rate']
    gendict = {uid:{var:np.array([np.nan,np.nan]) for var in allvarnames} for uid in uids}
    for uu,uid in enumerate(uids):
        for var in allvarnames:
            vals_temp = datadict[uid][var]
            cond = (~np.isnan(vals_temp)) & (~np.isinf(vals_temp))
            if np.sum(cond) >= n_tints_min:
                vals = vals_temp[cond]
                gendict[uid][var] = np.array([np.mean(vals),np.std(vals)])


    #saving stuff

    savename_stub = os.path.basename(tintfile).split('.h5')[0]
    #savename = os.path.join(dstpath,'%s__TSEL%s_quantities.h5'%(aRec.id,ttag))
    savename = os.path.join(dstpath,'%s_quantities_meanvar.h5'%savename_stub)
    with h5py.File(savename,'w') as hand:
        sgrp = hand.create_group('seg')
        for var in allvarnames:
            meanvec,stdvec = np.array([gendict[uid][var] for uid in uids]).T
            sgrp.create_dataset('%s_mean'%var,data=meanvec,dtype='f')
            sgrp.create_dataset('%s_std'%var,data=stdvec,dtype='f')

        hand.create_dataset('uids',data=uids)
        hand.create_dataset('n_isis',data=n_isis)
        hand.attrs['recid'] = aRec.id
        hand.attrs['ttag'] = ttag
        hand.attrs['n_tints'] = len(tints)
        hand.attrs['tfile'] = tintfile
        hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
        #hand.attrs['srcfile'] = __file__


    dstpath2 = dstpath+'_detail'
    savename2 = os.path.join(dstpath2,'%s_quantities_detail.h5'%savename_stub)
    with h5py.File(savename2,'w') as hand:
        grp = hand.create_group('tintwise')
        for var in allvarnames:
            data = np.vstack([datadict[uid][var] for uid in uids]).T
            grp.create_dataset(var,data=data,dtype='f')
        hand.create_dataset('uids',data=uids)
        hand.attrs['recid'] = aRec.id
        hand.attrs['ttag'] = ttag
        hand.attrs['n_tints'] = len(tints)
        hand.attrs['tfile'] = tintfile
        hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
        #hand.attrs['srcfile'] = __file__

