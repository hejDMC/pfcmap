import sys
import yaml
import os
import numpy as np
import h5py
import git
import pandas as pd

precut = 3.
ttags = ['prestim']
postcut = 0
block_licking = True
savepattern = '%s__TSEL%s{}__STATE%s%s.h5'.format(int(precut)) #% (aRec.id, ttag, statetag,reftag)

active_buff = np.array([-1,1])
quiet_buff = np.array([0,0])
#on_files = 'all'

badtimesdir = 'ZENODOPATH/preprocessing/bad_times_manual'
statedir = 'ZENODOPATH/quiet_active_detection/merged_quiet'
tablepath = 'config/datatables/allrecs_allprobes.xlsx'
dstdir = 'ZENODOPATH/preprocessing/metrics_extraction/timeselections/timeselections_quietactive'#


not_allowed=['-', '?']
#datasetlabs = ['HPC','MO','SS','AUD']#,'AUD','mPFC','HPC','MO','SS'
df = pd.read_excel(tablepath, sheet_name='sheet0')
allowed_bool = df['usable_gen'].isin(['run'])
isrec_bool = df['recid'].str.contains('probe',na=False)

#dataset_bool = df['datasetlab'].isin(datasetlabs)
bools_list = [allowed_bool,isrec_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
quietdet_srcs = df['quietdet_src'][cond_bool].values
recids = df['recid'][cond_bool].values

print('Number of recs: %i'%len(recids))

#



if not os.path.isdir(dstdir):
    os.makedirs(dstdir)

pathpath = 'PATHS/general_paths.yml'
with open(pathpath) as yfile: pdict = yaml.safe_load(yfile)

#ttag = 'prestim'

#N.B change Passive and Aversion to correspond to "FULL" setting
stimdict = {'Passive': {'stimnames': 'all'},
 'Context': {'stimnames': ['AudStim_10kHz']},
 'Opto': {'stimnames': ['AudStim_10kHz']},
 'Aversion': {'stimnames': ['AudStim_10kHz','passive_bluenoise']},
 'Association': {'stimnames': ['AudStim_10kHz']},
 'Detection': {'stimnames': ['AudStim_10kHz']},\
 'Attention':{'stimnames': ['lighton']}}


sys.path.append(pdict['code'])

from utils import data_classes as dc
from utils import data_handling as dh
from utils import tint_funcs as tf


cfgpath,dspath = dc.get_paths('Passive',pdict)#N.B.: we do not need specific configs for the other types of datasets


exptypes = ['Passive','Context','Association','Detection','Opto','Aversion']

buff = 1.
refselfn = lambda freemat,refdata:freemat[freemat[:,1]<=refdata[0]-buff,:]

purge_badtints =  lambda tints, badtints: np.vstack([tint for tint in tints if np.sum([(tint[0]<=badtime[1]) & (tint[1]>=badtime[0]) for badtime in badtints])== 0])
#tints and badtimes are samples x 2 arrays

only_before_ref_bools = [True, False]

for recid,quietdet_src in zip(recids,quietdet_srcs):


    ignore_quietactive = False
    print('###%s %s'%(recid,quietdet_src))

    if quietdet_src == 'self':
        recid_ref = str(recid)

    elif quietdet_src.count('from-other'):
        mother_probe = quietdet_src.split(':')[1]
        recid_stub,my_probe = recid.split('-')
        recid_ref = '%s-%s'%(recid_stub,mother_probe)
        print('Using mother probe %s for %s'%(recid_ref,recid))


    elif quietdet_src == 'ignore':
        ignore_quietactive = True
        print('Ignoring quietdet')

    else:
        assert 0, 'unknown quietdet_src "%s" in table'%quietdet_src


    recid_trad = recid.replace('-','_')

    filename_g = 'NWBEXPORTPATH/XXX_export/%s__XXX.h5' % (recid_trad)
    aRec = dc.RecPassive(filename_g,cfgpath,dspath)

    exptype = (aRec.h_info['exptype'][()]).decode()

    print('exptype: %s'%exptype)

    reftaglist = []
    if True in only_before_ref_bools and not exptype in ['Detection', 'Attention']:
        reftaglist += ['']
    if False in only_before_ref_bools:
        reftaglist += ['__all']
    print('reftags %s'%(str(reftaglist)))

    if exptype == 'Attention':
        stimtimes = aRec.h_info['intervals/trials/start_time'][()]

    else:
        stimnames = stimdict[exptype]['stimnames']
        stims_avail = list(aRec.h_info['stimulus/presentation'].keys())

        if stimdict[exptype]['stimnames'] == 'all':
            stimnames = stims_avail[:]

        else:
            stimnames_notavail = [stimn for stimn in  stimnames if not stimn in stimnames]
            if len(stimnames_notavail)>0:
                print('WARNING: STIM %s not available'%(str(stimnames_notavail)))
            stimnames = [stimn for stimn in stimnames if stimn in stims_avail]

        stimtimes = np.sort(
            np.hstack([aRec.h_info['stimulus/presentation/%s/timestamps' % stimname][()] for stimname in stimnames]))

    tints0 = stimtimes[:, None] + np.array([-precut, postcut])[None, :]

    # check whether there is a bad interval file and remove the tints that touch on the bad interval
    forbidden_tints = np.empty((0, 2))
    badint_file = os.path.join(badtimesdir, '%s__badtimes.h5' % (recid_trad))
    if os.path.isfile(badint_file):
        with h5py.File(badint_file, 'r') as fhand:
            badtimes = fhand['badtints'][()]
            print('found manually set bad times (N_bad=%i) --> excluding those ' % badtimes.shape[0])

            forbidden_tints = np.r_[forbidden_tints, badtimes]

    if block_licking:
        if np.size(aRec.lickmat) > 0:
            print('blocking lick-times')
            forbidden_tints = np.r_[forbidden_tints, aRec.lickmat]


    if np.size(aRec.spikesatmat) > 0:
        print('blocking spike saturations')
        forbidden_tints = np.r_[forbidden_tints, aRec.spikesatmat]

    if not ignore_quietactive:
        statefile = os.path.join(statedir, '%s__transAndOff.h5' % (recid_ref.replace('-','_')))
        with h5py.File(statefile, 'r') as fhand:
            quietmat = fhand['results/quiet_merge'][()]
            usable_spikeoff_str = fhand.attrs['off_used']
            assert usable_spikeoff_str in ['no', 'yes'], 'inadmissible string spikeoff detection: %s' % usable_spikeoff_str

        # filename = filenames[0]


        # if spike based offdetection was not usable, remove the lfp

        if usable_spikeoff_str == 'no':
            aRec.get_freetimes()
            arttimes = aRec.artblockmat.T
            print('spike-based offdet not usable --> excluding tints touching LFP artifacts (N_art=%i)' % arttimes.shape[0])
            forbidden_tints = np.r_[forbidden_tints, arttimes]

    # remove the collected not usable tints
    print('Number of forbidden time intervals', forbidden_tints.shape[0])
    print('tints prev: %i' % len(tints0))
    tints0 = purge_badtints(tints0, forbidden_tints)
    print('tints postpurge: %i' % len(tints0))


    for reftag in reftaglist:
        print('REFTAG: %s' % reftag)

        if reftag == '':
            tints = tf.select_relative_to_ref(aRec.h_info, tf.refkeydict[exptype], tints0, refselfn=refselfn)
        elif reftag == '__all':
            tints = tints0[:]
        else:
            assert 0, 'unknown reftag %s' % reftag
        # now classify the whole tint as quiet or active

        if not ignore_quietactive:
            active_tints0 = np.array([tint for tint in tints if np.sum(
                [dh.check_olap(quiet_int, tint + active_buff) for quiet_int in quietmat]) == 0])

            if block_licking:

                active_tints = active_tints0
            else:
                #rationale: when the mouse is licking it can not be inactive
                lick_adders = np.array([tint for tint in tints if
                                        np.sum([dh.check_olap(lick_int, tint + active_buff) for lick_int in aRec.lickmat]) > 0 \
                                        and not tint in active_tints0])
                if np.size(lick_adders) > 0:
                    active_tints = np.sort(np.vstack([active_tints0, lick_adders]), 0)
                else:
                    active_tints = active_tints0


            if np.size(active_tints) == 0:
                active_tints = np.empty((0, 2))

            quiet_tints = np.array([tint for tint in tints if np.sum(
                [dh.check_olap(quiet_int, tint + quiet_buff) for quiet_int in quietmat]) > 0 and not tint in active_tints])
            if np.size(quiet_tints) == 0:
                quiet_tints = np.empty((0, 2))

            un_tints = np.array([tint for tint in tints if not tint in np.r_[active_tints, quiet_tints]])

            if np.size(un_tints) == 0:
                un_tints = np.empty((0, 2))

        else:
            active_tints = tints[:]
            quiet_tints = np.empty((0, 2))
            un_tints = np.empty((0, 2))
            print('!Setting all active because quietdet_src=%s!'%quietdet_src)

        print('rec:%s , ref:%s  ; active: %s, quiet: %s, undef: %s'%(recid,recid_ref, str(active_tints.shape), str(quiet_tints.shape), str(un_tints.shape)))

        # for states quiet, active, unclassified, for pre and post
        for statetag, gentints in zip(['active', 'quiet', 'uncertain'], [active_tints, quiet_tints, un_tints]):
            for ttag in ttags:
                savename = os.path.join(dstdir,
                                        savepattern % (recid, ttag, statetag, reftag))  # '%s__TSEL%s__STATE%s%s.h5'
                print(savename)

                if ttag == 'prestim':
                    mytints = gentints - np.array([0, postcut])
                elif ttag == 'poststim':
                    mytints = gentints + np.array([precut, 0])
                with h5py.File(savename, 'w') as hand:
                    ds = hand.create_dataset('tints', data=mytints)
                    ds.attrs['recid'] = recid
                    ds.attrs['ttag'] = ttag
                    ds.attrs['statetag'] = statetag
                    ds.attrs['quietdet_src'] = quietdet_src
                    ds.attrs['githash'] = git.Repo(search_parent_directories=True).head.object.hexsha
                    #ds.attrs['srcfile'] = __file__

#as a sanity check: print nicely for each recid N active!

ttag = 'prestim'
statetag = 'active'
reftag = '__all'
for recid in recids:

    #recid = '273855_20200928-probe0'
    tintfile = os.path.join(dstdir, savepattern % (recid, ttag, statetag, reftag))  # '%s__TSEL%s__STATE%s%s.h5'

    with h5py.File(tintfile,'r') as fhand: mytints = fhand['tints'][()]
    print(recid,len(mytints))