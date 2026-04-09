from glob import glob
import os
import numpy as np
import h5py
import git
import sys
import yaml
import pandas as pd
from rich.progress import track

precut = 3.
ttags = ['prestim']
postcut = 0
block_licking = True
savepattern = '%s__TSEL%s{}__STATE%s%s.h5'.format(int(precut)) #% (aRec.id, ttag, statetag,reftag)

active_buff = np.array([-1,1])
quiet_buff = np.array([0,0])
#on_files = 'all'


dsdir = 'D:/Carlen/Pete_Rudebeck/NWB'
dstdir = 'D:/Carlen/Intermediate/preprocessing/metrics_extraction/timeselections/timeselections_Rudebeck'#
savedir = 'D:/Carlen/Pete_Rudebeck'

#

myfiles = glob(os.path.join(dsdir,'*.nwb'))

if not os.path.isdir(dstdir):
    os.makedirs(dstdir)

pathpath = 'PATHS/general_paths.yml'
with open(pathpath) as yfile: pdict = yaml.safe_load(yfile)

#ttag = 'prestim'



sys.path.append(pdict['code'])

from utils import data_classes as dc
from utils import data_handling as dh
from utils import tint_funcs as tf


exptypes = ['Passive']

buff = 1.
refselfn = lambda freemat,refdata:freemat[freemat[:,1]<=refdata[0]-buff,:]

#purge_badtints =  lambda tints, badtints: np.vstack([tint for tint in tints if np.sum([(tint[0]<=badtime[1]) & (tint[1]>=badtime[0]) for badtime in badtints])== 0])
#tints and badtimes are samples x 2 arrays

only_before_ref_bools = [True, False]
cfgpath = "C:/Users/etien/PycharmProjects/pfcmap/python/config/rec_configPassive.yml"
dspath = "C:/Users/etien/PycharmProjects/pfcmap/python/config/dspaths.yml"

tintdur = 3
ttag = 'passive'
savepattern = '%s__TSEL{}{}.h5'.format(ttag,int(tintdur)) #

sr_psth = 2000
k_std = (10/1000)*sr_psth #first number: width in ms
k_win = k_std*4

prestim = 1
poststim = 2
def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )
gauss_kernel = np.array([gaussian( x , k_std) for x in np.arange( -k_win, k_win+1)])
psth_tvec_temp = np.arange(-prestim,poststim,1/sr_psth)
psth_tvec = psth_tvec_temp[:-1]+0.5/sr_psth
post_bool = psth_tvec>1

psth_mat = np.empty((0,len(psth_tvec)))
nspikes_tot = np.empty((0))
nspikes_psth = np.empty((0))
i=0
for datafile in track(myfiles, description = 'Processing'):
    i+=1
    filename = os.path.basename(datafile)
    
    with h5py.File(datafile,'r') as hand:
        trialstarts = hand['/intervals/trials/start_time'][()]
        trialstops = hand['/intervals/trials/stop_time'][()]
        recid = hand['identifier'][()].decode()

        FS = hand['/units/FS'][()]
        RS = hand['/units/RS'][()]

        uids = hand['/units/id'][RS]
        
        psth_file = np.zeros((len(uids),len(psth_tvec)))
        nspikes_pre = np.zeros(len(uids))
        

        for uu,uid in enumerate(uids):
            i1, i0 = hand['/units/spike_times_index'][uid], hand['/units/spike_times_index'][uid-1] if uid>0 else 0

            ustimes = hand['/units/spike_times'][i0:i1]
            nspikes_tot = np.append(nspikes_tot, len(ustimes))
            nspikes_psth = np.append(nspikes_psth, 0)
            
            rastermat = np.empty((0,2))
            psth_unit = np.zeros((len(trialstops),len(psth_tvec)))
            for tt,tint in enumerate(trialstops):
                stimes = ustimes[(ustimes >= tint-prestim) & (ustimes <= tint+poststim)] - tint
                nspikes = len(stimes)
                nspikes_psth[-1] += nspikes
                #nspikes_pre[uu] = np.sum(stimes<0.)
                #nspikes_pre[uu] = len(stimes)
                if nspikes > 0:
                    rastermat = np.r_[rastermat,np.array([stimes,np.ones(len(stimes))*tt]).T]
                    psth_temp,_ = np.histogram(rastermat[:,0],psth_tvec_temp)
                    psth_s = np.convolve( psth_temp, gauss_kernel, mode='same' )
                else:
                    psth_s = psth_tvec*0.
                
                psth_unit[tt] = psth_s
            
            psth_file[uu] = np.nanmean(psth_unit,axis=0)
    psth_mat = np.concatenate((psth_mat, psth_file),axis=0)


saveFile = os.path.join(savedir,'Pete_trialstop_PSTH.h5')
print(saveFile)
if os.path.isfile(saveFile):
    os.remove(saveFile)
    
with h5py.File(saveFile,'w') as hand:
    hand.create_dataset('psth',data=psth_mat,dtype='f')
    hand.create_dataset('psth_tvec',data=psth_tvec,dtype='f')
    hand.create_dataset('nspikes_tot',data=nspikes_tot,dtype='f')
    hand.create_dataset('nspikes_psth',data=nspikes_psth,dtype='f')

    hand.attrs['dataset'] = 'Pete_Rudebeck'
    hand.attrs['alignment'] = 'Trial stops'
    hand.attrs['sr_psth'] = sr_psth
    hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
    hand.attrs['srcfile'] = __file__



'''

    tintmat = np.stack((trialstarts-tintdur,trialstarts),axis=1)

    #aRec = dc.RecPassive(datafile,cfgpath,dspath)
    savename = os.path.join(dstdir,savepattern % (recid))
    with h5py.File(savename, 'w') as hand:
        ds = hand.create_dataset('tints', data=tintmat)
        ds.attrs['recid'] = recid
        ds.attrs['ttag'] = ttag

        ds.attrs['pre'] = tintdur
        ds.attrs['post'] = 0
        ds.attrs['githash'] = git.Repo(search_parent_directories=True).head.object.hexsha
        # ds.attrs['srcfile'] = __file__
    

'''



'''
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


####
'''