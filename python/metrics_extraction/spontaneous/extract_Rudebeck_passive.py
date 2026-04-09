from glob import glob
import os
import numpy as np
import h5py
import git
import sys
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import track, Progress

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

tintdur = 1
ttag = 'prestim'
 #

# intertrials = np.empty((0))
with Progress() as progress:
    task = progress.add_task(f"Extraction of trial start times. Current file : None", total = len(myfiles))
    for datafile in myfiles:
        progress.update(task, description=f"Extraction of trial start times. Current file : {os.path.basename(datafile)}", advance=1)

        with h5py.File(datafile,'r') as hand:
            trialstarts = hand['/intervals/trials/start_time'][()]
            trialstops = hand['/intervals/trials/stop_time'][()]
            recid = hand['identifier'][()].decode()
            
        trialstops = np.insert(trialstops, 0, 0)
        #intertrials = np.append(intertrials, trialstarts - trialstops[:-1])
        intertrials =  trialstarts - trialstops[:-1]
        for tintdur in np.arange(1,4):
        # Keep only trials where the intertrial interval is above tintdur+0.5s
            valid_trials = trialstarts[intertrials > (tintdur + 0.5)]
            tintmat = np.stack((valid_trials-tintdur, valid_trials),axis=1)

            savepattern = '%s__TSEL{}{}.h5'.format(ttag,int(tintdur))
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
savedir = 'D:/Carlen/Pete_Rudebeck'
saveFile = os.path.join(savedir, 'Trial_intervals.h5')
with h5py.File(saveFile, 'w') as hand:
    hand.create_dataset('trialintervals', data=intertrials)


ax = sns.histplot(intertrials, stat='proportion', binrange=(0,5), binwidth=0.1, color='b')
height = ax.get_ylim()
ax.vlines([1,2,3], ymin=height[0], ymax=height[1], linestyles='-', colors='firebrick')
for i in range(1,4):
    ax.text(x=i, y=height[1]*1.1, s='{:.1f}%'.format(100*np.sum(intertrials>i)/len(intertrials)), color='firebrick')
ax.set_ylim([0,height[1]*1.2])
ax.set_title("Intertrial intervals above selected durations")
ax.set_xlabel("Duration")
plt.show()


'''