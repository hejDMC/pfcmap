from glob import glob
import os
import numpy as np
import h5py
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt

savedir = 'D:/Carlen/Pete_Rudebeck'

saveFile = os.path.join(savedir,'Pete_trialstop_PSTH.h5')
with h5py.File(saveFile, 'r') as hand:
    psth_mat = hand['psth'][()]
    psth_tvec = hand['psth_tvec'][()]
    nspikes_tot = hand['nspikes_tot'][()]
    nspikes_psth = hand['nspikes_psth'][()]


post_bool = psth_tvec>1
nonzeroids = np.any(psth_mat[:,post_bool],axis=1)

psth_nonan = psth_mat[nonzeroids]
psth_normed = (psth_nonan -np.mean(psth_nonan[:,post_bool],axis=1)[:,None]) /np.std(psth_nonan[:,post_bool],axis=1)[:,None]
tbounds = [-0.95,1.95]
tbool = (psth_tvec>=tbounds[0]) & (psth_tvec<=tbounds[1])


f,ax = plt.subplots(figsize=(4,3))
f.subplots_adjust(left=0.15, right=0.95,bottom=0.2,top=0.85)
ax.plot(psth_tvec[tbool],np.mean(psth_normed[:,tbool],axis=0),'k')
#ax.fill_between(psth_tvec[tbool],np.mean(psth_normed[:,tbool],axis=0)-np.std(psth_normed[:,tbool],axis=0),\
#        np.mean(psth_normed[:,tbool],axis=0)+np.std(psth_normed[:,tbool],axis=0),color='grey',zorder=-2,ec='none')
ax.axvline(0,color='silver',zorder=-9)
ax.set_xlim(tbounds)
ax.set_xlabel('time [s]')
ax.set_ylabel('firing [z]')
plt.show()

'''
n_examples =  np.min([20,np.sum(nonzeroids)])
u_space = 7
sel_inds = np.random.choice(np.flatnonzero(nonzeroids),n_examples,replace=False)
f,ax = plt.subplots(figsize=(5,10))
for uu, psth in enumerate(psth_mat[sel_inds]):
    psth_n = (psth -np.mean(psth[post_bool])) /np.std(psth[post_bool])
    myline = ax.plot(psth_tvec,psth_n+u_space*uu)
    ax.axhline(u_space*uu,color=myline[0].get_color(),alpha=0.2,zorder=-10)
ax.axvline(0,color='k',zorder=-9,alpha=0.5)
#ax.set_xlim([-1,2])
ax.set_ylim([-5,uu*4+8])
ax.set_ylabel('firing [z] + offset')
ax.set_xlabel('time [s]')
plt.show()
'''