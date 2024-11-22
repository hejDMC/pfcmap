import os
import numpy as np
import sys
import yaml
import h5py
import git
import matplotlib.pyplot as plt


filename,pathpath,dstpath,tintfile,exptype = sys.argv[1:]

recid = os.path.basename(filename).split('.')[0]

stylepath = 'config/presentation.mplstyle'

plt.style.use(stylepath)

sr_psth = 2000
k_std = (10/1000)*sr_psth #first number: width in ms
k_win = k_std*4

def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )
gauss_kernel = np.array([gaussian( x , k_std) for x in np.arange( -k_win, k_win+1)])
#f,ax = plt.subplots()
#ax.plot(gauss_kernel)

with open(pathpath) as yfile: pdict = yaml.safe_load(yfile)

#print('uuuuuu',filename)
sys.path.append(pdict['workspace_dir'])
from pfcmap.python.utils import data_classes as dc
from pfcmap.python.utils import data_handling as dh

figpath = pdict['fig_path']+'/psth'

cfgpath,dspath = dc.get_paths('Passive',pdict)#N.B.: we do not need specific configs for the other types of datasets

aRec = dc.RecPassive(filename,cfgpath,dspath)
print('processing %s'%recid)

with h5py.File(tintfile,'r') as hand:
    tints = hand['tints'][()]
    prestim = hand['tints'].attrs['pre']
    poststim = hand['tints'].attrs['post']

print('tints: %s'%str(tints.shape))
psth_tvec_temp = np.arange(-prestim,poststim,1/sr_psth)
psth_tvec = psth_tvec_temp[:-1]+0.5/sr_psth
pre_bool = psth_tvec<0.
#tints = tints0 + np.array([0,0.7])[None,:]

uids = aRec.unit_ids.astype(int)

unitdict = {unitid:np.unique(dh.grab_spikes_unit(aRec, unitid)) for unitid in uids}

tintdurs = np.diff(tints)[:,0]

psth_mat = np.zeros((len(uids),len(psth_tvec)))
nspikes_pre = np.zeros(len(uids))
nspikes = np.zeros(len(uids))


for uu,uid in enumerate(uids):
    spiketimes = unitdict[uid]
    rastermat = np.empty((0,2))
    for tt,tint in enumerate(tints):
        stimes = spiketimes[(spiketimes >= tint[0]) & (spiketimes <= tint[1])] - prestim - tint[0]
        nspikes[uu] = len(stimes)
        nspikes_pre[uu] = np.sum(stimes<0.)
        if nspikes_pre[uu] > 0:
            rastermat = np.r_[rastermat,np.array([stimes,np.ones(len(stimes))*tt]).T]
            psth_temp,_ = np.histogram(rastermat[:,0],psth_tvec_temp)
            psth_s = np.convolve( psth_temp, gauss_kernel, mode='same' )
        else:
            psth_s = psth_tvec*np.nan
        psth_mat[uu] = psth_s



savename_stub = os.path.basename(tintfile).split('.h5')[0]
#savename = os.path.join(dstpath,'%s__TSEL%s_quantities.h5'%(aRec.id,ttag))
kstd_ms = k_std/sr_psth*1000
savename_stubx = '%s_psth_ks%i'%(savename_stub,int(kstd_ms))
savename = os.path.join(dstpath,'%s.h5'%(savename_stubx))
print('saving to %s'%savename)

with h5py.File(savename,'w') as hand:
    hand.create_dataset('psth',data=psth_mat,dtype='f')
    hand.create_dataset('psth_tvec',data=psth_tvec,dtype='f')
    hand.create_dataset('nspikes',data=nspikes,dtype='i')
    hand.create_dataset('nspikes_pre',data=nspikes_pre,dtype='i')

    hand.create_dataset('uids',data=uids)
    hand.attrs['recid'] = aRec.id
    hand.attrs['n_tints'] = len(tints)
    hand.attrs['sr_psth'] = sr_psth
    hand.attrs['kstd_ms'] = kstd_ms
    hand.attrs['tfile'] = tintfile
    hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
    hand.attrs['srcfile'] = __file__



def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figpath, '%s__%s.png'%(savename_stubx,nametag))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    f.suptitle('%s - kstd:%s'%(recid,kstd_ms))

    fig.savefig(figname)
    if closeit: plt.close(fig)

print('plotting %s to %s'%(recid,figpath))


nonnaninds = np.unique(np.where(~np.isnan(psth_mat))[0])
n_examples =  np.min([20,len(nonnaninds)])
u_space = 7
sel_inds = np.random.choice(nonnaninds,n_examples,replace=False)
f,ax = plt.subplots(figsize=(5,10))
for uu, psth in enumerate(psth_mat[sel_inds]):
    psth_n = (psth -np.mean(psth[pre_bool])) /np.std(psth[pre_bool])
    myline = ax.plot(psth_tvec,psth_n+u_space*uu)
    ax.axhline(u_space*uu,color=myline[0].get_color(),alpha=0.2,zorder=-10)
ax.axvline(0,color='k',zorder=-9,alpha=0.5)
ax.set_xlim([-0.5,0.65])
ax.set_ylim([-5,uu*4+8])
ax.set_ylabel('firing [z] + offset')
ax.set_xlabel('time [s]')
figsaver(f,'examples')


#grand average
psth_nonan = psth_mat[nonnaninds]
psth_normed = (psth_nonan -np.mean(psth_nonan[:,pre_bool],axis=1)[:,None]) /np.std(psth_nonan[:,pre_bool],axis=1)[:,None]
tbounds = [-0.5,0.65]
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
figsaver(f,'grand_average')



