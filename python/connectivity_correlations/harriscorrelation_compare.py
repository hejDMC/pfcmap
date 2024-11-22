import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

run1 = 'runC00dMP3_brain'

stags = ['Carlen','IBL']
if run1.count('dMP'):
    ncluststr = '8'
    run2 = 'runIBLPasdMP3_brain_pj'
else:
    ncluststr = '5'
    run2 = 'runIBLPasdMI3_brain_pj'
cmethod = 'ward'
flav = 'enr'



pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S

genfigdir = pathdict['figdir_root'] + '/hierarchy/harris_correlations/compare_%s__%s'%(run1,run2)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(genfigdir, nametag + '__%sVs%s.%s'%(stags[1],stags[0],S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

cfile1 = os.path.join(pathdict['statsdict_dir'],'harriscorr__%s__ncl%s_%s_%s.h5'%(run1,ncluststr,cmethod,flav))
cfile2 = os.path.join(pathdict['statsdict_dir'],'harriscorr__%s__ncl%s_%s_%s.h5'%(run2,ncluststr,cmethod,flav))


dsdict = {}
for cfile,stag in zip([cfile1,cfile2],stags):
    with h5py.File(cfile,'r') as hand:
        dsdict[stag] = uloader.unpack_statshand(hand, remove_srcpath=True)

ncl = int(ncluststr)
cmap = mpl.cm.get_cmap(S.cmap_clust)#
norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cdict_clust = {lab:cmap(norm(lab)) for lab in np.arange(ncl)}

xvec = np.arange(ncl)
for statsflav in ['kendall','pearson']:
    corrmat = np.vstack([dsdict[stag]['/%s/corrvec'%(statsflav)] for stag in stags]).T

    for ii,stag in enumerate(stags):
        #ii = 1
        #stag = stags[ii]
        corrvec = corrmat[:,ii]
        f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
        f.subplots_adjust(left=0.3,bottom=0.3)
        blist = ax.bar(xvec,corrvec,color='k')
        ax.set_xticks(xvec)
        tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')
        for cc in np.arange(ncl):
            col = cdict_clust[cc]
            tlabs[cc].set_color(col)
            blist[cc].set_color(col)
            #ax.xaxis.get_ticklabels()
        ax.set_ylabel(statsflav)
        ax.set_xlabel('category')
        ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
        ax.set_title(stag)
        ax.set_ylim([-0.9,0.9])
        for pos in ['top','right']:ax.spines[pos].set_visible(False)
        figsaver(f,'ONLY%s_corr%s'%(stag,statsflav))



ymax = 0.75
for statsflav in ['kendall','pearson']:
    corrmat = np.vstack([dsdict[stag]['/%s/corrvec'%(statsflav)] for stag in stags]).T
    f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
    f.subplots_adjust(left=0.3,bottom=0.3)
    xvec = np.arange(ncl)
    for ii,[stag,offset,alph] in enumerate(zip(stags,[-0.2,0.2],[0.5,1.])):
        #ii = 1
        #stag = stags[ii]
        corrvec = corrmat[:,ii]
        xvec_off = np.arange(ncl)+offset

        blist = ax.bar(xvec_off,corrvec,color='k',width=0.3,alpha=alph)
        for cc in np.arange(ncl):
            col = cdict_clust[cc]
            blist[cc].set_color(col)

    ax.set_xticks(xvec)
    tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        tlabs[cc].set_color(col)
        #ax.xaxis.get_ticklabels()
    ax.set_ylabel(statsflav)
    ax.set_xlabel('category')
    ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
    ax.set_title(' and '.join(stags))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,which='both',color='grey',linestyle=':',zorder=-10)
    for pos in ['top','right']:ax.spines[pos].set_visible(False)
    figsaver(f,'compare%s_corr%s'%(''.join(stags),statsflav))

