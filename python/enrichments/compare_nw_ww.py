import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kendalltau,pearsonr
import pandas as pd

pathpath,run1,run2,stag1,stag2,ncluststr1,ncluststr2, cmethod = sys.argv[1:]
stags = [stag1,stag2]

'''
run1 = 'runC00dMP3_brain'
run2 = 'runC00dMI3_brain'
stags = ['ww','nw']
ncluststr1 = '8'
ncluststr2 = '5'
cmethod = 'ward'
'''


pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S


statsfile1 = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(run1,ncluststr1,cmethod))
statsfile2 = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(run2,ncluststr2,cmethod))

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_base =  pathdict['figdir_root'] + '/nw_ww_comparison'
figdir_mother = os.path.join(figdir_base,'%s__vs__%s'%(run2,run1))

outfile = os.path.join(figdir_mother,'correlation_values.xlsx')

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%sVs%s.%s'%(stags[1],stags[0],S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




thistag = 'laydepth'
reftag = 'refall'
depth_checker = lambda aval: aval.count('|deep')

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr

#find out which labels are availabe for both
avaldict_temp = {}
for statsfile,stag in zip([statsfile1,statsfile2],stags):
    with h5py.File(statsfile,'r') as hand:
        statshand = hand['regs'][reftag][thistag]
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        #presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval.count('|sup') and not aval=='na' \
        #                        and not (S.check_pfc(aval) and not aval.count('|deep'))])
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and depth_checker(aval)])
        avaldict_temp[stag] = mystats['avals1'][presel_inds]

titlestr = 'deepCtx'


avals_avail = np.array([lab for lab in avaldict_temp[stags[0]] if lab in avaldict_temp[stags[1]]])

dsdict = {}
for statsfile,stag in zip([statsfile1,statsfile2],stags):
    with h5py.File(statsfile,'r') as hand:
        statshand = hand['regs'][reftag][thistag]
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([np.where(mystats['avals1']==aval)[0][0] for aa, aval in enumerate(avals_avail)])
        assert (mystats['avals1'][presel_inds] == avals_avail).all(),'mismatching avalues'
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        dsdict[stag] = sdict



statsfn = S.enr_fn
X1,X2 = [statsfn(dsdict[stag]) for stag in stags]
nregs = len(avals_avail)

ncl1,ncl2 = int(ncluststr1),int(ncluststr2)



cmap = mpl.cm.get_cmap(S.cmap_clust)  #

norm1 = mpl.colors.Normalize(vmin=0, vmax=ncl1 - 1)
norm2 = mpl.colors.Normalize(vmin=0, vmax=ncl2 - 1)

cdict_clust1 = {lab: cmap(norm1(lab)) for lab in np.arange(ncl1)}
cdict_clust2 = {lab: cmap(norm2(lab)) for lab in np.arange(ncl2)}

corrdict = {}
for cfn,cfnlab in zip([kendalltau,pearsonr],['KT','CC']):
    corrmat = np.zeros((ncl1,ncl2))
    pmat = np.zeros((ncl1,ncl2))
    for ii in np.arange(ncl1):
        for jj in np.arange(ncl2):
            corrmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).correlation#[0,1]
            pmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).pvalue#[0,1]
    corrdict[cfnlab] = {'pvalues':pmat,'corrvalues':corrmat}



    f,axarr = plt.subplots(1,2,figsize=(1.5+ncl1*0.3, 0.5+ncl2*0.3),gridspec_kw={'width_ratios':[1,0.1]})
    ax,cax = axarr
    f.subplots_adjust(left=0.15,bottom=0.15,right=0.82,wspace=0.1)
    im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='PiYG_r')
    ax.set_xlabel(stags[0])
    ax.set_ylabel(stags[1])
    ax.set_xticks(np.arange(ncl1))
    ax.set_yticks(np.arange(ncl2))
    tlabs = ax.set_xticklabels(np.arange(ncl1)+1,fontweight='bold')
    for cc in np.arange(ncl1):
        tlabs[cc].set_color(cdict_clust1[cc])
    tlabs = ax.set_yticklabels(np.arange(ncl2)+1,fontweight='bold')
    for cc in np.arange(ncl2):
        tlabs[cc].set_color(cdict_clust2[cc])
    ax.set_aspect('equal')
    f.colorbar(im,cax=cax,extend='neither')
    ax.set_title(titlestr)
    cax.set_title(cfnlab)
    ax.text(0.99,1.01,'N:%i'%(nregs),transform=ax.transAxes,ha='right',va='bottom')
    cax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    f.tight_layout()
    figsaver(f,'clustcorr_%s_%s_%s'%(reftag,titlestr,cfnlab))

# save matrices as xlsx
with pd.ExcelWriter(outfile) as writer:
    for cfnlab in corrdict.keys():
        for flav in corrdict[cfnlab].keys():
            outmat = corrdict[cfnlab][flav]
            df = pd.DataFrame(data=outmat,columns=np.arange(ncl2)+1,index=np.arange(ncl1)+1)
            df.to_excel(writer, sheet_name='%s_%s'%(cfnlab,flav))
