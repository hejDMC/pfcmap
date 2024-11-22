import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kendalltau,pearsonr
import pandas as pd


pathpath,run1,run2,stag1,stag2,ncluststr,cmethod = sys.argv[1:]

stags = [stag1,stag2]

'''
run1 = 'runC00dMP3_brain'
run2 = 'runIBLPasdMP3_brain_pj'
stags = ['Carlen','IBL']
ncluststr = '8'
cmethod = 'ward'
pathpath = 'PATHS/filepaths_carlen.yml'

'''

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S


statsfile1 = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(run1,ncluststr,cmethod))
statsfile2 = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(run2,ncluststr,cmethod))

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_base =  pathdict['figdir_root'] + '/dataset_comparison'
figdir_mother = os.path.join(figdir_base,'%s__vs__%s'%(run2,run1))
outfile = os.path.join(figdir_mother,'correlation_values.xlsx')


def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%sVs%s.%s'%(stags[1],stags[0],S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


reftag = 'refall'
thistag = 'laydepth'
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
ncl = int(ncluststr)

cmap = mpl.cm.get_cmap(S.cmap_clust)#
norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cdict_clust = {lab:cmap(norm(lab)) for lab in np.arange(ncl)}

corrdict = {tag:{} for tag in ['IBLvsCarlen']+stags}
vmin,vmax = -1,1
extend = 'neither'

for cfn,cfnlab in zip([kendalltau,pearsonr],['KT','CC']):
    corrmat = np.zeros((ncl,ncl))
    pmat = np.zeros((ncl,ncl))
    for ii in np.arange(ncl):
        for jj in np.arange(ncl):
            corrmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).correlation#[0,1]
            pmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).pvalue#[0,1]


    corrdict['IBLvsCarlen'][cfnlab] = {'pvalues':pmat,'corrvalues':corrmat}



    f,axarr = plt.subplots(1,2,figsize=(3.6, 3.35),gridspec_kw={'width_ratios':[1,0.1]})
    ax,cax = axarr
    f.subplots_adjust(left=0.15,bottom=0.15,right=0.82,wspace=0.1)
    im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap='PiYG_r')
    ax.set_xlabel(stags[0])
    ax.set_ylabel(stags[1])
    ax.set_xticks(np.arange(ncl))
    ax.set_yticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    tlabs = ax.set_yticklabels(np.arange(ncl)+1,fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    ax.set_aspect('equal')
    f.colorbar(im,cax=cax,extend=extend)
    ax.set_title(titlestr)
    cax.set_title(cfnlab)
    ax.text(0.99,1.01,'N:%i'%(nregs),transform=ax.transAxes,ha='right',va='bottom')
    figsaver(f,'clustcorr_%s_%s_%s'%(reftag,titlestr,cfnlab))


    xvec = np.arange(ncl)
    corrvec = np.diag(corrmat)
    mcorr = np.mean(corrvec)
    f, ax = plt.subplots(figsize=(1.5 + 0.2 * ncl, 2))
    f.subplots_adjust(left=0.3, bottom=0.3)
    blist = ax.bar(xvec, corrvec, color='k')
    ax.set_xticks(xvec)
    tlabs = ax.set_xticklabels(xvec + 1, fontweight='bold')
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        tlabs[cc].set_color(col)
        blist[cc].set_color(col)
        # ax.xaxis.get_ticklabels()
    ax.axhline(mcorr,color='grey',linestyle='--')
    ax.text(0.99,mcorr,'%1.2f'%mcorr,transform=mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData),ha='right',va='bottom',color='grey')
    ax.set_ylabel(cfn.__name__)
    ax.set_xlabel('category')
    ax.set_xlim([xvec.min() - 0.5, xvec.max() + 0.5])
    ax.set_title(' vs '.join(stags))
    for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
    figsaver(f,'clustcorrDiag_%s_%s_%s'%(reftag,titlestr,cfnlab))




    for mymat,stag in zip([X1,X2],stags):

        ncl = int(ncluststr)
        corrmat = np.zeros((ncl,ncl))
        pmat = np.zeros((ncl,ncl))

        for ii in np.arange(ncl):
            for jj in np.arange(ncl):
                corrmat[ii,jj] = cfn(mymat[:,ii],mymat[:,jj]).correlation#[0,1]
                pmat[ii,jj] = cfn(mymat[:,ii],mymat[:,jj]).pvalue#[0,1]

        corrdict[stag][cfnlab] = {'pvalues':pmat,'corrvalues':corrmat}




        f,axarr = plt.subplots(1,2,figsize=(3.6, 3.35),gridspec_kw={'width_ratios':[1,0.1]})
        ax,cax = axarr
        f.subplots_adjust(left=0.15,bottom=0.15,right=0.82,wspace=0.1)
        im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap='PiYG_r')
        ax.set_xlabel(stag)
        ax.set_ylabel(stag)
        ax.set_xticks(np.arange(ncl))
        ax.set_yticks(np.arange(ncl))
        tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
        for cc in np.arange(ncl):
            tlabs[cc].set_color(cdict_clust[cc])
        tlabs = ax.set_yticklabels(np.arange(ncl)+1,fontweight='bold')
        for cc in np.arange(ncl):
            tlabs[cc].set_color(cdict_clust[cc])
        ax.set_aspect('equal')
        f.colorbar(im,cax=cax,extend=extend)
        ax.set_title(titlestr)
        cax.set_title(cfnlab)
        ax.text(0.99,1.01,'N:%i'%(nregs),transform=ax.transAxes,ha='right',va='bottom')
        figsaver(f,'%sONLY_clustcorr_%s_%s_%s'%(stag,reftag,titlestr,cfnlab))


with pd.ExcelWriter(outfile) as writer:
    for stag in corrdict.keys():
        for cfnlab in corrdict[stag].keys():
            for flav in corrdict[stag][cfnlab].keys():
                outmat = corrdict[stag][cfnlab][flav]
                df = pd.DataFrame(data=outmat,columns=np.arange(ncl)+1,index=np.arange(ncl)+1)
                df.to_excel(writer, sheet_name='%s_%s_%s'%(stag,cfnlab,flav))


alabs = np.array([aval.split('|')[0] for aval in avals_avail])
xvec = np.arange(nregs)
for ii in np.arange(ncl):
    f,ax = plt.subplots(figsize=(5,3.5))
    f.subplots_adjust(bottom=0.18)
    vec1,vec2 = X1[:,ii],X2[:,ii]
    statsstrlist = ['%s:%1.2f'%(ctag,cfn(vec1,vec2).correlation) for cfn,ctag in zip([kendalltau,pearsonr],['KT','CC'])]
    ax.set_title('%s clust%i   %s'%(titlestr,ii+1,'; '.join(statsstrlist)))
    ax.plot(xvec,vec1,'ko')
    ax.plot(xvec,vec2,'ro')
    ax.plot(xvec,vec1,'k',alpha=0.2)
    ax.plot(xvec,vec2,'r',alpha=0.2)
    ax.set_xticks(xvec)
    ax.set_xticklabels(alabs,rotation=-90)
    ax.text(0.99,0.98,stags[0],color='k',transform=ax.transAxes,ha='right',va='top')
    ax.text(0.99,0.90,stags[1],color='r',transform=ax.transAxes,ha='right',va='top')
    ax.set_ylabel('enrichment')
    figsaver(f,'diag_comparisons/clust%i_comparison_%s_%s'%(ii+1,reftag,titlestr))

diff_matz = np.zeros((ncl,nregs))
diff_mat = np.zeros((ncl,nregs))

for ii in np.arange(ncl):
    v1,v2 = X1[:,ii],X2[:,ii]
    z1,z2 = [(myv-np.mean(myv))/np.std(myv) for myv in [v1,v2]]
    diff_matz[ii] = z2-z1
    diff_mat[ii] = v2-v1


for mydmat,difftag in zip([diff_mat,diff_matz],['absdiff','reldiff']):
    vlim = np.max(np.abs(mydmat))
    meandiff = np.abs(mydmat).mean(axis=0)
    sortinds = np.argsort(meandiff)
    #sortinds =cfns.get_sortidx_featdist(diff_mat.T, linkmethod='ward')#np.argsort(meandiff)
    f,axarr = plt.subplots(1,3,gridspec_kw={'width_ratios':[1,0.1,0.1]})
    ax,aax,cax = axarr
    im = ax.imshow(mydmat[:,sortinds].T,origin='lower',aspect='auto',cmap='PRGn_r',vmin=-vlim,vmax=vlim)
    aax.imshow(meandiff[sortinds][:,None],origin='lower',aspect='auto',cmap='inferno')
    aax.set_axis_off()
    aax.set_title(r'$\overline{|\delta|}$')
    cax.set_title('diff [z]')
    ax.set_title(' vs '.join(stags))
    ax.set_yticks(np.arange(nregs))
    ax.set_yticklabels(alabs[sortinds])
    ax.set_xticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl) + 1, fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    f.colorbar(im,cax=cax)
    figsaver(f,'diag_comparisons/regionalDifferences__%s_%s_%s'%(reftag,titlestr,difftag))