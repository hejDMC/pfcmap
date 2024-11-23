import sys
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


pathpath,myrun = sys.argv[1:]

#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runC00dMP3_brain'


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

savedir_output = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__corrstats'%myrun)

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(savedir_output, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)


from scipy.stats import kendalltau,pearsonr,linregress
nfeats = len(somfeats)
featlabs = [somfeat.split('_')[0] for somfeat in somfeats]
ranges = {'B_mean':[-0.6,0.6],\
          'M_mean':[-0.7,0.7],
          'rate_mean':[np.log10(0.34),np.log10(150)]}
ranges = {'B_mean':[-0.5,0.5],\
          'M_mean':[-0.5,0.5],
          'rate_mean':[-0.22,1.6]}
tickdict = {'B_mean':np.arange(-0.4,0.5,0.2),\
            'M_mean':np.arange(-0.4,0.5,0.2),\
            'rate_mean':np.arange(0.,1.4,0.4)}


nbins = 50
cmap = 'inferno'

Units_pfc = [U for U in Units if S.check_pfc(U.area)]


for Usel,seltag in zip([Units,Units_pfc],['_allunits','_pfc']):

    figsaver_spec = lambda fig,nametag: figsaver(fig, 'corrstats%s/%s_%s'%(seltag,nametag,seltag), closeit=True)

    X = np.vstack([[getattr(U,somfeat) for U in Usel] for somfeat in somfeats])



    for cfn,cfnlab in zip([kendalltau,pearsonr],['KT','CC']):
        #cfn = pearsonr
        #cfnlab = 'CC'
        corrmat = np.zeros((nfeats,nfeats))
        pmat = np.zeros((nfeats,nfeats))
        for ii in np.arange(nfeats):
            for jj in np.arange(nfeats):
                result = cfn(X[ii],X[jj])
                corrmat[ii,jj] = result.correlation
                pmat[ii,jj] = result.pvalue

        vminmax = np.abs(np.array([np.min(np.tril(corrmat)),np.max(np.tril(corrmat))])).max()
        f,axarr = plt.subplots(1,2,figsize=(2.7, 2.0),gridspec_kw={'width_ratios':[1,0.1]})
        f.subplots_adjust(right=0.87)
        ax,cax = axarr
        f.subplots_adjust(left=0.15,bottom=0.15,right=0.78,wspace=0.1)
        im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='PiYG_r')
        for ii in np.arange(nfeats-1):
            for jj in np.arange(ii+1,nfeats):
                ax.text(ii,jj,'%1.2f'%(corrmat[ii,jj]),ha='center',va='center',fontsize=8)
                ax.text(jj,ii,'%1.2e'%(pmat[ii,jj]),ha='center',va='center',fontsize=8)

        ax.set_xticks(np.arange(nfeats))
        ax.set_yticks(np.arange(nfeats))
        ax.set_xticklabels(featlabs,fontweight='bold')
        ax.set_yticklabels(featlabs,fontweight='bold',rotation=90)
        ax.set_aspect('equal')
        f.colorbar(im,cax=cax)
        cax.set_title(cfnlab)
        #f.suptitle('%s N=%i'%(myrun,X.shape[1]),fontsize=8)
        ax.text(-0.1,1.01,'%s N:%i'%(myrun,X.shape[1]),transform=ax.transAxes,ha='left',va='bottom',fontsize=8)
        figsaver_spec(f,'featcorrmat_%s'%(cfnlab))

    for ii in np.arange(nfeats-1):
        for jj in np.arange(ii+1,nfeats):
            #ii = 0
            #jj = 2

            xvec,yvec = X[ii], X[jj]

            #lsq_res = linregress(xvec,yvec)
            #x_simp = np.array([xvec.min(),xvec.max()])
            #ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='r', zorder=-10)

            xminmax = ranges[somfeats[ii]]#xvec.min(),xvec.max()
            yminmax = ranges[somfeats[jj]]#yvec.min(),yvec.max()
            binsx,binsy = np.linspace(*xminmax,nbins), np.linspace(*yminmax,nbins)
            H, xedges, yedges = np.histogram2d(xvec, yvec, bins=(binsx,binsy))

            XX, YY = np.meshgrid(xedges, yedges)
            f,axarr = plt.subplots(1,2,figsize=(3.7, 3.0),gridspec_kw={'width_ratios':[1,0.1]})
            ax,cax = axarr
            #f,ax = plt.subplots(figsize=(3.,3.))
            f.subplots_adjust(left=0.2,bottom=0.2,right=0.85)
            #im = ax.pcolormesh(XX, YY, H.T, cmap=cmap)
            im = ax.imshow(H.T, interpolation='nearest', cmap=cmap, extent=[*xminmax,*yminmax],origin='lower',aspect='auto')
            ax.set_xlim(xminmax)
            ax.set_ylim(yminmax)
            ax.set_xlabel(featlabs[ii])
            ax.set_ylabel(featlabs[jj])
            ax.set_xticks(tickdict[somfeats[ii]])
            ax.set_yticks(tickdict[somfeats[jj]])

            if not somfeats[ii].count('rate'):
                ax.axvline(0,color='cyan',linestyle=':')

            if not somfeats[jj].count('rate'):
                ax.axhline(0,color='cyan',linestyle=':')

            ax.set_box_aspect(1)
            f.colorbar(im,cax=cax)
            cax.set_title('#')
            f.suptitle('%s N:%i'%(myrun,len(xvec)))
            figsaver_spec(f,'%s_vs_%s'%(featlabs[jj],featlabs[ii]))

#ax.set_xlim(ranges[somfeats[ii]])
#ax.set_ylim(ranges[somfeats[jj]])


