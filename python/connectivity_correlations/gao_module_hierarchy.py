import yaml
import sys
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from glob import glob
import pandas as pd
import matplotlib as mpl
import h5py

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
ncluststr = '8'
cmethod = 'ward'
'''



basetags = ['levels','emat_zerod','emat']
reftags = ['refall','refPFC']
statstag = 'enr'
roi_tag = 'gaoRois'
seltags = ['deepRois','nolayRois']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python import settings as S
from pfcmap.python.utils import unitloader as uloader
#from pfcmap.python.utils import tessellation_tools as ttools
gao_hier_path = pathdict['gao_hierarchy_file']
rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

genfigdir = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/Nc%s/%s/enrichments/rois/%s'%(myrun,kshape[0],kshape[1],ncluststr,cmethod,roi_tag))


def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(genfigdir, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


with open(gao_hier_path, newline='') as csvfile:
    hdata = list(csv.reader(csvfile, delimiter=','))[1:]
hdict = {int(float(el[0])): float(el[1]) for el in hdata}
allrois_gao = np.sort(np.array(list(hdict.keys())))
hvec = np.array([hdict[key] for key in allrois_gao])

for reftag in reftags:
    for basetag in basetags:
        for thistag in seltags:

            def make_savename(plotname):
                return '%s/%s__%s/%s/%s__%s_%s_%s_%s'%(reftag,thistag,statstag,basetag,plotname,reftag,thistag,statstag,basetag)

            srcfiles = glob(os.path.join(genfigdir,reftag,'%s__%s'%(thistag,statstag),basetag,'clustered_flatmaps_labeldict__*.xlsx'))
            assert len(srcfiles)==1, 'not exactly one srcfile  %s'%(str(srcfiles))
            srcfile = srcfiles[0]



            #srcfiles = glob(os.path.join(tempsrcdir,'*.xlsx'))




            df = pd.read_excel(srcfile, sheet_name=None)#reads alll sheets
            ncvec = np.sort(np.array([int(val.replace('ncl','')) for val in df.keys()]))
            nclusters = ncvec[0]
            key = 'ncl%i'%nclusters
            ddf = df[key]
            rois_data = np.array(ddf.loc[:,ddf.columns[0]])
            labeldict = {}
            color_dict = {nc:{} for nc in ncvec}
            for nclusters in ncvec:
                key = 'ncl%i'%nclusters
                ddf = df[key]
                assert (rois_data==np.array(ddf.loc[:,ddf.columns[0]])).all(),'mismatching roi data: %s'%key
                labels = np.array(ddf.loc[:, 'lab'])
                ulabels = np.unique(labels)
                ccolors =  ddf.loc[:, 'color']
                color_dict[nclusters] = {ulab:eval(ccolors[np.where(labels==ulab)[0][0]]) for ulab in ulabels}
                labeldict[nclusters] = labels


            hinds = np.array([np.where(allrois_gao == roi)[0][0] for roi in rois_data])
            htemp = hvec[hinds]
            h_cond = ~np.isnan(htemp)
            hiervec = htemp[h_cond]
            myrois = rois_data[h_cond]
            #labeldict2 = {nclusters:labeldict[nclusters][h_cond] for nclusters in ncvec}

            lhdict = {nc:{} for nc in ncvec}
            for nclusters in ncvec:
                labels = labeldict[nclusters][h_cond]
                #labels = labeldict2[nclusters]
                ulabs = np.unique(labels)
                lhdict[nclusters] = {ulab:hiervec[labels==ulab] for ulab in ulabs}



            #mpl.use('qtagg',force=True)#['TKAgg','GTKAgg','Qt4Agg','WXAgg']
            #f,ax = plt.subplots()
            #plt.show()

            #ncreal_vec = np.array([len(lhdict[ncl]) for ncl in ncvec ])
            jit=-0.1
            Nruns = len(ncvec)
            f,axarr = plt.subplots(1,Nruns,figsize=(9,2.5),gridspec_kw={'width_ratios':list(ncvec)},sharey=True)
            for aa,nclusters in enumerate(ncvec):
                ax = axarr[aa]
                subd = lhdict[nclusters]
                xvec = np.array(list(subd.keys()))
                for xx in xvec:
                    vals = subd[xx]
                    col = color_dict[nclusters][xx]
                    jit_vec = np.random.uniform(-jit,jit,len(vals))
                    meanval = np.mean(vals)
                    medval = np.median(vals)
                    ax.plot(np.zeros_like(vals)+xx+jit_vec,vals,'o',mfc='none',mec=col,ms=5,mew=1)
                    for avgval,acol in zip([meanval,medval],['k','grey']):
                        ax.plot([xx-0.2,xx+0.2],[avgval,avgval],color=acol)
                ax.set_xticks(xvec)
                tlabs = ax.set_xticklabels(xvec,fontweight='bold')
                for xx,tlab in zip(xvec,tlabs):
                    tlab.set_color(color_dict[nclusters][xx])
                ax.set_xlabel('module')
                ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
            axarr[0].set_ylabel('h-score')
            f.suptitle('%s %s %s %s'%(myrun,reftag,thistag,basetag))
            f.tight_layout()
            figsaver(f,make_savename('hscorePerModule'))



            # save lhdict as hdf for statsuse
            outfile = os.path.join(os.path.join(genfigdir,make_savename('hscorePerModule'))+'.h5')
            with h5py.File(outfile,'w') as hand:
                mgrp = hand.create_group('modules_hierarchy')
                for key,subd in lhdict.items():
                    grp = mgrp.create_group('ncl%i'%key)
                    for modnum,modvals in subd.items():
                        grp.create_dataset('mod%i'%modnum,data=modvals,dtype='f')
                hand.create_dataset('rois_used',data=myrois,dtype='i')
                hand.create_dataset('h_scores',data=hiervec,dtype='f')

