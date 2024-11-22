import sys
import yaml
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]
#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runCrespZP_brain'
#ncluststr = '8'
#cmethod = 'ward'

ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from pfcmap.python.utils import clustering as cfns

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)


cmap = mpl.cm.get_cmap(S.cmap_clust)#


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)


kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/Nc%s'%(myrun,kshape[0],kshape[1],ncluststr))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


#reftag,layerlist,datasets,tasks,tsel,statetag,utypes,nnodes = [rundict[key] for key in ['reftag','layerlist','datasets','tasks','tsel','state','utypes','nnodes']]
#wmetrics,imetrics,wmetr_weights,imetr_weights = [rundict[metrtype][mytag] for mytag in ['features','weights'] for metrtype in ['wmetrics','imetrics']]#not strictly necessary: gets called in uloader


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')))

assert (somfeats == somdict['features']).all(),'mismatching features'
assert (weightvec == somdict['featureweights']).all(),'mismatching weights'




get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])


#project on map
refmean,refstd = somdict['refmeanstd']
featureweights = somdict['featureweights']
dmat = np.vstack([get_features(elobj) for elobj in Units])
wmat = uloader.reference_data(dmat,refmean,refstd,featureweights)

#set BMU for each
weights = somdict['weights']
allbmus = somh.get_bmus(wmat,weights)
for bmu,U in zip(allbmus,Units):
    U.set_feature('bmu',bmu)



X = weights.T

cfndict = cfns.get_cfn_dict()

#cmethod = 'ward'
clusterfn = cfndict[cmethod]
#labels0 = clusterfn(ncl,X)
#labels = cfns.sort_labels(labels0,X,somfeats,sortfeat,avgfn=lambda x:np.median(x))

ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cdict_clust = {lab:cmap(norm(lab)) for lab in np.unique(labels)}


#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])


f = plt.figure(figsize=(10, 10))
f.suptitle('%s_Nc%i'%(myrun,ncl))
ax = f.add_subplot(projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=labels,cmap=cmap)
for ff,labelfn in enumerate([ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]):
    labelfn(somfeats[ff])
figsaver(f,'%s/featspace_3d'%cmethod)


tbound = [-0.25,0.65]

#get the psth-cut for each unit
psth_dir = os.path.join(S.timescalepath,'psth')

recids = np.unique([U.recid for U in Units])

for recid in recids:
    #print(recid)
    repl_dict = {'RECID': recid, 'mystate': rundict['state'], 'REFTAG': rundict['reftag'],'__PCAscoresMODE':''}
    psthfile_tag = uloader.replace_by_dict(S.responsefile_pattern, repl_dict)
    psth_file = os.path.join(psth_dir,psthfile_tag)

    with h5py.File(psth_file,'r') as hand:
        huids = hand['uids'][()]
        psth_tvec = hand['psth_tvec'][()]
        psth = hand['psth'][()]

    pre_bool =psth_tvec<0

    psth_normed = (psth -np.mean(psth[:,pre_bool],axis=1)[:,None]) /np.std(psth[:,pre_bool],axis=1)[:,None]
    psth_cut = psth_normed[:,(psth_tvec<=tbound[1])&(psth_tvec>=[tbound[0]])]
    recUs = [U for U in Units if U.recid==recid]
    for U in recUs:
        uidx = int(np.where(huids==U.uid)[0])
        U.set_feature('psth',psth_cut[uidx])

from scipy.stats import scoreatpercentile as sap
tvec = psth_tvec[(psth_tvec<=tbound[1])&(psth_tvec>=[tbound[0]])]
cat_dict = {cat:np.vstack([U.psth for U in Units if U.clust==cat]) for cat in np.arange(ncl)}
nvec = np.array([len(cat_dict[cat])for cat in np.arange(ncl)])
allpsth = np.vstack([cat_dict[cat] for cat in np.arange(ncl)])
vmax = sap(allpsth,99.5)
vmin = sap(allpsth,0.5)




ncols = 4
nrows = int(np.ceil(ncl/ncols))
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

cmap = 'RdBu_r'
f = plt.figure(figsize=(13,1+nrows*4))
f.subplots_adjust(left=0.08)
gs0 = gridspec.GridSpec(nrows, ncols, figure=f,wspace=0.5,hspace=0.3)
counter = 0
for rownum in np.arange(nrows):
    for colnum in np.arange(ncols):
        if counter == ncl: break
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[rownum,colnum],\
                                                height_ratios=[1,0.35],hspace=0.01)
        ax = f.add_subplot(gs00[0])
        psthmat = cat_dict[counter]
        im = ax.imshow(psthmat,aspect='auto',interpolation='nearest',origin='lower',extent=[*tbound,1,len(psthmat)],norm=norm, cmap=cmap)

        ax.set_title('clust%i'%(counter+1))
        ax.set_xticklabels([])


        ax2 = f.add_subplot(gs00[1])
        sapscores = [sap(psthmat,myscore,axis=0) for myscore in [25,50,75]]
        ax2.fill_between(tvec,sapscores[0],sapscores[2],color='grey',alpha=0.5,lw=0)
        ax2.plot(tvec,sapscores[1],color='k')
        ax2.axvline(0.,color='silver',zorder=-10)
        if counter == 0:
            pos = ax.get_position()
            cax = f.add_axes( [0.92, pos.y0, 0.01, pos.height])
            cb = f.colorbar(im, cax=cax,extend='both')

        #if counter>=ncl-ncols:
        ax2.set_xlabel('time [s]')
        #else:
        #    ax2.set_xticklabels([])
        if colnum == 0:
            ax.set_ylabel('n units')
            ax2.set_ylabel('avg z')
        for myax in [ax,ax2]:
            myax.set_xlim(tbound)
        counter += 1
    else: continue
    break
cax.set_title('z')
f.suptitle('%s  %s'%(myrun,cmethod))
f.tight_layout()
figsaver(f,'%s/psth_collected'%cmethod)


#now the median
tracemat = np.vstack([np.median(cat_dict[cc],axis=0) for cc in np.arange(ncl)] )

f, ax = plt.subplots(figsize=(4.5, 3))
f.subplots_adjust(bottom=0.18, left=0.2, right=0.83)
for cc in np.arange(ncl):
    col = cdict_clust[cc]
    ax.plot(tvec,tracemat[cc],color=col,lw=2)
    ax.text(1.01,0.99-cc*0.1,'clust%i'%(cc+1),color=col,transform=ax.transAxes,ha='left',va='top',fontweight='bold')
ax.axvline(0.,color='silver',zorder=-10)
ax.set_xlim(tbound)
ax.set_ylabel('avg.z')
ax.set_xlabel('time [s]')
f.suptitle('%s  %s'%(myrun,cmethod))
f.tight_layout()
figsaver(f,'%s/psth_medians'%cmethod)