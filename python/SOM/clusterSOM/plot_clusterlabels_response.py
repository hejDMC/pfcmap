import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

#myrun = 'runC00dMI3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'
#myrun = 'runIBLPasdM3_brain_pj'
'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZP_brain'#'runC00dMP3_brain'
ncluststr = '8'
cmethod = 'ward'
'''
#kickout_lay3pfc = True


ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S

psth_dir = os.path.join(S.metricsextr_path,'psth')

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)
cmap = mpl.cm.get_cmap(S.cmap_clust)#

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
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)

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

n_binsU = 30#when plotting the feature distributions

ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cdict_clust = {lab:cmap(norm(lab)) for lab in np.unique(labels)}


#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])


### START PSTH PREP
tbound = [-0.25,0.65]#get the psth-cut for each unit

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

####END PSTH PREP


tvec = psth_tvec[(psth_tvec <= tbound[1]) & (psth_tvec >= [tbound[0]])]
cat_dict = {cat: np.vstack([U.psth for U in Units if U.clust == cat]) for cat in np.arange(ncl)}
nvec = np.array([len(cat_dict[cat]) for cat in np.arange(ncl)])
allpsth = np.vstack([cat_dict[cat] for cat in np.arange(ncl)])
tracemat = np.vstack([np.median(cat_dict[cc], axis=0) for cc in np.arange(ncl)])

max_div = 0.8
scalefac = 0.8/tracemat.max()

f, ax = plt.subplots(figsize=(0.3 + ncl * 0.2, 1.2))
f.subplots_adjust(bottom=0.01, top=0.75)
for cc in np.arange(ncl):
    col = cdict_clust[cc]
    ax.plot(tracemat[cc]*scalefac+cc,tvec,color=col,lw=1.5)
    ax.axvline(cc,color='silver',zorder=-10)
ax.set_ylim([-0.1,0.5])
ax.set_yticks([])
for val in np.arange(0,0.45,0.2):
    ax.axhline(val,color='silver',linestyle=':')
ax.set_xlim([0-scalefac,ncl-1+scalefac])
ax.invert_yaxis()
ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

ax.xaxis.set_ticks_position('top')
ax.set_xticks(np.arange(ncl))
tlabs = ax.set_xticklabels(np.arange(ncl) + 1, fontweight='bold')
for cc in np.arange(ncl):
    tlabs[cc].set_color(cdict_clust[cc])
    # ax.xaxis.get_ticklabels()
ax.tick_params(axis='both', which='both', length=0)
ax.tick_params(axis='x', which='major', pad=-1)
figsaver(f,'%s/clusterlabels_psth'%(cmethod))
