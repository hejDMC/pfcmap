import sys
import yaml
import os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

pathpath,myrun = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
'''

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from pfcmap.python.utils import som_plotting as somp


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)


somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)


kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/examples/imetrics'%(myrun,kshape[0],kshape[1]))

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

#select units (randomly or evenly)
N_rand = 3
N_pop = 4
N_per_sel = 1

first_col = np.linspace(0,kshape[1]-1,3).astype(int)*kshape[0]
last_col = first_col+kshape[0]-1
mid_col = first_col+kshape[0]//2
in_bet1 = np.arange(kshape[1]/4,kshape[1],kshape[1]/2).astype(int)*kshape[0] + kshape[0]//4
in_bet2 = in_bet1 + kshape[0]//2

nvec = np.array([len(allbmus[allbmus==somnode]) for somnode in np.arange(np.prod(kshape))])
sortinds = nvec.argsort()
populars = sortinds[-N_pop:]

selected_nodes = np.sort(np.r_[first_col,last_col,mid_col,in_bet1,in_bet2,populars])
#rands = np.random.choice( np.delete(np.arange(np.prod(kshape)),selected_nodes),N_rand,replace=False)
#selected_nodes = np.r_[selected_nodes,rands]

showtints_min = 100 if not rundict['datasets'] == ['IBL_Passive'] else 90
recids_allowed = []
recids_temp = np.unique([[U.recid,U.dataset] for U in Units],axis=0)
for recid,mydataset in recids_temp:
    tintfile = uloader.get_tintfile_rec(recid, mydataset, rundict, timescalepath=S.timescalepath)
    with h5py.File(tintfile, 'r') as hand: ntints = hand['tints'][()].shape[0]
    if ntints>=showtints_min:
        recids_allowed.append(recid)




#examples of spiking
src_pool = np.hstack([glob(pathdict['src_dirs']['nwb_dirpatterns'][dset]) for dset in rundict['datasets']])

get_ridx = lambda unit_inds, my_unit: [unit_inds[my_unit], [unit_inds[my_unit - 1] if my_unit > 0 else 0][0]]
borders = np.array([-0.25,0.25])
getspikes = lambda stimes,tinterval: stimes[(stimes>=tinterval[0]) & (stimes<=tinterval[1])]
reftag = '__all' if rundict['reftag'] == 'all' else ''

def get_tintfile(U):
    tintfile = uloader.get_tintfile_rec(recid,U.dataset,rundict,timescalepath=S.timescalepath)
    return tintfile



Usel_dict = {}
for tag,allowedrecids in zip(['all','tintsmin%i'%showtints_min],[recids_temp[:,0],recids_allowed]):
    Usel = np.array([])
    for node in selected_nodes:
        Upool = [U for U in Units if U.bmu==node and U.recid in allowedrecids]
        myUs = np.random.choice(Upool,size=N_per_sel,replace=False)
        Usel = np.r_[Usel,myUs]
    Usel = np.r_[Usel,np.random.choice([U for U in Units if U.recid if U.recid in allowedrecids and not U.bmu in populars],size=N_rand,replace=False)]
    selbmus = np.array([U.bmu for U in Usel])
    sortinds = selbmus.argsort()
    Usel_dict[tag] = Usel[sortinds]


#plot indices on the map
sizefac = 0.5
hw_hex = 0.35
hex_dict = somp.get_hexgrid(kshape, hw_hex=hw_hex * sizefac)
cmap = 'plasma'
alphahex = 0.5
textcol = 'k'
fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                 sizefac=sizefac)

allcounts = np.zeros(np.prod(kshape))
for bmu in allbmus:
    allcounts[bmu] += 1


for tag in ['all','tintsmin%i'%showtints_min]:
    Usel = Usel_dict[tag]

    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])
    f.suptitle(myrun)
    f.subplots_adjust(right=0.85)
    somp.plot_hexPanel(kshape, allcounts, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, alphahex=alphahex, hexcol=cmap)  # rankvec

    norm = mpl.colors.Normalize(vmin=allcounts.min(), vmax=allcounts.max())
    cb = somp.plot_cmap(cax, cmap, norm,alpha=alphahex)
    cax.set_title('# hits')
    u_counter = 0
    for somnode,pos in hex_dict.items():
        csum = np.cumsum([1 for U in Usel if U.bmu==somnode])
        if np.size(csum)>0:
            mystr =','.join(list((csum+u_counter).astype(str)))
            center = pos[0]
            ax.text(center[0], center[1], mystr, fontsize=10, ha='center',
                                va='center', color=textcol,fontweight='bold')
            u_counter+=len(csum)
    if tag == 'all':
        figsaver(f,'example_map')
    else:
        figsaver(f,'%s/example_map'%tag)



    for uu,U in enumerate(Usel):
        #uu = 10
        #U = Usel[uu]
        srcfile = [fname for fname in src_pool if fname.count(U.recid)][0]
        with h5py.File(srcfile,'r') as hand:
            utinds = hand['units/spike_times_index'][()]
            r1, r0 = get_ridx(utinds,U.uid)
            spiketimes = hand['units/spike_times'][r0:r1]


        tintfile = get_tintfile(U)

        with h5py.File(tintfile,'r') as hand: tints0 = hand['tints'][()]
        tints = tints0 + borders[None,:]
        spikelist = [getspikes(spiketimes, tint)-tint[0] for tint in tints]
        rastermat = np.vstack([np.hstack(spikelist),np.hstack([np.ones(len(subl))+xx for xx,subl in enumerate(spikelist)])])
        framedur = float(np.diff(tints0)[0])
        tadder = -framedur if rundict['tsel'] == 'prestim' else 0.
        featstr = ','.join(['%s:%1.1f'%(feat[:1],getattr(U, feat)) for feat in somfeats])
        f,ax = plt.subplots(figsize=(3,2.5))
        f.subplots_adjust(left=0.23,bottom=0.22,right=0.95)
        f.suptitle('[%i] bmu%i %s u:%i %s (%s)'%(uu+1,U.bmu,U.recid.split('_')[0][:8],U.uid,U.region,featstr),fontsize=7)
        ax.plot(rastermat[0]+borders[0]+tadder,rastermat[1],'k.',ms=2)
        ax.set_ylabel('trials')
        ax.set_xlabel('time [s]')
        if rundict['tsel'] == 'prestim':
            ax.set_xlim([borders[0]+tadder,borders[1]])
            ax.axvline(0,color='silver',zorder=-10)
            ax.axvline(tadder,color='silver',zorder=-10)
        elif rundict['tsel'] == 'poststim':
            ax.set_xlim([borders[0],framedur+borders[1]])
            ax.axvline(0,color='silver',zorder=-10)
            ax.axvline(framedur,color='silver',zorder=-10)
        ax.set_ylim([0.5,tints.shape[0]+0.5])
        if tag.count('tint'):
            ax.set_ylim([0.5,showtints_min+0.5])
        #figsaver(f,'uexample_%i'%(uu+1))
        if tag == 'all':
            figsaver(f, 'uexample_%i'%(uu+1))
        else:
            figsaver(f, '%s/uexample_%i'%(tag,uu+1))

