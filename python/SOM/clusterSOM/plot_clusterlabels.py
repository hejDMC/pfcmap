import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

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





diverge_dict = {'B':0,'LvR':1,'Lv':1,'Rho':0,'M':0}
diverge_dict2 = {key+'_mean':val for key,val in diverge_dict.items()}
diverge_dict.update(diverge_dict2)


def get_cmap_and_vminmax(feattag,values):
    if feattag in diverge_dict.keys():
        cmap = 'RdBu_r'
        dcenter = diverge_dict[feattag]
        vminmax = np.max(np.abs([values.min()-dcenter,values.max()-dcenter]))*np.array([-1,1])+dcenter
    else:
        cmap = 'binary'
        vminmax = [values.min(),values.max()]
    return cmap,vminmax

def get_cmap_and_vminmax2(feattag,values):
    if feattag in diverge_dict.keys():
        cmap = 'RdBu_r'
        dcenter = diverge_dict[feattag]
        vminmax = np.max(np.abs([values.min()-dcenter,values.max()-dcenter]))*np.array([-1,1])+dcenter
    else:
        cmap = 'magma'
        vminmax = [values.min(),values.max()]
    return cmap,vminmax


repl_dict = {'rate':r'$\nu$','B':r'$B$','M':r'$M$'}
featlabs = [repl_dict[somfeat.split('_')[0]] for somfeat in somfeats]

meanfn = lambda vals: np.median(vals)
radius_transform = lambda val: val*0.5#np.sqrt(val/np.pi)
val_offset = 0.2
scale_fac = 0.85
nfeats = len(somfeats)
mode = 'mixed'
color_mode = 'default'
gridon = True
max_rad = radius_transform((1 + val_offset) * scale_fac) + 0.05
x_bound = [-max_rad, ncl - 1 + max_rad]
y_bound = [-max_rad, nfeats + max_rad]

for cmapvminmaxgetter,cmaptag in zip([get_cmap_and_vminmax2,get_cmap_and_vminmax],['magma','binary']):
    cmapfn_dict = {}
    for ff,somfeat in enumerate(somfeats):
        featurevalues = (weights[ff]/weightvec[ff]*refstd[ff])+refmean[ff]
        mycmap,vminmax = cmapvminmaxgetter(somfeat,featurevalues)
        norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
        cmap_feat = mpl.cm.get_cmap(mycmap)

        cmapfn_dict[somfeat] = [ mpl.cm.get_cmap(mycmap),mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])]
        #cmapfn_dict[somfeat] = lambda value: mpl.cm.get_cmap(mycmap)(mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])(value))




    circleval_dict = {}
    circleval_dict_rel = {}
    circleval_dict_mixed = {}

    for ff,somfeat in enumerate(somfeats):
        ctransfn = lambda value: cmapfn_dict[somfeat][0](cmapfn_dict[somfeat][1](value))
        meanvals = np.array([meanfn([getattr(U,somfeat) for U in Units if U.clust==cc]) for cc in np.arange(ncl)])
        colorvals = np.array([ctransfn(val) for val in meanvals])

        if somfeat in diverge_dict:
            myvals = np.abs(meanvals)
        else:
            myvals = meanvals.astype(float)
        valrange = np.array([myvals.min(),myvals.max()])
        normed_vals = ((myvals-valrange[0])/(valrange[1]-valrange[0])+val_offset)*scale_fac
        circleval_dict[somfeat] = [normed_vals,colorvals]

        normed_vals_rel = ((meanvals-meanvals.min())/(meanvals.max()- meanvals.min())+val_offset)*scale_fac
        ref_circ = ((0-meanvals.min())/(meanvals.max()- meanvals.min())+val_offset)*scale_fac
        circleval_dict_rel[somfeat] = [normed_vals_rel,np.array([cdict_clust[cc] for cc in np.arange(ncl)]),ref_circ]
        circleval_dict_mixed[somfeat] = [normed_vals_rel,colorvals]



    f,ax = plt.subplots(figsize=(0.3+ncl*0.2,1.4))
    f.subplots_adjust(bottom=0.01,top=0.75)
    for ff,somfeat in enumerate(somfeats):
        normed_vals,colorvals = circleval_dict[somfeat]
        if mode == 'abs':
            normed_vals,colorvals = circleval_dict[somfeat]
            ec = 'k'
        elif mode == 'rel':
            normed_vals,colorvals,ref_circ = circleval_dict_rel[somfeat]
            ec = 'w'
        elif mode == 'mixed':
            ec = 'k'
            normed_vals,colorvals = circleval_dict_mixed[somfeat]

        #if color_mode == 'colored':
        #    circles = [plt.Circle((cc, nfeats-ff), radius_transform(normed_vals[cc]), facecolor=colorvals[cc],edgecolor=cdict_clust[cc],linewidth=2) for cc in np.arange(ncl)]
        circles = [plt.Circle((cc, nfeats-ff), radius_transform(normed_vals[cc]), facecolor=colorvals[cc],edgecolor=ec,linewidth=1) for cc in np.arange(ncl)]

        for circle in circles:
            ax.add_patch(circle)
        if somfeat in diverge_dict and mode == 'rel':
            circles_ref = [plt.Circle((cc, nfeats-ff), radius_transform(ref_circ), facecolor='none',edgecolor='grey',linewidth=1) for cc in np.arange(ncl)]
            for circle in circles_ref:
                ax.add_patch(circle)

    for cc in np.arange(ncl):
        if color_mode == 'colored':ax.plot([cc,cc],[1,y_bound[1]],color=cdict_clust[cc],zorder=-10,lw=2)
        else:ax.plot([cc,cc],[1,y_bound[1]],color='k',zorder=-10)

    if gridon:
        for ff in np.arange(nfeats):
            ax.plot([x_bound[0],ncl-1],[ff+1,ff+1],color='silver',linestyle='-',zorder=-12)
    ax.spines[['right', 'top','bottom','left']].set_visible(False)
    ax.set_xlim(x_bound)
    ax.xaxis.set_ticks_position('top')
    ax.set_ylim(y_bound)
    ax.set_yticks(np.arange(nfeats)+1)
    ax.set_yticklabels(featlabs[::-1])
    ax.set_xticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
        #ax.xaxis.get_ticklabels()
    ax.tick_params(axis='both', which='both',length=0)
    ax.tick_params(axis='x', which='major', pad=-1)
    ax.set_aspect("equal")
    f.tight_layout()
    figsaver(f,'%s/clusterlabels_%s_CMAP%s'%(cmethod,mode,cmaptag))
