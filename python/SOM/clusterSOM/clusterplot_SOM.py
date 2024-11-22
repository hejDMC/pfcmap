import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import scoreatpercentile as sap

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]


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
from pfcmap.python.utils import som_plotting as somp
from pfcmap.python.utils import filtering as filt


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
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




nfeats = len(somfeats)
diverge_dict = {'B':0,'LvR':1,'Lv':1,'Rho':0,'M':0}
diverge_dict2 = {key+'_mean':val for key,val in diverge_dict.items()}
diverge_dict.update(diverge_dict2)


#condensed feature display
percs = np.array([10,50,90])
nperc = len(percs)
scoremat = np.zeros((nfeats,ncl,nperc))
for ff,somfeat in enumerate(somfeats):
    for cc in np.arange(ncl):
        uvals = np.array([getattr(U,somfeat) for U in Units if U.clust==cc])
        scoremat[ff,cc] = np.array([sap(uvals,perc) for perc in percs])

left_indent,right_lim = [0.41,0.97] if ncl<=7 else [0.35,0.99]
for ff,somfeat in enumerate(somfeats):
    f,ax = plt.subplots(figsize=(0.37+0.14*ncl,1.5))
    f.subplots_adjust(left=left_indent,right=right_lim,bottom=0.25)#left=0.35,
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        ax.plot([cc,cc],[scoremat[ff,cc,0],scoremat[ff,cc,-1]],color=col)
        ax.plot(cc,scoremat[ff,cc,1],'o',mfc=col,mec='none',ms=5)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='normal')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    if somfeat in diverge_dict:
        abslim = np.max(np.abs([scoremat[ff].min(),scoremat[ff].max()]))+0.02
        ax.set_ylim([-abslim,abslim])
        ax.axhline(0,color='silver',zorder=-10)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    elif somfeat.count('rate'):
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

    ax.set_title(somfeats[ff].split('_')[0],pad=-2)
    #f.tight_layout()
    figsaver(f,'%s/features_condensed/%s'%(cmethod,somfeat))








if not myrun.count('IBL'):

    if len(somfeats) == 3:
        f = plt.figure(figsize=(10, 10))
        f.suptitle('%s_Nc%i'%(myrun,ncl))
        ax = f.add_subplot(projection='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2],c=labels,cmap=cmap)
        for ff,labelfn in enumerate([ax.set_xlabel,ax.set_ylabel,ax.set_zlabel]):
            labelfn(somfeats[ff])
        figsaver(f,'%s/featspace_3d'%cmethod)


    sg_on = True
    for ff,feat in enumerate(somfeats):
        #ff = 0
        #feat = somfeats[0]
        featvals = (weights[ff]/weightvec[ff]*refstd[ff])+refmean[ff]

        f,axarr = plt.subplots(2,1,figsize=(4,3),gridspec_kw={'height_ratios':[0.3,1.]})
        f.subplots_adjust(left=0.2,bottom=0.2,hspace=0.02)

        ax2,ax = axarr
        for lab in np.unique(labels):
            protovals = featvals[labels==lab]
            unitvals = np.array([getattr(U,feat) for U in Units if U.clust==lab])
            hist,bins = np.histogram(unitvals,n_binsU)
            bw = np.diff(bins)[0]
            plotvals = filt.savitzky_golay(hist,7,3) if sg_on else hist[:]
            ax.plot(bins[:-1]+bw/2,plotvals/len(unitvals),color=cdict_clust[lab])
            ax2.plot(protovals,lab*np.ones_like(protovals),'o',mfc=cdict_clust[lab],mec='none',alpha=0.5)
        ax.set_ylabel('prob.')
        featlab = S.featfndict[feat]['repl'] if feat in S.featfndict else feat
        ax.set_xlabel(featlab)
        ax2.set_axis_off()
        ax2.set_ylim([labels.min()-0.5,labels.max()+0.5])
        ax2.set_xlim(ax.get_xlim())
        ax.set_ylim([0,ax.get_ylim()[1]])
        figsaver(f,'%s/features/%s'%(cmethod,feat))


    #plot on SOM
    #for plotting on SOM
    sizefac = 0.25
    hw_hex = 0.35
    fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                        sizefac=sizefac)

    dcolors = np.array([cdict_clust[ii][:3] for ii in np.unique(labels)])#np.array([cdict_clust[lab] for lab in labels])

    f, ax = plt.subplots(figsize=(fwidth, fheight))
    f.subplots_adjust(left=l_w / fwidth, right=1. - r_w / fwidth-0.1, bottom=b_h / fheight,
                      top=1. - t_h / fheight-0.05, wspace=0.05)  # -tspace/float(fheight)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_axis_off()
    somp.plot_hexPanel(kshape,labels,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                     ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',quality_map=dcolors)#rankvec
    figsaver(f,'%s/SOM'%(cmethod))


#counts: units per cluster
clusts = np.unique(labels)
cvec = np.array([cdict_clust[clust] for clust in clusts])

countvec = np.array([np.sum([1 for U in Units if U.clust==clust]) for clust in clusts])
xvec = np.arange(len(clusts))
f,ax = plt.subplots(figsize=(1.5+0.2*len(clusts),2))
f.subplots_adjust(left=0.3,bottom=0.3)
blist = ax.bar(xvec,countvec,color='k')
ax.set_xticks(xvec)
tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')

for cc,col in enumerate(cvec):
    blist[cc].set_color(col)
    tlabs[cc].set_color(col)
    #ax.xaxis.get_ticklabels()
ax.set_ylabel('counts')
ax.set_xlabel('category')
ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
for pos in ['top','right']:ax.spines[pos].set_visible(False)
figsaver(f, '%s/clust_counts'%cmethod)
with open(os.path.join(figdir_mother,cmethod,'clustcounts.txt'), 'w') as f:
    f.write('COUNTS category\n\n')
    for attrval,count in zip(xvec+1,countvec):
        f.write('cat.%s: %i\n'%(attrval,count))


