import sys
import yaml
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

pathpath,myrun = sys.argv[1:]
#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runC00dMP_M'


nclustvec = np.arange(2,21)


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import clustering as cfns
from pfcmap.python import settings as S
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python.utils import som_plotting as somp



stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/nclust'%(myrun,kshape[0],kshape[1]))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



#project on map
refmean,refstd = somdict['refmeanstd']
featureweights = somdict['featureweights']

#set BMU for each
weights = somdict['weights']



X = weights.T

for link_method in ['centroid','complete','ward']:
    f,axarr = cfns.plot_distmat(X,link_method=link_method,axlab='SOM nodes', cmap='viridis_r',showticks=True)
    ax = axarr[0]
    ax.set_xticks([])
    ax.set_yticks([])
    f.suptitle('%s linkage:%s' % (myrun,link_method))
    figsaver(f,'distmats/distmat_%s'%link_method)

#labelgetter = cfns.get_clusterfn_dict(roivals)
cfndict = cfns.get_cfn_dict()


#########################
#plot the hexagons with successive colors

sizefac = 0.7
hw_hex = 0.35

fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                    sizefac=sizefac)
cmap = mpl.cm.get_cmap(S.cmap_clust)#


for cmethod in cfndict.keys():

    ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)

    k_ratio = np.divide(*somdict['kshape'])
    NCvec = np.sort(list(ddict.keys()))
    f,axarr = plt.subplots(1,len(NCvec),figsize=(len(NCvec)*k_ratio*2.2,1.5/k_ratio))
    f.text(0.005,0.99,'%s\n%s'%(myrun,cmethod),ha='left',va='top')
    f.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.85)
    for nn,ncl in enumerate(NCvec):
        #nn =0
        #ncl = n_clusts_dict[nn]
        labels = ddict[ncl]
        ax = axarr[nn]
        ax.set_title(ncl)
        norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
        cdict_clust = {lab:cmap(norm(lab)) for lab in np.unique(labels)}
        dcolors = np.array([cdict_clust[ii][:3] for ii in np.unique(labels)])#np.array([cdict_clust[lab] for lab in labels])
        #print(ncl,np.unique(labels),dcolors.shape)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_axis_off()
        somp.plot_hexPanel(kshape, labels, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                           , scalefree=True, return_scale=False, alphahex=1, idcolor='k', quality_map=dcolors)  # rankvec

        figsaver(f,'SOMclust/SOMclust_%s'%cmethod)



for cmethod in ['ward','kmeans','gmm']:
    clustfn = cfndict[cmethod]



    evalfns = cfns.get_evalfn_dict()
    evaldict = cfns.eval_asfn_nclust(X,nclustvec,clustfn,evalfns)

    #randdict = cfns.get_randdict(roivals,nclustvec,clustfn,evalfns,nrep = nreps)
    #scoredict = cfns.rand_to_scoredict(randdict,percentiles = [20,50,80])



    for randbool,randtag in zip(['False'],['']):
        flist = cfns.plot_clustquality(evaldict,nclustvec,show_rand=randbool)#,scoredict=scoredict



        for f,evalkey in zip(flist,evaldict.keys()):
            figsaver(f,'%s/quality/%s%s'%(cmethod,evalkey,randtag))

for lmethod in ['centroid','ward','complete']:
    f = cfns.plot_thorndike(nclustvec, X, linkage_method=lmethod)
    figsaver(f,'thorndike/%s'%(lmethod))