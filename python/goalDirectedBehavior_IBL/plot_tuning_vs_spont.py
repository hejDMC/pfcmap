import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import matplotlib as mpl


#'''
pathpath = 'PATHS/filepath_IBL.yml'#
myrun = 'runIBLPasdMP3_brain_pj'
ncluststr = '8'
cmethod = 'ward'
#'''
ncl = int(ncluststr)

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S
from pfcmap_paper.utils import graphs as gfns

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
statsdict_dir = pathdict['statsdict_dir']#
srcfile = os.path.join(statsdict_dir,'statsdictTuningVsSpont__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

statsdict_spont = uloader.load_dict_from_hdf5(srcfile)

tuning_attr_names = ['stimTuned','choiceTuned','feedbTuned','taskResp']



figdir_base = '/FIGURES/tuning_vs_spontaneous'

figdir_mother = os.path.join(figdir_base,'tuningVsSpont__%s'%(myrun))

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__tuningVsSpont_%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

N_tuningattrs = len(tuning_attr_names)




#create a "typical" statsdict, but fused from the different significant tunings

matkeys = ['levels','matches','meanshuff','pofs','stdshuff']
sdict = {key:np.zeros((N_tuningattrs,ncl))*np.nan for key in matkeys}
for aa,attr in enumerate(tuning_attr_names):
    matchdict = statsdict_spont[attr]
    for key in matkeys:
        sdict[key][aa] = matchdict[key][matchdict['avals1']=='signif'][0]
sdict['avals1'] = tuning_attr_names
sdict['avals2'] = matchdict['avals2']
sdict['a1'] = 'cTuning'
sdict['SI'] = 1

#dict_keys(['SI', 'a1', 'a2', 'avals1', 'avals2', 'cattr', 'levels', 'matches', 'meanshuff', 'pofs', 'stdshuff'])
plotsel_stats = [ 'z from shuff','perc of score', 'p-levels','orig data counts']

f,axarr = gfns.plot_all_stats(sdict, plotsel=plotsel_stats,sortinds=np.arange(N_tuningattrs), zlim=8,zmat_levelbounded=True)
f.subplots_adjust(left=0.12)
f.set_figheight(2.)
for ax in axarr[::2]:
    ax.set_yticks(np.arange(N_tuningattrs))
    ax.set_xlabel('spont activity')
    ax.set_aspect('equal')
axarr[0].set_yticklabels(tuning_attr_names)
axarr[0].set_ylabel('')
ctrl_inds = np.array([1,3,7])
for axref, axcol in zip(axarr[ctrl_inds-1], axarr[ctrl_inds]):
    #for axref, axcol in zip(axarr[::2][:2], axarr[1::2][:2]):
    pos = axref.get_position()
    cpos = axcol.get_position()
    axcol.set_position([pos.x1 + 0.2 * (cpos.x0 - pos.x1), pos.y0, cpos.width, pos.height])  # [l
figsaver(f, 'enrichmentmats')


#todo make a bar graph of the fraction of significantly tuned units overall, and write the number on top
n_vec = np.zeros(N_tuningattrs)
frac_vec = np.zeros(N_tuningattrs)
nunit_mat = np.zeros((N_tuningattrs,ncl))

for aa,attr in enumerate(tuning_attr_names):
    matchdict = statsdict_spont[attr]
    N_split = matchdict['matches'].sum(axis=1)
    N_tuned = N_split[matchdict['avals1']=='signif'][0]
    frac_vec[aa] = N_tuned/N_split.sum()
    n_vec[aa] = N_tuned
    nunit_mat[aa] =  matchdict['matches'].sum(axis=0)#should be roughly the same for all, just take the first
    print(N_split.sum())#last one has fewer because there is nan!

tattrs_short = [attr.replace('Tuned','').replace('Resp','') for attr in tuning_attr_names]
N_tot = nunit_mat[0].sum()
f,ax = plt.subplots(figsize=(3.,3))
ax.bar(np.arange(N_tuningattrs),frac_vec,color='k')
for aa,myval in enumerate(n_vec):
    ax.text(aa,0.4,'%i\n%1.3f'%(myval,frac_vec[aa]),ha='center',va='top',color='r',fontweight='bold')
ax.set_xticks(np.arange(N_tuningattrs))
ax.set_xticklabels(tattrs_short,rotation=-90)
ax.set_ylabel('frac.')
ax.set_title('tunings - Nunits %i'%(N_tot))
f.tight_layout()
figsaver(f, 'fracsAndCounts_tunedUnits')


cmap = mpl.cm.get_cmap(S.cmap_clust)#
norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cdict_clust = {lab:cmap(norm(lab)) for lab in np.arange(ncl)}
cvec = np.array([cdict_clust[clust] for clust in np.arange(ncl)])

f,ax = plt.subplots(figsize=(3.,3))
blist=ax.bar(np.arange(ncl),nunit_mat[0]/N_tot,color='k')
for cc in np.arange(ncl):
    ax.text(cc,0.05,'%i - %1.3f'%(nunit_mat[0,cc],nunit_mat[0,cc]/N_tot),ha='center',\
            va='bottom',color='k',rotation=-90)
ax.set_xticks(np.arange(ncl))
ax.set_xticklabels(np.arange(ncl)+1)
ax.set_ylabel('frac.')
ax.set_title('spont. unit - Nunits %i'%(nunit_mat[0].sum()))
tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
for cc,col in enumerate(cvec):
    blist[cc].set_color(col)
    tlabs[cc].set_color(col)
f.tight_layout()
figsaver(f, 'fracsAndCounts_spontUnits')



