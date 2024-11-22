import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


pathpath,tempstr = sys.argv[1:]
my_runsets = tempstr.split('___')
print('Comparing  %s'%str(my_runsets))

'''
my_runsets = [key for key in S.runsets.keys() if not key.count('resp')] 

pathpath = 'PATHS/filepaths_carlen.yml'
'''
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
figdir_base =  pathdict['figdir_root'] + '/overall_comparison'

runset_str = '_'.join(my_runsets)
figdir_mother = os.path.join(figdir_base,runset_str)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(runset_str,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




thistag = 'laydepth'
reftag = 'refall'
depth_checker = lambda aval: aval.count('|deep')

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr




statsfiles = []
for runset in my_runsets:
    runname = S.runsets[runset]
    ncl = S.nclust_dict[runset]
    statsfile = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%i_%s.h5'%(runname,ncl,S.cmethod))
    #print(runname,ncl,os.path.isfile(statsfile))
    statsfiles += [statsfile]



#find out which labels are available
avaldict_temp = {}
for statsfile,stag in zip(statsfiles,my_runsets):
    with h5py.File(statsfile,'r') as hand:
        statshand = hand['regs'][reftag][thistag]
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        #presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval.count('|sup') and not aval=='na' \
        #                        and not (S.check_pfc(aval) and not aval.count('|deep'))])
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and depth_checker(aval)])
        avaldict_temp[stag] = mystats['avals1'][presel_inds]

titlestr = 'deepCtx'
all_avail_areas = np.unique(np.hstack([avaldict_temp[runset] for runset in my_runsets if not runset.count('IBL')]))
all_stripped_dict = {S.strip_to_area(area):area for area in all_avail_areas}
sorted_areas = np.array([all_stripped_dict[area] for area in S.PFC_sorted if area in all_stripped_dict] +\
                        [area for area in all_avail_areas if not S.strip_to_area(area) in S.PFC_sorted])

dsdict = {}
for statsfile,stag in zip(statsfiles,my_runsets):
    with h5py.File(statsfile,'r') as hand:
        statshand = hand['regs'][reftag][thistag]
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)

        presel_inds = np.array([np.where(mystats['avals1']==aval)[0][0] for aa, aval in enumerate(sorted_areas)])
        assert (mystats['avals1'][presel_inds] == sorted_areas).all(),'mismatching avalues'
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['countbools'] = countvec[presel_inds]>S.Nmin_maps
        dsdict[stag] = sdict

statsfn = S.enr_fn


X_dict = {}
for stag in my_runsets:
    X_temp = statsfn(dsdict[stag])
    X_temp[dsdict[stag]['countbools']==False] = np.nan
    X_dict[stag] = X_temp

allX = np.hstack([X.flatten() for X in list(X_dict.values())])
x_bounds = np.array([np.nanmin(allX),np.nanmax(allX)])
bound_amp = np.diff(x_bounds)[0]
my_ylim = x_bounds+np.array([-0.1,0.1])*bound_amp

#cdict_temp = {'spont':{'ww':'k','nw':'grey'},'IBL':{'ww':'orange','nw':'darkgoldenrod'}}
#cdict = {'%s%s'%(key2,key1):cdict_temp[key1][key2] for key1 in cdict_temp.keys() for key2 in cdict_temp[key1].keys()}#clumsy-lazy way of setting columns
cdict = {'wwspont': 'firebrick', 'nwspont': 'mediumblue',\
         'wwIBL': 'darkorange', 'nwIBL': 'skyblue',\
         'wwresp':'sienna','nwresp':'teal'}

runset_sel_dict = {utype_tag:[runset for runset in my_runsets if runset.count(utype_tag)] for utype_tag in ['nw','ww']}
xlab_simp = [replace_fn(lab) for lab in sorted_areas]
N_regs = len(all_avail_areas)
xvec = np.arange(N_regs)
def make_nice_axes(ax):
    ax.set_xticks(xvec)
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    ax.set_xticklabels(xlab_simp, rotation=-90)
    ax.set_ylabel('enrichment')
    myxlim = np.array(ax.get_xlim())
    ax.fill_between(myxlim,np.zeros_like(myxlim)-2,np.zeros_like(myxlim)+2,color='grey',alpha=0.5)
    ax.set_ylim(my_ylim)
    ax.set_xlim(myxlim)

    ax.get_figure().tight_layout()



#contrasting runs
for utype in ['nw','ww']:
    ncl = S.nclust_dict[runset_sel_dict[utype][0]]
    for cc in np.arange(ncl):
        #vals1,vals2 = [X_dict[runset][:,cc] for runset in runset_sel_dict[utype]]
        for line_mode in ['conn','default']:
            f,ax = plt.subplots(figsize=(4,3))
            f.subplots_adjust(bottom=0.18)
            for runset in runset_sel_dict[utype]:
                col = cdict[runset]
                if line_mode == 'conn':
                    ax.plot(xvec,X_dict[runset][:,cc],'.-',color=col,ms=12,mec='none',mfc=col,lw=0)
                elif line_mode == 'default':
                    ax.plot(xvec,X_dict[runset][:,cc],'.-',color=col,ms=12,mec='none',mfc=col,lw=1)
            if line_mode == 'conn' and len(my_runsets)>2:
                for xx in np.arange(N_regs):
                    v1,v2 = [X_dict[runset][xx,cc] for runset in runset_sel_dict[utype]]
                    if np.mean([v1,v2])!=np.nan:
                        ax.plot([xx,xx],[v1,v2],color='silver',zorder=-5)#cdict[runset_sel_dict[utype][0]]
            for rr,runset in enumerate(runset_sel_dict[utype]):
                ax.text(0.99,0.99-rr*0.1,runset,color=cdict[runset],ha='right',va='top',transform=ax.transAxes)

            ax.set_title('clust %i'%(cc+1))
            make_nice_axes(ax)
            figsaver(f,'contrasting_datasets/linemode_%s/%s_clust%i_%s'%(line_mode,utype,cc+1,line_mode))


# now one just per runsel



cmap = mpl.cm.get_cmap(S.cmap_clust)
for runset in my_runsets:
    ncl = S.nclust_dict[runset]
    norm = mpl.colors.Normalize(vmin=0, vmax=ncl - 1)
    cdict_clust = {lab: cmap(norm(lab)) for lab in np.arange(ncl)}
    f, ax = plt.subplots(figsize=(4, 3))
    f.subplots_adjust(bottom=0.18)
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        ax.plot(xvec,X_dict[runset][:,cc],'.-',color=col,ms=12,mec='none',mfc=col,lw=0)
    for xx in np.arange(N_regs):
        vals = X_dict[runset][xx,:]
        if not len(np.unique(vals))==1:
            minval = np.nanmin(vals)
            maxval = np.nanmax(vals)
            ax.plot([xx,xx],[minval,maxval],color='k',zorder=-5)
    ax.set_title('%s'%runset)
    make_nice_axes(ax)
    figsaver(f,'all_clust_per_dataset/%s_allclustEnr'%(runset))





