import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import kendalltau,pearsonr




pathpath,tempstr = sys.argv[1:]
my_runsets = tempstr.split('___')
print('Comparing  %s'%str(my_runsets))

'''

pathpath = 'PATHS/filepaths_carlen.yml'
'''
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
'''
my_runsets = [key for key in S.runsets.keys() if not key.count('resp')] 

'''
stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
figdir_base =  pathdict['figdir_root'] + '/overall_comparison_statsenhanced'


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


Xstats_dict = {'N':{},'plevels':{},'NperReg':{}}
for stag in my_runsets:
    Xp_temp = dsdict[stag]['levels']
    Xp_temp[dsdict[stag]['countbools']==False] = np.nan
    Xstats_dict['plevels'][stag] = Xp_temp
    Xn_temp = dsdict[stag]['matches']
    Xn_temp[dsdict[stag]['countbools']==False] = np.nan
    Xstats_dict['N'][stag] = Xn_temp
    Xstats_dict['NperReg'][stag] = dsdict[stag]['matches'].sum(axis=1)



allX = np.hstack([X.flatten() for X in list(X_dict.values())])
x_bounds = np.array([np.nanmin(allX),np.nanmax(allX)])
bound_amp = np.diff(x_bounds)[0]
my_ylim = x_bounds+np.array([-0.1,0.1])*bound_amp

runset_sel_dict = {utype_tag:[runset for runset in my_runsets if runset.count(utype_tag)] for utype_tag in ['nw','ww']}
xlab_simp = [replace_fn(lab) for lab in sorted_areas]
N_regs = len(all_avail_areas)
xvec = np.arange(N_regs)


cdict2 = {'wwspont': 'k', 'nwspont': 'k',\
         'wwIBL': 'grey', 'nwIBL': 'grey',\
         'wwresp':'sienna','nwresp':'teal'}


def make_nice_axes2(ax):
    ax.set_xticks(np.arange(np.sum(cond)))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
    ax.set_xticklabels(np.array(xlab_simp)[cond], rotation=-90)
    ax.set_ylabel('enrichment')
    myxlim = np.array(ax.get_xlim())
    #ax.fill_between(myxlim,np.zeros_like(myxlim)-2,np.zeros_like(myxlim)+2,color='grey',alpha=0.5)
    ax.set_ylim(my_ylim)
    ax.set_xlim(myxlim)

    ax.get_figure().tight_layout()

line_mode = 'conn'

#contrasting runs
# plot N on top
# plot significance

def paper_axes(ax,xlab='',ylab=''):
	for pos in ['top','right']:ax.spines[pos].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.tick_params(tickdir='out',width=1.,pad=0.3,length=3)
	ax.set_ylabel(ylab)
	ax.set_xlabel(xlab)

def loggify_y(ax):
	ax2 = ax.twinx()
	ax2.set_ylim(10**np.array(ax.get_ylim()))
	ax2.set_yscale('log')
	for pos in ['top','right','left','bottom']:ax.spines[pos].set_visible(False)
	ax2.yaxis.set_label_position('left')
	ax2.yaxis.set_ticks_position('left')
	paper_axes(ax2,xlab=ax.get_xlabel(),ylab=ax.get_ylabel())
	ax.set_ylabel('')
	ax.set_yticks([])
	return ax2


for utype in ['nw','ww']:
    ncl = S.nclust_dict[runset_sel_dict[utype][0]]
    for cc in np.arange(ncl):
        all_countbools = np.vstack([dsdict[stag]['countbools'] for stag in runset_sel_dict[utype]])
        cond = all_countbools.sum(axis=0) == len(all_countbools)

        f,axarr = plt.subplots(2,1,figsize=(4,4),gridspec_kw={'height_ratios':[0.3,1]})
        nax,ax = axarr
        f.subplots_adjust(bottom=0.18)
        tempmat = np.empty((0,np.sum(cond)))
        for runset,offset in zip(runset_sel_dict[utype],[-0.2,0.2]):
            col = cdict2[runset]
            Nvec = Xstats_dict['NperReg'][runset][cond]
            pvec = Xstats_dict['plevels'][runset][cond,cc]
            Navail = len(pvec)
            myx = np.arange(Navail)
            xvec_off = myx + offset
            enrvals = X_dict[runset][cond,cc]
            blist = nax.bar(xvec_off, np.log10(Nvec), color=col, width=0.4, alpha=1)
            signif = pvec!=0
            ax.plot(myx[signif],enrvals[signif],'.',color=col,ms=12,mec='none',mfc=col,lw=0)
            ax.plot(myx[signif==False],enrvals[signif==False],'x',color=col,ms=6,mec=col,mfc=col,lw=3)

            tempmat = np.r_[tempmat,enrvals[None,:]]
        for xx,[v1,v2] in enumerate(tempmat.T):
            ax.plot([xx,xx],[v1,v2],color='silver',zorder=-5)#cdict[runset_sel_dict[utype][0]]
        for rr,runset in enumerate(runset_sel_dict[utype]):
            ax.text(0.99,0.99-rr*0.1,runset,color=cdict2[runset],ha='right',va='top',transform=ax.transAxes)
        res = [myfn(tempmat[0],tempmat[1]) for myfn in [kendalltau,pearsonr]]
        statsstr = ', '.join(['%s:%1.2f (p:%1.2e)'%(rlab,myres.statistic,myres.pvalue) for rlab,myres in zip(['tau','R'],res)])
        nax.set_title('clust %i - %s'%(cc+1,statsstr),fontsize=8)
        nax.set_xticklabels([])
        make_nice_axes2(ax)
        nax.set_xlim(ax.get_xlim())
        nax.set_xticks(ax.get_xticks())

        ax.axhline(0,color='silver',alpha=1,linestyle=':')

        nax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        nax.set_ylabel('count')
        nax.set_ylim([1,nax.get_ylim()[1]])
        loggify_y(nax)
        for myax in [ax,nax]: myax.spines[['right', 'top']].set_visible(False)
        f.tight_layout()
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 25/55)#setting the aspect ratio as wanted
        figsaver(f,'contrasting_datasets/linemode_%s/%s_clust%i_%s'%(line_mode,utype,cc+1,line_mode))
