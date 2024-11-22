import sys
import yaml
import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import scoreatpercentile as sap
from matplotlib.patches import Rectangle



pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

#nb. just edit the roimap path
'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZI_brain'
cmethod = 'ward'
ncluststr = '5'
'''


#for psth
tbound = [-0.25,0.65]#get the psth-cut for each unit
ncols_psth = 4
cmap_psth = 'RdBu_r'

zeta_pattern = 'RECID__%s__%s%s__TSELpsth2to7__STATEactive__all__zeta.h5'%(myrun,cmethod,ncluststr)
#'152417_20191023-probe0__runC00dMI3_brain__ward5__TSELpsth2to7__STATEactive__all__zeta.h5'
with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
zeta_dir =  os.path.join(pathdict['src_dirs']['zeta'],'%s__%s%s'%(myrun,cmethod,ncluststr))

savepath_gen = pathdict['savepath_gen']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import som_helpers as somh


psth_dir = os.path.join(S.timescalepath,'psth')

statsfn = S.enr_fn


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)
cmap = mpl.cm.get_cmap(S.cmap_clust)#

kshape = somdict['kshape']

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i/clustering/Nc%s/%s/psth'%(myrun,kshape[0],kshape[1],ncluststr,cmethod))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

def figsaverZETA(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother.replace('psth','zeta'), nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)
#figsaver


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)#rois from path = False  gets us the Gao rois

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

### START PSTH PREP
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

    pre_bool = psth_tvec < 0
    psth_normed = (psth -np.mean(psth[:,pre_bool],axis=1)[:,None]) /np.std(psth[:,pre_bool],axis=1)[:,None]
    psth_cut = psth_normed[:,(psth_tvec<=tbound[1])&(psth_tvec>=[tbound[0]])]
    recUs = [U for U in Units if U.recid==recid]
    for U in recUs:
        uidx = int(np.where(huids==U.uid)[0])
        U.set_feature('psth',psth_cut[uidx])

####END PSTH PREP
###zeta prep
zeta_flav = 'onset_time_(half-width)'
zeta_pthr = 0.01
missing_zeta = 999

uloader.get_set_zeta_delay(recids,zeta_dir,zeta_pattern,Units,zeta_feat=zeta_flav,zeta_pthr=zeta_pthr,missingval=missing_zeta)


nondescr_units = [U.id for U in Units if U.zeta_respdelay==missing_zeta]
len(nondescr_units)
#end zeta prep


ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
#ddict2 = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=False)


clustlist = rundict['clustering'][cmethod]


ncl = int(ncluststr)


print('doing %s %s nclust %i'%(myrun,cmethod,ncl))
#ncl = clustlist[1]
labels = ddict[ncl]
#labels = S.sort_labels(myrun, ncl, labels_orig)
titlestr = '%s ncl:%i'%(myrun,ncl)

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cdict_clust = {lab:cmap(norm(lab)) for lab in np.unique(labels)}

#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])



delay_dict = {cat: np.vstack([U.zeta_respdelay for U in Units if U.clust == cat]) for cat in np.arange(ncl)}


tvec = psth_tvec[(psth_tvec <= tbound[1]) & (psth_tvec >= [tbound[0]])]
cat_dict = {cat: np.vstack([U.psth for U in Units if U.clust == cat]) for cat in np.arange(ncl)}
nvec = np.array([len(cat_dict[cat]) for cat in np.arange(ncl)])
allpsth = np.vstack([cat_dict[cat] for cat in np.arange(ncl)])
vmax = sap(allpsth, 99.5)
vmin = sap(allpsth, 0.5)


nrows = int(np.ceil(ncl / ncols_psth))


norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
f = plt.figure(figsize=(13, 1 + nrows * 4))
f.subplots_adjust(left=0.08)
gs0 = gridspec.GridSpec(nrows, ncols_psth, figure=f, wspace=0.5, hspace=0.3)
counter = 0
for rownum in np.arange(nrows):
    for colnum in np.arange(ncols_psth):
        if counter == ncl: break
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[rownum, colnum], \
                                                height_ratios=[1, 0.35], hspace=0.01)
        ax = f.add_subplot(gs00[0])
        psthmat = cat_dict[counter]
        im = ax.imshow(psthmat, aspect='auto', interpolation='nearest', origin='lower',
                       extent=[*tbound, 1, len(psthmat)], norm=norm, cmap=cmap_psth)

        ax.set_title('clust%i' % (counter + 1))
        ax.set_xticklabels([])

        ax2 = f.add_subplot(gs00[1])
        ax2.text(0.98,0.98,'N:%i'%(psthmat.shape[0]),ha='right',va='top',transform=ax2.transAxes)
        sapscores = [sap(psthmat, myscore, axis=0) for myscore in [25, 50, 75]]
        ax2.fill_between(tvec, sapscores[0], sapscores[2], color='grey', alpha=0.5, lw=0)
        ax2.plot(tvec, sapscores[1], color='k')
        ax2.axvline(0., color='silver', zorder=-10)
        if counter == 0:
            pos = ax.get_position()
            cax = f.add_axes([0.92, pos.y0, 0.01, pos.height])
            cb = f.colorbar(im, cax=cax, extend='both')

        # if counter>=ncl-ncols:
        ax2.set_xlabel('time [s]')
        # else:
        #    ax2.set_xticklabels([])
        if colnum == 0:
            ax.set_ylabel('n units')
            ax2.set_ylabel('avg z')
        for myax in [ax, ax2]:
            myax.set_xlim(tbound)
        counter += 1
cax.set_title('z')
f.suptitle('%s  %s' % (myrun, cmethod))
f.tight_layout()
figsaver(f, 'psth_panels')

# as a tower with and without zeta on top
psthtower = np.vstack([cat_dict[cc] for cc in np.arange(ncl)[::-1]])
delaytower = np.vstack([delay_dict[cc] for cc in np.arange(ncl)[::-1]])[:,0]
n_per_clust = np.array([len(cat_dict[cc]) for cc in np.arange(ncl)[::-1]])
mycvec =  np.array([cdict_clust[cc] for cc in np.arange(ncl)[::-1]])
ncum = np.cumsum(n_per_clust)
nstartstop = np.vstack([np.r_[0,ncum[:-1]],ncum]).T

f,axarr = plt.subplots(1,3,figsize=(5.5,14),gridspec_kw={'width_ratios':[0.05,1,0.05]})
f.subplots_adjust(wspace=0.3)
clax,ax,cax = axarr
im = ax.imshow(psthtower, aspect='auto', interpolation='nearest', origin='lower',
                   extent=[*tbound, 0.5, len(psthtower)+0.5], norm=norm, cmap=cmap_psth)
for n_bord in ncum[:-1]:
    ax.axhline(n_bord,color='k')
    clax.axhline(n_bord,color='k')

for cc,[n0,n1] in enumerate(nstartstop):
    clax.add_patch(Rectangle((0., n0), 1, n1 - n0,facecolor=mycvec[cc],linewidth=0))
ax.yaxis.set_tick_params(rotation=90)
clax.set_ylim(ax.get_ylim())
clax.set_xlim([0,1])
clax.set_axis_off()
ax.set_xlim(tbound)
cb = f.colorbar(im, cax=cax, extend='both')
ax.set_xlabel('time [s]')
cax.set_title('z')
f.suptitle('%s  %s' % (myrun, cmethod))

figsaver(f, 'psth_stacked',closeit=False)
yposvec_gen = np.arange(len(delaytower))+1
ax.plot(np.zeros(np.sum(delaytower==missing_zeta)),yposvec_gen[delaytower==missing_zeta],'x',color='k',ms=4)
ax.plot(delaytower[delaytower<missing_zeta],yposvec_gen[delaytower<missing_zeta],'.',color='k',ms=1)
ax.plot(np.zeros(np.sum(np.isnan(delaytower)))-0.2,yposvec_gen[np.isnan(delaytower)],'.',color='k',ms=1)
figsaver(f, 'psth_stacked_ZETA',closeit=True)

#myboxes = [Rectangle((0.,n0), 1, n1-n0) for n0,n1 in nstartstop]


#
# myinds = np.where((~np.isnan(delaytower)) & (delaytower < 100))[0]
# idx = myinds[50]
# f,ax = plt.subplots()
# ax.plot(tvec,psthtower[idx],color='k')
# ax.axvline(delaytower[idx],color='r')



# now the median
tracemat = np.vstack([np.median(cat_dict[cc], axis=0) for cc in np.arange(ncl)])
f, ax = plt.subplots(figsize=(4.5, 3))
f.subplots_adjust(bottom=0.18, left=0.2, right=0.83)
for cc in np.arange(ncl):
    col = cdict_clust[cc]
    ax.plot(tvec, tracemat[cc], color=col, lw=2)
    ax.text(1.01, 0.99 - cc * 0.1, 'clust%i' % (cc + 1), color=col, transform=ax.transAxes, ha='left', va='top',
            fontweight='bold')
ax.axvline(0., color='silver', zorder=-10)
ax.set_xlim(tbound)
ax.set_ylabel('avg.z')
ax.set_xlabel('time [s]')
f.suptitle('%s  %s' % (myrun, cmethod))
f.tight_layout()
figsaver(f, 'psth_medians')

#frac of signif per clust

frac_vec = np.array([len(np.where((~np.isnan(delay_dict[cc])) & (delay_dict[cc] < missing_zeta))[0])/len(delay_dict[cc]) for cc in np.arange(ncl)])
xvec = np.arange(ncl)
f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
f.subplots_adjust(left=0.3,bottom=0.3)
blist = ax.bar(xvec,frac_vec,color='k')
for ii,val in enumerate(frac_vec):
    ax.text(xvec[ii],0.1,'%1.3f'%val,ha='center',va='bottom',rotation=90)
ax.set_xticks(xvec)
tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')
for cc in np.arange(ncl):
    col = cdict_clust[cc]
    blist[cc].set_color(col)
    tlabs[cc].set_color(col)
    #ax.xaxis.get_ticklabels()
ax.set_ylabel('frac')
ax.set_xlabel('category')
#ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
for pos in ['top','right']:ax.spines[pos].set_visible(False)
figsaverZETA(f, 'frac_signif')

# distr of vals
xmax = 0.35
#norm_mode = 'count'
#norm_mode = 'prob'
#norm_mode = 'prob/bin'

for norm_mode in ['count','prob/bin','prob']:
    f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
    f.subplots_adjust(left=0.3,bottom=0.3)
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        mydelays = delay_dict[cc]
        usedlays = mydelays[(~np.isnan(delay_dict[cc])) & (delay_dict[cc] < missing_zeta)]
        n_delays = len(usedlays)
        mybins = np.linspace(0.,sap(usedlays,97.5),int(np.sqrt(n_delays)))
        hist, bins = np.histogram(usedlays,mybins)
        bw = np.diff(bins)[0]
        plotvals = hist[:] #filt.savitzky_golay(hist, 7, 3) if sg_on else



        if norm_mode == 'count':normed = plotvals# /np.sum(plotvals)/bw
        elif norm_mode == 'prob':normed = plotvals/np.sum(plotvals)
        elif norm_mode == 'prob/bin':normed = plotvals/(np.sum(plotvals)*bw)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_centers = np.insert(bin_centers, 0, 0)
        bin_centers = np.insert(bin_centers, len(bin_centers), bin_centers[-1] + np.diff(bins)[-1])
        showvals = np.insert(normed, 0, 0)
        showvals = np.insert(showvals, len(showvals), 0)
        ax.plot(bin_centers, showvals,'-', color=col)#,mfc=col,mec=col
        #print(norm_mode,np.sum(normed))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    ax.grid(axis='x',color='silver',linestyle='--')
    ax.set_ylabel(norm_mode)
    ax.set_xlabel('delay [s]')
    ax.set_ylim([0, ax.get_ylim()[1]])
    #ax.set_xlim([0,sap(delaytower[(~np.isnan(delaytower)) & (delaytower < 999)],95)])
    ax.set_xlim([0,xmax])
    figsaverZETA(f, 'delay_distr_%s'%norm_mode.replace('/','X'))


'''### renaming the zeta files

from glob import glob
replace_dict = {'runCrespI_brain__ward5':'runCrespZI_brain__ward5',\
                'runCrespP_brain__ward8':'runCrespZP_brain__ward8'}

for oldname,newname in replace_dict.items():
    #oldname,newname = list(replace_dict.items())[0]
    mydir = os.path.join(pathdict['src_dirs']['zeta'],oldname)
    os.chdir(mydir)
    myfiles = glob(os.path.join(mydir,'*__zeta.h5'))
    myfiles = glob('*__zeta.h5')

    for myfile in myfiles:
        newfilename = myfile.replace(oldname,newname)
        os.rename(myfile,newfilename)
    #os.rename(mydir,mydir.replace(oldname,newname))
'''

