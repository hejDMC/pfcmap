import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import h5py
import matplotlib as mpl
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac
from scipy import stats


pathpath = 'PATHS/filepaths_IBLTuning.yml'#
myrun =  'runIBLTuning_utypeP'



with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S
from pfcmap_paper.utils import graphs as gfns
from pfcmap_paper.utils import tessellation_tools as ttools
from pfcmap_paper.utils import clustering as cfns

gao_hier_path = pathdict['gao_hierarchy_file']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
statsdict_dir = pathdict['statsdict_dir']#
srcfile = os.path.join(pathdict['statsdict_dir'],'statsdictTuning_gaoRois__%s.h5'%(myrun))

statsdict_rois = uloader.load_dict_from_hdf5(srcfile)

tuning_attr_names = ['stimTuned','choiceTuned','feedbTuned','taskResp']


figdir_base = 'FIGURES/tuning_gaoCorrelations'

figdir_mother = os.path.join(figdir_base,'tuningGaoCorr__%s'%(myrun))

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__gaoCorr_%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


flatmapfile = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')
region_file = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCregions.h5')
with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}


roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}
statstags = ['kendall', 'pearson']

#now import the gao hierarchy index
with open(gao_hier_path, newline='') as csvfile:
    hdata = list(csv.reader(csvfile, delimiter=','))[1:]
hdict = {int(float(el[0])):float(el[1]) for el in hdata}
allrois_gao = np.sort(np.array(list(hdict.keys())))
hvec = np.array([hdict[key] for key in allrois_gao])

show_theil = False
show_lsq = True


replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr

matkeys = ['levels','matches','meanshuff','pofs','stdshuff']
N_tuningattrs = len(tuning_attr_names)
attr = tuning_attr_names[0]#just one attribute
n_roi_entries = len(statsdict_rois[attr]['matches'])
Nmin_maps = int(S.Nmin_maps)

sum_mat = np.vstack([statsdict_rois[attr]['matches'].sum(axis=1) for attr in tuning_attr_names])#should be the same for all...
minnperroi = np.min(sum_mat,axis=0)
deep_cond = np.array([myroiname.count('|deep') for myroiname in  statsdict_rois[attr]['avals1']]).astype(bool)
avail_rois = statsdict_rois[attr]['avals1'][(minnperroi>=Nmin_maps)&(deep_cond)]
avail_inds = np.arange(n_roi_entries)[(minnperroi>=Nmin_maps)&(deep_cond)]
N_avail = len(avail_inds)
sdict = {key:np.zeros((N_avail,N_tuningattrs))*np.nan for key in matkeys}
for aa,attr in enumerate(tuning_attr_names):
    matchdict = statsdict_rois[attr]
    for key in matkeys:
        sdict[key][:,aa] = matchdict[key][avail_inds,matchdict['avals2']=='signif']
sdict['alabels'] = np.array([replace_fn(aval) for aval in matchdict['avals1'][avail_inds]])
sdict['avals2'] = tuning_attr_names
sdict['a1'] = 'cTuning'
sdict['SI'] = 1

statstag = 'enr'
statsfn = S.enr_fn

X = statsfn(sdict)

rois_data = np.array([aval.split('|')[0] for aval in sdict['alabels']]).astype(int)
hinds = np.array([np.where(allrois_gao == roi)[0][0] for roi in rois_data])
htemp = hvec[hinds]
h_cond = ~np.isnan(htemp)
xvec = htemp[h_cond]
ymat = X[h_cond]
myrois = rois_data[h_cond]

titlestr = '%s'%(myrun)


cvec = np.array([roi_colors[str(roi)] for roi in myrois])

x_simp = np.array([xvec.min(), xvec.max()])
col = 'k'

for aa,attr in enumerate(tuning_attr_names):
    f, ax = plt.subplots(figsize=(3.5, 3.))
    f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
    yvec = ymat[:, aa]
    # ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
    ax.scatter(xvec, yvec, c=cvec, marker='o')

    kt = stats.kendalltau(xvec, yvec)
    tau = kt.correlation
    pval = kt.pvalue
    pr = stats.pearsonr(xvec, yvec)

    if show_theil:
        ts = stats.theilslopes(yvec, xvec, method='separate')
        ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color=col)
        ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color=col)
    if show_lsq:
        lsq_res = stats.linregress(xvec, yvec)
        ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
    ax.set_title('%s; tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (attr, tau, pval, pr.correlation, pr.pvalue),
                 color=col, fontsize=8)
    ax.set_xlabel('h-score')
    ax.set_ylabel('enrichment')
    f.tight_layout()
    figsaver(f, 'corrgraphs/enrVsH_%s'%attr)

    lims = [ax.get_xlim(), ax.get_ylim()]
    f, ax = plt.subplots(figsize=(9, 8.))
    f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
    for zz, [x, y, area] in enumerate(zip(xvec, yvec, myrois)):
        ax.text(x, y, str(area), ha='center', va='center', color=cvec[zz])
    # ax.set_xlim([xvec.min()-0.1,xvec.max()+0.1])
    # ax.set_ylim([yvec.min()-0.5,yvec.max()+0.5])
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    kt = stats.kendalltau(xvec, yvec)
    tau = kt.correlation
    pval = kt.pvalue
    ts = stats.theilslopes(yvec, xvec, method='separate')
    ax.set_title('%s; tau:%1.2f, p:%1.2e' % (attr, tau, pval), color=col)
    ax.set_xlabel('h-score')
    ax.set_ylabel('enrichment')
    f.suptitle(titlestr, fontsize=8)
    f.tight_layout()
    figsaver(f, 'corrgraphs/enrVsH_text_%s'%attr)

# write a list with tau and pearson, save it

statsfile = os.path.join(figdir_mother,'stats__%s.txt' % (myrun))
writedict = {statstag: {'pvec': np.zeros(N_tuningattrs), 'corrvec': np.zeros(N_tuningattrs)} for statstag in
             statstags}  # for easier hdf5 access

with open(statsfile, 'w') as hand:
    hand.write('N: %i\n%s\n%s\n' % (len(myrois), str(list(myrois)), titlestr))
    for aa,attr in enumerate(tuning_attr_names):
        yvec = ymat[:, aa]
        kt = stats.kendalltau(xvec, yvec)
        pr = stats.pearsonr(xvec, yvec)
        mystr = '%s Kendall-Tau: (%1.2f, %1.3e)   Pearson-R: (%1.2f, %1.3e)' % (
            attr, kt.correlation, kt.pvalue, pr.correlation, pr.pvalue)
        hand.write(mystr + '\n')
        writedict['kendall']['pvec'][aa], writedict['kendall']['corrvec'][aa] = kt.pvalue, kt.correlation
        writedict['pearson']['pvec'][aa], writedict['pearson']['corrvec'][aa] = pr.pvalue, pr.correlation

writedict['rois'] = [str(myroi) for myroi in myrois]

corrfilename = os.path.join(pathdict['statsdict_dir'],
                            'gaocorrTuning__%s__enr.h5' % (myrun))

uloader.save_dict_to_hdf5(writedict, corrfilename, strtype='S10')

# display as bar-plot correlation as bar plot
tattrs_short = [attr.replace('Tuned','').replace('Resp','') for attr in tuning_attr_names]

ymax = 0.75
clvec = np.arange(N_tuningattrs)
for statstag in statstags:
    corrvec = writedict[statstag]['corrvec']
    f, ax = plt.subplots(figsize=(1.5 + 0.2 * N_tuningattrs, 2))
    f.subplots_adjust(left=0.3, bottom=0.3)
    blist = ax.bar(clvec, corrvec, color='k')
    ax.set_xticks(clvec)
    tlabs = ax.set_xticklabels(clvec + 1)
    ax.set_ylabel(statstag)
    ax.set_xlim([clvec.min() - 0.5, clvec.max() + 0.5])
    ax.set_title(titlestr, fontsize=8)
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='both', color='grey', linestyle=':', zorder=-10)
    for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
    ax.set_xticklabels(tattrs_short,rotation=-90)
    # ax.set_yticks(np.arange(-0.6,0.7,0.2))
    for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
    f.tight_layout()
    figsaver(f,'corrbars_%s' % statstag)

