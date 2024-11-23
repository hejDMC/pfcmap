import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


pathpath,myrun = sys.argv[1:]


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from pfcmap.python.utils import som_plotting as somp

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)


kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i'%(myrun,kshape[0],kshape[1]))

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

#metricsfiles = S.get_metricsfiles(srcdir,tablepath=pathdict['tablepath'])

Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc = (not myrun.count('_brain')),rois_from_path=False)
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


sizefac = 0.7
hw_hex = 0.35

unique_regs = np.unique([U.area for U in Units])
regs = [reg for reg in S.PFC_sorted if reg in unique_regs]


n_regs = len(regs)
n_nodes = np.prod(kshape)

region_mat = np.zeros((n_regs, n_nodes))

layers_allowed = ['5','6']


for rr,reg in enumerate(regs):
    #reg = region_keys[rr]
    reginds = np.array([ee for ee,U in enumerate(Units) if U.area.count(reg) and  S.check_layers(U.layer,layers_allowed)])
    regdata = wmat[reginds]
    bmus = somh.get_bmus(regdata,weights)
    for bmu in bmus:
        region_mat[rr,bmu] += 1


fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                    sizefac=sizefac)

for rr,reg in enumerate(regs):
    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])
    somp.plot_hexPanel(kshape, region_mat[rr], ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, hexcol=S.cmap_count, alphahex=1., idcolor='k')

    norm = mpl.colors.Normalize(vmin=region_mat[rr].min(), vmax=region_mat[rr].max())
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    ax.set_title('hits %s'%reg)
    cax.tick_params(axis='y', which='major', labelsize=15)
    f.tight_layout()
    figsaver(f, 'hitmaps_deep_pfc/hits_pfc_%s'%(reg))

classical_pfc_regs = [ 'ACAd',  'ACAv',  'PL', 'ILA']
MOs = ['MOs']
orb_regs = ['ORBm', 'ORBvl', 'ORBl']
rest_of_pfc = [ 'FRP', 'AId', 'AIv']

regs_sel_dict = {'classicalPFC':classical_pfc_regs,\
                 'classicalPFCAndMOs':classical_pfc_regs+MOs,\
                 'classicalPFCAndMOsAndOrb':classical_pfc_regs+MOs+orb_regs,\

                 'fullGaoPFC':classical_pfc_regs+MOs+orb_regs+rest_of_pfc,\
                 'orbsOnly':orb_regs,\
                 'FRPAIdAIv':rest_of_pfc}

for reg_mode,reg_sel in regs_sel_dict.items():
    #reg_sel = regs_sel_dict[reg_mode]
    sel_inds = np.array([regs.index(selreg) for selreg in reg_sel])
    countsperhex = region_mat[sel_inds].sum(axis=0)

    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])
    somp.plot_hexPanel(kshape, countsperhex, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, hexcol=S.cmap_count, alphahex=1., idcolor='k')

    norm = mpl.colors.Normalize(vmin=countsperhex.min(), vmax=countsperhex.max())
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    ax.set_title('hits %s' % reg_mode)
    cax.tick_params(axis='y', which='major', labelsize=15)
    f.tight_layout()
    figsaver(f, 'hitmaps_deep_pfc/merged_subregions/hits_pfc_%s' % (reg_mode))



sel_inds = np.array([regs.index(selreg) for selreg in regs_sel_dict['fullGaoPFC']])
countsperhex_ref = region_mat[sel_inds].sum(axis=0)
vminmax = [countsperhex_ref.min(),countsperhex_ref.max()]
regs_sel_dict2 = {key:regs_sel_dict[key] for key in ['fullGaoPFC','classicalPFC']}
regs_sel_dict2['gaoPFC_woAI'] = classical_pfc_regs+MOs+orb_regs+['FRP']

for reg_mode,reg_sel in regs_sel_dict2.items():
    #reg_sel = regs_sel_dict[reg_mode]
    sel_inds = np.array([regs.index(selreg) for selreg in reg_sel])
    countsperhex = region_mat[sel_inds].sum(axis=0)

    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])
    somp.plot_hexPanel(kshape, countsperhex, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, hexcol=S.cmap_count, alphahex=1., idcolor='k',vminmax=vminmax)

    norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    ax.set_title('hits %s' % reg_mode)
    cax.tick_params(axis='y', which='major', labelsize=15)
    f.tight_layout()
    figsaver(f, 'hitmaps_deep_pfc/merged_subregions/same_colorrange/hits_pfc_%s' % (reg_mode))
