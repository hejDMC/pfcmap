import sys
import yaml
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats
import h5py


pathpath,myrun,ncluststr,cmethod,get_rois_bool_str,roi_tag = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZP_brain'
get_rois_bool_str = '1'
cmethod = 'ward'
ncluststr = '8'
roi_tag = 'gaoRois'
get_rois_bool_str = '0'
'''
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import category_matching as matchfns
from pfcmap.python.utils import tessellation_tools as ttools



get_rois_bool = bool(int(get_rois_bool_str))

#kickout_lay3pfc = True
signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000


outfile = os.path.join(pathdict['statsdict_dir'],'statsdictZETA_rois_%s__%s.h5'%(roi_tag,myrun))

gao_hier_path = pathdict['gao_hierarchy_file']

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s_zeta/zeta_enrichments/rois/%s'%(myrun,roi_tag))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

zeta_flav = 'onset_time_(half-width)'
zeta_pthr = 0.01
missing_zeta = 999
zeta_pattern = 'RECID__%s__%s%s__TSELpsth2to7__STATEactive__all__zeta.h5'%(myrun,cmethod,ncluststr)
zeta_dir =  os.path.join(pathdict['src_dirs']['zeta'],'%s__%s%s'%(myrun,cmethod,ncluststr))


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']


region_file = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCregions.h5')
if roi_tag == 'dataRois':
    flatmapfile = str(S.roimap_path)
elif roi_tag == 'gaoRois':
    flatmapfile = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')
else:
    assert 0, 'unknown roi_tag: %s'%roi_tag


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=get_rois_bool)

#now set the zeta:

recids = np.unique([U.recid for U in Units])
uloader.get_set_zeta_delay(recids,zeta_dir,zeta_pattern,Units,zeta_feat=zeta_flav,zeta_pthr=zeta_pthr,missingval=missing_zeta)
Units = [U for U in Units if not U.zeta_respdelay == missing_zeta]




for U in Units:
    U.set_feature('clust',int(U.zetap<=zeta_pthr))

Units_pfc = np.array([U for U in Units if not U.roi==0 and S.check_pfc(U.area)])

for U in Units:
    U.set_feature('roistr',str(U.roi))


uloader.assign_parentregs(Units,pfeatname='area',cond=lambda uobj: S.check_pfc(uobj.area)==False,na_str='na')
Units_ctx = np.array([U for U in Units if U in Units_pfc or U.area in S.ctx_regions])

print('pfc: areas',np.unique([U.area for U in Units_pfc]))
print('no pfc areas',np.unique([U.area for U in Units if not S.check_pfc(U.area)]))
print('ctx areas',np.unique([U.area for U in Units_ctx]))



#ux = [U for U in Units if U.area in ctx_regions]

for U in Units:
    if U in Units_pfc:
        U.set_feature('roistrreg',str(U.roi))
    else:
        U.set_feature('roistrreg',str(U.area))



for U in Units:
    if U in Units_ctx:
        laysimple = re.sub("[^0-9]", "", U.layer)
        if U in Units_pfc: U.set_feature('roilay','%s|%s'%(str(U.roi),laysimple))
        else: U.set_feature('roilay',str(U.area))
    else:
        U.set_feature('roilay',str(U.area))


#allroilays = [U.arealay for U in Units]
#myUs = [U for U in Units if U.arealay=='ACAd|5']

for U in Units:
    if U in Units_ctx:
        is_deep = S.check_layers(U.layer, ['5', '6'])
        laydepth = 'deep' if is_deep else 'sup'
        if U in Units_pfc:U.set_feature('roilaydepth', '%s|%s' % (str(U.roi), laydepth))
        else: U.set_feature('roilaydepth', str(U.area))
    else:
        U.set_feature('roilaydepth', str(U.area))





#const_attr='task'
#attr1 = 'roistr'
#attr2 = 'clust'
#Units_pfc56 = [U for Units in Units_pfc if  S.check_layers(U.layer, ['5','6'])]



#laydepth: split into superficical (123) and deep (56)

reftags = ['refall','refPFC']
statsdict = {reftag:{} for reftag in reftags}
attr2 = 'clust'
const_attr = 'task' #only not used in attr1=task!

for reftag, Usel in zip(['refall','refPFC'],[Units,Units_pfc]):
    statsdict[reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'roistr',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict[reftag]['lays'] = matchfns.get_all_shuffs(Usel,'roilay',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict[reftag]['laydepth'] = matchfns.get_all_shuffs(Usel,'roilaydepth',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)

uloader.save_dict_to_hdf5(statsdict, outfile)

print('Done statsdict calc zeta for %s'%myrun)
print('Continuing now with plotting!')

statstag = 'enr'

omnidict = {reftag:{statstag:{} for statstag in [statstag]} for reftag in reftags}

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr

with h5py.File(outfile,'r') as hand:

    for reftag in reftags:


        ####
        thistag = 'deepRois'
        statshand = hand[reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



        ####
        thistag = 'nolayRois'
        statshand = hand[reftag]['nolays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval=='0'])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'layeredRois'
        statshand = hand[reftag]['lays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}


        ####
        thistag = 'deepSupRois'
        statshand = hand[reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}



seltags = list(omnidict[reftag][statstag].keys())
unique_roiruns = ['deepRois','nolayRois']#







if roi_tag == 'gaoRois':
    import csv

    # now import the gao hierarchy index
    with open(gao_hier_path, newline='') as csvfile:
        hdata = list(csv.reader(csvfile, delimiter=','))[1:]
    hdict = {int(float(el[0])): float(el[1]) for el in hdata}
    allrois_gao = np.sort(np.array(list(hdict.keys())))
    hvec = np.array([hdict[key] for key in allrois_gao])


# prepare roi plotting
def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])


with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}


roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}
statsfn = S.enr_fn

mapstr_z = 'RdBu_r'
nancol = 'gainsboro'
zlim = 8
#for hier:
show_theil = False
show_lsq = True

for reftag in reftags:
    for thistag in unique_roiruns:
        #thistag = seltags[0]
        ddict = omnidict[reftag][statstag][thistag]

        sdict = ddict['src_stats']
        X = statsfn(sdict)

        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s'%(reftag,thistag,statstag,plotname,reftag,thistag,statstag)
        titlestr = '%s %s %s'%(myrun,thistag,reftag)

        # plot the zmat on the flatmap
        mean_shuff, std_shuff = [sdict[key] for key in ['meanshuff', 'stdshuff']]
        zmat = (sdict['matches'] - mean_shuff) / std_shuff
        zmat[sdict['levels'] == 0] = np.nan
        plotdict_z = {roi.split('|')[0]: zmat[rr] for rr, roi in enumerate(sdict['alabels'])}
        cmap_z = ttools.get_scalar_map(mapstr_z, [-zlim, zlim])

        f, axarr = plt.subplots(1,2,figsize=(6, 3))
        f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
        for cc,zeta_bool in enumerate(np.arange(2)):
            ax = axarr[cc]
            ax.set_title('issignif==%i' % (zeta_bool), color='k', fontweight='bold')
            ttools.colorfill_polygons(ax, polygon_dict, plotdict_z, subkey=cc, cmap=cmap_z, clab='', na_col='k', nancol=nancol,
                                      ec='k',
                                      show_cmap=cc == 0, mylimfn=set_mylim)
        f.suptitle(titlestr)
        figsaver(f, make_savename('clustspect_on_flatmap'))  # should be the same for all the distbaseMats!!!


        # plot against hierarchy when it is gao
        if roi_tag == 'gaoRois': #
            rois_data = np.array([aval.split('|')[0] for aval in sdict['avals1']]).astype(int)
            hinds = np.array([np.where(allrois_gao == roi)[0][0] for roi in rois_data])
            htemp = hvec[hinds]
            h_cond = ~np.isnan(htemp)
            xvec = htemp[h_cond]
            ymat = X[h_cond]
            myrois = rois_data[h_cond]

            cvec = np.array([roi_colors[str(roi)] for roi in myrois])

            x_simp = np.array([xvec.min(), xvec.max()])
            col = 'k'
            for ii in np.arange(2):
                f, ax = plt.subplots(figsize=(3.5, 3.))
                f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
                yvec = ymat[:, ii]
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
                ax.set_title('issignif==%i; tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (ii, tau, pval, pr.correlation, pr.pvalue),
                             color=col, fontsize=8)
                ax.set_xlabel('h-score')
                ax.set_ylabel('enrichment')
                f.tight_layout()
                figsaver(f, make_savename('enrVsH_issignif%i' % (ii)))

                lims = [ax.get_xlim(), ax.get_ylim()]
                f, ax = plt.subplots(figsize=(9, 8.))
                f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
                yvec = ymat[:, ii]
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
                ax.set_title('issignif=%i; tau:%1.2f, p:%1.2e' % (ii, tau, pval), color=col)
                ax.set_xlabel('h-score')
                ax.set_ylabel('enrichment')
                f.suptitle(titlestr, fontsize=8)
                f.tight_layout()
                figsaver(f, make_savename('enrVsH_text_issignif%i' % (ii)))
