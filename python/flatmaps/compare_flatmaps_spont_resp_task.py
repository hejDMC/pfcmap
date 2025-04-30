import os
import h5py
import yaml
import sys
import numpy as np
from scipy.stats import pearsonr,kendalltau
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

pathpath = 'PATHS/filepaths_carlen.yml'
myrun_spont = 'runC00dMP3_brain'
ncluststr_spont = '8'
myrun_resp = 'runCrespZP_brain'
ncluststr_resp = '8'
cmethod = 'ward'
myrun_task = 'runIBLTuning_utypeP'
myrun_zeta = 'runCrespZP_brain'




dpi_png = 200

def figsaver(fig, nametag, fformat='svg',closeit=True):
    figname = os.path.join(figdir_mother, nametag + '.%s'%(fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    if fformat == 'svg':
        fig.savefig(figname)
    else:
        fig.savefig(figname,dpi=dpi_png)
    if closeit: plt.close(fig)


task_attr_names = ['stimTuned','choiceTuned','feedbTuned']#what we are after
flatmapName_dict = {'dataROIs':'flatmap_PFC_ntesselated_obeyRegions_res200.h5','iblROIs':'IBLflatmap_PFC_ntesselated_obeyRegions_res200.h5',\
                    'gaoROIs':'flatmap_PFCrois.h5'}
rastermapName_dict = {'dataROIs':'flatmap_PFC_ntesselated_obeyRegions_res200__rasterized.h5',\
                      'iblROIs':'IBLflatmap_PFC_ntesselated_obeyRegions_res200__rasterized.h5'}


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S
from pfcmap_paper.utils import tessellation_tools as ttools

figdir_mother = pathdict['figdir_root']+'flatmaps_and_flatmapComparisons'

statsfile_spont = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%('dataROIs',myrun_spont,ncluststr_spont,cmethod))
statsfile_resp = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%('dataROIs',myrun_resp,ncluststr_resp,cmethod))
statsfile_task = os.path.join(pathdict['statsdict_dir'],'sstatsdictTuning_rois_iblROIs__%s.h5'%(myrun_task))
statsfile_zeta = os.path.join(pathdict['statsdict_dir'], 'statsdictZETA_rois_%s__%s.h5' % ('dataROIs', myrun_zeta))
rasterdir = pathdict['tesselation_dir'].replace('flatmaps','flatmaps_rasterized')

statsfn = S.enr_fn

replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr




colldict = {}
for myrun,statsfile in zip([myrun_spont,myrun_resp],[statsfile_spont,statsfile_resp]):

    with h5py.File(statsfile,'r') as hand:
        statshand = hand['refPFC']['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
    countvec = mystats['matches'][()].sum(axis=1)
    presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
    sdictDrois = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
    sdictDrois['alabels'] = np.array([replace_fn(aval) for aval in sdictDrois['avals1']])
    sdictDrois['src'] = statshand.name
    sdictDrois_roinums = np.array([int(alab.split('|')[0]) for alab in sdictDrois['alabels']])
    XDrois = statsfn(sdictDrois)
    colldict[myrun] = {'map':'dataROIs','datamat':XDrois,'roinums':sdictDrois_roinums,'cats':np.arange(XDrois.shape[1])+1}

assert (colldict[myrun_spont]['roinums']== colldict[myrun_resp]['roinums']).all(),'there must be empty rois'#todo check this more elegantly on the map later!

#the zeta map
zeta_run = myrun_zeta+'_zeta'
with h5py.File(statsfile_zeta,'r') as hand:
    statshand = hand['refPFC']['laydepth']#from this select the deep ones
    mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
countvec = mystats['matches'][()].sum(axis=1)
presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
sdictDrois = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
sdictDrois['alabels'] = np.array([replace_fn(aval) for aval in sdictDrois['avals1']])
sdictDrois['src'] = statshand.name
sdictDrois_roinums = np.array([int(alab.split('|')[0]) for alab in sdictDrois['alabels']])
XDrois = statsfn(sdictDrois)[:,mystats['avals2']==1]#nsamples x 1
colldict[zeta_run] = {'map':'dataROIs','datamat':XDrois,'roinums':sdictDrois_roinums,'cats':np.array(['audResp'])}


#now the IBL task maps
N_taskattrs = len(task_attr_names)
statsdict_task = uloader.load_dict_from_hdf5(statsfile_task)

matkeys = ['levels','matches','meanshuff','pofs','stdshuff']
attr1 = task_attr_names[0]#just one of the attrs
n_roi_entries = len(statsdict_task[attr1]['matches'])
avals1 =statsdict_task[attr1]['avals1']
#set sdict nan where there is too little data in the roi! find general presel inds!
sum_mat = np.vstack([statsdict_task[attr]['matches'].sum(axis=1) for attr in task_attr_names])#should be the same for all...
minnperroi = np.min(sum_mat,axis=0)
deep_cond = np.array([myroiname.count('|deep') for myroiname in  avals1]).astype(bool)
cond_avail = (minnperroi>=S.Nmin_maps)&(deep_cond)
avail_rois = statsdict_task[attr1]['avals1'][cond_avail]
avail_inds = np.arange(n_roi_entries)[cond_avail]
N_avail = len(avail_inds)
sdict = {key:np.zeros((N_avail,N_taskattrs))*np.nan for key in matkeys}
for aa,attr in enumerate(task_attr_names):
    matchdict = statsdict_task[attr]
    for key in matkeys:
        sdict[key][:,aa] = matchdict[key][avail_inds,matchdict['avals2']=='signif']
sdict['alabels'] = np.array([replace_fn(aval) for aval in matchdict['avals1'][avail_inds]])
sdict['avals2'] = task_attr_names
sdict['a1'] = 'cTuning'
sdict_roinums = np.array([int(alab.split('|')[0]) for alab in sdict['alabels']])
Xtask = statsfn(sdict)

colldict[myrun_task] = {'map':'iblROIs','datamat':Xtask,'roinums':sdict_roinums,'cats':np.array(task_attr_names)}

#getting the polygon dicts
polygon_map_dicts = {}
for roitype,flatmapname in flatmapName_dict.items():
    flatmapfile = os.path.join(pathdict['tesselation_dir'],flatmapname)
    with h5py.File(flatmapfile,'r') as hand:
        polygon_map_dicts[roitype] = {key: hand[key][()] for key in hand.keys()}

#getting the raster dicts
#roitypes =  list(rastermapName_dict.keys())
raster_map_dict = {}
for roitype,rastername in rastermapName_dict.items():
    rasterfile = os.path.join(rasterdir,rastername)
    with h5py.File(rasterfile,'r') as hand:
        raster_map_dict[roitype] = {dsname: hand[dsname][()] for dsname in ['rastermat','valuevec']}


###preparing plotting
cmapclust = mpl.cm.get_cmap(S.cmap_clust)#

ncl_resp = int(ncluststr_resp)
norm_resp = mpl.colors.Normalize(vmin=0, vmax=ncl_resp-1)
cdict_clust = {myrun_resp:{lab+1:cmapclust(norm_resp(lab)) for lab in np.arange(ncl_resp)}}
ncl_spont = int(ncluststr_spont)
norm_spont = mpl.colors.Normalize(vmin=0, vmax=ncl_spont-1)
cdict_clust[myrun_spont] = {lab+1:cmapclust(norm_spont(lab)) for lab in np.arange(ncl_spont)}
cdict_clust[myrun_task] = {lab:'k' for lab in task_attr_names}
cdict_clust[zeta_run] = {'audResp':'k'}

labfeat_dict = {myrun_spont:['spont%i'%(mycl+1) for mycl in np.arange(ncl_spont)],\
                myrun_resp:['resp%i'%(mycl+1) for mycl in np.arange(ncl_spont)],\
                myrun_task:[tasktag.replace('Tuned','') for tasktag in task_attr_names],\
                zeta_run:['audResp']}



#################################
# plot the polygon dicts for all
all_runs = list(colldict.keys())
mapstr_z = 'RdBu_r'
zlims = [5,8]
def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])


for myrun,mydict in colldict.items():
    #mydict = colldict[myrun]
    feats = mydict['cats']
    N_feats = len(feats)
    maptype = mydict['map']
    polygon_dict = polygon_map_dicts[maptype]

    for zlim in zlims:
        cmap_z = ttools.get_scalar_map(mapstr_z, [-zlim, zlim])

        f, axarr = plt.subplots(1, N_feats, figsize=(N_feats*2+0.05, 2))
        f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
        for ff,feat in enumerate(feats):
            plotdict_z = {str(roinum):roidata for roinum,roidata in zip(mydict['roinums'],mydict['datamat'][:,ff])}
            ax = axarr[ff] if len(feats)>1 else axarr
            ttools.colorfill_polygons(ax, polygon_dict, plotdict_z, subkey=aa, cmap=cmap_z, clab='E',
                                      na_col='grey', nancol='grey',
                                      ec='k',
                                      show_cmap=ff==0, mylimfn=set_mylim)  #
            #ax.set_title(feat,pad=-30)
            ax.set_title(labfeat_dict[myrun][ff],pad=-30,color=cdict_clust[myrun][feat],fontweight='bold')

            ax.set_aspect('equal')
        f.suptitle(myrun,fontsize=8)
        figsaver(f,'orig_rois/fm_%s__Z%i_origRois'%(myrun,zlim))

# get the raster-values for each run
rasterval_dict = {myrun:{} for myrun in all_runs}
for myrun,mydict in colldict.items():
    maptype = mydict['map']
    feats = mydict['cats']
    rastermat = raster_map_dict[maptype]['rastermat']
    for ff,feat in enumerate(feats):
        enrraster = np.zeros_like(rastermat)
        enrraster[np.isnan(rastermat)] = np.nan
        for roinum,roidata in zip(mydict['roinums'],mydict['datamat'][:,ff]):
            enrraster[rastermat==roinum] = roidata
        rasterval_dict[myrun][feat] = enrraster

# make a superdict with different smoothing levels for the rasterized
sigmavec = np.array([5,10,20,30,40,50])
blurdict = {sigma:{myrun:{} for myrun in all_runs} for sigma in sigmavec}
for sigma in sigmavec:
    for myrun,subd in rasterval_dict.items():
        for feat,enrraster in subd.items():
            blurred_raster = ttools.nan_gaussian_blur(enrraster, sigma=sigma)
            blurdict[sigma][myrun][feat] = blurred_raster

blurdict[0] = {key:val for key,val in rasterval_dict.items()}#zero-blur is the original
sigmavec_run = np.r_[0,sigmavec]



for sigma in sigmavec_run:
    for myrun,subd in blurdict[sigma].items():
        N_feats = len(subd)
        feats = list(subd.keys())
        for zlim in zlims:
            #if (myrun==zeta_run) and (sigma==30): assert 0,'here is your break!'
            f, axarr = plt.subplots(1, N_feats, figsize=(N_feats*2+0.05, 2))
            f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
            for ff,feat in enumerate(feats):
                featraster = subd[feat]
                ax = axarr[ff] if len(feats)>1 else axarr
                im = ax.imshow(featraster[::-1],cmap=mapstr_z,origin='lower',vmin=-zlim,vmax=zlim)
                ax.set_axis_off()
                #ax.set_title(feats[ff],pad=-30)
                ax.set_title(labfeat_dict[myrun][ff],pad=-30,color=cdict_clust[myrun][feat],fontweight='bold')
                #make colorbar:
                if ff == 0:
                    cmap_temp = im.get_cmap()
                    pos = ax.get_position()
                    cax = f.add_axes(
                        [pos.x0 + pos.width * 0.2, pos.y0 + 0.05, pos.width / 12, pos.height / 4])  # [left, bottom, width, height]
                    cax.set_title('E')
                    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap_temp, norm=mpl.colors.Normalize(vmin=-zlim, vmax=zlim), orientation='vertical')

                ax.set_aspect('equal')
            f.suptitle('%s   blur: %i'%(myrun,sigma),fontsize=8)
            figsaver(f,'rasterblur%i/fm_%s__Z%i_blur%i'%(sigma,myrun,zlim,sigma))

#################
###CORRELATION MATRICES (kendall and pearson)
###calculate correlation matrix betweeen the rasterized
# save xlsx and plot

#first extend the blurdict with a field combining zeta and task
from copy import deepcopy
myrun_task_ext = myrun_task+'_InclAudResp'
labfeat_dict[myrun_task_ext] = labfeat_dict[myrun_task]+labfeat_dict[zeta_run]
for sigma in sigmavec_run:
    blurdict[sigma][myrun_task_ext] = deepcopy(blurdict[sigma][myrun_task])
    blurdict[sigma][myrun_task_ext].update(deepcopy(blurdict[sigma][zeta_run]))

run_combinations = [(myrun_spont,myrun_resp),(myrun_spont,myrun_task),(myrun_spont,myrun_task_ext),(myrun_resp,myrun_task)]
run_axlabeldict = {myrun_spont:'spont.cat.',myrun_resp:'resp.cat.',myrun_task:'',myrun_task_ext:''}

cfn_dict = {'pearson':pearsonr,'ktau':kendalltau,'csim':lambda vals1,vals2: np.dot(vals1, vals2)/(np.linalg.norm(vals1)*np.linalg.norm(vals2))}
corrmethods = list(cfn_dict.keys())

#calculate
corrdict_super = {sigma:{(r1,r2):{} for r1,r2 in run_combinations} for sigma in sigmavec_run}

for sigma in sigmavec_run:
    for run1,run2 in run_combinations:
        #run1,run2 = run_combinations[1]
        #if (run2==myrun_task_ext) and (sigma==30): assert 0,'here is your break!'

        corrdict = corrdict_super[sigma][(run1,run2)]
        dict1 = blurdict[sigma][run1]
        dict2 = blurdict[sigma][run2]
        feats1 = list(dict1.keys())
        feats2 = list(dict2.keys())
        X1 = np.vstack([dict1[feat][~np.isnan(dict1[feat])] for feat in feats1])
        X2 = np.vstack([dict2[feat][~np.isnan(dict2[feat])] for feat in feats2])
        corrdict['features'] = {'feats1':feats1,'feats2':feats2}
        corrdict['N'] = X1.shape[1]
        corrdict['corrmats'] = {}
        assert X1.shape[1]==X2.shape[1],'mismatching values for correlation'
        n1,n2 = len(feats1),len(feats2)
        for corrmethod in corrmethods:
            #corrmethod = corrmethods[1]
            cfn = cfn_dict[corrmethod]
            corrmat =  np.zeros((n1,n2)) if corrmethod=='csim' else np.zeros((n1,n2,2))
            for ff1 in np.arange(n1):
                for ff2 in np.arange(n2):
                    corrmat[ff1,ff2] = cfn(X1[ff1],X2[ff2])
            corrdict['corrmats'][corrmethod] = corrmat

#plot corrmats and save to xlsx
corrlims = [1.,0.5]
corrcmap = 'RdBu_r'
for sigma in sigmavec_run:
    #sigma = sigmavec[3]#
    blur_folder = 'rasterblur%i'%sigma
    for run_comb in run_combinations:

        #run_comb = run_combinations[1]
        run1,run2 = run_comb
        #if (run2==myrun_task_ext) and (sigma==30): assert 0,'here is your break!'

        corrdict = corrdict_super[sigma][run_comb]
        feats1, feats2 = [corrdict['features'][feattag] for feattag in ['feats1','feats2']]
        n1,n2 = len(feats1),len(feats2)
        runc_lab = ('_vs_').join(run_comb)
        outfile = os.path.join(figdir_mother,blur_folder,'%s__correlations.xlsx'%runc_lab)
        #if not os.path.isdir(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile))
        with pd.ExcelWriter(outfile) as writer:
            for corrmethod in corrmethods:
                if corrmethod == 'csim':
                    cmat = corrdict['corrmats'][corrmethod]
                    df = pd.DataFrame(data=cmat, columns=labfeat_dict[run2], index=labfeat_dict[run1])
                    df.to_excel(writer, sheet_name='%s'%(corrmethod))
                else:
                    cmat,pmat = corrdict['corrmats'][corrmethod].transpose(2,0,1)
                    for flav,outmat in zip(['corr','pvals'],[cmat,pmat]):
                        df = pd.DataFrame(data=outmat, columns=labfeat_dict[run2], index=labfeat_dict[run1])
                        df.to_excel(writer, sheet_name='%s_%s'%(corrmethod,flav))
        for corrmethod in corrmethods:
            #corrmethod = corrmethods[0]
            corrmat = corrdict['corrmats'][corrmethod] if corrmethod == 'csim' else corrdict['corrmats'][corrmethod][:,:,0]
            for corrlim in corrlims:
                for corrcmap in ['PiYG_r','RdBu_r','RdGy_r']:
                    f,ax = plt.subplots(figsize=(2+n1*0.3,1+n2*0.3))
                    ax.set_title('%s %s\n blur:%i'%(run1,run2,sigma),fontsize=8)
                    im = ax.imshow(corrmat.T,origin='lower',vmin=-corrlim,vmax=corrlim,cmap=corrcmap)
                    ax.set_xticks(np.arange(n1))
                    ax.set_yticks(np.arange(n2))
                    xtlabs = ax.set_xticklabels(feats1)
                    ytlabs = ax.set_yticklabels(feats2)
                    ax.set_xlabel(run_axlabeldict[run1])
                    ax.set_ylabel(run_axlabeldict[run2])
                    if not type(feats1[0]) == np.str_:
                        for cc in np.arange(n1):
                            xtlabs[cc].set_color(cdict_clust[run1][feats1[cc]])
                            xtlabs[cc].set_fontweight('bold')
                    if not type(feats2[0]) == np.str_:
                        for cc in np.arange(n2):
                            ytlabs[cc].set_color(cdict_clust[run2][feats2[cc]])
                            ytlabs[cc].set_fontweight('bold')

                    cb = f.colorbar(im)
                    cb.set_label(corrmethod,rotation=-90,labelpad=10)
                    ax.set_aspect('equal')
                    f.tight_layout()
                    figsaver(f,'%s/corr_%s__%s_cz%i_blur%i_%s'%(blur_folder,runc_lab,corrmethod,corrlim*10,sigma,corrcmap))




##############################
##### SAME FOR GAO
gao_folder = 'gao_rois'

statsfile_spont_gao = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%('gaoROIs',myrun_spont,ncluststr_spont,cmethod))
statsfile_resp_gao = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%('gaoROIs',myrun_resp,ncluststr_resp,cmethod))
statsfile_task_gao = os.path.join(pathdict['statsdict_dir'],'sstatsdictTuning_gaoRois__%s.h5'%(myrun_task))
statsfile_zeta_gao = os.path.join(pathdict['statsdict_dir'], 'statsdictZETA_rois_%s__%s.h5' % ('gaoROIs', myrun_zeta))


colldictGao = {}

for myrun,statsfile in zip([myrun_spont,myrun_resp],[statsfile_spont_gao,statsfile_resp_gao]):

    with h5py.File(statsfile,'r') as hand:
        statshand = hand['refPFC']['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
    countvec = mystats['matches'][()].sum(axis=1)
    presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
    sdictDrois = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
    sdictDrois['alabels'] = np.array([replace_fn(aval) for aval in sdictDrois['avals1']])
    sdictDrois['src'] = statshand.name
    sdictDrois_roinums = np.array([int(alab.split('|')[0]) for alab in sdictDrois['alabels']])
    XDrois = statsfn(sdictDrois)
    colldictGao[myrun] = {'map':'gaoROIs','datamat':XDrois,'roinums':sdictDrois_roinums,'cats':np.arange(XDrois.shape[1])+1}

with h5py.File(statsfile_zeta_gao,'r') as hand:
    statshand = hand['refPFC']['laydepth']#from this select the deep ones
    mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
countvec = mystats['matches'][()].sum(axis=1)
presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|deep')])
sdictDrois = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
sdictDrois['alabels'] = np.array([replace_fn(aval) for aval in sdictDrois['avals1']])
sdictDrois['src'] = statshand.name
sdictDrois_roinums = np.array([int(alab.split('|')[0]) for alab in sdictDrois['alabels']])
XDrois = statsfn(sdictDrois)[:,mystats['avals2']==1]#nsamples x 1
colldictGao[zeta_run] = {'map':'gaoROIs','datamat':XDrois,'roinums':sdictDrois_roinums,'cats':np.array(['audResp'])}



#now the IBL task maps
statsdict_task = uloader.load_dict_from_hdf5(statsfile_task_gao)

attr1 = task_attr_names[0]#just one of the attrs
n_roi_entries = len(statsdict_task[attr1]['matches'])
avals1 =statsdict_task[attr1]['avals1']
#set sdict nan where there is too little data in the roi! find general presel inds!
sum_mat = np.vstack([statsdict_task[attr]['matches'].sum(axis=1) for attr in task_attr_names])#should be the same for all...
minnperroi = np.min(sum_mat,axis=0)
deep_cond = np.array([myroiname.count('|deep') for myroiname in  avals1]).astype(bool)
cond_avail = (minnperroi>=S.Nmin_maps)&(deep_cond)
avail_rois = statsdict_task[attr1]['avals1'][cond_avail]
avail_inds = np.arange(n_roi_entries)[cond_avail]
N_avail = len(avail_inds)
sdict = {key:np.zeros((N_avail,N_taskattrs))*np.nan for key in matkeys}
for aa,attr in enumerate(task_attr_names):
    matchdict = statsdict_task[attr]
    for key in matkeys:
        sdict[key][:,aa] = matchdict[key][avail_inds,matchdict['avals2']=='signif']
sdict['alabels'] = np.array([replace_fn(aval) for aval in matchdict['avals1'][avail_inds]])
sdict['avals2'] = task_attr_names
sdict['a1'] = 'cTuning'
sdict_roinums = np.array([int(alab.split('|')[0]) for alab in sdict['alabels']])
Xtask = statsfn(sdict)

colldictGao[myrun_task] = {'map':'gaoROIs','datamat':Xtask,'roinums':sdict_roinums,'cats':np.array(task_attr_names)}
polygon_dict_gao = polygon_map_dicts['gaoROIs']





#plotting the gao maps
for myrun,mydict in colldictGao.items():
    #mydict = colldict[myrun]
    feats = mydict['cats']
    N_feats = len(feats)
    maptype = mydict['map']
    for zlim in zlims:
        cmap_z = ttools.get_scalar_map(mapstr_z, [-zlim, zlim])

        f, axarr = plt.subplots(1, N_feats, figsize=(N_feats*2+0.05, 2))
        f.subplots_adjust(wspace=0.001, left=0.01, right=0.99, bottom=0.02, top=0.85)
        for ff,feat in enumerate(feats):
            plotdict_z = {str(roinum):roidata for roinum,roidata in zip(mydict['roinums'],mydict['datamat'][:,ff])}
            ax = axarr[ff] if len(feats)>1 else axarr
            ttools.colorfill_polygons(ax, polygon_dict_gao, plotdict_z, subkey=aa, cmap=cmap_z, clab='E',
                                      na_col='grey', nancol='grey',
                                      ec='k',
                                      show_cmap=ff==0, mylimfn=set_mylim)  #
            #ax.set_title(feat,pad=-30)
            ax.set_title(labfeat_dict[myrun][ff],pad=-30,color=cdict_clust[myrun][feat],fontweight='bold')

            ax.set_aspect('equal')
        f.suptitle(myrun,fontsize=8)
        figsaver(f,'%s/fm_%s__Z%i_gaoROIs'%(gao_folder,myrun,zlim))

run_combinations_pure = [runcomb for runcomb in run_combinations if not str(runcomb).count('Incl')]+[(myrun_spont,zeta_run)]#removing the merged
corrdict_gao = {(r1,r2):{} for r1,r2 in run_combinations_pure}
for run1,run2 in run_combinations_pure:
    #run1,run2 = run_combinations[1]
    corrdict = corrdict_gao[(run1,run2)]
    dict1 = colldictGao[run1]
    dict2 = colldictGao[run2]
    feats1 = dict1['cats']
    feats2 = dict2['cats']

    corrdict['features'] = {'feats1':feats1,'feats2':feats2}
    labels1 = dict1['roinums']
    labels2 = dict2['roinums']
    common_labels = np.array(list(set(labels1).intersection(set(labels2))))
    inds1 = np.array([np.where(labels1 == lab)[0][0] for lab in common_labels])
    inds2 = np.array([np.where(labels2 == lab)[0][0] for lab in common_labels])

    X1 = dict1['datamat'][inds1].T
    X2 = dict2['datamat'][inds2].T
    corrdict['N'] = X1.shape[1]
    corrdict['corrmats'] = {}
    assert X1.shape[1]==X2.shape[1],'mismatching values for correlation'
    n1,n2 = len(feats1),len(feats2)
    for corrmethod in corrmethods:
        #corrmethod = corrmethods[1]
        cfn = cfn_dict[corrmethod]
        corrmat = np.zeros((n1, n2)) if corrmethod == 'csim' else np.zeros((n1, n2, 2))
        for ff1 in np.arange(n1):
            for ff2 in np.arange(n2):
                corrmat[ff1,ff2] = cfn(X1[ff1],X2[ff2])
        corrdict['corrmats'][corrmethod] = corrmat


#now a separate correlation matrixe merged from zeta and task

taskd = corrdict_gao[(myrun_spont,myrun_task)]
respd = corrdict_gao[(myrun_spont,zeta_run)]
corrdict_gao[(myrun_spont,myrun_task_ext)] = {}
joindict = corrdict_gao[(myrun_spont,myrun_task_ext)]
joindict['features'] = {'feats1':taskd['features']['feats1'],\
                        'feats2':np.r_[taskd['features']['feats2'],respd['features']['feats2']]}
joindict['N'] = taskd['N']
joindict['corrmats'] = {}
for corrmethod in corrmethods:
    joindict['corrmats'][corrmethod] = np.concatenate([taskd['corrmats'][corrmethod],respd['corrmats'][corrmethod]],axis=1)#(8,3,2)--> (8,4,2)


#now write and plto it
for run_comb in run_combinations:
    #run_comb = run_combinations[1]
    run1,run2 = run_comb
    #if (run2==myrun_task_ext) and (sigma==30): assert 0,'here is your break!'

    corrdict = corrdict_gao[run_comb]
    feats1, feats2 = [corrdict['features'][feattag] for feattag in ['feats1','feats2']]
    n1,n2 = len(feats1),len(feats2)
    runc_lab = ('_vs_').join(run_comb)
    outfile = os.path.join(figdir_mother,gao_folder,'%s__correlations.xlsx'%runc_lab)
    if not os.path.isdir(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile))
    with pd.ExcelWriter(outfile) as writer:
        for corrmethod in corrmethods:
            if corrmethod == 'csim':
                cmat = corrdict['corrmats'][corrmethod]
                df = pd.DataFrame(data=cmat, columns=labfeat_dict[run2], index=labfeat_dict[run1])
                df.to_excel(writer, sheet_name='%s'%(corrmethod))
            else:

                cmat,pmat = corrdict['corrmats'][corrmethod].transpose(2,0,1)
                for flav,outmat in zip(['corr','pvals'],[cmat,pmat]):
                    df = pd.DataFrame(data=outmat, columns=labfeat_dict[run2], index=labfeat_dict[run1])
                    df.to_excel(writer, sheet_name='%s_%s'%(corrmethod,flav))
    for corrmethod in corrmethods:
        #corrmethod = corrmethods[0]
        corrmat = corrdict['corrmats'][corrmethod] if corrmethod == 'csim' else corrdict['corrmats'][corrmethod][:,:,0]
        for corrlim in corrlims:
            for corrcmap in ['PiYG_r', 'RdBu_r', 'RdGy_r']:

                f,ax = plt.subplots(figsize=(2+n1*0.3,1+n2*0.3))
                ax.set_title('%s %s gaoRois'%(run1,run2),fontsize=8)
                im = ax.imshow(corrmat.T,origin='lower',vmin=-corrlim,vmax=corrlim,cmap=corrcmap)
                ax.set_xticks(np.arange(n1))
                ax.set_yticks(np.arange(n2))
                xtlabs = ax.set_xticklabels(feats1)
                ytlabs = ax.set_yticklabels(feats2)
                ax.set_xlabel(run_axlabeldict[run1])
                ax.set_ylabel(run_axlabeldict[run2])
                if not type(feats1[0]) == np.str_:
                    for cc in np.arange(n1):
                        xtlabs[cc].set_color(cdict_clust[run1][feats1[cc]])
                        xtlabs[cc].set_fontweight('bold')
                if not type(feats2[0]) == np.str_:
                    for cc in np.arange(n2):
                        ytlabs[cc].set_color(cdict_clust[run2][feats2[cc]])
                        ytlabs[cc].set_fontweight('bold')

                cb = f.colorbar(im)
                cb.set_label(corrmethod,rotation=-90,labelpad=10)
                ax.set_aspect('equal')
                f.tight_layout()
                figsaver(f,'%s/corr_%s__%s_cz%i_blur%i_%s'%(gao_folder,runc_lab,corrmethod,corrlim*10,sigma,corrcmap))

