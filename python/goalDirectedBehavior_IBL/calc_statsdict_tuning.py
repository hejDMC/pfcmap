import os
import numpy as np
import git
import yaml
import sys


pathpath = 'PATHS/sompaths_units_IBLpassive.yml'#
myrun = 'runIBLPasdMP3_brain_pj'
ncluststr = '8'
cmethod = 'ward'



ncl = int(ncluststr)
nshuff = 2000
signif_levels = np.array([0.05,0.01,0.001])
tuning_attr_names = ['stimTuned','choiceTuned','feedbTuned','taskResp']
blk_condition = 'detail'
pthr_cccp = 0.05
pthr_cccp_naive = 0.001
cccpthr = 0.51
pthr_taskresp = 0.001

outdir = '/results/paper/goalDirectedTuningIBL/pfcOnly/%s'%myrun#

#layers_allowed = ['5','6']
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap_paper.utils import IBLtask_funcs as iblfns
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S
from pfcmap_paper.utils import som_helpers as somh
from pfcmap_paper.utils import category_matching as matchfns

outfile_spont = os.path.join(pathdict['statsdict_dir'],'statsdictTuningVsSpont__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))
outfile_rois = os.path.join(pathdict['statsdict_dir'],'statsdictTuning_rois_iblROIs__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

savepath_gen = pathdict['savepath_gen']

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= True)#todo include not only the passive ones, we also need coordinates! just being lazy...

print('unqiue_areas',np.unique([U.area for U in Units]))

unit_filter = lambda uobj: True #S.check_layers(uobj.layer,layers_allowed)

Usel = [U for U in Units if unit_filter(U)]


#####GETTTING CLUSTERLABELS SPONT ACTIVTY: START
print('getting clusterlabels')
somfile = uloader.get_somfile(rundict, myrun, savepath_gen)
somdict = uloader.load_som(somfile)
assert (somfeats == somdict['features']).all(), 'mismatching features'
assert (weightvec == somdict['featureweights']).all(), 'mismatching weights'


get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])

# project on map
refmean, refstd = somdict['refmeanstd']
featureweights = somdict['featureweights']
dmat = np.vstack([get_features(elobj) for elobj in Usel])
wmat = uloader.reference_data(dmat, refmean, refstd, featureweights)

# set BMU for each
weights = somdict['weights']
allbmus = somh.get_bmus(wmat, weights)
for bmu, U in zip(allbmus, Usel):
    U.set_feature('bmu',bmu)

ddict = uloader.extract_clusterlabeldict(somfile, cmethod, get_resorted=True)
labels = ddict[ncl]

# categorize units
for U in Usel:
    U.set_feature('clust', labels[getattr(U, 'bmu')])
#####GETTTING CLUSTERLABELS SPONT ACTIVTY: END

#####GETTTING TUNING SIGNIFICANCE BOOLS: START
recids = np.unique([U.recid for U in Usel])
#assign tunings for stim, choice, feedback
cccp_rec_dict, dimdict = iblfns.load_cccp_data(recids, outdir, myrun)#30 recs!
iblfns.assing_tunings(Usel, cccp_rec_dict, dimdict, get_significance=True, pthr=pthr_cccp, pthr_naive=pthr_cccp_naive,
                      cccpthr=cccpthr)

#simplify tuning attributes:
def signifbool_to_tag(mybool):
    if np.isnan(mybool):
        return 'na'
    elif mybool==True:
        return 'signif'
    elif mybool==False:
        return 'ns'
    else:
        assert 0, 'unknown booltype'

for U in Usel:
    for attrname,featname in zip(['stimTuned','choiceTuned','feedbTuned'],['stimulus','choice','feedback']):

        setattr(U,attrname,signifbool_to_tag(U.tsignif[featname][blk_condition]))
#now task engagement
taskresp_dict = iblfns.load_taskresp_data(recids,outdir,myrun)
iblfns.assing_taskresp(Usel, taskresp_dict, pthr=pthr_taskresp)
for U in Usel:
    setattr(U,'taskResp',signifbool_to_tag(U.issignif_task))
#####GETTTING TUNING SIGNIFICANCE BOOLS: END

#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS SPON ACT CATEGORY: START

attr2 = 'clust'
const_attr = ['task','area']
statsdict_spont = {}
for attr1 in tuning_attr_names:
    Usubsel = [U for U in Usel if not getattr(U,attr1)=='na']
    statsdict_spont[attr1] = matchfns.get_all_shuffs(Usubsel,attr1,attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
uloader.save_dict_to_hdf5(statsdict_spont, outfile_spont)
#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS SPON ACT CATEGORY: END


#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS ROI: START
#todo re-run once the proper ROIs are there

for U in Usel:
    is_deep = S.check_layers(U.layer, ['5', '6'])
    laydepth = 'deep' if is_deep else 'sup'
    if U.roi == 0:
        U.set_feature('roilaydepth', 'na')
    else:
        U.set_feature('roilaydepth', '%s|%s' % (str(U.roi), laydepth))

attr1 = 'roilaydepth'
const_attr = 'task'
statsdict_rois = {}

for attr2 in tuning_attr_names:
    Usubsel = [U for U in Usel if not getattr(U,attr2)=='na' and not U.roilaydepth=='na']
    statsdict_rois[attr2] = matchfns.get_all_shuffs(Usubsel,attr1,attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
uloader.save_dict_to_hdf5(statsdict_rois, outfile_rois)

#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS ROI: END
