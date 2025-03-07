import os
import numpy as np
import yaml
import sys


pathpath = 'PATHS/filepaths_IBLTuning.yml'#
myrun =  'runIBLTuning_utypeP'

layers_allowed = ['5','6']

#roi_file = 'IBLflatmap_PFC_ntesselated_obeyRegions_res200.h5'
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
from pfcmap_paper.utils import category_matching as matchfns

#my_roimappath = os.path.join(os.path.dirname(S.roimap_path),roi_file)

outfile_rois = os.path.join(pathdict['statsdict_dir'],'statsdictTuning_rois_iblROIs__%s.h5'%(myrun))

savepath_gen = pathdict['savepath_gen']

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])

Units, somfeats, weightvec = uloader.load_all_units_for_som(metricsfiles, rundict, check_pfc=True,check_wavequality=False)



print('unqiue_areas',np.unique([U.area for U in Units]))

unit_filter = lambda uobj: S.check_layers(uobj.layer,layers_allowed)

Usel = [U for U in Units if unit_filter(U)]

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
#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS ROI: START

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
    Usubsel = [U for U in Usel if not getattr(U,attr2)=='na' and not U.roilaydepth=='na' and not np.sum(np.abs([U.u,U.v]))==0]
    statsdict_rois[attr2] = matchfns.get_all_shuffs(Usubsel,attr1,attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
uloader.save_dict_to_hdf5(statsdict_rois, outfile_rois)

#####CACLCULATING AND SAVING ENRICHMENT STATSDICT FOR TUNING VS ROI: END
