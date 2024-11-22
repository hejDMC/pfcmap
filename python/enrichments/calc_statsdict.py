import sys
import yaml
import os
import numpy as np
import re


pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

#myrun = 'runC00dMI3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'
#myrun = 'runIBLPasdM3_brain_pj'
#pathpath = 'PATHS/filepaths_IBL.yml'#
#myrun = 'runIBLPasdMP3_brain_pj'

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
#myrun = 'runCrespZP_brain'#
ncluststr = '8'
cmethod = 'ward'
'''


#kickout_lay3pfc = True
signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000


savegendspath = '%s/%s'%(cmethod,ncluststr)


ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']
outfile = os.path.join(pathdict['statsdict_dir'],'statsdict__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python.utils import category_matching as matchfns



rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_gen)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')))

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



ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
labels = ddict[ncl]




#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])

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
        U.set_feature('arealay','%s|%s'%(U.area,laysimple))
        U.set_feature('layctx',laysimple)
        if U in Units_pfc: U.set_feature('roilay','%s|%s'%(str(U.roi),laysimple))
        else: U.set_feature('roilay',str(U.area))
    else:
        U.set_feature('arealay',str(U.area)) #there can be a few ones in PFC that have no layer assigned
        U.set_feature('roilay',str(U.area))
        U.set_feature('layctx','na')


#allroilays = [U.arealay for U in Units]
#myUs = [U for U in Units if U.arealay=='ACAd|5']

for U in Units:
    if U in Units_ctx:
        is_deep = S.check_layers(U.layer, ['5', '6'])
        laydepth = 'deep' if is_deep else 'sup'
        U.set_feature('arealaydepth', '%s|%s' % (U.area, laydepth))
        if U in Units_pfc:U.set_feature('roilaydepth', '%s|%s' % (str(U.roi), laydepth))
        else: U.set_feature('roilaydepth', str(U.area))
    else:
        U.set_feature('arealaydepth', str(U.area))
        U.set_feature('roilaydepth', str(U.area))

for U in Units: #just to make the shuffling function work when nothing should stay constant for shuffling
    U.set_feature('fake_const','a')


#const_attr='task'
#attr1 = 'roistr'
#attr2 = 'clust'
#Units_pfc56 = [U for Units in Units_pfc if  S.check_layers(U.layer, ['5','6'])]



#laydepth: split into superficical (123) and deep (56)
supertags = ['rois','regs','tasks','utype','recid','layers']

reftags = ['refall','refPFC']
statsdict = {supertag:{subtag:{} for subtag in reftags} for supertag in supertags}
attr2 = 'clust'
const_attr = 'task' #only not used in attr1=task!

for reftag, Usel in zip(['refall','refPFC'],[Units,Units_pfc]):
    statsdict['rois'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'roistr',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['rois'][reftag]['lays'] = matchfns.get_all_shuffs(Usel,'roilay',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['rois'][reftag]['laydepth'] = matchfns.get_all_shuffs(Usel,'roilaydepth',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['regs'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'area',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['regs'][reftag]['lays'] = matchfns.get_all_shuffs(Usel,'arealay',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['regs'][reftag]['laydepth'] = matchfns.get_all_shuffs(Usel,'arealaydepth',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['tasks'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'task',attr2,const_attr='fake_const',nshuff=nshuff,signif_levels=signif_levels)
    statsdict['utype'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'utype',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['recid'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'recid',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['layers'][reftag]['justlays'] = matchfns.get_all_shuffs(Usel,'layctx',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)

uloader.save_dict_to_hdf5(statsdict, outfile)
