import sys
import yaml
import os
import numpy as np


pathpath,myrun,ncluststr,cmethod = sys.argv[1:]


signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000


savegendspath = '%s/%s'%(cmethod,ncluststr)


ncl = int(ncluststr)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']
outfile = os.path.join(pathdict['statsdict_dir'],'statsdictSpecCtrl__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))


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

uloader.assign_parentregs(Units,pfeatname='preg',cond=lambda uobj: True,na_str='na')
Units_ctx = np.array([U for U in Units if U.preg in S.ctx_regions+['PFC']])

layers_allowed_in_ctx = ['5','6']

Us_filtered = []
for U in Units:
    if U.preg in S.ctx_regions+['PFC']:
        if S.check_layers(U.layer,layers_allowed_in_ctx):
            genarea = U.area if S.check_pfc(U.area) else U.preg
            U.set_feature('genArea','%s|d'%genarea)
            Us_filtered += [U]
    else:
        if U.preg!='na':
            U.set_feature('genArea',str(U.preg))
            Us_filtered += [U]

print('Unique genAreas',np.unique([U.genArea for U in Us_filtered]))

Us_filtered_pfc_only = [U for U in Us_filtered if S.check_pfc(U.area)]
print('Unique genAreas PFC',np.unique([U.genArea for U in Us_filtered_pfc_only]))



supertags = ['tasks','recid','layers']

reftags = ['refall','refPFC']
statsdict = {supertag:{subtag:{} for subtag in reftags} for supertag in supertags}
attr2 = 'clust'

for reftag, Usel in zip(['refall','refPFC'],[Us_filtered,Us_filtered_pfc_only]):

    ###  TASKS: QUESTION: If we disregard area, is something more enriched in certain tasks?
    #### --> shuffle areas; exclude area='na' in the general selection; use only layer 56 in ctx
    const_attr = 'genArea' #only not used in attr1=task!
    mystats = matchfns.get_all_shuffs(Usel,'task',attr2,const_attr='genArea',nshuff=nshuff,signif_levels=signif_levels)
    mystats['const_attr_subsel'] =     np.unique([getattr(U,const_attr) for U in Usel])
    statsdict['tasks'][reftag]['area_ctrl'] = mystats

    ###recids: QUESTION: If we disregard task and area, are some recordings particularly enriched in something?
    #### --> shuffle areas and tasks; exclude area='na' in the general selection; use only layer 56 in ctx
    const_attr=['genArea','task']
    mystats = matchfns.get_all_shuffs(Usel,'recid',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    unique_combs =  np.unique(['__'.join([getattr(U,my_ca) for my_ca in const_attr]) for U in Usel])
    mystats['const_attr_subsel'] =    unique_combs
    statsdict['recid'][reftag]['area_task_ctrl'] = mystats


#layers: QUESTION If we disregard task and area, are some layers particularly enriched in something?
#### --> shuffle areas and tasks, exclude area='na' in the general selection; use only layers in ctx


Us_filtered_ctx = []
for U in Units:
    if U.preg in S.ctx_regions + ['PFC']:
        genarea = U.area if S.check_pfc(U.area) else U.preg
        U.set_feature('genAreaCtx', '%s' % genarea)
        Us_filtered_ctx += [U]


print('Unique genAreaCtx',np.unique([U.genAreaCtx for U in Us_filtered_ctx]))

Us_filtered_ctx_pfc_only = [U for U in Us_filtered_ctx if S.check_pfc(U.area)]
print('Unique genAreaCtx PFC',np.unique([U.genAreaCtx for U in Us_filtered_ctx_pfc_only]))

for reftag, Usel in zip(['refall','refPFC'],[Us_filtered_ctx,Us_filtered_ctx_pfc_only]):
    const_attr= ['genAreaCtx','task']
    mystats = matchfns.get_all_shuffs(Usel,'layer',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    unique_combs =  np.unique(['__'.join([getattr(U,my_ca) for my_ca in const_attr]) for U in Usel])
    mystats['const_attr_subsel'] =    unique_combs
    statsdict['layers'][reftag]['area_task_ctrl'] = mystats

uloader.save_dict_to_hdf5(statsdict, outfile,strtype='S30')#S20 to accomodate for the long unique_combs and the recids

#tood just plot it now






