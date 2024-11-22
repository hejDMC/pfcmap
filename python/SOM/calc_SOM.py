import sys
import yaml
import os
import numpy as np




pathpath,myrun = sys.argv[1:]
#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runCrespZP_brain'

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python.utils import my_somz as somz
from pfcmap.python import settings as S
from pfcmap.python.utils import unitloader as uloader


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

#reftag,layerlist,datasets,tasks,tsel,statetag,utypes,nnodes = [rundict[key] for key in ['reftag','layerlist','datasets','tasks','tsel','state','utypes','nnodes']]
#wmetrics,imetrics,wmetr_weights,imetr_weights = [rundict[metrtype][mytag] for mytag in ['features','weights'] for metrtype in ['wmetrics','imetrics']]#not strictly necessary: gets called in uloader



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])

Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)





recids = np.unique([U.recid for U in Units])

get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])

#somUnits = [U for U in Units if check_fullfeat(U)]
dmat = np.vstack([get_features(elobj) for elobj in Units])
refmean,refstd = dmat.mean(axis=0),dmat.std(axis=0)#save for later to show data
zmat = (dmat-refmean[None,:])/refstd[None,:]# standardize
wmat = zmat*weightvec[None,:]#wheight

#somh.reference_datamat(datamat,refmean,refstd,**kwargs) --> to reference things

kshape = somh.get_kshape(wmat, nneurons=rundict['nnodes'])#som dimension

# now calculate the SOM!
startweights = somh.getInitialMap_pca(wmat, kshape[::-1])
# print 'rep: ',ii
map = somz.SelfMap(wmat, [], topology='hex', Ntop=kshape, iterations=S.niter, som_type='batch')
# now we have map.distLib etc
map.create_map(inputs_weights=startweights)
weights = map.weights
qe = somh.get_quantisationError(wmat, weights)
te = somh.get_topologicalError(wmat, weights, kshape)



somsavepath = os.path.join(savepath_gen,'%s_SOM_kShape%i_%i.h5'%(myrun,kshape[0],kshape[1]))
uloader.save_som(somsavepath,myrun,rundict,kshape,weights,weightvec,refmean,refstd,somfeats,qe,te,recids,S.niter)
#somdict = uloader.load_som(somsavepath)
