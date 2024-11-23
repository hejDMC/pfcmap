#save clustering labels in som's hdf5

import sys
import yaml
import os
import numpy as np
import h5py


pathpath,myrun = sys.argv[1:]
#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runC00dMP3_brain'

nclustvec = np.arange(2,11)
overwrite_cmethod = True

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import clustering as cfns



rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)


somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

weights = somdict['weights']
somfeats = list(somdict['features'])

X = weights.T

#print('XXXX',somfeats)
if 'Rho' in somfeats:
    sortfeat = 'Rho'
elif 'Rho_mean' in somfeats:
    sortfeat = 'Rho_mean'
elif 'LvR' in somfeats:
    sortfeat = 'LvR'
elif 'LvR_mean' in somfeats:
    sortfeat = 'LvR_mean'
elif 'M_mean' in somfeats:
    sortfeat = 'M_mean'
elif 'peak2Trough' in somfeats:
    sortfeat = 'peak2Trough'
else:
    sortfeat = somfeats[0]
#sortfeat = 'LvR' if 'LvR' in somfeats else 'peak2Trough'

print('Clustering sortfeat: %s'%sortfeat)
cfndict = cfns.get_cfn_dict()
for cmethod,clusterfn in cfndict.items():
    #cmethod = 'ward'
    clusterfn = cfndict[cmethod]
    cm_lab = 'clustering/'+cmethod

    if overwrite_cmethod:
        with h5py.File(somfile,'r+') as hand:
            if cm_lab in hand: del hand[cm_lab]

    for ncl in nclustvec:
        ncl_lab = str(ncl)

        labels0 = clusterfn(ncl,X)
        labels = cfns.sort_labels(labels0,X,somfeats,sortfeat,avgfn=lambda x:-np.median(x))
        #labels = cfns.sort_labels_featsimilarity(labels0,X,somfeats,sortfeat,avgfn=lambda x:np.median(x,axis=0),mode='low_to_high')
        with h5py.File(somfile,'r+') as hand:
            cm_hand = hand[cm_lab] if cm_lab in hand else hand.create_group(cm_lab)
            if ncl_lab in cm_hand:
                del cm_hand[ncl_lab]
            ds = cm_hand.create_dataset(ncl_lab,data=labels,dtype='i')
            ds.attrs['sortfeat'] = str(sortfeat)


