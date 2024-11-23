
import yaml
import os
import sys
import h5py
import numpy as np
from numpy import array

pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZI_brain'
cmethod = 'ward'


label_sorter = {'runC00dMP3_brain':{9:np.array([5,1,7,2,3,4,8,9,6])-1,\
                                    8:np.array([4,1,6,2,3,7,8,5])-1,\
                                    7:np.array([3,1,5,2,6,7,4])-1,\
                                    6:np.array([4,1,2,5,6,3])-1},\
                'runC00dMI3_brain':{9:np.array([1,5,6,2,3,7,9,8,4])-1,\
                                    8:np.array([1,5,6,2,3,8,7,4])-1,\
                                    7:np.array([1,4,5,2,7,6,3])-1,\
                                    6:np.array([1,2,4,6,5,3])-1,\
                                    5:np.array([1,2,4,5,3])-1},\

                'runCrespZP_brain':{10: array([3, 6, 7, 9, 8, 1, 2, 5, 4, 0]),\
                                    9: array([3, 5, 6, 8, 7, 1, 4, 2, 0]),\
                                    8: array([3, 5, 6, 7, 1, 4, 2, 0]), \
                                    7: array([3, 5, 6, 1, 4, 2, 0]), \
                                    6: array([3, 4, 5, 2, 1, 0]), \
                                    5: array([3, 4, 2, 1, 0])},\

                'runCrespZI_brain':{10: array([8, 5, 7, 4, 9, 1, 2, 6, 3, 0]), \
                                    9: array([7, 4, 6, 3, 8, 1, 5, 2, 0]), \
                                    8: array([6, 3, 4, 7, 1, 5, 2, 0]), \
                                    7: array([4, 3, 6, 1, 5, 2, 0]), \
                                    6: array([3, 2, 5, 4, 1, 0]), \
                                    5: array([3, 2, 4, 1, 0])}


}
'''

'''

if not myrun in label_sorter:
    sys.exit()

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)
somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)


with h5py.File(somfile, 'r') as hand:
    ddict = {int(key): vals[()] for key, vals in hand['clustering/%s' % cmethod].items()}


new_labeldict = label_sorter[myrun]
with h5py.File(somfile, 'r+') as hand:
    #cgrp = hand['clustering/%s' % cmethod]
    regrp_name = 'clustering_resorted/%s'%cmethod
    if regrp_name in hand:
        del hand[regrp_name]
    regrp = hand.create_group('clustering_resorted/%s'%cmethod)

    for ncl,sortarray in new_labeldict.items():
        print('writing resorting %s %s ncl%i'%(somfile,regrp_name,ncl))
        labels_orig = ddict[ncl]
        labels = np.zeros_like(labels_orig)
        for newval, oldval in enumerate(sortarray):
            labels[labels_orig == oldval] = newval
        #print('Ncl%i orig:%s, new:%s'%(ncl,np.unique()))
        ds = regrp.create_dataset(str(ncl),data=labels,dtype='i')

#ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)