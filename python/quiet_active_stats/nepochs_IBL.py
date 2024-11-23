import sys
import yaml
import os
import h5py
import pandas as pd
from scipy.stats import scoreatpercentile as sap
import numpy as np
from glob import glob


pathpath = 'PATHS/filepaths_IBL.yml'

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

metrics_dir =  pathdict['src_dirs']['metrics']

tintfilepath = os.path.join(S.metricsextr_path, 'timeselections', 'timeselections_IBL_Passive')


metrfiles =  glob(os.path.join(metrics_dir,'*'))

recids = [os.path.basename(mfile).replace('metrics_','').replace('.h5','') for mfile in metrfiles]


tintdict = {}
startdict = {}
for recid in recids:
    tintfile = os.path.join(tintfilepath,'%s__TSELpassive3.h5'%recid)

    if os.path.isfile(tintfile):
        with h5py.File(tintfile, 'r') as hand:
            tints = hand['tints'][()]
            ntints = tints.shape[0]
            starttime = tints[0,0]
    else:
        ntints = 0
        starttime = np.nan
        print('%s: tintfile %s not found, setting ntints=0' % (recid, ntints))
    tintdict[recid] = ntints
    startdict[recid] = starttime/60.

allstarts = list(startdict.values())
print([sap(allstarts,perc) for perc in  [25,50,75]])