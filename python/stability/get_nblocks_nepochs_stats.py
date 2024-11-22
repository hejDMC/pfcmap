import yaml
import sys
import os
import h5py
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile as sap

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

"""
pathpath = 'PATHS/filepaths_carlen.yml'
myrun =  'runC00dMI3_brain'
cmethod = 'ward'
ncluststr = '5'
"""

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import evol_helpers as ehelpers
from pfcmap.python import settings as S

statstag = 'TSELprestim3__STATEactive__all'
savepath_evol = os.path.join(pathdict['savepath_gen'].replace('SOMs','category_evolutions'))
genfigdir = pathdict['figdir_root'] + '/stability'

tasks_sorted = ['Passive','Attention','Aversion_ctrl','Aversion_stim','Context','Detection']#later in the script, it will be checked whether those are all available
ncl = int(ncluststr)

targetdir = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl))
if not os.path.isdir(targetdir): os.makedirs(targetdir)


recids = S.get_allowed_recids_from_table(tablepath=pathdict['tablepath'],sheet='sheet0',not_allowed=['-', '?'])

###calculating stuff

tempdict = {}
not_available = []
for recid in recids:
    evolfile = os.path.join(savepath_evol, '%s__%s__%s%i__%s__catevols.h5' % (recid, myrun, cmethod, ncl, statstag))
    if os.path.isfile(evolfile):
        with h5py.File(evolfile) as statshand:
            tempdict[recid] = {key.replace('/',''):val for key,val in uloader.unpack_statshand(statshand, remove_srcpath=True).items()}
    else:
        not_available += [recid]

tasktypes = np.unique([tempdict[recid]['task_full'] for recid in tempdict.keys()])

if set(tasks_sorted)==set(tasktypes):
    pass
else:
    tasks_sorted = [task for task in tasks_sorted if task in tasktypes]
    tasks_sorted += [task for task in tasktypes if not task in tasks_sorted]
    print('WARNING manual task sorting set did not match present task set')

datadict = {}
for tasktype in tasktypes:
    myrecs = [recid for recid in tempdict.keys() if tempdict[recid]['task_full']==tasktype]
    datadict[tasktype] = {recid:tempdict[recid] for recid in myrecs}



trials_per_block_min = 15 #that is overall!!!!
nvals_min = 12

epb_vec = np.array([])
tpb_vec = np.array([])
nb_vec = np.array([])
nbpu_vec = np.array([])
for tasktype in tasktypes:
    myrecs = list(datadict[tasktype].keys())
    for recid in myrecs:
        tpb_u = datadict[tasktype][recid]['trials_per_block_units']#3 x nunits
        tpb = datadict[tasktype][recid]['trials_per_block']
        tpb_overall = tpb[tpb>trials_per_block_min]
        tpb_vec = np.hstack([tpb_vec,tpb_overall])
        nblocks_overall = len(tpb_overall)
        nb_vec = np.hstack([nb_vec,nblocks_overall])
        n_blocks_per_unit = np.array([np.sum(udata>=nvals_min) for udata in tpb_u[tpb>trials_per_block_min].T])
        nbpu_vec = np.hstack([nbpu_vec,n_blocks_per_unit])
        n_epochs_per_block = np.hstack([ udata[udata>nvals_min] for udata in tpb_u[tpb>trials_per_block_min].T])
        epb_vec = np.hstack([epb_vec,n_epochs_per_block])




scoredict = {}
percs = [0,10,25,50,75,90,100]
for tag,myvec in zip(['epochs_per_block [units]','epochs_per_block [recs]','N blocks [units]','N blocks [recs]'],\
                     [epb_vec,tpb_vec,nbpu_vec,nb_vec]):
    scoredict[tag] = {}
    for perc in percs:
        scoreval = sap(myvec,perc,interpolation_method='lower')
        scoredict[tag][perc] = int(scoreval)
df = pd.DataFrame.from_dict(scoredict).transpose()

outfile = os.path.join(targetdir,'percstats_blocks_and_epochs__%s_ncl%i.xlsx'%(myrun,ncl))
with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='blocks_and_epochs')

