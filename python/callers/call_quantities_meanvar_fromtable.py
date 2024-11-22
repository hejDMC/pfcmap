import subprocess
import sys
import yaml
import os
from glob import glob
import pandas as pd
import numpy as np


print ('starting script Meanvar Quantities!')
pathpath = 'PATHS/general_paths.yml'
with open(pathpath,'r') as f: pdict = yaml.safe_load(f)
genpath = pdict['datapath_metricsextraction']


script = 'metrics_extraction/spontaneous/extract_quantities_meanvar.py'
logfile =  '%s/ts_meanvar.log'%pdict['logfolder']


ttags = ['prestim']

timeselfolder = 'timeselections_quietactive' #'timeselections'
statetags = ['active','quiet']
#reftags = ['__full']#['','__all']
reftags = ['__all','']
missing_tints = []
prestim_dur = 3.

tintfilepath = os.path.join(genpath,timeselfolder)
nwbdir = pdict['nwbdir']
dstpath = os.path.join(genpath,'quantities_all_meanvar','Carlen_quietactive')
donelist = []


tablepath = 'config/datatables/allrecs_allprobes.xlsx'
#recids_ok,allrecs,df = fhs.read_ok_recs_from_excel(tablepath,sheetname='sheet0',not_allowed=['-', '?'],output='verbose')



#not_allowed=['-', '?']
sheet = 'sheet0'
df = pd.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
allrecrows = df['recid']
isrec_bool = allrecrows.str.contains('probe',na=False)
allowed_bool = ~df['usable_gen'].isin(['?','-'])#df['usable_gen'].isin(['run'])
bools_list = [allowed_bool,isrec_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
recids = list(allrecrows[cond_bool].values)
datasetlabs = np.unique(df['datasetlab'][cond_bool])
exptypelist = list(df['exptype'][cond_bool])

exptypes = np.unique(exptypelist)
nwb_pool = np.hstack([np.hstack([np.array(glob(os.path.join(nwbdir,dslab,exptype,'*.nwb'))) for exptype in exptypes]) for dslab in datasetlabs])



n_recs = len(recids)



for rr,recid in enumerate(recids):
    print('\n\n ####    PROCESSING %s %i/%i     ####\n'%(recid,rr+1,n_recs))
    #recid = recids[rr]
    exptype = exptypelist[rr]
    nwb_file = [fname for fname in nwb_pool if os.path.basename(fname).count(recid)]
    assert len(nwb_file) == 1, 'not exactly one matching nwbfile, len=%i'%(len(nwb_file))
    nwb_file = nwb_file[0]
    for ttag in ttags:
        print ('##%s ##'%(ttag))
        for statetag in statetags:
            for reftag in reftags:
                tintfile =  os.path.join(tintfilepath, '%s__TSEL%s%i__STATE%s%s.h5' % (recid, ttag,prestim_dur, statetag,reftag))
                if os.path.isfile(tintfile): #and (tintfile.count('518804_2022') or tintfile.count('560171_2022'))
                    print('%s calling: %s tsel: %s'%(os.path.basename(logfile).split('.')[0],recid,tintfile))

                    subprocess.call([sys.executable,script,nwb_file,pathpath,ttag,dstpath,tintfile,logfile,exptype])
                    donelist += [tintfile]
                else:
                    print('NOTINTFILE', tintfile)
                    missing_tints += [tintfile]




