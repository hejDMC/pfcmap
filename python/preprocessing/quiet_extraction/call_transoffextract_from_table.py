import sys
import subprocess

import yaml
import pandas as pd
import numpy as np

script = 'preprocessing/quiet_extraction/extract_from_off_and_trans.py'


pathpath = 'PATHS/merge_quiet_paths.yml'

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

sys.path.append(pathdict['code']['workspace'])



tablepath = 'config/datatables/allrecs_allprobes.xlsx'
sheet = 'lick_selection'
df = pd.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

#tag_bool = df['run_tag'] == 'R'
isrec_bool = df['recid'].str.contains('probe',na=False)
allowed_bool = df['run_tag'].isin(['RRR','R'])
self_bool = df['quietdet_src'].str.contains('self',na=False)
bools_list = [self_bool,allowed_bool,isrec_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
recids =  list(df['recid'][cond_bool].values)
anatlists = [ [df['quietdet_regions'][cond_bool].iloc[ii]] for ii in np.arange(np.sum(cond_bool))]#list(df['on regions'][cond_bool])
srclist = list(df['quietdet_src'][cond_bool].values)
print ('Obtaining quiet episodes from LFP (transients) and spikes (OFF) %i'%(len(recids)))



for rr,recid in enumerate(recids):
    qsrc = srclist[rr]
    if anatlists[rr] == ['PFC']:
        anatlist = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']
    else:
        anatlist = anatlists[rr]

    rname_trad = recid.replace('-', '_')

    print ('###DOING %s %s'%(recid,str(anatlist)))
    subprocess.call([sys.executable, script,rname_trad,pathpath,str(anatlist),qsrc])
#N.B.: for PFC recordings the regs of interest ("anatlist") are always:['ACA', 'ILA', 'MOs', 'ORB', 'PL']
