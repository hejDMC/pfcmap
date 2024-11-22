import sys
import subprocess
import os
import yaml
import pandas as pd
import numpy as np


#whether offdet is usable will be obtained directly from offdetfile, licking is obtained directly from aRec, opto is blocked according to stimdir
script = 'preprocessing/quiet_extraction/extract_transients.py'
pathpath = 'PATHS/transient_paths.yml'

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

collection_dir = pathdict['save_dirs']['collection_dir']
nwbpath = pathdict['src_dirs']['nwbpath']
chanflav = 'regionchans'

sys.path.append(pathdict['code']['workspace'])



tablepath = 'config/datatables/allrecs_allprobes.xlsx'
sheet = 'lick_selection'
df = pd.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

tag_bool = df['run_tag'] == 'RRR'
self_bool = ~df['offdet_src'].str.contains('from-other',na=False)
bools_list = [self_bool,tag_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
recids =  list(df['recid'][cond_bool].values)
anatlists = [ [df['quietdet_regions'][cond_bool].iloc[ii]] for ii in np.arange(np.sum(cond_bool))]#list(df['on regions'][cond_bool])


print ('Spectral transient detection for %i recs'%(len(recids)))

for rr,recid in enumerate(recids):
    if anatlists[rr]==['PFC']:
        anatlist = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']
    else:
        anatlist = anatlists[rr]
    rname_trad = recid.replace('-', '_')
    collection_file = os.path.join(collection_dir, '%s__transients.h5' %rname_trad )
    print(rname_trad,anatlist)

    #if not os.path.isfile(collection_file):
    filename_g = 'NWBEXPORTPATH/XXX_export/%s__XXX.h5' %rname_trad
    subprocess.call([sys.executable, script, pathpath, filename_g, chanflav,str(anatlist)])
