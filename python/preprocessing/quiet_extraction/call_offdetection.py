#serves both for detection and plotting
import sys
import subprocess
from glob import glob
import os
import yaml
import numpy as np
import pandas as pd

script_det = 'preprocessing/quiet_extraction/run_offperiod_detection.py'
script_plot = 'preprocessing/quiet_extraction/run_offperiod_plotting.py'#'routines/run_offperiod_detection.py'upstateheightCorr_quick.py
pathpath = 'PATHS/offstate_paths.yml'


tablepath = 'config/datatables/allrecs_allprobes.xlsx'
sheet = 'lick_selection'
df = pd.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

cond_bool = df['run_tag'] == 'RRR'
recids =  list(df['recid'][cond_bool].values)
anatlists = [ [df['quietdet_regions'][cond_bool].iloc[ii]] for ii in np.arange(np.sum(cond_bool))]#list(df['on regions'][cond_bool])

data_dir = 'NWBEXPORTPATH/XXX_export/YYY__XXX.h5'
for rr,recid in enumerate(recids):
     if anatlists[rr]==['PFC']:
          anatlist = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']
     else:
          anatlist = anatlists[rr]
     print(recid,anatlist)
     filename_g = data_dir.replace('YYY',recid.replace('-','_'))
     for script in [script_det,script_plot]:
          #subprocess.call([sys.executable,script, pathpath, filename_g,str(anatlist)])
          subprocess.call([sys.executable,script, pathpath, filename_g,str()])

#N.B.: for PFC recordings the regs of interest ("anatlist") are always:['ACA', 'ILA', 'MOs', 'ORB', 'PL']

