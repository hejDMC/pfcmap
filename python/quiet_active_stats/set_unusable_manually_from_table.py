
import h5py
import os
import yaml
import pandas
import numpy as np



tablepath = 'config/datatables/allrecs_allprobes.xlsx'
sheet = 'lick_selection'
df = pandas.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

allowed_bool = ~df['usable_gen'].isin(['-','?'])
tag_bool = df['run_tag'] == 'RRR'

bools_list = [allowed_bool,tag_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)


recids = df['recid'][cond_bool].values
offdet_qualities = df['offdet_q'][cond_bool].values
offdet_srcs = df['offdet_src'][cond_bool].values

pathpath = 'PATHS/offstate_paths.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
outdir = pathdict['outdir']


#unusable_recs = ['259112_20200914_probe0','258419_20200921_probe0']
for recid,osrc in zip(recids,offdet_srcs):
    outfile = os.path.join(outdir,'%s__offstates.h5'%(recid.replace('-','_')))

    if osrc.strip() != 'self':
        print(recid,'not_usable')
        tag = 'no'
    else:
        tag = 'yes'
        print(recid,'USABLE')


    with h5py.File(outfile, 'r+') as fdest:
        fdest.attrs['usable'] = tag
        fdest.attrs['offdet_src'] = osrc

