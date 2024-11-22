import sys
import subprocess
from glob import glob
import os
import yaml

lick_exptypes = ['Detection']#'Detection',,'Association' --> association sucks, ditch it!
script1 = 'preprocessing/lick_extraction/run_licktrace_icadet.py'
scripts = [script1]#script1,

pathpath = 'PATHS/artifact_paths.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
nwbpath = 'DANDIPATH'#pathdict['src_dirs']['nwbpath']

lfppath = pathdict['src_dirs']['lfppath']
genpath = lfppath.replace('lfp','XXX')

my_exports = glob(os.path.join(lfppath,'*'))
export_ids = [os.path.basename(myfile).split('__')[0] for myfile in my_exports]

for exptype in lick_exptypes:
    print ('#### EXPTYPE %s'%exptype)
    nwbfiles = glob(os.path.join(nwbpath, exptype,'*.nwb'))
    nwbids = [os.path.basename(file).replace('-', '_').split('.')[0] for file in nwbfiles]#\if not os.path.basename(file).split('_')[0] in ['PL033','PL034']]
    recids = [recid for recid in nwbids if recid in export_ids]
    for recid in recids:
        recfilename = os.path.join(genpath,'%s__XXX.h5' % recid)
        for script in scripts:
            subprocess.call([sys.executable, script, pathpath, recfilename])


