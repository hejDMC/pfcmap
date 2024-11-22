import subprocess
import sys
import yaml

script = 'flatmaps/plot_flatmap_countstats.py'
scripts = [script]#[script_calc]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]#
#setkeys = ['wwspont','nwspont']

roi_tags = ['gaoRois','dataRois']

for key in setkeys:
    myrun = S.runsets[key]
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        for roi_tag in roi_tags:
            print('running %s on %s with %s'%(script,myrun,roi_tag))
            subprocess.call([sys.executable, script, pathpath,myrun,roi_tag])

