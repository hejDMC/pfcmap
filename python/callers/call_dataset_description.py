import subprocess
import sys
import yaml

script_unit_couting = 'dataset_general/unit_counts.py'
script_feature_correlations = 'dataset_general/feature_correlations.py'
script_paraminteraction = 'dataset_general/parameter_interactions.py'

scripts = [script_unit_couting,script_feature_correlations]
scripts = [script_paraminteraction]#script_gao,
scripts = [script_unit_couting]#script_gao,

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]


for key in setkeys:
    myrun = S.runsets[key]
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        if script == script_paraminteraction and key.count('IBL'):
            pass
        else:
            print('running %s on %s'%(script,myrun))
            subprocess.call([sys.executable, script, pathpath,myrun])
