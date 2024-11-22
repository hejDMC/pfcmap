import subprocess
import sys
import yaml

script = 'flatmaps/plot_flat_zeta_responder_and_hierarchy.py'
scripts = [script]#[script_calc]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['wwspont','nwspont']

boolstr_dict = {'gaoRois':'0','dataRois':'1'}

for key in setkeys:
    myrun = S.runsets[key]
    ncluststr = str(S.nclust_dict[key])
    pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        for roi_tag,get_rois_bool_str in boolstr_dict.items():
            print('running %s on %s with %s'%(script,myrun,roi_tag))
            subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod,get_rois_bool_str,roi_tag])
