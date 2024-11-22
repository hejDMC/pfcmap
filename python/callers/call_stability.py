import subprocess
import sys
import yaml

script_calc = 'stability/get_transition_stats_per_rec.py'
script_plot = 'stability/plot_stability.py'

scripts = [script_calc,script_plot]#

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

#setkeys = [key for key in S.runsets.keys() if not key.count('resp')]
setkeys = ['wwspont','nwspont']#
for key in setkeys:
    ncluststr = str(S.nclust_dict[key])
    myrun = S.runsets[key]

    for script in scripts:
        print('running %s on %s ncl %s'%(script,myrun,ncluststr))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])
