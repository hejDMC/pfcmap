import subprocess
import sys
import yaml

script = 'SOM/clusterSOM/plot_psth_and_zeta.py'

scripts = [script]#

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]
for key in setkeys:
    ncluststr = str(S.nclust_dict[key])
    myrun = S.runsets[key]

    for script in scripts:
        print('running %s on %s ncl %s'%(script,myrun,ncluststr))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])
