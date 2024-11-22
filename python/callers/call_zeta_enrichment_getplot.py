import subprocess
import sys
import yaml

script = 'enrichments/plot_zeta_responder_enrichment.py'

scripts = [script]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['wwspont','nwspont']

for key in setkeys:
    myrun = S.runsets[key]
    ncluststr = str(S.nclust_dict[key])
    for script in scripts:
        print('running %s on %s'%(script,myrun))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])