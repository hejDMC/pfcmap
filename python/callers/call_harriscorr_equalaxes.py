import subprocess
import sys
import yaml

script = 'connectivity_correlations/harris_vs_enrichment_ENHequalaxes.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

#setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['nwspont','nwIBL']#['wwspont','wwIBL']
runset_collections = [[key for key in S.runsets.keys() if not key.count('resp') and key.count(utype)] for utype in ['nw','ww']]

for runset in runset_collections:
    setstr = '___'.join(runset)
    print(setstr)
    print('running %s on %s'%(script,setstr))

    subprocess.call([sys.executable, script, pathpath,setstr])

