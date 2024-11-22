import subprocess
import sys
import yaml

script_compare = 'enrichments/compare_datasets.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if not key.count('resp')]

setkeys_pairs = [['wwspont','wwIBL'],['nwspont','nwIBL']]
stags = ['Carlen','IBL']
pathpath = 'PATHS/filepaths_carlen.yml'

scripts = [script_compare]

for key_pair in setkeys_pairs:
    key1,key2 = key_pair
    run1 = S.runsets[key1]
    run2 = S.runsets[key2]
    ncluststr = str(S.nclust_dict[key1])
    for script in scripts:
        print('running %s on %s and %s'%(script,run1,run2))

        subprocess.call([sys.executable, script, pathpath,run1,run2,stags[0],stags[1], ncluststr,S.cmethod])