import subprocess
import sys
import yaml

script_compare = 'enrichments/compare_nw_ww.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if not key.count('resp')]

setkeys_pairs = [['wwspont','nwspont'],['wwIBL','nwIBL']]
setkeys_pairs = [['wwresp','nwresp']]
stags = ['ww','nw']




scripts = [script_compare]

for key_pair in setkeys_pairs:
    key1,key2 = key_pair
    run1 = S.runsets[key1]
    run2 = S.runsets[key2]
    ncl1 = str(S.nclust_dict[key1])
    ncl2 = str(S.nclust_dict[key2])
    for script in scripts:
        print('running %s on %s and %s'%(script,run1,run2))

        if key1.count('IBL'):
            pathpath = 'PATHS/filepaths_IBL.yml'
        else:
            pathpath = 'PATHS/filepaths_carlen.yml'

        subprocess.call([sys.executable, script, pathpath,run1,run2,stags[0],stags[1], ncl1,ncl2,S.cmethod])