import subprocess
import sys
import yaml

script = 'enrichments/compare_nwwwIBLCarlen.py'
script = 'enrichments/compare_nwwwIBLCarlen_statsmode.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if not key.count('resp')]
#setkeys = [key for key in S.runsets.keys() if key.count('resp')]


setstr = '___'.join(setkeys)
print(setstr)
scripts = [script]


for script in scripts:
    print('running %s on %s'%(script,setstr))

    subprocess.call([sys.executable, script, pathpath,setstr])