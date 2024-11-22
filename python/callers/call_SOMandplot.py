import subprocess
import sys
import yaml

script_SOM = 'SOM/calc_SOM.py'
script_somplot = 'SOM/plot_SOM.py'
script_brainplot = 'SOM/plot_SOM_brain.py'
script_hitplot = 'SOM/plot_SOMhits_resolved.py'
script_clusterSOM = 'SOM/clusterSOM/cluster_SOM.py'
script_nclust = 'SOM/clusterSOM/identify_nclust_SOM.py'
script_examples = 'SOM/plot_example_imetrics_in_hex.py'
Carlen_only_scripts = [script_SOM,script_nclust,script_clusterSOM]
scripts = [script_SOM,script_somplot,script_brainplot,script_hitplot,script_clusterSOM,script_nclust]
#scripts = [script_examples]
scripts = [script_SOM,script_somplot,script_brainplot,script_hitplot,script_clusterSOM,script_nclust]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['wwspont']

for key in setkeys:
    myrun = S.runsets[key]
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        if not (script in Carlen_only_scripts and myrun.count('IBL')):
            print('running %s on %s'%(script,myrun))

            subprocess.call([sys.executable, script, pathpath,myrun])
