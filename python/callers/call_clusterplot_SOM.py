import subprocess
import sys
import yaml

script_plot = 'SOM/clusterSOM/clusterplot_SOM.py'
script_labels_spont = 'SOM/clusterSOM/plot_clusterlabels.py'
script_labels_resp = 'SOM/clusterSOM/plot_clusterlabels_response.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['wwspont']


scripts = [script_plot,'labelscript']


for key in setkeys:
    myrun = S.runsets[key]
    ncluststr = str(S.nclust_dict[key])
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        if script == 'labelscript':
            script = script_labels_resp if key.count('resp') else script_labels_spont
        print('running %s on %s'%(script,myrun))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])