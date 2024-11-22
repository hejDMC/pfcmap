import subprocess
import sys
import yaml

script_gao = 'connectivity_correlations/gao_vs_enrichment.py'
script_harris = 'connectivity_correlations/harris_vs_enrichment.py'

script_harris_zeta = 'connectivity_correlations/harris_vs_zeta_enrichment.py'
script_gao_modules = 'connectivity_correlations/gao_module_hierarchy.py'

scripts = [script_gao]#script_gao,

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

#setkeys = [key for key in S.runsets.keys() if key.count('resp')]
#setkeys = ['nwspont','nwIBL']#['wwspont','wwIBL']
setkeys = [key for key in S.runsets.keys()]

for key in setkeys:
    ncluststr = str(S.nclust_dict[key])
    myrun = S.runsets[key]
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        print('running %s on %s ncl %s'%(script,myrun,ncluststr))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])
