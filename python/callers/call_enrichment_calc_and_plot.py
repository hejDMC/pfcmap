import subprocess
import sys
import yaml

script_calc = 'enrichments/calc_statsdict.py'
script_plot = 'enrichments/plot_enrichments.py'
script_calc_ctrl = 'enrichments/calcctrlEnrichments_area_layer_recid.py'
script_plot_ctrl = 'enrichments/plot_ctrlEnrichments.py'
script_polarization = 'enrichments/enrichment_polarization.py'
script_tables = 'enrichments/write_enrichment_tables.py'
scripts = [script_calc,script_plot,script_calc_ctrl,script_plot_ctrl,script_polarization]
scripts = [script_plot,script_plot_ctrl]
scripts = [script_tables]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys()] #if key.count('resp')
#setkeys = ['wwspont','nwspont']

for key in setkeys:
    myrun = S.runsets[key]
    ncluststr = str(S.nclust_dict[key])
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        print('running %s on %s'%(script,myrun))

        subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod])