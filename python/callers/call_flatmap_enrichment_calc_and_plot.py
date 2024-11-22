import subprocess
import sys
import yaml

script_calc = 'flatmaps/calc_flatmap_statsdict.py'
script_plot = 'flatmaps/plot_flatmap_enrichments.py'
script_tables = 'flatmaps/write_flatmap_enrichment_tables.py'

scripts = [script_calc,script_plot]#[script_calc]
scripts = [script_plot]
scripts = [script_tables]

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() ]#if key.count('resp')
#setkeys = ['wwspont','nwspont']

boolstr_dict = {'gaoRois':'0','dataRois':'1'}

for key in setkeys:
    myrun = S.runsets[key]
    ncluststr = str(S.nclust_dict[key])
    if myrun.count('IBL'):
        pathpath = 'PATHS/filepaths_IBL.yml'
    else:
        pathpath = 'PATHS/filepaths_carlen.yml'

    for script in scripts:
        for roi_tag,get_rois_bool_str in boolstr_dict.items():
            print('running %s on %s with %s'%(script,myrun,roi_tag))
            if script == script_calc:
                subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod,get_rois_bool_str,roi_tag])
            else:
                subprocess.call([sys.executable, script, pathpath,myrun, ncluststr,S.cmethod,roi_tag])
