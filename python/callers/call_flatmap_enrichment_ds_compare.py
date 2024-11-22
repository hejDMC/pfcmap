import subprocess
import sys
import yaml

script_compare = 'flatmaps/compare_flatmap_enrichments.py'

pathpath = 'PATHS/filepaths_carlen.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

setkeys = [key for key in S.runsets.keys() if not key.count('resp')]

setkeys_pairs = [['wwspont','wwIBL']]
stags = ['Carlen','IBL']
pathpath = 'PATHS/filepaths_carlen.yml'

scripts = [script_compare]
reftags = ['refall','refPFC']
datatags = ['deepRois','nolayRois']
roi_tags = ['dataRois','gaoRois']


for key_pair in setkeys_pairs:
    for roi_tag in roi_tags:
        key1,key2 = key_pair
        run1 = S.runsets[key1]
        run2 = S.runsets[key2]
        ncluststr = str(S.nclust_dict[key1])
        for data_tag in datatags:
            for reftag in reftags:
                for script in scripts:
                    print('running %s on %s and %s %s %s %s'%(script,run1,run2,data_tag,reftag,roi_tag))

                    subprocess.call([sys.executable, script, pathpath,run1,run2,stags[0],stags[1], ncluststr,S.cmethod,roi_tag,data_tag,reftag])