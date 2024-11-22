import sys
import subprocess
from glob import glob
import os

script = 'preprocessing/lick_extraction/lickarts_from_piezo.py'
mat_dir = 'ZENODOPATH/preprocessing/licktraces'
pathpath = 'PATHS/lickdet_piezo_paths.yml'
allfiles = glob(os.path.join(mat_dir,'*.mat'))
allfiles = [os.path.join(mat_dir,'%s-probe0.mat'%tag) for tag in ['PL084','PL086']]


for myfile in allfiles:
    print ('\n#### EXTRACTING PIEZO LICKS from  %s'%myfile)
    subprocess.call([sys.executable, script,myfile,pathpath])


#PL80_