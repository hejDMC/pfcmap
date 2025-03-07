import csv
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import h5py


pathpath = 'PATHS/filepaths_carlen.yml'


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import tessellation_tools as ttools

genfigdir = pathdict['figdir_root'] + '/hierarchy/gao_correlations'
gao_hier_path = pathdict['gao_hierarchy_file']


flatmapfile = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')
with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

with open(gao_hier_path, newline='') as csvfile:
    hdata = list(csv.reader(csvfile, delimiter=','))[1:]
hdict = {str(int(float(el[0]))): float(el[1]) for el in hdata}


def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])

cmapstr = 'RdBu_r'
myvals = np.array([list(hdict.values())])
myvals = myvals[~np.isnan(myvals)]
ampval = (np.abs(np.array([myvals.min(),myvals.max()]))).max()
mycmap = ttools.get_scalar_map(cmapstr, [-ampval, ampval])



f, ax = plt.subplots(figsize=(4, 4))
ttools.colorfill_polygons(ax, polygon_dict, hdict, cmap=mycmap, na_col='grey', ec='k',nancol='grey',\
                          show_cmap=True,clab='H-score', mylimfn=set_mylim)
f.savefig(os.path.join(genfigdir,'gaoHierarchy_flatmap.svg'))