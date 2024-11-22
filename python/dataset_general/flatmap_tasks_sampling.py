import numpy as np
import h5py
import yaml
import sys
import os
import matplotlib.pyplot as plt



pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dM3_brain'



with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

tesselation_dir = pathdict['tesselation_dir']

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S

figdir_gen = pathdict['figdir_root'] + '/dataset'

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_gen, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])

Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')))


Units_pfc = [U for U in Units if S.check_pfc(U.area) and not U.roi==0]


v_fac = -1

roimage_file = os.path.join(tesselation_dir,'flatmap_PFCregions.h5')
with h5py.File(roimage_file,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

def plot_flatmap(axobj,**kwargs):
    for key in polygon_dict.keys():
        points = polygon_dict[key]
        axobj.plot(points[:, 0], points[:, 1] * v_fac, **kwargs)

X = np.array([[U.u,U.v] for U in Units_pfc])

for attr,cdict in zip(['task','layer'],[S.cdict_task,S.cdict_layer]):
    sortinds = np.random.permutation(np.arange(len(X)))
    if attr == 'layer':
        colorvec = np.array([cdict[getattr(U,attr)[0]] for U in Units_pfc])
    else:
        colorvec = np.array([cdict[getattr(U,attr)] for U in Units_pfc])
    f,ax = plt.subplots(figsize=(5,4))
    plot_flatmap(ax,color='grey',lw=0.8)
    ax.scatter(X[sortinds,0],X[sortinds,1]*-1,s=1,c=colorvec[sortinds])
    ii = 0
    for key,col in cdict.items():
        if col in colorvec:
            ax.text(0.25,0.5-ii*0.05,key,color=col,transform=ax.transAxes,ha='left',va='top',fontweight='bold')
            ii+=1
    ax.set_axis_off()
    figsaver(f, 'flatmapprobes_%s__%s'%(attr,myrun), closeit=True)
