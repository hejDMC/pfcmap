import sys
import yaml
import os
import numpy as np
from matplotlib import pyplot as plt


pathpath,myrun = sys.argv[1:]

#pathpath = 'PATHS/filepaths_carlen.yml'
#myrun = 'runC00dMP3_brain'


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

savedir_output = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__parameterInteractions'%myrun)

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(savedir_output, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import evol_helpers as ehelpers#just for plotting the cmap individually


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)

uloader.assign_parentregs(Units,pfeatname='preg',cond=lambda uobj: True,na_str='na')
Us_filtered = []
for U in Units:
    if U.preg in S.ctx_regions+['PFC']:
        genarea = U.area if S.check_pfc(U.area) else U.preg
        U.set_feature('genArea',genarea)
        Us_filtered += [U]
    else:
        if U.preg!='na':
            U.set_feature('genArea',str(U.preg))
            Us_filtered += [U]



#U_presel = [U for U in Units if S.get_parentreg(U.region) in S.parent_dict.keys()]
attrs = ['genArea','layer','task','recid']
tasks_sorted = ['Passive','Attention','Aversion','Context','Detection']#


PFC_units = [U for  U in Us_filtered if S.check_pfc(U.area)]

cmap_frac = 'binary'
cmap_count = 'viridis'
grid_col = 'k'
vminmax = [0,1]

for reftag,Usel in zip(['refall','refPFC'],[Us_filtered,PFC_units]):
    aval_dict = {attr:[aval for aval in np.unique([getattr(U,attr) for U in Usel]) if not aval in ['NA']] for attr in attrs}

    #sorting things
    aval_dict['genArea'] = [pfc_area for pfc_area in S.PFC_sorted if pfc_area in aval_dict['genArea']]+[parea for parea in S.parents_ordered if parea in aval_dict['genArea']]
    aval_dict['task'] = [task for task in tasks_sorted if task in aval_dict['task']]


    for a1 in attrs:
        for a2 in [attr for attr in attrs if attr!=a1]:
            #attr_pair = (attrs[0],attrs[1])
            #a1,a2 = attr_pair
            avals1,avals2 = np.array(aval_dict[a1]),np.array(aval_dict[a2])
            n1,n2 = len(avals1),len(avals2)
            countmat = np.zeros((n1,n2))
            for aa1,aval1 in enumerate(avals1):
                for aa2,aval2 in enumerate(avals2):
                    countmat[aa1,aa2] = len([U for U in Usel if getattr(U,a1)==aval1 and getattr(U,a2)==aval2])

            #cutting countmat together
            col_cond = ~np.all(countmat==0,axis=0)
            row_cond = ~np.all(countmat==0,axis=1)
            avals_row = avals1[row_cond]
            avals_col = avals2[col_cond]
            countmat = countmat[row_cond]
            countmat = countmat[:,col_cond]

            fracmat = countmat/countmat.sum(axis=1)[:,None]


            nrows = len(avals_row)
            ncols = len(avals_col)
            row_boost = 3 if a1.count('recid') else 1
            col_boost = 3 if a2.count('recid') else 1
            x_rot = 0 if a2.count('layer') else 90

            f,ax = plt.subplots(figsize=(col_boost+ncols*0.3,row_boost+nrows*0.3))
            ax.imshow(fracmat,cmap=cmap_frac,vmin=vminmax[0],vmax=vminmax[1])
            ax.hlines(y=np.arange(nrows)-0.5,xmin=np.full(nrows, 0)-0.5,xmax=np.full(nrows, ncols) - 0.5,color=grid_col)
            ax.set_xticks(np.arange(ncols))
            ax.set_yticks(np.arange(nrows))
            ax.set_xticklabels(avals_col,rotation=x_rot)
            ax.set_yticklabels(avals_row)
            f.tight_layout()
            figsaver(f,'%s/fracs/fracmat_%s_and_%s_%s'%(reftag,a1,a2,reftag))

            f,ax = plt.subplots(figsize=(col_boost+ncols*0.3,row_boost+nrows*0.3))
            im = ax.imshow(countmat,cmap=cmap_count)
            vminmax_counts = list(np.array(im.get_clim()).astype(int))
            ax.set_xticks(np.arange(ncols))
            ax.set_yticks(np.arange(nrows))
            ax.set_xticklabels(avals_col,rotation=x_rot)
            ax.set_yticklabels(avals_row)
            f.tight_layout()
            figsaver(f,'%s/counts/countmat_%s_and_%s_%s'%(reftag,a1,a2,reftag))


            f, ax = ehelpers.plot_cmap(cmap_count, vminmax_counts)
            figsaver(f,'%s/counts/countmat_%s_and_%s_CMAP_%s'%(reftag,a1,a2,reftag))


    # plot a single cmap for the 0-1 normed fracs
    f, ax = ehelpers.plot_cmap(cmap_frac, vminmax)
    figsaver(f,'%s/fracs/cmap_frac_%s'%(reftag,reftag))

#figsaver_ds(f, 'orig_block_matchmat_cmap')