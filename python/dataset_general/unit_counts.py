import sys
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt


pathpath,myrun = sys.argv[1:]

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)
savedir_output = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__countstats'%myrun)
if not os.path.isdir(savedir_output):os.makedirs(savedir_output)
outfilename = os.path.join(savedir_output,'regionlayerstats__%s.xlsx'%myrun)
outfilename_recwise = os.path.join(savedir_output,'recwise_regionlayerstats__%s.xlsx'%myrun)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)


uloader.assign_parentregs(Units,pfeatname='preg',cond=lambda uobj:True,na_str='na')


def colorrowbg_by_parent(row,column):
    val = row[column[0]]
    #print('row\n %s \n val\n%s\n\n'%(row,val))
    col = S.parent_colors[val] if val in S.parent_colors else 'white'
    return ['background-color: %s'%col]*len(row)


if not myrun.count('IBL'):
    urecids = sorted(np.unique([U.recid for U in Units]))
    df_dict_recs = {}

    for recid in urecids:
        U_supersel = [U for U in Units if U.recid == recid]
        uregs = np.unique([U.region for U in U_supersel])
        reg_dict = {}
        for reg in uregs:
            Usel = [U for U in U_supersel if U.region==reg]
            counts = len(Usel)
            areas = np.unique([U.area for U in Usel])
            assert len(areas) == 1, 'multi areas for %s --> %s'%(reg,str(areas))
            area = areas[0]
            pregs = np.unique([U.preg for U in Usel])
            assert len(areas) == 1, 'multi pregs for %s --> %s'%(reg,str(pregs))
            preg = pregs[0]
            layers = np.unique([U.layer for U in Usel])
            assert len(layers) == 1, 'multi layers for %s --> %s'%(reg,str(layers))
            layer = layers[0]
            reg_dict[reg] = [counts,area,preg,layer]
        df = pd.DataFrame.from_dict(reg_dict, orient='index',columns=['N','area','parent','layer']).sort_values(by=['N'], ascending=False)
        df_dict_recs[recid] = df.style.apply(colorrowbg_by_parent,column=['parent'],axis=1)

    with pd.ExcelWriter(outfilename_recwise, engine='openpyxl') as writer:
        for recid in urecids:
            df_dict_recs[recid].to_excel(writer, sheet_name=recid)







#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(savedir_output, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



uregs = np.unique([U.region for U in Units])
reg_dict = {}
for reg in uregs:
    Usel = [U for U in Units if U.region==reg]
    counts = len(Usel)
    nrecs = len(np.unique([U.recid for U in Usel]))
    ntasks = len(np.unique([U.task for U in Usel]))
    areas = np.unique([U.area for U in Usel])
    assert len(areas) == 1, 'multi areas for %s --> %s'%(reg,str(areas))
    area = areas[0]
    pregs = np.unique([U.preg for U in Usel])
    assert len(areas) == 1, 'multi pregs for %s --> %s'%(reg,str(pregs))
    preg = pregs[0]
    layers = np.unique([U.layer for U in Usel])
    assert len(layers) == 1, 'multi layers for %s --> %s'%(reg,str(layers))
    layer = layers[0]
    reg_dict[reg] = [counts,area,preg,layer,nrecs,ntasks]

df_allregs = pd.DataFrame.from_dict(reg_dict, orient='index',columns=['N','area','parent','layer','recs','tasks']).sort_values(by=['N'], ascending=False)




df_allregs_styled = df_allregs.style.apply(colorrowbg_by_parent,column=['parent'],axis=1)

df_dict = {}
df_dict['allregs'] = df_allregs_styled


layer_keys = ['1','2','4','5','6']

statstags = ['pfc']
if myrun.count('brain'): statstags += ['parentregs']

for statstag in statstags:
    if statstag == 'pfc':
        attr = 'area'
        ulocvals = np.unique([getattr(U,attr) for U in Units])
        mylocvals = np.array([area for area in S.PFC_sorted if area in ulocvals])
    elif statstag == 'parentregs':
        attr = 'preg'
        ulocvals = np.unique([U.preg for U in Units])
        mylocvals = np.unique([U.preg for U in Units if not U.preg == 'na'])

    statsmat = np.zeros((len(mylocvals)+1,len(layer_keys)+1))
    used_inds = []
    for rr,reg in enumerate(mylocvals):
        for ll,layerkey in enumerate(layer_keys):
            myinds = [uu for uu,U in enumerate(Units) if S.check_layers(U.layer,layerkey) and getattr(U,attr)==reg]
            statsmat[rr,ll] = len(myinds)
            used_inds += myinds
    for rr,reg in enumerate(mylocvals):
        myinds = [uu for uu,U in enumerate(Units) if getattr(U,attr)==reg and uu not in used_inds]
        statsmat[rr,-1] = len(myinds)
        used_inds += myinds
    otherregs = [reg for reg in ulocvals if not reg in mylocvals]
    for reg in otherregs:
        for ll,layerkey in enumerate(layer_keys):
            myinds = [uu for uu,U in enumerate(Units) if S.check_layers(U.layer,layerkey) and getattr(U,attr)==reg]
            #print('%s %s hits: %i'%(reg,layerkey,len(myinds)))
            statsmat[-1,ll] += len(myinds)
            used_inds += myinds
    for reg in otherregs:
        myinds = [uu for uu,U in enumerate(Units) if getattr(U,attr)==reg and uu not in used_inds]
        statsmat[-1,-1] += len(myinds)
        used_inds += myinds

    if statstag == 'pfc':
        statsmat_ext = np.vstack([statsmat,statsmat.sum(axis=0)])
        statsmat_ext =  np.hstack([statsmat_ext,statsmat_ext.sum(axis=1)[:,None]])
        my_labels = list(np.array(list(mylocvals)+ ['other']))

    elif statstag == 'parentregs':
        statsmat_sortinds = np.argsort(statsmat.sum(axis=1))[::-1]
        statsmat = statsmat[statsmat_sortinds]
        statsmat_ext = np.vstack([statsmat, statsmat.sum(axis=0)])
        statsmat_ext = np.hstack([statsmat_ext, statsmat_ext.sum(axis=1)[:, None]])
        my_labels = list(np.array(list(mylocvals) + ['other'])[statsmat_sortinds])

    df = pd.DataFrame(data=statsmat_ext, columns=layer_keys + ['na', 'SUM'], index=my_labels + ['SUM'],
                      dtype=int)  # ,index=myregs+['other']
    df_dict[statstag] = df


with pd.ExcelWriter(outfilename, engine='openpyxl') as writer:
        for statstag in ['allregs','pfc','parentregs']:
            df_dict[statstag].to_excel(writer, sheet_name=statstag)






for statstag in statstags:
    if statstag == 'pfc':
        cvec = [mpl.colors.to_rgb(S.cdict_pfc[reg]) for reg in S.PFC_sorted]
        countvec = np.array([len([U for U in Units if U.area==area]) for area in S.PFC_sorted])
        ticklabs = [el for el in S.PFC_sorted]
    elif statstag == 'parentregs':
        cvec = [mpl.colors.to_rgb(S.parent_colors[reg]) for reg in S.parents_ordered]
        countvec = np.array([len([U for U in Units if U.preg==preg]) for preg in S.parents_ordered])
        ticklabs = [el for el in S.parents_ordered]

    N = len(countvec)
    xvec = np.arange(N)

    f,ax = plt.subplots(figsize=(1.5+0.35*N,2))
    f.subplots_adjust(left=0.12,bottom=0.3)
    blist = ax.bar(xvec,countvec,color='k')
    ax.set_xticks(xvec)
    tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')

    for cc,col in enumerate(cvec):
        blist[cc].set_color(col)
        tlabs[cc].set_color(col)
        if countvec[cc] == countvec.max():
            yanch = countvec.max()/4.
            textcol='k'
        else:
            yanch = countvec[cc]+countvec.max()/10
            textcol = col
        ax.text(cc,yanch,'%i'%(countvec[cc]),color=textcol,va='bottom',ha='center',rotation=90,fontsize=12,fontweight='bold')
    ax.set_xticklabels(ticklabs,rotation=90)
    ax.set_ylabel('counts')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
    for pos in ['top','right']:ax.spines[pos].set_visible(False)
    figsaver(f, 'regioncounts_%s'%statstag)



