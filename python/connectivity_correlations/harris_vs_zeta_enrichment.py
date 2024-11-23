import sys
import yaml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

'''
myrun = 'runCrespZP_brain'
pathpath = 'PATHS/filepaths_carlen.yml'
ncluststr = '8'
cmethod = 'ward'
'''


signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000
ctx_only = True


ncl = int(ncluststr)


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']
tablepath_hierarchy =  pathdict['harris_hierarchy_file']
genfigdir = pathdict['figdir_root'] + '/hierarchy/harris_correlations_zeta'


sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from scipy import stats
from pfcmap.python.utils import category_matching as matchfns

layercheck = lambda mylay: S.check_layers(mylay,['5','6'])
titlestr = '%s |deep' % (myrun)

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
cmap = mpl.cm.get_cmap(S.cmap_clust)#



rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

zeta_flav = 'onset_time_(half-width)'
zeta_pthr = 0.01
missing_zeta = 999
zeta_pattern = 'RECID__%s__%s%s__TSELpsth2to7__STATEactive__all__zeta.h5'%(myrun,cmethod,ncluststr)
zeta_dir =  os.path.join(pathdict['src_dirs']['zeta'],'%s__%s%s'%(myrun,cmethod,ncluststr))


somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

#figsaver
targetdir = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl))
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(targetdir, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)
#figsaver



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)#rois from path = False  gets us the Gao rois





#now set the zeta:
recids = np.unique([U.recid for U in Units])

uloader.get_set_zeta_delay(recids,zeta_dir,zeta_pattern,Units,zeta_feat=zeta_flav,zeta_pthr=zeta_pthr,missingval=missing_zeta)
Units = [U for U in Units if not U.zeta_respdelay == missing_zeta]




for U in Units:
    U.set_feature('clust',int(U.zetap<=zeta_pthr))


cdict_clust = {0:'grey',1:'k'}






uareas = np.unique([U.area for U in Units])


df = pd.read_excel(tablepath_hierarchy, sheet_name='hierarchy_all_regions')
avec = np.array([el for el in  df['areas']])
hvec = np.array([el for el in df['CC+TC+CT iterated']])
ctx_bool = np.array([el=='C' for el in  df['CortexThalamus']])
ctx_avec = avec[ctx_bool]


solitary_avec = [area for area in avec if not area in uareas]
solitary_uareas = [area for area in uareas if not area in avec]


areaIBL_to_areaH = {'AMd':'AM','AMv':'AM',\
                  'LGd-co':'LGd','LGd-ip':'LGd','LGd-sh':'LGd',\
                  'MGd':'MG','MGm':'MG','MGv':'MG',\
                    'AId':'Aid','AIp':'Aip','AIv':'Aiv'}

areaH_to_areaIBL = {val:[key for key in areaIBL_to_areaH.keys() if areaIBL_to_areaH[key]==val] for val in np.unique(list(areaIBL_to_areaH.values()))}
#N.B: 'PF' is unmatched

for U in Units:
    if U.area in areaIBL_to_areaH.keys():
        areaH =  areaIBL_to_areaH[U.area]
    elif U.area in avec:
        areaH = str(U.area)
    else:
        areaH = 'NA'
    U.set_feature('areaH', areaH)



for U in Units:
    if U.areaH in ctx_avec and layercheck(U.layer):
        if U.area in areaIBL_to_areaH.keys():
            areaHC =  areaIBL_to_areaH[U.area]
        else:
            areaHC = str(U.area)
    else:
        areaHC = 'NA'

    U.set_feature('areaHC', areaHC)



uareasH = np.unique([U.areaH for U in Units])
uareasHC = np.unique([U.areaHC for U in Units])

solitary_avec = [area for area in avec if not area in uareasH]

for U in Units:
    U.set_feature('fake_const','a')


if ctx_only:
    attr1 = 'areaHC'#'areaH
    subfolder = 'ctx_only'
else:
    attr1 = 'areaH'
    subfolder = 'allareas'


statsdict = matchfns.get_all_shuffs(Units, attr1, 'clust', const_attr='fake_const',
                                                                    nshuff=nshuff, signif_levels=signif_levels)




countvec = statsdict['matches'].sum(axis=1)

presel_inds = np.array([aa for aa, aval in enumerate(statsdict['avals1']) if countvec[aa] > S.Nmin_maps and not aval == 'NA'])
areas = statsdict['avals1'][presel_inds]

inds_harris = np.array([int(np.where(avec==area)[0]) for area in areas])
xvec = hvec[inds_harris]#hierarchy sorted according to area as in our enrichment data
x_simp = np.array([xvec.min(),xvec.max()])










show_theil = False
show_lsq = True

#making the color dict and setting up the plotting
cdict_ext = dict(S.cdict_pfc,**{newkey:S.cdict_pfc[oldkey] for newkey,oldkey in zip(['Aid','Aiv'],['AId','AIv'])})
is_pfc = np.array([area in cdict_ext.keys() for area in areas]).astype(bool)

cdict_ext.update({area:'limegreen' for area in areas if area.count('SS')})
cdict_ext.update({area:'teal' for area in areas if area.count('VIS')})
cdict_ext.update({area:'aqua' for area in areas if area.count('AUD')})
pfc_marker = 'o'
no_pfc_marker = 's'


cvec = np.array([cdict_ext[area] if area in cdict_ext.keys() else 'k' for area in areas])
statslabs = {'enr':'enrichment','frc':'fraction'}
xlsx_dict = {fntag:{} for fntag in ['enr', 'frc'] }

for fntag, statsfn in zip(['enr', 'frc'], [S.enr_fn, S.frac_fn]):

    X = statsfn(statsdict)[presel_inds]  #

    xlsx_dict[fntag] = X

    for ii in np.arange(2):
        f, ax = plt.subplots(figsize=(3.5, 3.))
        f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
        col = cdict_clust[ii]
        yvec = X[:,ii]
        #ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
        ax.scatter(xvec[is_pfc],yvec[is_pfc],c=cvec[is_pfc],marker=pfc_marker)
        ax.scatter(xvec[~is_pfc],yvec[~is_pfc],c=cvec[~is_pfc],marker=no_pfc_marker)

        kt = stats.kendalltau(xvec, yvec)
        tau = kt.correlation
        pval = kt.pvalue
        pr = stats.pearsonr(xvec,yvec)

        if show_theil:
            ts = stats.theilslopes(yvec, xvec, method='separate')
            ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color=col)
            ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color=col)
        if show_lsq:
            lsq_res = stats.linregress(xvec, yvec)
            ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
        ax.set_title('signif=%i; tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (ii, tau, pval,pr.correlation,pr.pvalue), color=col,fontsize=8)
        ax.set_xlabel('h-score')
        ax.set_ylabel(statslabs[fntag])
        f.suptitle(titlestr,fontsize=8)
        f.tight_layout()

        figsaver(f, '%s/%sVsH_signif%i' % (subfolder,fntag,ii), closeit=True)

        lims = [ax.get_xlim(),ax.get_ylim()]
        f, ax = plt.subplots(figsize=(9,8.))
        f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
        col = cdict_clust[ii]
        yvec = X[:,ii]
        for x,y,area in zip(xvec,yvec,areas):
            ax.text(x, y, area, ha='center',va='center')
        #ax.set_xlim([xvec.min()-0.1,xvec.max()+0.1])
        #ax.set_ylim([yvec.min()-0.5,yvec.max()+0.5])
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        kt = stats.kendalltau(xvec, yvec)
        tau = kt.correlation
        pval = kt.pvalue
        ts = stats.theilslopes(yvec, xvec, method='separate')
        ax.set_title('signif=%i; tau:%1.2f, p:%1.2e' % (ii, tau, pval), color=col)
        ax.set_xlabel('h-score')
        ax.set_ylabel(statslabs[fntag])
        f.suptitle(titlestr,fontsize=8)
        f.tight_layout()

        figsaver(f, '%s/%sVsH_text_signif%i' % (subfolder,fntag,ii), closeit=True)




    # write a list with tau and pearson, save it
    statsfile = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl), '%s/%s_stats__%sZETA.txt'%(subfolder,fntag,myrun))
    statstags = ['kendall','pearson']
    writedict = {statstag:{'pvec':np.zeros(2),'corrvec':np.zeros(2)} for statstag in statstags}#for easier hdf5 access

    with open(statsfile, 'w') as hand:
        hand.write('N: %i\n%s\n%s\n'%(len(areas),str(list(areas)),titlestr))
        for ii in np.arange(2):
            yvec = X[:,ii]
            kt = stats.kendalltau(xvec, yvec)
            pr = stats.pearsonr(xvec,yvec)
            mystr = 'cat%i Kendall-Tau: (%1.2f, %1.3e)   Pearson-R: (%1.2f, %1.3e)'%(ii+1,kt.correlation,kt.pvalue,pr.correlation,pr.pvalue)
            hand.write(mystr+'\n')
            writedict['kendall']['pvec'][ii],writedict['kendall']['corrvec'][ii] = kt.pvalue,kt.correlation
            writedict['pearson']['pvec'][ii],writedict['pearson']['corrvec'][ii] = pr.pvalue,pr.correlation

    writedict['areas'] = areas

    corrfilename = os.path.join(pathdict['statsdict_dir'],'harriscorrZETA__%s__ncl%s_%s_%s.h5'%(myrun,ncluststr,cmethod,fntag))

    uloader.save_dict_to_hdf5(writedict, corrfilename,strtype='S10')





    xxvec = np.arange(2)
    for ii,statsflav in enumerate(['kendall','pearson']):
        corrvec = writedict[statsflav]['corrvec']
        f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
        f.subplots_adjust(left=0.3,bottom=0.3)
        blist = ax.bar(xxvec,corrvec,color='k')
        ax.set_xticks(xxvec)
        tlabs = ax.set_xticklabels(xxvec+1,fontweight='bold')
        for cc in np.arange(2):
            col = cdict_clust[cc]
            tlabs[cc].set_color(col)
            blist[cc].set_color(col)
            #ax.xaxis.get_ticklabels()
        ax.set_ylabel(statsflav)
        ax.set_xlabel('category')
        ax.set_xlim([xxvec.min()-0.5,xxvec.max()+0.5])
        ax.set_title(myrun)
        ax.set_ylim([-0.9,0.9])
        for pos in ['top','right']:ax.spines[pos].set_visible(False)
        f.tight_layout()
        figsaver(f, '%s/%s_corrbars_%s'%(subfolder,fntag,statsflav), closeit=True)

