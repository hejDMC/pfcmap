import sys
import yaml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau,pearsonr
from matplotlib import rc




pathpath,tempstr = sys.argv[1:]
my_runsets = tempstr.split('___')
print('getting data from  %s'%str(my_runsets))

cmethod = 'ward'

signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000
ctx_only = True

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from scipy import stats
from pfcmap.python.utils import category_matching as matchfns


tablepath_hierarchy =  pathdict['harris_hierarchy_file']
genfigdir = pathdict['figdir_root'] + '/hierarchy/harris_correlations_equalaxes'


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
cmap = mpl.cm.get_cmap(S.cmap_clust)#


#getting the hierarchy from the tables
df = pd.read_excel(tablepath_hierarchy, sheet_name='hierarchy_all_regions')
avec = np.array([el for el in  df['areas']])
hvec = np.array([el for el in df['CC+TC+CT iterated']])
ctx_bool = np.array([el=='C' for el in  df['CortexThalamus']])
ctx_avec = avec[ctx_bool]

areaIBL_to_areaH = {'AMd':'AM','AMv':'AM',\
                  'LGd-co':'LGd','LGd-ip':'LGd','LGd-sh':'LGd',\
                  'MGd':'MG','MGm':'MG','MGv':'MG',\
                    'AId':'Aid','AIp':'Aip','AIv':'Aiv'}

areaH_to_areaIBL = {val:[key for key in areaIBL_to_areaH.keys() if areaIBL_to_areaH[key]==val] for val in np.unique(list(areaIBL_to_areaH.values()))}
layercheck = lambda mylay: S.check_layers(mylay,['5','6'])

superdict = {}
for runset in my_runsets:
    myrun = S.runsets[runset]

    if myrun.count('IBL'): pathpath = 'PATHS/filepaths_IBL.yml'
    else: pathpath =  'PATHS/filepaths_carlen.yml'

    with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)


    ncl = S.nclust_dict[runset]

    rundict_path = pathdict['configs']['rundicts']
    srcdir = pathdict['src_dirs']['metrics']
    savepath_gen = pathdict['savepath_gen']

    rundict_folder = os.path.dirname(rundict_path)
    rundict = uloader.get_myrun(rundict_folder,myrun)

    somfile = uloader.get_somfile(rundict,myrun,savepath_gen)

    somdict = uloader.load_som(somfile)


    metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
    Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)#rois from path = False  gets us the Gao rois

    assert (somfeats == somdict['features']).all(),'mismatching features'
    assert (weightvec == somdict['featureweights']).all(),'mismatching weights'


    get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])


    #project on map
    refmean,refstd = somdict['refmeanstd']
    featureweights = somdict['featureweights']
    dmat = np.vstack([get_features(elobj) for elobj in Units])
    wmat = uloader.reference_data(dmat,refmean,refstd,featureweights)

    #set BMU for each
    weights = somdict['weights']
    allbmus = somh.get_bmus(wmat,weights)
    for bmu,U in zip(allbmus,Units):
        U.set_feature('bmu',bmu)




    ddict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
    labels = ddict[ncl]


    #categorize units
    for U in Units:
        U.set_feature('clust',labels[U.bmu])

    uareas = np.unique([U.area for U in Units])


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



    for U in Units:
        U.set_feature('fake_const','a')


    if ctx_only:
        attr1 = 'areaHC'#'areaH
        subfolder = 'ctx_only'
    else:
        attr1 = 'areaH'
        subfolder = 'allareas'




    mystats = matchfns.get_all_shuffs(Units, attr1, 'clust', const_attr='fake_const',
                            nshuff=nshuff, signif_levels=signif_levels)

    countvec = mystats['matches'].sum(axis=1)

    presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa] > S.Nmin_maps and not aval == 'NA'])


    sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
    sdict['avals2'] = mystats['avals2']


    superdict[myrun] = sdict



runset_str = '_'.join(my_runsets)

targetdir = os.path.join(genfigdir,'%s__ncl%i'%(runset_str,ncl))
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(targetdir, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname,transparent=True)
    if closeit: plt.close(fig)


myruns = list(superdict.keys())
all_areas = np.unique(np.hstack([superdict[myrun]['avals1'] for myrun in myruns]))


inds_harris_all = np.array([int(np.where(avec==area)[0]) for area in all_areas])
xvec_all = hvec[inds_harris_all]
xspan = xvec_all.max()-xvec_all.min()
myxlim = [xvec_all.min()-0.06*xspan,xvec_all.max()+0.06*xspan]

plotdict = {myrun:{'yvals':{}} for myrun in myruns}

statsfn = S.enr_fn
for myrun in myruns:
    X = statsfn(superdict[myrun])
    for cc,myvals in enumerate(X.T):
        plotdict[myrun]['yvals'][cc] = myvals
    areas = superdict[myrun]['avals1']
    plotdict[myrun]['areas'] = areas
    inds_harris = np.array([int(np.where(avec==area)[0]) for area in areas])
    xvec = hvec[inds_harris]#
    plotdict[myrun]['xvec'] = xvec
    plotdict[myrun]['xsimp'] = np.array([xvec.min(),xvec.max()])

ylim_dict = {}
for cc in np.arange(ncl):
    allyvals =  np.hstack([plotdict[myrun]['yvals'][cc] for myrun in myruns])
    yamp = allyvals.max()-allyvals.min()
    ylim_dict[cc] = np.array([allyvals.min()-yamp*0.06,allyvals.max()+yamp*0.06])


cdict_ext_prime = dict(S.cdict_pfc,**{newkey:S.cdict_pfc[oldkey] for newkey,oldkey in zip(['Aid','Aiv'],['AId','AIv'])})
#tdo adjust colors to what is actually shown
cdict_ext = {key:val for key,val in cdict_ext_prime.items()}
cdict_ext.update({area:'#6eb657ff' for area in all_areas if area.count('SS')})#limegreen
cdict_ext.update({area:'#397f80ff' for area in all_areas if area.count('VIS')})#teal
cdict_ext.update({area:'#d72f8aff' for area in all_areas if area.count('MOp')})
cdict_ext.update({area:'#a17d57ff' for area in all_areas if area.count('Aip')})
cdict_ext.update({area:'#e9983fff' for area in all_areas if area.count('RSP')})
cdict_ext.update({area:'#88cbdbff' for area in all_areas if area.count('AUD')})#aqua
pfc_marker = 'o'
no_pfc_marker = 's'

#mpl.rcParams['pdf.fonttype'] = 42#orig: 3
plt.rcParams['svg.fonttype'] = 'none' #orig: path
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
plt.rcParams['path.simplify'] = True  # Simplify the path
plt.rcParams['path.simplify_threshold'] = 1.0  # Set the threshold high to avoid small artifacts

for myrun in myruns:
    xvals = plotdict[myrun]['xvec']
    myareas = plotdict[myrun]['areas']
    for cc in np.arange(ncl):
        f, ax = plt.subplots(figsize=(1.312*1.34, 1.1*1.34))
        f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
        yvec = plotdict[myrun]['yvals'][cc]
        res = [myfn(xvals,yvec) for myfn in [kendalltau,pearsonr]]
        statsstr = ', '.join(['%s:%1.2f (p:%1.2e)'%(rlab,myres.statistic,myres.pvalue) for rlab,myres in zip(['tau','R'],res)])
        ax.set_title('clust %i - %s'%(cc+1,statsstr),fontsize=4.5)
        for xx,yy, mylab in zip(xvals,yvec,myareas):
            [mark,col] = [pfc_marker,'k'] if mylab in cdict_ext_prime.keys() else [no_pfc_marker,cdict_ext[mylab]]
            if mylab == 'Aiv':mylab = 'AIv'
            if mylab == 'Aid':mylab = 'AId'
            if mylab == 'Aip':mylab = 'AIp'
            ax.plot([xx],[yy],marker=mark,ms=3.5,mec='none',mfc=col)
            [fweight,fs] = ['bold',5] if mylab in cdict_ext_prime.keys() else ['normal',4.5]
            ax.annotate(mylab,[xx,yy],color=col,xytext=(xx-0.01,yy+0.01),xycoords='data',ha='right',va='bottom',\
                        fontsize=fs,fontweight=fweight)
        #if res[1].pvalue<=0.05:
        lsq_res = stats.linregress(xvals, yvec)
        x_simp = plotdict[myrun]['xsimp']
        ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
        ax.set_xlim(myxlim)
        ax.set_ylim(ylim_dict[cc])
        for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
        ax.set_xlabel('hierarchy score (anatomy)',fontsize=6)
        ax.set_ylabel('enrichment',fontsize=6)
        #design wishes
        for pos in ['bottom', 'left']:
            ax.spines[pos].set_linewidth(0.5)
        ax.xaxis.set_tick_params(width=0.5)
        ax.yaxis.set_tick_params(width=0.5)
        ax.tick_params(axis='both', which='major', labelsize=5,length=1.5)
        figsaver(f,'%s/enr_vs_harris__%s_cat%i.svg'%(myrun,myrun,cc+1))

