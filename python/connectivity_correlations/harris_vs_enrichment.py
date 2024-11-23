import sys
import yaml
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


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

genfigdir = pathdict['figdir_root'] + '/hierarchy/harris_correlations'

tablepath_hierarchy = pathdict['harris_hierarchy_file']
rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from scipy import stats

from pfcmap.python.utils import category_matching as matchfns


#if myrun.count('IBL'):
#    layercheck = lambda mylay: True
#    titlestr = '%s alllays' % (myrun)


layercheck = lambda mylay: S.check_layers(mylay,['5','6'])
titlestr = '%s |deep' % (myrun)



stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
cmap = mpl.cm.get_cmap(S.cmap_clust)#



rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

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

norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(labels)}






#categorize units
for U in Units:
    U.set_feature('clust',labels[U.bmu])


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

    for ii in np.arange(ncl):
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
        ax.set_title('cat.%i; tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (ii + 1, tau, pval,pr.correlation,pr.pvalue), color=col,fontsize=8)
        ax.set_xlabel('h-score')
        ax.set_ylabel(statslabs[fntag])
        f.suptitle(titlestr,fontsize=8)
        figsaver(f, '%s/%sVsH_cat%i' % (subfolder,fntag,ii + 1), closeit=True)

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
        ax.set_title('cat.%i; tau:%1.2f, p:%1.2e' % (ii + 1, tau, pval), color=col)
        ax.set_xlabel('h-score')
        ax.set_ylabel(statslabs[fntag])
        f.suptitle(titlestr,fontsize=8)
        figsaver(f, '%s/%sVsH_text_cat%i' % (subfolder,fntag,ii + 1), closeit=True)


    if fntag == 'enr':
        cstr = str(S.cmap_div)
        #zlim = np.max(np.abs([X.min(),X.max()]))
        #vmin,vmax = -zlim,zlim
        vmin,vmax,extd = -5,5,'both'
    else:
        cstr = str(S.cmap_rate)
        vmin,vmax = X.min(),X.max()
        extd = 'neither'
    hsortinds = np.argsort(xvec)
    csorted = X[hsortinds]
    nsorted = np.log10(countvec[presel_inds][hsortinds])
    hsorted = xvec[hsortinds]
    asorted = areas[hsortinds]
    f,axarr = plt.subplots(1,4,gridspec_kw={'width_ratios':[1,1,10,1]},figsize=(4.5,7))
    hax,nax,ax,cax = axarr
    him = hax.imshow(hsorted[:,None], cmap=S.cmap_hier, aspect='auto', origin='lower', vmin=hsorted.min(), vmax=hsorted.max())
    im = ax.imshow(csorted, cmap=cstr, aspect='auto', origin='lower',vmin=vmin,vmax=vmax)
    f.colorbar(im,cax=cax,extend=extd)
    nim = nax.imshow(nsorted[:,None], cmap=S.cmap_count, aspect='auto', origin='lower', vmin=nsorted.min(), vmax=nsorted.max())
    for myax in [ax,nax]:
        myax.set_yticks([])
    for myax in [hax,nax]:
        myax.set_xticks([])
    ax.set_xticks(np.arange(ncl))
    ax.set_xticklabels(np.arange(ncl)+1)
    hax.set_yticks(np.arange(len(hsorted)))
    hax.set_yticklabels(asorted)
    #hax.set_ylabel('roi')
    hax.set_title('hier')
    ax.set_title('clusters')
    nax.set_title('lgN')
    cax.set_title(fntag)
    f.suptitle(titlestr)
    f.tight_layout()
    figsaver(f, '%s/%sVsHierarchy_summary'%(subfolder,fntag), closeit=True)

    # write a list with tau and pearson, save it
    statsfile = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl), '%s/%s_stats__%s.txt'%(subfolder,fntag,myrun))
    statstags = ['kendall','pearson']
    writedict = {statstag:{'pvec':np.zeros(ncl),'corrvec':np.zeros(ncl)} for statstag in statstags}#for easier hdf5 access

    with open(statsfile, 'w') as hand:
        hand.write('N: %i\n%s\n%s\n'%(len(areas),str(list(areas)),titlestr))
        for ii in np.arange(ncl):
            yvec = X[:,ii]
            kt = stats.kendalltau(xvec, yvec)
            pr = stats.pearsonr(xvec,yvec)
            mystr = 'cat%i Kendall-Tau: (%1.2f, %1.3e)   Pearson-R: (%1.2f, %1.3e)'%(ii+1,kt.correlation,kt.pvalue,pr.correlation,pr.pvalue)
            hand.write(mystr+'\n')
            writedict['kendall']['pvec'][ii],writedict['kendall']['corrvec'][ii] = kt.pvalue,kt.correlation
            writedict['pearson']['pvec'][ii],writedict['pearson']['corrvec'][ii] = pr.pvalue,pr.correlation

    writedict['areas'] = areas

    corrfilename = os.path.join(pathdict['statsdict_dir'],'harriscorr__%s__ncl%s_%s_%s.h5'%(myrun,ncluststr,cmethod,fntag))

    uloader.save_dict_to_hdf5(writedict, corrfilename,strtype='S10')





    xxvec = np.arange(ncl)
    for ii,statsflav in enumerate(['kendall','pearson']):
        corrvec = writedict[statsflav]['corrvec']
        f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
        f.subplots_adjust(left=0.3,bottom=0.3)
        blist = ax.bar(xxvec,corrvec,color='k')
        ax.set_xticks(xxvec)
        tlabs = ax.set_xticklabels(xxvec+1,fontweight='bold')
        for cc in np.arange(ncl):
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
        figsaver(f, '%s/%s_corrbars_%s'%(subfolder,fntag,statsflav), closeit=True)



####################
##################PCA on the enrichment matrix

statsfn = S.enr_fn
fntag = 'enr'
X = statsfn(statsdict)[presel_inds]  #


XT = X.T
dataZ = (XT-XT.mean(axis=0))/XT.std(axis=0)#zscore
ncomps = int(ncl)

pca = PCA(n_components=ncomps)
#pca.fit(dataZ.T)
pcs = pca.fit_transform(dataZ.T)



f,ax = plt.subplots(figsize=(3,3))
ax.bar(np.arange(1,ncomps+1),pca.explained_variance_ratio_*100,color='k')
ax.set_xticks(np.arange(1,ncomps+1))
ax.set_ylabel('var explained [%]')
ax.set_xlabel('PC')
f.tight_layout()
figsaver(f, '%s/%s_PCAexplVar' % (subfolder,fntag), closeit=True)


corrmat_pca = np.zeros((2,ncl))
corr_pca_hier = np.zeros((2))
for pp,mypc in enumerate(pcs[:,:2].T):
    corr_pca_hier[pp] = np.corrcoef(xvec,mypc)[0,1]
    for xx,enrvals in enumerate(X.T):
        corrmat_pca[pp,xx] = np.corrcoef(enrvals,mypc)[0,1]





f,ax = plt.subplots(figsize=(3.5,3))
f.subplots_adjust(left=0.2,bottom=0.2)
for cc,pair in enumerate(corrmat_pca.T):
    ax.plot(pair[0],pair[1],'o',mfc=cdict_clust[cc],mec='none',ms=8)
ax.plot(corr_pca_hier[0],corr_pca_hier[1],'o',mfc='none',mec='k',ms=8)


for rad in [0.5,1.]:
    circle = plt.Circle((0,0), rad, fc='none', lw=1, ec='silver',zorder=-10)
    ax.add_patch(circle)
ax.axhline(0,color='silver',zorder=-10)
ax.axvline(0,color='silver',zorder=-10)
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

ax.set_xlim([-1.05,1.05])
ax.set_ylim([-1.05,1.05])
ax.set_xlabel('PC1 corr')
ax.set_ylabel('PC2 corr')
ax.set_aspect('equal')
f.suptitle(titlestr)
figsaver(f, '%s/%s_PCA_clustercorr_with_hierarchy' % (subfolder,fntag), closeit=True)



col = 'k'
npc_show = 4
for ii in np.arange(npc_show):

    f, ax = plt.subplots(figsize=(3.5, 3.))
    f.subplots_adjust(left=0.25, bottom=0.17, right=0.95)
    yvec = pcs[:, ii]
    # ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
    ax.scatter(xvec[is_pfc], yvec[is_pfc], c=cvec[is_pfc], marker=pfc_marker)
    ax.scatter(xvec[~is_pfc], yvec[~is_pfc], c=cvec[~is_pfc], marker=no_pfc_marker)

    kt = stats.kendalltau(xvec, yvec)
    tau = kt.correlation
    pval = kt.pvalue
    pr = stats.pearsonr(xvec, yvec)

    if show_theil:
        ts = stats.theilslopes(yvec, xvec, method='separate')
        ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color='k')
        ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color='k')
    if show_lsq:
        lsq_res = stats.linregress(xvec, yvec)
        ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
    ax.set_title('tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (tau, pval, pr.correlation, pr.pvalue), color=col,
                 fontsize=8)
    ax.set_xlabel('h-score')
    ax.set_ylabel('PC%i'%(ii+1))
    f.suptitle(titlestr, fontsize=8)
    figsaver(f, '%s/%sVsH_PC%i' % (subfolder, fntag, ii+1), closeit=True)


    ###NOW THE TEXT
    lims = [ax.get_xlim(), ax.get_ylim()]
    f, ax = plt.subplots(figsize=(9, 8.))
    f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
    for x, y, area in zip(xvec, yvec, areas):
        ax.text(x, y, area, ha='center', va='center')
    # ax.set_xlim([xvec.min()-0.1,xvec.max()+0.1])
    # ax.set_ylim([yvec.min()-0.5,yvec.max()+0.5])
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    kt = stats.kendalltau(xvec, yvec)
    tau = kt.correlation
    pval = kt.pvalue
    ts = stats.theilslopes(yvec, xvec, method='separate')
    ax.set_title('tau:%1.2f, p:%1.2e' % (tau, pval), color=col)
    ax.set_xlabel('h-score')
    ax.set_ylabel('PC%i'%(ii+1))
    f.suptitle(titlestr, fontsize=8)
    figsaver(f, '%s/%sVsH_text_PC%i' % (subfolder, fntag, ii+1), closeit=True)



if not myrun.count('resp'):
    ###################################
    #### B/M/rate vs hierarchy -->
    def check_translated_present(area_u,area_target):
        if not area in areaH_to_areaIBL.keys():
            return False
        else:
            return area_u in areaH_to_areaIBL[area_target]

    fmat = np.zeros((len(areas),len(somfeats)))
    for aa,area in enumerate(areas):
        usel =[ U for U in Units if (U.area==area or check_translated_present(U.area,area)) and not getattr(U,attr1)=='NA' and layercheck(U.layer)]#last cond not strictly necessary as contained in attr check
        #print(area,len(usel),np.unique([U.layer for U in usel]))
        fmat[aa] = np.mean(np.array([[getattr(U,somfeat) for somfeat in somfeats] for U in usel]),axis=0)


    fntag = 'RAW'
    xlsx_dict[fntag] = fmat

    col = 'k'
    for ii,somfeat in enumerate(somfeats):

        f, ax = plt.subplots(figsize=(3.5, 3.))
        f.subplots_adjust(left=0.25, bottom=0.17, right=0.95)
        yvec = fmat[:, ii]
        # ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
        ax.scatter(xvec[is_pfc], yvec[is_pfc], c=cvec[is_pfc], marker=pfc_marker)
        ax.scatter(xvec[~is_pfc], yvec[~is_pfc], c=cvec[~is_pfc], marker=no_pfc_marker)

        kt = stats.kendalltau(xvec, yvec)
        tau = kt.correlation
        pval = kt.pvalue
        pr = stats.pearsonr(xvec, yvec)

        if show_theil:
            ts = stats.theilslopes(yvec, xvec, method='separate')
            ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color='k')
            ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color='k')
        if show_lsq:
            lsq_res = stats.linregress(xvec, yvec)
            ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
        ax.set_title('tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (tau, pval, pr.correlation, pr.pvalue), color=col,
                     fontsize=8)
        ax.set_xlabel('h-score')
        ax.set_ylabel(somfeat.split('_')[0])
        f.suptitle(titlestr, fontsize=8)
        figsaver(f, '%s/%sVsH_feat%s' % (subfolder, fntag, somfeat.split('_')[0]), closeit=True)


        ###NOW THE TEXT
        lims = [ax.get_xlim(), ax.get_ylim()]
        f, ax = plt.subplots(figsize=(9, 8.))
        f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
        for x, y, area in zip(xvec, yvec, areas):
            ax.text(x, y, area, ha='center', va='center')
        # ax.set_xlim([xvec.min()-0.1,xvec.max()+0.1])
        # ax.set_ylim([yvec.min()-0.5,yvec.max()+0.5])
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        kt = stats.kendalltau(xvec, yvec)
        tau = kt.correlation
        pval = kt.pvalue
        ts = stats.theilslopes(yvec, xvec, method='separate')
        ax.set_title('tau:%1.2f, p:%1.2e' % (tau, pval), color=col)
        ax.set_xlabel('h-score')
        ax.set_ylabel(somfeat.split('_')[0])
        f.suptitle(titlestr, fontsize=8)
        figsaver(f, '%s/%sVsH_text_feat%s' % (subfolder, fntag, somfeat.split('_')[0]), closeit=True)


    ###################
    #### PCA on the metrics data
    FT = fmat.T
    dataZ = (FT-FT.mean(axis=0))/FT.std(axis=0)#zscore
    ncomps = int(len(somfeats))

    pca = PCA(n_components=ncomps)
    #pca.fit(dataZ.T)
    pcs = pca.fit_transform(dataZ.T)



    f,ax = plt.subplots(figsize=(3,3))
    ax.bar(np.arange(1,ncomps+1),pca.explained_variance_ratio_*100,color='k')
    ax.set_xticks(np.arange(1,ncomps+1))
    ax.set_ylabel('var explained [%]')
    ax.set_xlabel('PC')
    f.tight_layout()
    figsaver(f, '%s/%s_PCAexplVar' % (subfolder,fntag), closeit=True)


    corrmat_pca = np.zeros((2,len(somfeats)))
    corr_pca_hier = np.zeros((2))
    for pp,mypc in enumerate(pcs[:,:2].T):
        corr_pca_hier[pp] = np.corrcoef(xvec,mypc)[0,1]
        for xx,enrvals in enumerate(fmat.T):
            corrmat_pca[pp,xx] = np.corrcoef(enrvals,mypc)[0,1]


    f,ax = plt.subplots(figsize=(3.5,3))
    f.subplots_adjust(left=0.2,bottom=0.2)
    for cc,pair in enumerate(corrmat_pca.T):
        #ax.plot(pair[0],pair[1],'o',mfc=cdict_clust[cc],mec='none',ms=8)
        ax.text(pair[0],pair[1],somfeats[cc].split('_')[0],ha='center',va='center')
    ax.plot(corr_pca_hier[0],corr_pca_hier[1],'o',mfc='none',mec='k',ms=8)


    for rad in [0.5,1.]:
        circle = plt.Circle((0,0), rad, fc='none', lw=1, ec='silver',zorder=-10)
        ax.add_patch(circle)
    ax.axhline(0,color='silver',zorder=-10)
    ax.axvline(0,color='silver',zorder=-10)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

    ax.set_xlim([-1.05,1.05])
    ax.set_ylim([-1.05,1.05])
    ax.set_xlabel('PC1 corr')
    ax.set_ylabel('PC2 corr')
    ax.set_aspect('equal')
    f.suptitle(titlestr)
    figsaver(f, '%s/%s_PCA_clustercorr_with_hierarchy' % (subfolder,fntag), closeit=True)


    #write down the xlsx dict
    outfile = os.path.join(targetdir,subfolder,'enr_frac_raw_values__%s.xlsx'%myrun)

    with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
        for dskey in xlsx_dict.keys():
            if dskey == 'RAW': columns = list(somfeats)
            else: columns = ['cl%i'%(ii+1) for ii in np.arange(ncl)]
            df = pd.DataFrame.from_dict({area:np.r_[xlsx_dict[dskey][aa],np.array(xvec[aa])] for aa,area in enumerate(areas)}, orient='index', columns=columns+['h-score'])
            df.to_excel(writer, sheet_name=dskey)
