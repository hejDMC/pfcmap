import os
import sys
import yaml
import h5py
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]
'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
ncluststr = '8'
cmethod = 'ward'
'''
ncl = int(ncluststr)



with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
genfigdir = pathdict['figdir_root'] + '/hierarchy/gao_correlations'

gao_hier_path = pathdict['gao_hierarchy_file']



from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import tessellation_tools as ttools

basetag = 'emat'#that doesn't matter here
datafile = os.path.join(pathdict['statsdict_dir'],'enrichmentdict__%s__rois_gaoRois__%s__ncl%s_%s.h5'%(basetag,myrun,ncluststr,cmethod))

targetdir = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl))


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
cmap = mpl.cm.get_cmap(S.cmap_clust)#

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(targetdir, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)
#figsaver

#getting the flatmap files and prepare coloring
flatmapfile = os.path.join(pathdict['tesselation_dir'],'flatmap_PFCrois.h5')
region_file = os.path.join(pathdict['tesselation_dir'],'flatmap_PFCregions.h5')
with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(np.arange(ncl))}
roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}



statstags = ['kendall', 'pearson']

reftags = ['refall','refPFC']
datatags = ['deepRois','nolayRois']
data_dict = {reftag:{} for reftag in reftags}

with h5py.File(datafile,'r') as hand:
    for reftag in reftags:
        for datatag in datatags:
            statshand = hand[reftag]['enr'][datatag]['src_stats']  # from this select the deep ones
            mystats = uloader.unpack_statshand(statshand, remove_srcpath=True)
            data_dict[reftag][datatag] = mystats

#now import the gao hierarchy index
with open(gao_hier_path, newline='') as csvfile:
    hdata = list(csv.reader(csvfile, delimiter=','))[1:]
hdict = {int(float(el[0])):float(el[1]) for el in hdata}
allrois_gao = np.sort(np.array(list(hdict.keys())))
hvec = np.array([hdict[key] for key in allrois_gao])

show_theil = False
show_lsq = True




#this is only for the raw somfeat correlations
rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)
metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)#rois from path = False  gets us the Gao rois


for reftag in reftags:
    for datatag in datatags:

        sdict = data_dict[reftag][datatag]
        X = S.enr_fn(sdict)
        rois_data = np.array([aval.split('|')[0] for aval in sdict['avals1']]).astype(int)
        hinds = np.array([np.where(allrois_gao==roi)[0][0] for roi in rois_data])
        htemp = hvec[hinds]
        h_cond = ~np.isnan(htemp)
        xvec = htemp[h_cond]
        ymat = X[h_cond]
        myrois = rois_data[h_cond]

        titlestr = '%s %s %s' % (myrun,datatag,reftag)


        def make_savename(plotname):
            return '%s/%s/%s__%s_%s_enr' % (reftag, datatag, plotname, reftag, datatag)


        cvec = np.array([roi_colors[str(roi)] for roi in myrois])

        x_simp = np.array([xvec.min(),xvec.max()])

        for ii in np.arange(ncl):
            f, ax = plt.subplots(figsize=(3.5, 3.))
            f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
            col = cdict_clust[ii]
            yvec = ymat[:,ii]
            # ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
            ax.scatter(xvec, yvec, c=cvec, marker='o')

            kt = stats.kendalltau(xvec, yvec)
            tau = kt.correlation
            pval = kt.pvalue
            pr = stats.pearsonr(xvec, yvec)

            if show_theil:
                ts = stats.theilslopes(yvec, xvec, method='separate')
                ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color=col)
                ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color=col)
            if show_lsq:
                lsq_res = stats.linregress(xvec, yvec)
                ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
            ax.set_title('cat.%i; tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (ii + 1, tau, pval, pr.correlation, pr.pvalue),
                         color=col, fontsize=8)
            ax.set_xlabel('h-score')
            ax.set_ylabel('enrichment')
            f.tight_layout()
            figsaver(f,make_savename(('enrVsH_cat%i'% (ii + 1))))

            lims = [ax.get_xlim(), ax.get_ylim()]
            f, ax = plt.subplots(figsize=(9, 8.))
            f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
            col = cdict_clust[ii]
            yvec = ymat[:, ii]
            for zz,[x, y, area] in enumerate(zip(xvec, yvec, myrois)):
                ax.text(x, y, str(area), ha='center', va='center',color=cvec[zz])
            # ax.set_xlim([xvec.min()-0.1,xvec.max()+0.1])
            # ax.set_ylim([yvec.min()-0.5,yvec.max()+0.5])
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
            kt = stats.kendalltau(xvec, yvec)
            tau = kt.correlation
            pval = kt.pvalue
            ts = stats.theilslopes(yvec, xvec, method='separate')
            ax.set_title('cat.%i; tau:%1.2f, p:%1.2e' % (ii + 1, tau, pval), color=col)
            ax.set_xlabel('h-score')
            ax.set_ylabel('enrichment')
            f.suptitle(titlestr, fontsize=8)
            f.tight_layout()
            figsaver(f,make_savename(('enrVsH_text_cat%i'% (ii + 1))))


        # write a list with tau and pearson, save it

        statsfile = os.path.join(targetdir, make_savename('stats') + '__%s.txt'%(myrun))
        writedict = {statstag: {'pvec': np.zeros(ncl), 'corrvec': np.zeros(ncl)} for statstag in
                     statstags}  # for easier hdf5 access

        with open(statsfile, 'w') as hand:
            hand.write('N: %i\n%s\n%s\n' % (len(myrois), str(list(myrois)), titlestr))
            for ii in np.arange(ncl):
                yvec = ymat[:, ii]
                kt = stats.kendalltau(xvec, yvec)
                pr = stats.pearsonr(xvec, yvec)
                mystr = 'cat%i Kendall-Tau: (%1.2f, %1.3e)   Pearson-R: (%1.2f, %1.3e)' % (
                ii + 1, kt.correlation, kt.pvalue, pr.correlation, pr.pvalue)
                hand.write(mystr + '\n')
                writedict['kendall']['pvec'][ii], writedict['kendall']['corrvec'][ii] = kt.pvalue, kt.correlation
                writedict['pearson']['pvec'][ii], writedict['pearson']['corrvec'][ii] = pr.pvalue, pr.correlation

        writedict['rois'] = [str(myroi) for myroi in myrois]

        corrfilename = os.path.join(pathdict['statsdict_dir'],
                                    'gaocorr__%s__ncl%s_%s_enr.h5' % (myrun, ncluststr, cmethod))

        uloader.save_dict_to_hdf5(writedict, corrfilename, strtype='S10')


        #display as bar-plot correlation as bar plot
        ymax = 0.75
        clvec = np.arange(ncl)
        for statstag in statstags:
            corrvec = writedict[statstag]['corrvec']
            f, ax = plt.subplots(figsize=(1.5 + 0.2 * ncl, 2))
            f.subplots_adjust(left=0.3, bottom=0.3)
            blist = ax.bar(clvec, corrvec, color='k')
            ax.set_xticks(clvec)
            tlabs = ax.set_xticklabels(clvec + 1, fontweight='bold')
            for cc in np.arange(ncl):
                col = cdict_clust[cc]
                tlabs[cc].set_color(col)
                blist[cc].set_color(col)
                # ax.xaxis.get_ticklabels()
            ax.set_ylabel(statstag)
            ax.set_xlabel('category')
            ax.set_xlim([clvec.min() - 0.5, clvec.max() + 0.5])
            ax.set_title(titlestr,fontsize=8)
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            ax.set_ylim([-ymax, ymax])
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, which='both', color='grey', linestyle=':', zorder=-10)
            for pos in ['top', 'right']: ax.spines[pos].set_visible(False)

            #ax.set_yticks(np.arange(-0.6,0.7,0.2))
            for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
            figsaver(f, make_savename('corrbars_%s'%statstag))



        XT = ymat.T
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
        figsaver(f, make_savename('PCAexplVar'), closeit=True)


        corrmat_pca = np.zeros((2,ncl))
        corr_pca_hier = np.zeros((2))
        for pp,mypc in enumerate(pcs[:,:2].T):
            corr_pca_hier[pp] = np.corrcoef(xvec,mypc)[0,1]
            for xx,enrvals in enumerate(ymat.T):
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
        figsaver(f, make_savename('PCAclustcorrPC1PC2'), closeit=True)



        col = 'k'
        npc_show = 4
        for ii in np.arange(npc_show):

            f, ax = plt.subplots(figsize=(3.5, 3.))
            f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
            yvec = pcs[:, ii]
            # ax.plot(xvec, yvec, 'o', mec='none', mfc=cvec)
            ax.scatter(xvec, yvec, c=cvec, marker='o')
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
            figsaver(f, make_savename('PCA_enrVsH_PC%i'%(ii+1)), closeit=True)


            ###NOW THE TEXT
            lims = [ax.get_xlim(), ax.get_ylim()]
            f, ax = plt.subplots(figsize=(9, 8.))
            f.subplots_adjust(left=0.1, bottom=0.1, right=0.95)
            for x, y, area in zip(xvec, yvec, myrois):
                ax.text(x, y, str(area), ha='center', va='center')
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
            figsaver(f, make_savename('PCA_enrVsH_text_PC%i'%(ii+1)), closeit=True)

        if not myrun.count('resp'):
            #now correlation against the raw, it makes no difference, whether refall or refPFC for this
            if reftag == 'refall':

                titlestr = '%s %s' % (myrun,datatag)
                #['deepRois', 'nolayRois']
                if datatag == 'deepRois':
                    layercheck = lambda mylay: S.check_layers(mylay,['5','6'])
                elif datatag == 'nolayRois':
                    layercheck = lambda mylay: True
                else:
                    assert 0, 'unknow datatag %s'%datatag


                fmat = np.zeros((len(myrois), len(somfeats)))
                for aa, myroi in enumerate(myrois):
                    usel = [U for U in Units if U.roi==myroi and layercheck(U.layer)]  # last cond not strictly necessary as contained in attr check
                    # print(area,len(usel),np.unique([U.layer for U in usel]))
                    fmat[aa] = np.mean(np.array([[getattr(U, somfeat) for somfeat in somfeats] for U in usel]), axis=0)


                for ii,somfeat in enumerate(somfeats):
                    f, ax = plt.subplots(figsize=(3.5, 3.))
                    f.subplots_adjust(left=0.19, bottom=0.17, right=0.95)
                    yvec = fmat[:, ii]

                    ax.scatter(xvec, yvec, c=cvec, marker='o')

                    kt = stats.kendalltau(xvec, yvec)
                    tau = kt.correlation
                    pval = kt.pvalue
                    pr = stats.pearsonr(xvec, yvec)

                    if show_theil:
                        ts = stats.theilslopes(yvec, xvec, method='separate')
                        ax.plot(x_simp, ts[1] + ts[0] * x_simp, '-', color=col)
                        ax.fill_between(x_simp, ts[1] + ts[2] * x_simp, ts[1] + ts[3] * x_simp, lw=0., alpha=0.2, color=col)
                    if show_lsq:
                        lsq_res = stats.linregress(xvec, yvec)
                        ax.plot(x_simp, lsq_res[1] + lsq_res[0] * x_simp, color='grey', zorder=-10)
                    ax.set_title('tau:%1.2f, p:%1.2e; R:%1.2f,p:%1.2e' % (tau, pval, pr.correlation, pr.pvalue),
                                 color=col, fontsize=8)
                    ax.set_xlabel('h-score')
                    ax.set_ylabel(somfeat.split('_')[0])
                    f.suptitle(titlestr, fontsize=8)
                    f.tight_layout()
                    figsaver(f, make_savename('%sVsH_feat%s' %('RAW', somfeat.split('_')[0])), closeit=True)



