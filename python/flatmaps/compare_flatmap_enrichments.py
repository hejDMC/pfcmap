import os
import yaml
import h5py
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import kendalltau,pearsonr
import pandas as pd


pathpath,run1,run2,stag1,stag2,ncluststr,cmethod,roi_tag,thistag,reftag = sys.argv[1:]


'''
run1 = 'runC00dMP3_brain'
run2 = 'runIBLPasdMP3_brain_pj'
stag1,stag2 = ['Carlen','IBL']
ncluststr = '8'
cmethod = 'ward'
pathpath = 'PATHS/filepaths_carlen.yml'
roi_tag = 'gaoRois'
thistag = 'deepRois'
reftag = 'refall'
'''
stags = [stag1,stag2]


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import tessellation_tools as ttools

basetag = 'emat'#that doesn't matter here
statsfile1 = os.path.join(pathdict['statsdict_dir'],'enrichmentdict__%s__rois_%s__%s__ncl%s_%s.h5'%(basetag,roi_tag,run1,ncluststr,cmethod))
statsfile2 = os.path.join(pathdict['statsdict_dir'],'enrichmentdict__%s__rois_%s__%s__ncl%s_%s.h5'%(basetag,roi_tag,run2,ncluststr,cmethod))

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_base =  pathdict['figdir_root'] + '/dataset_comparison_rois/%s/%s/%s'%(reftag,roi_tag,thistag)
figdir_mother = os.path.join(figdir_base,'%s__vs__%s'%(run2,run1))
outfile = os.path.join(figdir_mother,'correlation_values.xlsx')

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%sVs%s_%s_%s_%s.%s'%(stags[1],stags[0],thistag,roi_tag,reftag,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




avaldict_temp = {}
for statsfile,stag in zip([statsfile1,statsfile2],stags):
    with h5py.File(statsfile,'r') as hand:
        statshand =hand[reftag]['enr'][thistag]['src_stats']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        #presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval.count('|sup') and not aval=='na' \
        #                        and not (S.check_pfc(aval) and not aval.count('|deep'))])
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps])
        avaldict_temp[stag] = mystats['avals1'][presel_inds]


avals_avail = np.array([lab for lab in avaldict_temp[stags[0]] if lab in avaldict_temp[stags[1]]])


dsdict = {}
for statsfile,stag in zip([statsfile1,statsfile2],stags):
    with h5py.File(statsfile,'r') as hand:
        statshand = hand[reftag]['enr'][thistag]['src_stats']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([np.where(mystats['avals1']==aval)[0][0] for aa, aval in enumerate(avals_avail)])
        assert (mystats['avals1'][presel_inds] == avals_avail).all(),'mismatching avalues'
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([aval.split('|')[0] for aval in sdict['avals1']])
        dsdict[stag] = sdict


titlestr = str(roi_tag)
statsfn = S.enr_fn
nregs = len(avals_avail)
ncl = int(ncluststr)

cmap = mpl.cm.get_cmap(S.cmap_clust)#
norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cdict_clust = {lab:cmap(norm(lab)) for lab in np.arange(ncl)}

corrdict = {tag:{} for tag in ['IBLvsCarlen']+stags}
vmin,vmax = -1,1
extend = 'neither'

#getting the flatmap files and prepare coloring
if roi_tag == 'dataRois':
    flatmapfile = str(S.roimap_path)
elif roi_tag == 'gaoRois':
    flatmapfile = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')
else:
    assert 0, 'unknown roi_tag: %s'%roi_tag

region_file = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCregions.h5')
with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cmap_clust = mpl.cm.get_cmap(S.cmap_clust)#
cdict_clust = {lab:cmap_clust(norm(lab)) for lab in np.unique(np.arange(ncl))}
roidict_regs = ttools.get_regnamedict_of_rois(region_file,polygon_dict)
roi_colors = {myroi: S.cdict_pfc[reg] for myroi, reg in roidict_regs.items()}
sorted_rois = np.array([roi for area in S.PFC_sorted  for roi in avals_avail if roidict_regs[roi.split('|')[0]]==area])
cvec = np.array([roi_colors[roi.split('|')[0]] for roi in sorted_rois])
sortinds = np.array([np.where(avals_avail==roi)[0][0] for roi in sorted_rois])
assert (avals_avail[sortinds] == sorted_rois).all(),'sorting mismatch'
X1,X2 = [statsfn(dsdict[stag])[sortinds] for stag in stags]

for cfn,cfnlab in zip([kendalltau,pearsonr],['KT','CC']):
    corrmat = np.zeros((ncl,ncl))
    pmat = np.zeros((ncl,ncl))
    for ii in np.arange(ncl):
        for jj in np.arange(ncl):
            corrmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).correlation#[0,1]
            pmat[ii,jj] = cfn(X1[:,ii],X2[:,jj]).pvalue#[0,1]


    corrdict['IBLvsCarlen'][cfnlab] = {'pvalues':pmat,'corrvalues':corrmat}



    f,axarr = plt.subplots(1,2,figsize=(3.6, 3.35),gridspec_kw={'width_ratios':[1,0.1]})
    ax,cax = axarr
    f.subplots_adjust(left=0.15,bottom=0.15,right=0.82,wspace=0.1)
    im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap='PiYG_r')
    ax.set_xlabel(stags[0])
    ax.set_ylabel(stags[1])
    ax.set_xticks(np.arange(ncl))
    ax.set_yticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    tlabs = ax.set_yticklabels(np.arange(ncl)+1,fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    ax.set_aspect('equal')
    f.colorbar(im,cax=cax,extend=extend)
    ax.set_title(titlestr)
    cax.set_title(cfnlab)
    ax.text(0.99,1.01,'N:%i'%(nregs),transform=ax.transAxes,ha='right',va='bottom')
    figsaver(f,'clustcorr_%s_%s_%s'%(reftag,titlestr,cfnlab))


    xvec = np.arange(ncl)
    corrvec = np.diag(corrmat)
    mcorr = np.mean(corrvec)
    f, ax = plt.subplots(figsize=(1.5 + 0.2 * ncl, 2))
    f.subplots_adjust(left=0.3, bottom=0.3)
    blist = ax.bar(xvec, corrvec, color='k')
    ax.set_xticks(xvec)
    tlabs = ax.set_xticklabels(xvec + 1, fontweight='bold')
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        tlabs[cc].set_color(col)
        blist[cc].set_color(col)
        # ax.xaxis.get_ticklabels()
    ax.axhline(mcorr,color='grey',linestyle='--')
    ax.text(0.99,mcorr,'%1.2f'%mcorr,transform=mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData),ha='right',va='bottom',color='grey')
    ax.set_ylabel(cfn.__name__)
    ax.set_xlabel('category')
    ax.set_xlim([xvec.min() - 0.5, xvec.max() + 0.5])
    ax.set_title(' vs '.join(stags))
    for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
    figsaver(f,'clustcorrDiag_%s_%s_%s'%(reftag,titlestr,cfnlab))


    for mymat,stag in zip([X1,X2],stags):

        ncl = int(ncluststr)
        corrmat = np.zeros((ncl,ncl))
        pmat = np.zeros((ncl,ncl))

        for ii in np.arange(ncl):
            for jj in np.arange(ncl):
                corrmat[ii,jj] = cfn(mymat[:,ii],mymat[:,jj]).correlation#[0,1]
                pmat[ii,jj] = cfn(mymat[:,ii],mymat[:,jj]).pvalue#[0,1]

        corrdict[stag][cfnlab] = {'pvalues':pmat,'corrvalues':corrmat}




        f,axarr = plt.subplots(1,2,figsize=(3.6, 3.35),gridspec_kw={'width_ratios':[1,0.1]})
        ax,cax = axarr
        f.subplots_adjust(left=0.15,bottom=0.15,right=0.82,wspace=0.1)
        im = ax.imshow(corrmat.T,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap='PiYG_r')
        ax.set_xlabel(stag)
        ax.set_ylabel(stag)
        ax.set_xticks(np.arange(ncl))
        ax.set_yticks(np.arange(ncl))
        tlabs = ax.set_xticklabels(np.arange(ncl)+1,fontweight='bold')
        for cc in np.arange(ncl):
            tlabs[cc].set_color(cdict_clust[cc])
        tlabs = ax.set_yticklabels(np.arange(ncl)+1,fontweight='bold')
        for cc in np.arange(ncl):
            tlabs[cc].set_color(cdict_clust[cc])
        ax.set_aspect('equal')
        f.colorbar(im,cax=cax,extend=extend)
        ax.set_title(titlestr)
        cax.set_title(cfnlab)
        ax.text(0.99,1.01,'N:%i'%(nregs),transform=ax.transAxes,ha='right',va='bottom')
        figsaver(f,'%sONLY_clustcorr_%s_%s_%s'%(stag,reftag,titlestr,cfnlab))

with pd.ExcelWriter(outfile) as writer:
    for stag in corrdict.keys():
        for cfnlab in corrdict[stag].keys():
            for flav in corrdict[stag][cfnlab].keys():
                outmat = corrdict[stag][cfnlab][flav]
                df = pd.DataFrame(data=outmat,columns=np.arange(ncl)+1,index=np.arange(ncl)+1)
                df.to_excel(writer, sheet_name='%s_%s_%s'%(stag,cfnlab,flav))




allX = np.hstack([X.flatten() for X in [X1,X2]])
x_bounds = np.array([np.nanmin(allX),np.nanmax(allX)])
bound_amp = np.diff(x_bounds)[0]
my_ylim = x_bounds+np.array([-0.1,0.1])*bound_amp
cdict = {'Carlen': 'firebrick',\
         'IBL': 'darkorange'}


alabs = np.array([aval.split('|')[0] for aval in avals_avail[sortinds]])
xvec = np.arange(nregs)

for line_mode in ['conn','default']:
    for ii in np.arange(ncl):
        f, ax = plt.subplots(figsize=(4, 3))
        f.subplots_adjust(bottom=0.18)
        vec1,vec2 = X1[:,ii],X2[:,ii]
        statsstrlist = ['%s:%1.2f'%(ctag,cfn(vec1,vec2).correlation) for cfn,ctag in zip([kendalltau,pearsonr],['KT','CC'])]
        ax.set_title('%s clust%i   %s'%(titlestr,ii+1,'; '.join(statsstrlist)))
        for Xmat,runset in zip([X1,X2],[stags[0],stags[1]]):
            col = cdict[runset]
            if line_mode == 'conn':
                ax.plot(xvec, Xmat[:, ii], '.-', color=col, ms=12, mec='none', mfc=col, lw=0)
            elif line_mode == 'default':
                ax.plot(xvec, Xmat[:, ii], '.-', color=col, ms=12, mec='none', mfc=col, lw=1)
        if line_mode == 'conn':
            for xx in np.arange(len(alabs)):
                v1, v2 = X1[xx, ii],X2[xx, ii]
                if np.mean([v1, v2]) != np.nan:
                    ax.plot([xx, xx], [v1, v2], color='silver', zorder=-5)  # cdict[runset_sel_dict[utype][0]]
        #ax.plot(xvec,vec1,'o')
        #ax.plot(xvec,vec2,'ro')
        #ax.plot(xvec,vec1,'k',alpha=0.2)
        #ax.plot(xvec,vec2,'r',alpha=0.2)
        ax.set_xticks(xvec)
        tlabs = ax.set_xticklabels(alabs,rotation=-90)
        for aa in np.arange(len(alabs)):
            tlabs[aa].set_color(cvec[aa])
        ax.text(0.99,0.98,stags[0],color=cdict[stags[0]],transform=ax.transAxes,ha='right',va='top')
        ax.text(0.99,0.90,stags[1],color=cdict[stags[1]],transform=ax.transAxes,ha='right',va='top')
        ax.set_ylabel('enrichment')
        myxlim = np.array(ax.get_xlim())
        ax.fill_between(myxlim, np.zeros_like(myxlim) - 2, np.zeros_like(myxlim) + 2, color='grey', alpha=0.5)
        ax.set_ylim(my_ylim)
        ax.set_xlim(myxlim)
        f.tight_layout()
        figsaver(f,'diag_comparisons/%s/clust%i_comparison_%s_%s'%(line_mode,ii+1,reftag,titlestr))


diff_matz = np.zeros((ncl,nregs))
diff_mat = np.zeros((ncl,nregs))

for ii in np.arange(ncl):
    v1,v2 = X1[:,ii],X2[:,ii]
    z1,z2 = [(myv-np.mean(myv))/np.std(myv) for myv in [v1,v2]]
    diff_matz[ii] = z2-z1
    diff_mat[ii] = v2-v1


for mydmat,difftag in zip([diff_mat,diff_matz],['absdiff','reldiff']):
    vlim = np.max(np.abs(mydmat))
    meandiff = np.abs(mydmat).mean(axis=0)
    sortinds = np.argsort(meandiff)
    #sortinds =cfns.get_sortidx_featdist(diff_mat.T, linkmethod='ward')#np.argsort(meandiff)
    f,axarr = plt.subplots(1,3,gridspec_kw={'width_ratios':[1,0.1,0.1]})
    ax,aax,cax = axarr
    im = ax.imshow(mydmat[:,sortinds].T,origin='lower',aspect='auto',cmap='PRGn_r',vmin=-vlim,vmax=vlim)
    aax.imshow(meandiff[sortinds][:,None],origin='lower',aspect='auto',cmap='inferno')
    aax.set_axis_off()
    aax.set_title(r'$\overline{|\delta|}$')
    cax.set_title('diff [z]')
    ax.set_title(' vs '.join(stags))
    ax.set_yticks(np.arange(nregs))
    ax.set_yticklabels(alabs[sortinds])
    ax.set_xticks(np.arange(ncl))
    tlabs = ax.set_xticklabels(np.arange(ncl) + 1, fontweight='bold')
    for cc in np.arange(ncl):
        tlabs[cc].set_color(cdict_clust[cc])
    f.colorbar(im,cax=cax)
    figsaver(f,'diag_comparisons/regionalDifferences__%s_%s_%s'%(reftag,titlestr,difftag))


# compare two ds as in harris enrichment, but for gao rois, side by side

if roi_tag == 'gaoRois':
    import csv
    gao_hier_path = pathdict['gao_hierarchy_file']

    with open(gao_hier_path, newline='') as csvfile:
        hdata = list(csv.reader(csvfile, delimiter=','))[1:]
    hdict = {int(float(el[0])): float(el[1]) for el in hdata}
    allrois_gao = np.sort(np.array(list(hdict.keys())))
    hvec = np.array([hdict[key] for key in allrois_gao])

    alabs2 = np.array([int(val) for val in alabs])

    hinds = np.array([np.where(allrois_gao==roi)[0][0] for roi in alabs2])
    htemp = hvec[hinds]
    h_cond = ~np.isnan(htemp)

    X_list = [myX[h_cond] for myX in [X1,X2]]
    my_hier = htemp[h_cond]

    for cfn,statsflav in zip([kendalltau,pearsonr],['kendall','pearson']):
        corrmat = np.vstack([np.array([cfn(my_hier,subx).correlation for subx in myX.T]) for myX in X_list]).T

        f,ax = plt.subplots(figsize=(1.5+0.2*ncl,2))
        f.subplots_adjust(left=0.3,bottom=0.3)
        xvec = np.arange(ncl)
        for ii,[stag,offset,alph] in enumerate(zip(stags,[-0.2,0.2],[0.5,1.])):
            #ii = 1
            #stag = stags[ii]
            corrvec = corrmat[:,ii]
            xvec_off = np.arange(ncl)+offset

            blist = ax.bar(xvec_off,corrvec,color='k',width=0.3,alpha=alph)
            for cc in np.arange(ncl):
                col = cdict_clust[cc]
                blist[cc].set_color(col)

        ax.set_xticks(xvec)
        tlabs = ax.set_xticklabels(xvec+1,fontweight='bold')
        for cc in np.arange(ncl):
            col = cdict_clust[cc]
            tlabs[cc].set_color(col)
            #ax.xaxis.get_ticklabels()
        ax.set_ylabel(statsflav)
        ax.set_xlabel('category')
        ax.set_xlim([xvec.min()-0.5,xvec.max()+0.5])
        ax.set_title(' and '.join(stags))
        #ax.set_ylim([-0.9,0.9])
        for pos in ['top','right']:ax.spines[pos].set_visible(False)
        figsaver(f,'compareHierCorr_%s_corr%s'%(''.join(stags),statsflav))


    #now the values for both individually, side by side (previous was for only rois where BOTH ds have sufficient data)

    corrfilename1 = os.path.join(pathdict['statsdict_dir'],
                                'gaocorr__%s__ncl%s_%s_enr.h5' % (run1, ncluststr, cmethod))
    corrfilename2 = os.path.join(pathdict['statsdict_dir'],
                                'gaocorr__%s__ncl%s_%s_enr.h5' % (run2, ncluststr, cmethod))

    for statsflav in ['kendall', 'pearson']:

        corrlist = []
        for cfile in [corrfilename1,corrfilename2]:
            with h5py.File(cfile,'r') as hand:
                corrlist += [hand['%s/corrvec'%(statsflav)][()]]

        corrmat = np.vstack(corrlist).T

        f, ax = plt.subplots(figsize=(1.5 + 0.2 * ncl, 2))
        f.subplots_adjust(left=0.3, bottom=0.3)
        xvec = np.arange(ncl)
        for ii, [stag, offset, alph] in enumerate(zip(stags, [-0.2, 0.2], [0.5, 1.])):
            # ii = 1
            # stag = stags[ii]
            corrvec = corrmat[:, ii]
            xvec_off = np.arange(ncl) + offset

            blist = ax.bar(xvec_off, corrvec, color='k', width=0.3, alpha=alph)
            for cc in np.arange(ncl):
                col = cdict_clust[cc]
                blist[cc].set_color(col)

        ax.set_xticks(xvec)
        tlabs = ax.set_xticklabels(xvec + 1, fontweight='bold')
        for cc in np.arange(ncl):
            col = cdict_clust[cc]
            tlabs[cc].set_color(col)
            # ax.xaxis.get_ticklabels()
        ax.set_ylabel(statsflav)
        ax.set_xlabel('category')
        ax.set_xlim([xvec.min() - 0.5, xvec.max() + 0.5])
        ax.set_title(' and '.join(stags))
        # ax.set_ylim([-0.9,0.9])
        for pos in ['top', 'right']: ax.spines[pos].set_visible(False)
        figsaver(f, 'compareHierCorrDSwise_%s_corr%s' % (''.join(stags), statsflav))
