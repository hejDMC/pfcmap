import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import os
import pandas as pd
import numpy as np
import h5py
import git

pathpath = 'PATHS/general_paths.yml'
with open(pathpath,'r') as f: pdict = yaml.safe_load(f)
genpath = pdict['metricsextraction']
dstpath = os.path.join(genpath,'psth_scores')
timeselfolder = 'timeselections_psth' #'timeselections'
psth_path = os.path.join(genpath,'psth')
figpath = pdict['fig_path']+'/psth_pca'
stylepath = 'config/presentation.mplstyle'

statetag = 'active'
reftag = '__all'

ncomps_max = 30
kstd_ms = 10
tbounds = [0,0.65]

savestub = 'TSELpsth2to7__STATE%s%s_psth_ks%i'%(statetag, reftag,int(kstd_ms))

plt.style.use(stylepath)

tablepath = 'config/datatables/allrecs_allprobes.xlsx'

not_allowed=['-', '?']
sheet = 'sheet0'
df = pd.read_excel(tablepath, sheet_name=sheet)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
allrecrows = df['recid']
isrec_bool = allrecrows.str.contains('probe',na=False)
allowed_bool = ~df['usable_gen'].isin(not_allowed)
bools_list = [allowed_bool,isrec_bool]
cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
recids = list(allrecrows[cond_bool].values)
datasetlabs = np.unique(df['datasetlab'][cond_bool])
exptypelist = list(df['exptype'][cond_bool])

exptypes = np.unique(exptypelist)


# collect all psth in the relevant time window




datadict = {}
for recid in recids:
    print(recid)
    srctag = '%s__%s'%(recid, savestub)
    psthfile = os.path.join(psth_path,'%s.h5'%(srctag))

    with h5py.File(psthfile) as hand:

        nspikes,nspikes_pre,psth0,psth_tvec,uids = [hand[key][()] for key in  ['nspikes', 'nspikes_pre', 'psth', 'psth_tvec', 'uids']]
        n_tints = hand.attrs['n_tints']
        tfile = hand.attrs['tfile']
    pre_bool =psth_tvec<0
    dur_tot = psth_tvec[-1]-psth_tvec[0]
    rate_overall =  nspikes/dur_tot
    rate_pre = nspikes_pre/-psth_tvec.min()
    cond1 = rate_pre>0.1
    cond2 = rate_overall>0.1
    cond3 = ~np.isnan(psth0.min(axis=1))
    condlist = np.array([cond1,cond2,cond3])
    condbool = np.sum(condlist,axis=0)==len(condlist)
    psth = psth0[condbool]
    psth_normed = (psth -np.mean(psth[:,pre_bool],axis=1)[:,None]) /np.std(psth[:,pre_bool],axis=1)[:,None]
    tsel_bool = (psth_tvec>=tbounds[0]) & (psth_tvec<=tbounds[1])
    datadict[recid] = {'psth':psth_normed[:,tsel_bool],'uids':uids,'uids_psth':uids[condbool],'n_tints':n_tints,\
                       'tfile':tfile,'pfile':psthfile}


for mode in ['zscore','shape']:
    datamat = np.vstack([datadict[recid]['psth'] for recid in recids])

    if mode == 'shape':
        X = datamat/datamat.sum(axis=1)[:,None]
        X_mean = np.mean(X,axis=0)

    elif mode == 'zscore':
        X_mean = np.mean(datamat,axis=0)
        X = datamat-X_mean[None,:]

    else:
        assert 0, 'unknown mode %s'%mode

    def figsaver(fig, nametag, closeit=True):
        figname = os.path.join(figpath, '%s_%s__%s.png'%(savestub,mode,nametag))
        figdir = os.path.dirname(figname)
        if not os.path.isdir(figdir): os.makedirs(figdir)
        #f.suptitle('%s - %s'%(recid,mode))

        fig.savefig(figname)
        if closeit: plt.close(fig)


    expl_thr = 0.8 if mode=='shape' else 0.95
    covmat = np.cov(X.T)#featxfeat
    evals, evecs = np.linalg.eig(covmat)#feats,featxfeat
    v_explained = evals[:ncomps_max]/np.sum(evals)
    ncomps = np.where(np.cumsum(v_explained)>expl_thr)[0][0]+1
    evecs_used = evecs[:,:ncomps]

    #f,ax = plt.subplots()
    #ax.plot(psth_tvec[tsel_bool],X[200])

    f,ax = plt.subplots(figsize=(4,3))
    f.subplots_adjust(left=0.15,bottom=0.2)
    ax.plot(psth_tvec[tsel_bool],X_mean,'k')
    ax.set_ylabel('mean psth')
    ax.set_xlabel('time [s]')
    ax.set_xlim(tbounds)
    figsaver(f,'X_mean')

    f,ax = plt.subplots(figsize=(5,4))
    f.subplots_adjust(left=0.15,bottom=0.15,right=0.85)
    ax.plot(np.arange(ncomps_max)+1,v_explained,'k.-')
    ax2 = ax.twinx()
    ax2.plot(np.arange(ncomps_max)+1,np.cumsum(v_explained),'r.-')
    ax2.axhline(expl_thr,color='pink',zorder=-10)
    ax2.axvline(ncomps,color='pink',zorder=-10)
    ax2.set_ylabel('cumulative',color='r')
    ax2.text(ncomps+0.5,expl_thr-0.02,'Ncomps:%i'%ncomps,ha='left',va='top',color='r')
    ax.set_xlim([1,15])
    ax.set_ylabel('v-frac. explained')
    ax.set_xlabel('n_comps')
    figsaver(f,'varexpl')

    cmap = mpl.cm.get_cmap('jet')
    cnorm = mpl.colors.Normalize(vmin=0, vmax=ncomps-1)
    colvals = cmap(cnorm(np.arange(ncomps)))

    f,ax = plt.subplots(figsize=(5,4))
    f.subplots_adjust(left=0.2,bottom=0.15,right=0.85)
    for ee,evec in enumerate(evecs_used.T):
        col = colvals[ee]
        ax.plot(psth_tvec[tsel_bool],evec,color=col,lw=2,zorder=-ee)
        ax.text(1.01,0.99-ee*0.1,'evec%i'%(ee+1),color=col,fontweight='bold',transform=ax.transAxes,va='top',ha='left')
    ax.set_xlim(tbounds)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('evec. amp.')
    figsaver(f,'evecs')


    scores = np.dot(X,evecs_used)#nsamps x npcs --> just for control here

    #get the scores per recid and save
    for recid in recids:
        #recid = recids[0]
        psth_mat,uids, uids_psth,n_tints,tfile,pfile = [datadict[recid][key] for key in ['psth','uids','uids_psth','n_tints','tfile','pfile']]
        savestub_rec = os.path.basename(pfile).split('.')[0]
        score_mat = np.zeros((len(uids),ncomps))
        mat_counter = 0
        for uu,uid in enumerate(uids):
            if uid in uids_psth:
                upsth = psth_mat[mat_counter]
                uX = upsth-X_mean
                uScores = np.dot(uX[None,:],evecs_used)#nsamps x npcs
                mat_counter += 1
            else:
                uScores = np.ones(ncomps)*np.nan
            score_mat[uu] = uScores


        savename = os.path.join(dstpath,'%s__PCAscores%s.h5'%(savestub_rec,mode))
        #print('saving to %s'%savename)

        with h5py.File(savename,'w') as hand:
            hand.create_dataset('scores',data=score_mat,dtype='f')
            #hand.create_dataset('psth_tvec',data=psth_tvec,dtype='f')
            #hand.create_dataset('psth_mat',data=psth_mat,dtype='f')
            hand.create_dataset('evals',data=evals[:100],dtype='f')
            #hand.create_datastes('evecs',data=evecs_used,dtype='f')
            hand.create_dataset('tbounts',data=np.array(tbounds),dtype='f')
            hand.create_dataset('uids',data=uids,dtype='i')
            #hand.create_dataset('uids_psth',data=uids_psth,dtype='i')
            hand.attrs['recid'] = recid
            hand.attrs['n_tints'] = n_tints
            hand.attrs['kstd_ms'] = kstd_ms
            hand.attrs['tfile'] = tfile
            hand.attrs['pfile'] = pfile
            hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
            #hand.attrs['srcfile'] = __file__




'''
#rconX = np.dot(scores,evecs_used.T)
# check via reconstruction
rconX = np.dot(scores,evecs_used.T)

ii = 1800
f,ax = plt.subplots()
ax.plot(X[ii],'k')
ax.plot(rconX[ii],'g')

ii = 500
f,ax = plt.subplots()
ax.plot(datamat[ii],'k')
ax.plot(rconX[ii]+X_mean,'g')

U, S, V_T = np.linalg.svd(X, full_matrices=False)

#evecs2 = V_T.T
#evals2 = (S**2)/(len(X)-1)
explmat[ff] = evals/np.sum(evals)
evecmat[ff] = evecs[:,:Ncomps].T
'''

