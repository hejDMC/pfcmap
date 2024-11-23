import yaml
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

"""
pathpath = 'PATHS/filepaths_carlen.yml'
myrun =  'runC00dMP3_brain'
cmethod = 'ward'
ncluststr = '8'
"""

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import evol_helpers as ehelpers
from pfcmap.python import settings as S

statstag = 'TSELprestim3__STATEactive__all'
savepath_evol = os.path.join(pathdict['savepath_SOM'].replace('SOMs','category_evolutions'))
genfigdir = pathdict['figdir_root'] + '/stability'

tasks_sorted = ['Passive','Attention','Aversion_ctrl','Aversion_stim','Context','Detection']#later in the script, it will be checked whether those are all available
ncl = int(ncluststr)

targetdir = os.path.join(genfigdir,'%s_ncl%i'%(myrun,ncl))
if not os.path.isdir(targetdir): os.makedirs(targetdir)
trials_per_block_file = os.path.join(targetdir,'trials_per_block.xlsx')

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(targetdir, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




recids = S.get_allowed_recids_from_table(tablepath=pathdict['tablepath'],sheet='sheet0',not_allowed=['-', '?'])

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

cmap = mpl.cm.get_cmap(S.cmap_clust)#
norm = mpl.colors.Normalize(vmin=0, vmax=ncl-1)
cdict_clust = {lab:cmap(norm(lab)) for lab in np.arange(ncl)}




###preparing plotting
def statstitlefn(ax,tcmat):
    stabi = np.nansum(np.diag(tcmat))/np.nansum(tcmat)
    ax.set_title('SI: %1.2f - %i'%(stabi,np.nansum(tcmat)),fontsize=9)


def statstitlefn_probs(ax,tcmat):
    probmat = ehelpers.get_transprobmat(tcmat)
    avgprob = np.nanmean(np.diag(probmat))
    ax.set_title('%1.2f - %i'%(avgprob,tcmat.sum()),fontsize=9)

def statstitlefn_ratio(ax,tcmat):
    ratio_mat = ehelpers.get_origByExpected_transcounts(tcmat)
    stabi = np.nanmean(np.diag(ratio_mat))
    ax.set_title('%1.2f - %i'%(stabi,tcmat.sum()),fontsize=9)


def statstitlefn_cocoeff(ax,tcmat):
    cocoeffs = ehelpers.get_cocoeff_mat(tcmat)
    avgcoco = np.nanmean(np.diag(cocoeffs))
    ax.set_title('%1.2f - %i'%(avgcoco,tcmat.sum()),fontsize=9)

divnorm_oByE = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=8)


fn_dict = {'orig':lambda tcmat:tcmat,\
             'expected':lambda tcmat:ehelpers.get_expected_transcounts(tcmat),\
             'prob':lambda tcmat: ehelpers.get_transprobmat(tcmat),\
             'origByExp':lambda tcmat: ehelpers.get_origByExpected_transcounts(tcmat),\
           'cocoeff':lambda tcmat:ehelpers.get_cocoeff_mat(tcmat)}


label_dict = {'orig':'count',\
              'expected': 'expected count',\
              'prob':'prob(x+1=j|x=i)',\
              'origByExp':'count/count_exp',\
              'origByExp_climited':'count/count_exp',\
              'cocoeff':'CoiCoeff'}

plotflavs = ['orig','expected','prob','origByExp','cocoeff'] #also important for the order

plot_empty = lambda myax: [myax.set_facecolor('whitesmoke'),myax.set_xticks([]),myax.set_yticks([]),plt.setp(ax.spines.values(), color='w')]#for high-res matrix plot



###calculating stuff

tempdict = {}
not_available = []
for recid in recids:
    evolfile = os.path.join(savepath_evol, '%s__%s__%s%i__%s__catevols.h5' % (recid, myrun, cmethod, ncl, statstag))
    if os.path.isfile(evolfile):
        with h5py.File(evolfile) as statshand:
            tempdict[recid] = {key.replace('/',''):val for key,val in uloader.unpack_statshand(statshand, remove_srcpath=True).items()}
    else:
        not_available += [recid]

tasktypes = np.unique([tempdict[recid]['task_full'] for recid in tempdict.keys()])

if set(tasks_sorted)==set(tasktypes):
    pass
else:
    tasks_sorted = [task for task in tasks_sorted if task in tasktypes]
    tasks_sorted += [task for task in tasktypes if not task in tasks_sorted]
    print('WARNING manual task sorting set did not match present task set')

datadict = {}
for tasktype in tasktypes:
    myrecs = [recid for recid in tempdict.keys() if tempdict[recid]['task_full']==tasktype]
    datadict[tasktype] = {recid:tempdict[recid] for recid in myrecs}



with pd.ExcelWriter(trials_per_block_file) as writer:
    for tasktype in tasktypes:
        myrecs = list(datadict[tasktype].keys())
        tpb_list = [datadict[tasktype][recid]['trials_per_block'] for recid in myrecs]
        maxdim = np.max([len(tpb) for tpb in tpb_list])
        tpb_mat = np.zeros((len(myrecs),maxdim),dtype=int)-1
        for rr,myrec in enumerate(myrecs):
            vals = tpb_list[rr]
            tpb_mat[rr,:len(vals)] = vals

        df = pd.DataFrame(data=tpb_mat,columns=np.arange(maxdim)+1,index=myrecs,dtype=int)
        df.to_excel(writer, sheet_name=tasktype)

#--> review this table and come up with a sensible threshold!
trials_per_block_min = 15 #that is overall!!!!
label_mapper_aversion = {0:'T1',1:'O1',2:'O2',3:'T2',4:'A'}



######################################################
####### BASIC STATS WITHOUT GOING AFTER INDIVIDUAL TRANSITIONS, just asking: if you are of a certain category, will you be this category also in individual blocks?

cdict_clust_nan = {ii:cdict_clust[ii] for ii in np.arange(ncl)}
cdict_clust_nan.update({ncl:'grey'})


match_dict = {tasktype:{} for tasktype in tasktypes}
for tasktype in tasktypes:
    taskdict = datadict[tasktype]
    myrecs = list(taskdict.keys())
    for myrec in myrecs:
        #myrec = myrecs[0]
        sdict = taskdict[myrec]

        cids_full = sdict['clust_vec']
        cids_blocks = sdict['clust_blocks']

        match_mat = np.zeros((ncl,ncl+1))
        for cc in np.arange(ncl):
            block_matchers = cids_blocks[:,cids_full==cc].flatten()
            match_mat[cc] = np.array([np.sum(block_matchers==cc2) for cc2 in np.r_[np.arange(ncl),-1]])
        match_dict[tasktype][myrec] = match_mat


#def plot_match_panel()
namodes = ['woNA','withNA']
cmap_matches = 'Greys'
vminmax = [0, 1]
for tasktype in tasktypes:
    subdict = match_dict[tasktype]

    figsaver_ds = lambda fig, nametag: figsaver(fig, os.path.join('datasets', tasktype, nametag + '__%s' % tasktype),
                                                closeit=True)
    myrecs = list(subdict.keys())
    n_r = len(myrecs)

    for namode in namodes:
        f,axarr = plt.subplots(1,n_r,figsize=(n_r*1.8+2,3),gridspec_kw={'width_ratios':[1]*n_r})
        for rr,myrec in enumerate(myrecs):
            ax = axarr[rr]
            if namode == 'woNA':
                showmat = subdict[myrec][:,:ncl]
            elif namode == 'withNA':
                showmat = subdict[myrec]
            ehelpers.plot_block_orig_matchmat(ax,showmat,cdict_clust,cmap=cmap_matches,grid_col = 'k',write_col = 'm',vminmax = vminmax,write_data=False,write_counts=True,labelfs=8)
            ax.set_title(myrec,fontsize=8)#.split('_')[0]
        axarr[0].set_ylabel('gen. cat.')
        axarr[0].set_xlabel('block cat.')
        f.tight_layout()
        figsaver_ds(f,'orig_block_matchmat_allrecs_%s'%(namode))

        if namode == 'woNA':
            sum_mat = np.array([val[:,:ncl] for val in subdict.values()]).sum(axis=0)
        elif namode == 'withNA':
            sum_mat = np.array([val for val in subdict.values()]).sum(axis=0)

        f,ax = plt.subplots(figsize=(2.5,2))
        ehelpers.plot_block_orig_matchmat(ax,sum_mat,cdict_clust,cmap=cmap_matches,grid_col = 'k',write_col = 'm',vminmax = vminmax,write_data=False,write_counts=True,labelfs=10)
        ax.set_ylabel('gen. cat.')
        ax.set_xlabel('block cat.')
        ax.set_title(tasktype)
        f.tight_layout()
        figsaver_ds(f,'orig_block_matchmat_sumrecs_%s'%(namode))


        f,ax = plt.subplots(figsize=(3.5,3))
        ehelpers.plot_block_orig_matchmat(ax,sum_mat,cdict_clust,cmap=cmap_matches,grid_col = 'k',write_col = 'm',vminmax = vminmax,write_data=True,write_counts=True,labelfs=10)
        ax.set_ylabel('gen. cat.')
        ax.set_xlabel('block cat.')
        ax.set_title(tasktype)
        f.tight_layout()
        figsaver_ds(f,'orig_block_matchmat_sumrecs_TXT_%s'%(namode))

        f,ax = ehelpers.plot_cmap(cmap_matches,vminmax)
        figsaver_ds(f, 'orig_block_matchmat_cmap')


#grandsummary saver
figsaver_summary = lambda fig, nametag: figsaver(fig, os.path.join('summary', nametag + '__summary'),
                                                closeit=True)

for namode in namodes:

    if namode == 'woNA':
        sum_mat = np.array([val[:,:ncl] for subdict in match_dict.values() for val in subdict.values()]).sum(axis=0)
    elif namode == 'withNA':
        sum_mat = np.array([val for subdict in match_dict.values() for val in subdict.values()]).sum(axis=0)

    f, ax = plt.subplots(figsize=(2.5, 2))
    ehelpers.plot_block_orig_matchmat(ax, sum_mat, cdict_clust, cmap=cmap_matches, grid_col='k', write_col='m',
                                      vminmax=vminmax, write_data=False, write_counts=True, labelfs=10)
    ax.set_ylabel('gen. cat.')
    ax.set_xlabel('block cat.')
    ax.set_title('all tasks')
    f.tight_layout()
    figsaver_summary(f, 'orig_block_matchmat_sumrecs_%s' % (namode))


    f,ax = plt.subplots(figsize=(3.5,3))
    ehelpers.plot_block_orig_matchmat(ax,sum_mat,cdict_clust,cmap=cmap_matches,grid_col = 'k',write_col = 'm',vminmax = vminmax,write_data=True,write_counts=True,labelfs=10)
    ax.set_ylabel('gen. cat.')
    ax.set_xlabel('block cat.')
    ax.set_title('all tasks')
    f.tight_layout()
    figsaver_summary(f,'orig_block_matchmat_sumrecs_TXT_%s'%(namode))


    #plot avarage diagonal
    match_sum = sum_mat.sum(axis=1)
    frac_mat = sum_mat / match_sum[:, None]

    f,ax = plt.subplots(figsize=(2.5,2))
    ehelpers.plot_clustbars(ax,np.diag(frac_mat),cdict_clust,show_mean=True)
    ax.set_title('all tasks diag')
    ax.set_ylabel('frac matches')
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='minor',color='silver',linestyle='dashed')
    ax.set_ylim([0,1])
    f.tight_layout()
    figsaver_summary(f, 'orig_block_matchmat_sumrecs_DIAG_%s' % (namode))


#comparing datasets by fractions on the diagonal
diag_dict = {namode: {} for namode in namodes}
for tasktype in tasktypes:
    subdict = match_dict[tasktype]
    for namode in ['woNA','withNA']:
        if namode == 'woNA':
            matchmat = np.array([val[:, :ncl] for val in subdict.values()]).sum(axis=0)
        elif namode == 'withNA':
            matchmat = np.array([val for val in subdict.values()]).sum(axis=0)
        match_sum = matchmat.sum(axis=1)
        frac_mat = matchmat / match_sum[:, None]
        diag_dict[namode][tasktype]= frac_mat[np.arange(ncl),np.arange(ncl)]



for namode in namodes:
    mydict = diag_dict[namode]
    task_mat = np.array([mydict[tasktype] for tasktype in tasks_sorted])
    n_tasks = len(tasktypes)
    xvec = np.arange(n_tasks)
    f,ax = plt.subplots(figsize=(3.2,2.5))
    f.subplots_adjust(left=0.3,bottom=0.2,right=0.98)
    for cc in np.arange(ncl):
        ax.plot(task_mat[:,cc],xvec,'o',mec=cdict_clust[cc],mfc='none',color=cdict_clust[cc],mew=2)
    ax.set_yticks(np.arange(n_tasks))
    ax.set_yticklabels(tasks_sorted)
    ax.set_xlabel('frac. matched')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='major',color='silver',linestyle='-')
    f.tight_layout()
    figsaver_summary(f, 'orig_block_matchmat_sumrecs_DATASETS_DIAG_bubbles_%s' % (namode))


    f,ax = plt.subplots(figsize=(3.2,2.5))
    f.subplots_adjust(left=0.3,bottom=0.2,right=0.98)
    ax.imshow(task_mat,origin='lower',aspect='auto',cmap=cmap_matches,vmin=vminmax[0],vmax=vminmax[1])
    ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='x',flav='num',fs=12)
    ax.set_yticks(np.arange(n_tasks))
    ax.set_yticklabels(tasks_sorted)
    f.tight_layout()
    figsaver_summary(f, 'orig_block_matchmat_sumrecs_DATASETS_DIAG_matmode_%s' % (namode))

###gettings stats to plot trials per block and clustcounts overall
gendict = {tasktype: {} for tasktype in tasktypes}
for tasktype in datadict.keys():

    taskdict = datadict[tasktype]

    myrecs0 = list(taskdict.keys())
    '''
    if tasktype.count(
            'Aversion'):  # this way we also dont need to take care of the namepairs of Aversion types beeing correct
        myrecs = []
        for myrec in myrecs0:
            if list(taskdict[myrec]['blocknames']) == ['tone', 'opto', 'opto', 'tone', 'air']:
                myrecs += [myrec]
    else:'''
    myrecs = [myrec for myrec in myrecs0]

    gendict[tasktype]['tpb'] = np.hstack([taskdict[myrec]['trials_per_block'] for myrec in myrecs])
    clids_orig = np.hstack([taskdict[myrec]['clust_vec'] for myrec in myrecs])
    clids_blocks = np.hstack([taskdict[myrec]['clust_blocks'].flatten() for myrec in myrecs])

    gendict[tasktype]['ccounts_orig'] = np.array([np.sum(clids_orig==cc) for cc in np.arange(ncl)])
    gendict[tasktype]['ccounts_blocks'] = np.array([np.sum(clids_blocks==cc) for cc in np.r_[np.arange(ncl),-1]])
    gendict[tasktype]['n_recs'] = len(myrecs)
    gendict[tasktype]['n_units'] = len(clids_orig)





# plot ntrials/block per task
f, ax = plt.subplots(figsize=(3.2, 2.5))
f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
for tt,tasktype in enumerate(tasks_sorted):
    vals = gendict[tasktype]['tpb']
    ax.plot(vals,np.full(len(vals),tt), 'o', mec='k', mfc='none', mew=1,alpha=0.5)
    ax.plot(np.median(vals),tt,'|',color='r',ms=10,mew=3)
ax.set_yticks(np.arange(n_tasks))
ax.set_yticklabels(tasks_sorted)
ax.set_xlabel('trials/block')
ax.set_axisbelow(True)
ax.yaxis.grid(True, which='major', color='silver', linestyle='-')
f.tight_layout()
figsaver_summary(f,'support_stats_taskcompare/trials_per_block')


heights = np.array([gendict[task]['n_recs'] for task in tasks_sorted])
f, ax = plt.subplots(figsize=(3.2, 2.5))
f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
ax.barh(np.arange(n_tasks),heights,color='k')
ax.set_yticks(np.arange(n_tasks))
ax.set_yticklabels(tasks_sorted)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
ax.set_xlabel('n recs')
f.tight_layout()
figsaver_summary(f,'support_stats_taskcompare/n_recs')


heights = np.array([gendict[task]['n_units'] for task in tasks_sorted])
f, ax = plt.subplots(figsize=(3.2, 2.5))
f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
ax.barh(np.arange(n_tasks),heights,color='k')
ax.set_yticks(np.arange(n_tasks))
ax.set_yticklabels(tasks_sorted)
ax.set_xlabel('n units')
f.tight_layout()
figsaver_summary(f,'support_stats_taskcompare/n_units')





count_flavs = ['ccounts_orig', 'ccounts_blocks']
xvec = np.arange(n_tasks)

for namode in namodes:

        if namode == 'woNA':
            count_mats = [np.vstack([gendict[tasktype][cflav][:ncl] for tasktype in tasks_sorted]) for cflav in count_flavs]
        elif namode == 'withNA':
            count_mats = [np.vstack([gendict[tasktype][cflav] for tasktype in tasks_sorted]) for cflav in count_flavs]


        frac_mats = [count_mat/count_mat.sum(axis=1)[:,None] for count_mat in count_mats]

        maxfrac = np.hstack([fmat.flatten() for fmat in frac_mats]).max()#to obtain same x-range

        for ff,cflav in enumerate(count_flavs):

            if not (namode=='withNA' and cflav=='ccounts_orig'):#because this is the same as the woNA
                frac_mat = frac_mats[ff]
                f, ax = plt.subplots(figsize=(3.2, 2.5))
                f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
                for cc in np.arange(ncl):
                    ax.plot(frac_mat[:, cc], xvec, 'o', mec=cdict_clust[cc], mfc='none', color=cdict_clust[cc], mew=2)
                if namode == 'withNA' and frac_mat.shape[1]>ncl:
                    ax.plot(frac_mat[:, ncl], xvec, 'x', mec='k',mew=2)
                ax.set_yticks(np.arange(n_tasks))
                ax.set_yticklabels(tasks_sorted)
                ax.set_xlabel('frac. clust.')
                titlelab = cflav.replace('count','frac')
                ax.set_title(titlelab)
                ax.set_xlim([0,maxfrac+0.1*maxfrac])
                ax.set_axisbelow(True)
                ax.yaxis.grid(True, which='major', color='silver', linestyle='-')
                f.tight_layout()
                figsaver_summary(f,'support_stats_taskcompare/clustcomposition_%s'%(titlelab))

# cluster composition over trials per task type
blocks_show_dict = {'Passive':['B1', 'B2', 'B3', 'B4'],\

                    'Attention':['B1', 'B2', 'B3', 'B4'],\
                    'Aversion_ctrl':['tone', 'opto', 'opto', 'tone', 'air'],\
                    'Aversion_stim':['tone', 'opto', 'opto', 'tone', 'air'],\
                    'Context':['off1', 'on1', 'off2', 'on2', 'off3', 'on3', 'off4', 'on4', 'off5'],\
                    'Detection':['B1', 'B2', 'B3']
                    }# 'Opto':['B1', 'B2', 'B3'],\

perc_ratio_mat = np.zeros((n_tasks,ncl))
for tt,tasktype in enumerate(tasks_sorted):
    taskdict = datadict[tasktype]
    n_allowed = len(blocks_show_dict[tasktype])
    if tasktype.count('Aversion'):
        myrecs = [myrec for myrec in taskdict.keys() if list(taskdict[myrec]['blocknames'])==blocks_show_dict[tasktype]] #[ehelpers.get_namevec(taskdict[myrec]['blockvec'],taskdict[myrec]['blocknames'],label_mapper=label_mapper_aversion) for myrec in taskdict.keys()]
    else:
        nameslist = [ehelpers.get_namevec(taskdict[myrec]['blockvec'],taskdict[myrec]['blocknames']) for myrec in taskdict.keys()]
        myrecs = [myrec for rr,myrec in enumerate(taskdict.keys()) if list(nameslist[rr][:n_allowed])==blocks_show_dict[tasktype]]

    tpb_mat = np.vstack([datadict[tasktype][recid]['trials_per_block'][:n_allowed] for recid in myrecs])
    clids = np.hstack([datadict[tasktype][recid]['clust_vec'] for recid in myrecs])


    clid_blocks = [ taskdict[myrec]['clust_blocks'][:n_allowed] for myrec in myrecs]
    blcountmat = np.zeros((n_allowed,ncl+1))
    for bb in np.arange(n_allowed):

        alldata = np.hstack([blmat[bb] for blmat in clid_blocks])
        blcounts = np.array([np.sum(alldata==cc) for cc in np.r_[np.arange(ncl),-1]])
        blcountmat[bb] = blcounts

    xvec = np.arange(n_allowed)

    gen_counts = np.array([np.sum(clids==cc) for cc in np.arange(ncl)])
    gen_fracs = gen_counts/np.sum(gen_counts)
    overall_blocks = blcountmat[:,:ncl].sum(axis=0)
    frac_vec_blocks = overall_blocks/overall_blocks.sum()
    figsaver_ds = lambda fig, nametag: figsaver(fig, os.path.join('datasets', tasktype, nametag + '__%s' % tasktype),
                                                closeit=True)


    frac_perc = ((frac_vec_blocks-gen_fracs)/gen_fracs)*100
    perc_ratio_mat[tt]=frac_perc

    figsaver_ds = lambda fig, nametag: figsaver(fig, os.path.join('datasets', tasktype, nametag + '__%s' % tasktype),
                                                closeit=True)

    for namode in namodes:
        if namode == 'woNA':
            frac_mat = blcountmat[:,:ncl]/blcountmat[:,:ncl].sum(axis=1)[:,None]
        elif namode == 'withNA':
            frac_mat = blcountmat/blcountmat.sum(axis=1)[:,None]

        f,ax = plt.subplots(figsize=(2.5,2))
        for cc in np.arange(frac_mat.shape[1]):
            ax.plot(xvec,frac_mat[:,cc],'.-',mfc=cdict_clust_nan[cc],color=cdict_clust_nan[cc])
        if namode == 'woNA':
            for cc in np.arange(ncl):
                ax.plot([n_allowed-0.5],[gen_fracs[cc]],marker='o',mec= cdict_clust[cc],mew=1,mfc='none')
        ax.set_xticks(np.arange(n_allowed))
        ax.set_xticklabels(blocks_show_dict[tasktype],rotation=90)
        ax.set_ylabel('frac')
        ax.set_title(tasktype)
        f.tight_layout()
        ax.set_ylim([0.,ax.get_ylim()[1]])
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        figsaver_ds(f,'fracevol_through_blocks_%s'%namode)

        if namode=='woNA':
            f,ax = plt.subplots(figsize=(2.5,2))
            for tpb_rec in tpb_mat:
                ax.plot(np.arange(n_allowed),tpb_rec,'.-k',alpha=0.5)
            ax.set_xticks(np.arange(n_allowed))
            ax.set_xticklabels(blocks_show_dict[tasktype],rotation=90)
            ax.set_ylabel('N trials')
            ax.set_title(tasktype)
            f.tight_layout()
            figsaver_ds(f,'trials_per_block')


#overall frac ratio per dataset
xvec = np.arange(n_tasks)
f,ax = plt.subplots(figsize=(3.2,2.5))
f.subplots_adjust(left=0.3,bottom=0.2,right=0.98)
for cc in np.arange(ncl):
    ax.plot(perc_ratio_mat[:,cc],xvec,'o',mec=cdict_clust[cc],mfc='none',color=cdict_clust[cc],mew=2)
ax.set_yticks(np.arange(n_tasks))
ax.set_yticklabels(tasks_sorted)
ax.set_xlabel('% incr. blks')
ax.set_axisbelow(True)
ax.yaxis.grid(True, which='major',color='silver',linestyle='-')
f.tight_layout()
figsaver_summary(f,'perc_frac_increase_from_orig_to_blocks')




###############
######################################################
####################STATS CENTERED ON TRANSITION



dropped = {}
omnid = {}

for tasktype in datadict.keys():
    print('PROCESSING %s'%tasktype)

    taskdict = datadict[tasktype]

    myrecs = list(taskdict.keys())

    if tasktype.count('Aversion'):#this way we also dont need to take care of the namepairs of Aversion types beeing correct
        for myrec in myrecs:
            if not list(taskdict[myrec]['blocknames']) == [ 'tone','opto', 'opto', 'tone', 'air']:
                print('%s dropping %s due to bad blocknames %s'%(tasktype,myrec,str(taskdict[myrec]['blocknames'])))

                dropped[myrec] = taskdict.pop(myrec)

    myrecs = list(taskdict.keys())



    for myrec in myrecs:

        sdict = taskdict[myrec]
        tpb = sdict['trials_per_block']
        #tpb = np.array([50,10,50,50])
        transmats = sdict['transmats']
        blockvec = sdict['blockvec']
        blocknums = np.unique(blockvec[blockvec>-1])
        #print(blocknums,sdict['blocknames'])
        blocknum_inds = np.arange(len(blocknums))
        transpairs = [(blocknum_inds[bb],blocknum_inds[bb+1]) for bb in np.arange(len(blocknums)-1)]
        transmats2 = np.zeros_like(transmats)-1
        allowed_idx = np.where(tpb>trials_per_block_min)[0]
        if tasktype.count('Aversion'):     namevec = ehelpers.get_namevec(blockvec,sdict['blocknames'],label_mapper=label_mapper_aversion)
        else:namevec = ehelpers.get_namevec(blockvec,sdict['blocknames'])
        #print(namevec,sdict['blocknames'])
        sdict['namevec'] =  namevec
        namepairs = np.array([(namevec[bb],namevec[bb+1]) for bb in np.arange(len(namevec)-1)])
        sdict['namepairs'] = namepairs
        if tasktype == 'Context':
            name_bools = np.array([ehelpers.check_context_pair(namepair[0],namepair[1]) for namepair in namepairs])
        sdict['transpairs'] = transpairs
        sdict['allowed_block_inds'] = allowed_idx


        if len(allowed_idx)==0:
            dropped[myrec] = taskdict.pop(myrec)
            print('dropping %s'%myrec)

        else:
            for tt,[transmat,transpair] in enumerate(zip(transmats,transpairs)):
                tp0,tp1 = transpair
                if (tpb[tp0]>=trials_per_block_min) & (tpb[tp1]>=trials_per_block_min):
                    transmats2[tt] = transmat
            if tasktype == 'Context':
                name_bools = np.array([ehelpers.check_context_pair(namepair[0],namepair[1]) for namepair in namepairs])
                transmats2[name_bools==False,:,:] = -1
            sdict['T'] = transmats2
            sdict['transbool'] = np.array([(np.unique(tmat) != -1).all() for tmat in transmats2]).astype(bool)

            if (np.unique(transmats2) == -1).all():
                dropped[myrec] = taskdict.pop(myrec)
                print('dropping %s'%myrec)

        #print(sdict['T'].shape)

def lined_probfn(ax,tcmat,cmap='Greys',vminmax=[0,1],gridcol='k'):
    im = ehelpers.matplotfn_probs(ax,tcmat,cmap=cmap,titlefn=statstitlefn_probs,vminmax=vminmax)
    ncols = len(tcmat)
    ax.vlines(x=np.arange(ncols) - 0.5, ymin=np.full(ncols, 0) - 0.5, ymax=np.full(ncols, ncols) - 0.5, color=gridcol)
    return im

#ehelpers.matplotfn_probs(ax,tcmat,cmap='Greys',titlefn=statstitlefn_probs,vminmax=[0,1]),\
plfn_dict = {'orig':lambda ax,tcmat: ehelpers.matplotfn_counts(ax,tcmat,cmap='jet',titlefn=statstitlefn),\
             'expected':lambda ax,tcmat: ehelpers.matplotfn_counts(ax,ehelpers.get_expected_transcounts(tcmat),cmap='jet',titlefn=statstitlefn),\
             'prob':lambda ax,tcmat: lined_probfn(ax,tcmat,cmap='Greys',vminmax=[0,1],gridcol='k'),\
             'origByExp':lambda ax,tcmat: ehelpers.matplotfn_orig_by_expected(ax,tcmat,cmap='bwr',titlefn=statstitlefn_ratio),\
             'origByExp_climited': lambda ax,tcmat: ehelpers.matplotfn_orig_by_expected(ax,tcmat,cmap='bwr',titlefn=statstitlefn_ratio,divnorm=divnorm_oByE),\
             'cocoeff': lambda ax,tcmat: ehelpers.matplotfn_cocoeff(ax,tcmat,cmap='RdBu_r',titlefn=statstitlefn_cocoeff,vminmax=[-1,1])}

plotflavs2 = ['orig','expected','prob','origByExp_climited','cocoeff'] #also important for the order
plotflavs = ['orig','expected','prob','origByExp','cocoeff'] #also important for the order


for tasktype in datadict.keys():
    print('PROCESSING %s'%tasktype)

    taskdict = datadict[tasktype]





    figsaver_ds = lambda fig,nametag:figsaver(fig, os.path.join('datasets',tasktype,'transition_mats',nametag+'__%s'%tasktype), closeit=True)


    myrecs = sorted(list(taskdict.keys()))
    #for myrec in myrecs:
    #    print(myrec,taskdict[myrec]['namepairs'][taskdict[myrec]['transbool']])



    all_combinations = np.vstack([sdict['namepairs'][sdict['transbool']] for sdict in taskdict.values()])
    ucombs = np.unique(all_combinations,axis=0)
    csortinds = np.argsort(['%i%s'%(int(''.join(filter(str.isdigit, mystr))),mystr) for mystr in ucombs[:,0]])
    if tasktype.count('Aversion'):
        last_tags = np.array([val.replace('T','X').replace('A','Z2') for val in ucombs[:,1]])
        csortinds = np.argsort([('%i%s'%(int(''.join(filter(str.isdigit, mystr))),mystr)) for mystr in last_tags])
    overall_trans_pairs = ucombs[csortinds]



    maxNtrans = len(overall_trans_pairs)
    nrecs = len(myrecs)
    N_trans = len(overall_trans_pairs)

    sum_over_recs = np.zeros((maxNtrans,ncl,ncl))
    for myrec in myrecs:
        Tmats = taskdict[myrec]['T']
        for tt,trans_pair in enumerate(overall_trans_pairs):
            match_idx = np.where((taskdict[myrec]['namepairs'] == trans_pair).all(axis=1))[0]
            if len(match_idx)==1:
                tidx = int(match_idx)
                Tmat = Tmats[tidx]
                if Tmat.min() > -1:
                    sum_over_recs[tt] += Tmat

    omnid[tasktype] = {'pairs':overall_trans_pairs,'Tsum':sum_over_recs}

    sum_over_trans = np.zeros((nrecs,ncl,ncl))
    for rr,myrec in enumerate(myrecs):
        Tmats =  taskdict[myrec]['T']
        for Tmat in Tmats:
            if Tmat.min()>-1:
                sum_over_trans[rr] += Tmat


    grand_sum = sum_over_trans.sum(axis=0)
    assert (sum_over_recs.sum(axis=0) == grand_sum).all(),'grand sum over transitions does not match grand sum over recordings'


    f,axarr = plt.subplots(2,len(plotflavs),figsize=(10,3.3),gridspec_kw={'height_ratios':[1,0.1]})
    f.subplots_adjust(right=0.95,left=0.05,top=0.94,bottom=0.2)
    for ff,flav in enumerate(plotflavs2):
        ax,cax = axarr[:,ff]
        im = plfn_dict[flav](ax,grand_sum)
        ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='both',flav='num',fs=12)

        ax.set_aspect('equal')
        if flav == 'origByExp_climited':
            extend = 'max'
        else:
            extend = 'neither'
        cb = f.colorbar(im,cax=cax,orientation='horizontal',extend=extend)

        if flav.count('origByExp'):
            cb.locator = mpl.ticker.MaxNLocator(5,integer=True)
            cb.update_ticks()
        cax.set_xlabel(label_dict[flav])
    f.suptitle('%s  -   %s'%(tasktype,myrun))
    figsaver_ds(f,'grandSUMmats')


    stabi = np.diag(grand_sum).sum()/grand_sum.sum()
    f,axarr = plt.subplots(1,len(plotflavs),figsize=(10,2))
    f.subplots_adjust(left=0.08,top=0.8,wspace=0.6,right=0.95,bottom=0.2)
    for ff,flav in enumerate(plotflavs):
        ax = axarr[ff]
        heights = np.diag(fn_dict[flav](grand_sum))

        ehelpers.plot_clustbars(ax, heights, cdict_clust, show_mean=True)
        ax.set_ylabel(label_dict[flav],fontsize=9)
        if flav in ['prob','origByExp','cocoeff']:
            if flav in ['prob','cocoeff']:
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
            elif flav == 'origByExp':
                ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))

            ax.set_axisbelow(True)
            ax.yaxis.grid(True, which='minor', color='silver', linestyle='dashed')
            ax.yaxis.grid(True, which='major', color='silver', linestyle='dashed')

    axarr[1].sharey(axarr[0])
    f.suptitle('%s %s   -    i=j (diagonal)  -  SI: %1.2f'%(tasktype,myrun, stabi))
    f.tight_layout()
    figsaver_ds(f,'grandSUMdiags_bars')

    #some general trans stats
    # nrecs per trans

    transticklabs = ['>'.join(pair) for pair in overall_trans_pairs]
    recs_per_trans = np.zeros(N_trans,dtype=int)
    for myrec in myrecs:
        sdict = taskdict[myrec]

        mynamepairs = sdict['namepairs'][sdict['transbool']]
        for tt,trans_pair in enumerate(overall_trans_pairs):
            if trans_pair in mynamepairs:
                recs_per_trans[tt]+=1

    f,ax = plt.subplots(figsize=(2.5,2.5))
    ax.plot(np.arange(N_trans),recs_per_trans,'.k-')
    ax.set_xticks(np.arange(N_trans))
    ax.set_xticklabels(transticklabs,rotation=90)
    ax.set_ylabel('N recs')
    ax.set_title(tasktype)
    f.tight_layout()
    figsaver_ds(f,'Nrecs_vs_trans')


    f,ax = plt.subplots(figsize=(2.5,2.5))
    ax.plot(np.arange(N_trans),sum_over_recs.sum(axis=1).sum(axis=1),'.k-')
    ax.set_xticks(np.arange(N_trans))
    ax.set_xticklabels(transticklabs,rotation=90)
    ax.set_ylabel('N trans')
    ax.set_title(tasktype)
    f.tight_layout()
    figsaver_ds(f,'Ntrans_vs_rec')


    f,ax = plt.subplots(figsize=(2.5,2.5))
    ax.plot(np.arange(N_trans),sum_over_recs.sum(axis=1).sum(axis=1)/recs_per_trans,'.k-')
    ax.set_xticks(np.arange(N_trans))
    ax.set_xticklabels(transticklabs,rotation=90)
    ax.set_ylabel('trans/rec')
    ax.set_title(tasktype)
    f.tight_layout()
    figsaver_ds(f,'trans_per_rec')



    #sum over recs (as shown in the super matrix plot below) in a single line
    for flav in ['orig','prob','origByExp_climited','cocoeff']:

        matplotfn = plfn_dict[flav]

        f,axarr = plt.subplots(2,N_trans,figsize=(2.5+N_trans*1.4,3.3),gridspec_kw={'height_ratios':[1,0.1]})
        f.subplots_adjust(right=0.95,left=0.05,top=0.94,bottom=0.2)

        for tt,Tmat in enumerate(sum_over_recs):
            ax,cax = axarr[:,tt]
            im = matplotfn(ax,Tmat)
            if flav == 'orig':
                im.set_clim(vmin=sum_over_recs.min(),vmax=sum_over_recs.max())
            ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='both',flav='num',fs=12)
            ax.set_aspect('equal')
            #if tt == len(overall_trans_pairs)-1:
            if flav == 'origByExp_climited':
                extend = 'max'

            else:
                extend = 'neither'
            cb = f.colorbar(im,cax=cax,orientation='horizontal',extend=extend)

            if flav.count('origByExp'):
                cb.locator = mpl.ticker.MaxNLocator(5,integer=True)
                cb.update_ticks()
            cax.set_xlabel(label_dict[flav],fontsize=10)
            trans_str = '->'.join(overall_trans_pairs[tt])
            ax.set_title((r'$\bf{%s}$'+'\n%s; Nr%i')%(trans_str,ax.get_title(),recs_per_trans[tt]),fontsize=10)
        f.suptitle(tasktype)
        f.tight_layout()
        figsaver_ds(f,'sum_over_recs_%s'%flav)




    #diagonal of clusters vs transtion
    diag_dict_t = {}
    for flav in ['prob','origByExp','cocoeff']:
        diag_mat = np.zeros((N_trans,ncl))
        for tt,Tmat in enumerate(sum_over_recs):
            fullmat = fn_dict[flav](Tmat)
            diag_mat[tt] = np.diag(fullmat)
        diag_dict_t[flav] = diag_mat
    omnid[tasktype]['diagdict'] = diag_dict_t

    for flav in ['prob','origByExp','cocoeff']:
        diag_mat = diag_dict_t[flav]
        xvec = np.arange(N_trans)
        f, ax = plt.subplots(figsize=(2.5, 3))
        for cc in np.arange(ncl):
            ax.plot(xvec, diag_mat[:, cc], '.-', mfc=cdict_clust[cc], color=cdict_clust[cc])
        ax.set_xticks(np.arange(N_trans))
        ax.set_xticklabels(transticklabs,rotation=90)
        ax.set_ylabel(label_dict[flav].replace('count','#').replace('prob','P'))
        ax.set_title('%s %s'%(tasktype, 'diag.'))
        f.tight_layout()
        figsaver_ds(f, 'evol_through_trans_DIAG_%s' % flav)



    #supermatrix with all recs and transitions
    mfs = 8 #markerfontsize
    for flav in ['orig','prob','origByExp_climited','cocoeff']:
        matplotfn = plfn_dict[flav]
        #flav = 'origByExp'
        #matplotfn = lambda ax,tcmat: ehelpers.matplotfn_orig_by_expected(ax,tcmat,cmap='bwr',divnorm=divnorm_oByE)

        empty_axes = []
        clustvec = np.arange(ncl)+1
        N_trans = len(overall_trans_pairs)

        f,axarr = plt.subplots(nrecs+1,N_trans+1,figsize=(2.5+N_trans,2.5+nrecs))
        f.suptitle('%s   -   %s     %s'%(tasktype,myrun,flav))
        f.subplots_adjust(bottom=0.05,top=0.95,wspace=0.5,hspace=0.5)

        for rr,myrec in enumerate(myrecs):
            subaxs = axarr[rr]
            sdict = taskdict[myrec]
            Tmats = sdict['T']
            subaxs[0].set_ylabel(myrec.split('_')[0])
            for tt,trans_pair in enumerate(overall_trans_pairs):
                match_idx = np.where((sdict['namepairs'] == trans_pair).all(axis=1))[0]
                ax = subaxs[tt]

                if len(match_idx)==0:
                     plot_empty(ax)
                     empty_axes +=[ax]
                else:
                    tidx = int(match_idx)
                    Tmat = Tmats[tidx]
                    if Tmat.min()==-1:
                        plot_empty(ax)
                        empty_axes +=[ax]
                    else: matplotfn(ax,Tmat)

        for tt,trans_pair in enumerate(overall_trans_pairs):
            #pos = axarr[0,tt].get_position()
            #f.text(pos.x0,pos.y1+0.01,'->'.join(trans_pair),ha='left',va='bottom',fontsize=12)
            axarr[-1,tt].set_xlabel('->'.join(trans_pair))
        axarr[-1,-1].set_xlabel('SUM')
        #pos = axarr[0,-1].get_position()
        #f.text(pos.x0,pos.y1+0.01,'SUM',ha='left',va='bottom',fontsize=12)

        subaxs = axarr[-1]
        for tt,Tmat in enumerate(sum_over_recs):
            ax = subaxs[tt]
            matplotfn(ax, Tmat)
        sumlab = axarr[-1,0].set_ylabel('SUM')
        #spos = sumlab.get_position()
        for ax in axarr[:,0]:
            ax.yaxis.set_label_coords(-0.4,0.5)

        subaxs = axarr[:,-1]
        for tt,Tmat in enumerate(sum_over_trans):
            ax = subaxs[tt]
            matplotfn(ax, Tmat)

        ax = axarr[-1,-1]
        matplotfn(ax,grand_sum)

        for ax in axarr.flatten():
            if not ax in empty_axes:
                ax.set_xticks(np.arange(ncl))
                ax.set_yticks(np.arange(ncl))

        for ax in axarr[:,0]:
            if not ax in empty_axes:
                ax.set_xticklabels([])
                ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='y',flav='num',fs=mfs)

        for ax in axarr[-1,:]:
            if not ax in empty_axes:
                if not ax==axarr[-1,0]:ax.set_yticklabels([])
                ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='x',flav='num',fs=mfs)

        for ax in axarr[:,1:].flatten():
            if not ax in empty_axes:
                ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='y',flav=r'$\blacksquare$',fs=mfs)

        for ax in axarr[:-1].flatten():
            if not ax in empty_axes:
                ehelpers.set_color_ticklabs(ax,ncl,cdict_clust,which='x',flav=r'$\blacksquare$',fs=mfs)

        for ax in axarr.flatten():
            ax.set_aspect('equal')
        f.tight_layout()
        figsaver_ds(f,'allMats_%s'%flav)


##################GRAND SUMMARY OF TRANSITIONS!
figsaver_summary = lambda fig, nametag,closeit=True: figsaver(fig, os.path.join('summary','transition_mats', nametag + '__summary'),
                                                closeit=closeit)
exclude_tasktypes = ['Context']
tasks_summary = [task for task in tasks_sorted if not task in exclude_tasktypes]
titlestr = 'ALL tasks (wo %s)'%(''.join(exclude_tasktypes))
max_show = 6


T_list = [omnid[task]['Tsum'][:max_show] for task in tasks_summary]
Gsum = np.array([tmat.sum(axis=0) for tmat in T_list]).sum(axis=0)

f, axarr = plt.subplots(2, len(plotflavs), figsize=(10, 3.3), gridspec_kw={'height_ratios': [1, 0.1]})
f.subplots_adjust(right=0.95, left=0.05, top=0.94, bottom=0.2)
for ff, flav in enumerate(plotflavs2):
    ax, cax = axarr[:, ff]
    im = plfn_dict[flav](ax, Gsum)
    ehelpers.set_color_ticklabs(ax, ncl, cdict_clust, which='both', flav='num', fs=12)

    ax.set_aspect('equal')
    if flav == 'origByExp_climited':
        extend = 'max'
    else:
        extend = 'neither'
    cb = f.colorbar(im, cax=cax, orientation='horizontal', extend=extend)

    if flav.count('origByExp'):
        cb.locator = mpl.ticker.MaxNLocator(5, integer=True)
        cb.update_ticks()
    cax.set_xlabel(label_dict[flav])
f.suptitle('%s -   %s' % (titlestr,myrun))
figsaver_summary(f, 'grandSUM_allTasks')





#grand_sum
stabi = np.diag(Gsum).sum()/Gsum.sum()
f,axarr = plt.subplots(1,len(plotflavs),figsize=(10,2))
f.subplots_adjust(left=0.08,top=0.8,wspace=0.6,right=0.95,bottom=0.2)
for ff,flav in enumerate(plotflavs):
    ax = axarr[ff]
    heights = np.diag(fn_dict[flav](Gsum))

    ehelpers.plot_clustbars(ax, heights, cdict_clust, show_mean=True)
    ax.set_ylabel(label_dict[flav],fontsize=9)
    if flav in ['prob','origByExp','cocoeff']:
        if flav in ['prob','cocoeff']:
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        elif flav == 'origByExp':
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='minor', color='silver', linestyle='dashed')
        ax.yaxis.grid(True, which='major', color='silver', linestyle='dashed')

axarr[1].sharey(axarr[0])
f.suptitle('%s %s   -    i=j (diagonal)  -  SI: %1.2f'%(titlestr,myrun, stabi))
f.tight_layout()
figsaver_summary(f,'grandSUMdiags_allTasks_bars')






#now in one figure per dataset: SI, avg diag (from the diag dict,this per )
qual_dict = {task: {} for task in tasktypes}
diag_flavs = list(omnid[tasks_summary[0]]['diagdict'].keys())
for task in tasks_summary:
    Tmats = omnid[task]['Tsum']
    qual_dict[task]['SI'] = np.array([np.diag(tmat).sum()/np.sum(tmat) for tmat in Tmats])
    Tmats_exp = [fn_dict['expected'](tmat) for tmat in Tmats ]
    qual_dict[task]['SI_exp'] = np.array([np.diag(tmat).sum()/np.sum(tmat) for tmat in Tmats_exp])
    for key in diag_flavs:
        qual_dict[task][key] = omnid[task]['diagdict'][key].mean(axis=1)

cdict_task = {key:val for key,val in S.cdict_task.items()}
cdict_task['Aversion_ctrl'] = 'hotpink'
cdict_task['Aversion_stim'] = 'mediumvioletred'

label_dict_qual = {'prob': 'prob(x+1=i|x=i)',
                'origByExp': '#/#(exp) on diag',
                'SI': '#(diag)/#',\
                   'cocoeff':'CoiCoeff'}

max_n = np.max([len(qual_dict[task]['SI']) for task in tasks_summary])
max_n = np.min([max_n,max_show])

transticklabs_gen = ['T%i'%(ii+1) for ii in np.arange(max_n)]
for metr in ['SI']+diag_flavs:
    f,ax = plt.subplots(figsize=(3,2.5))
    for task in tasks_summary:
        vals = qual_dict[task][metr][:max_show]
        ax.plot(np.arange(len(vals)),vals,'.-',color=cdict_task[task])
        if metr == 'SI':
            vals2 = qual_dict[task]['SI_exp'][:max_show]
            ax.plot(np.arange(len(vals2)),vals2,'.-',color=cdict_task[task],alpha=0.5)
    ax.set_xticks(np.arange(max_n))
    ax.set_xticklabels(transticklabs_gen)
    ax.set_ylabel(label_dict_qual[metr])
    ax.set_title(titlestr)
    f.tight_layout()
    figsaver_summary(f,'qualityMetr_vsTrans_perTask_%s'%metr)


#stability vs trans over tasks
trans_mat_overall = np.zeros((max_show,ncl,ncl))
for task_transmats in T_list:
    trans_mat_overall[:len(task_transmats)] += task_transmats

stabi_vec = np.array([np.diag(tmat).sum()/np.sum(tmat) for tmat in trans_mat_overall])
tmats_exp = [fn_dict['expected'](tmat) for tmat in trans_mat_overall ]
stabi_vec_exp = np.array([np.diag(tmat).sum()/np.sum(tmat) for tmat in tmats_exp])
xvals = np.arange(max_show)
f, ax = plt.subplots(figsize=(3, 2.5))
ax.plot(xvals, stabi_vec, 'k.-')
ax.plot(xvals, stabi_vec_exp, '.-', color='k', alpha=0.5)
ax.set_xticks(xvals)
ax.set_xticklabels(transticklabs_gen)
ax.set_ylabel('stability')
ax.set_title(titlestr)
f.tight_layout()
figsaver_summary(f, 'qualityMetr_vsTrans_pooledTasks_SI')




# single panel dedicated to colors of tasks
f,ax = plt.subplots(figsize=(1.5,1.5))
for tt,task in enumerate(tasks_summary):
    ax.text(0.,0+tt*0.1,task,color=cdict_task[task],fontweight='bold',ha='left',va='top')
ax.set_ylim([-0.1,tt*0.1+0.1])
ax.set_axis_off()
figsaver_summary(f,'task_colors')

qual_dict_clust = {task: {} for task in tasktypes}
for task in tasks_summary:
    for dflav in diag_flavs:
        tmat_gen = omnid[task]['Tsum'][:max_n].sum(axis=0)
        tmat_metr = fn_dict[dflav](tmat_gen)
        qual_dict_clust[task][dflav] = np.diag(tmat_metr)


n_tasks_summary = len(tasks_summary)
yvec = np.arange(n_tasks_summary)

for dflav in diag_flavs:
    datamat = np.vstack([qual_dict_clust[task][dflav] for task in tasks_summary])

    f, ax = plt.subplots(figsize=(3.2, 2.5))
    f.subplots_adjust(left=0.3, bottom=0.2, right=0.98)
    for cc in np.arange(ncl):
        ax.plot(datamat[:, cc], yvec, 'o', mec=cdict_clust[cc], mfc='none', color=cdict_clust[cc], mew=2)
    ax.set_yticks(np.arange(n_tasks_summary))
    ax.set_yticklabels(tasks_summary)
    ax.set_xlabel(label_dict_qual[dflav])

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='major', color='silver', linestyle='-')
    ax.set_title('all trans.')
    f.tight_layout()
    if dflav == 'origByExp':
        ax.set_xlim([0,ax.get_xlim()[1]])
        figsaver_summary(f,'qualityMetr_clustresolved_perTask_%s'%dflav,closeit=False)
        ax.set_xlim([0,10])
        ax.axvline(1,color='k',linestyle=':',zorder=-10)
        ax.set_title('all trans. zoomed')
        figsaver_summary(f,'qualityMetr_clustresolved_perTask_ZOOM_%s'%dflav)
    else:
        figsaver_summary(f,'qualityMetr_clustresolved_perTask_%s'%dflav)





