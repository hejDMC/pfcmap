import os
import numpy as np
import h5py
import git
import yaml
import sys
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


pathpath = 'PATHS/filepaths_IBLTuning.yml'#
myrun =  'runIBLTuning_utypeP'
dset = 'IBL'

plot_examples = True
n_ex = 10
tdict_type = 'equaldur'#IBL styler
pthr = 0.001

#layers_allowed = ['5','6']
dsdir = 'IBLPATH'

dstdir = 'results/paper/tuningIBL/pfcOnly/%s'%myrun
if not os.path.isdir(dstdir): os.makedirs(dstdir)
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap_paper.utils import IBLtask_funcs as iblfns
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S


figdir_mother = 'FIGDIR/tuning_examples/%s_taskresp'%dset

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '.png')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
if dset == 'IBL':
    Units, somfeats, weightvec = uloader.load_all_units_for_som(metricsfiles, rundict, check_pfc=True,check_wavequality=False)
else:
    Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= True)
print('unqiue_areas',np.unique([U.area for U in Units]))


getspikes = lambda stimes,tinterval: stimes[(stimes>=tinterval[0]) & (stimes<=tinterval[1])]
unit_filter = lambda uobj: True #S.check_layers(uobj.layer,layers_allowed)

Usel = [U for U in Units if unit_filter(U)]

recids_sel = np.unique([U.recid for U in Usel])
trials_path = '/intervals/trials'

if tdict_type == 'equaldur':
    tdict = {}#corrected for all periods having the same duration to avoid statistical artifacts
    tdict['pre'] = {'avar':'start_time','tint':[-0.2,0.]}#alignment variable and time interval
    tdict['stim1'] = {'avar':'start_time','tint':[0.0,0.2]}
    tdict['stim2'] = {'avar':'start_time','tint':[0.2,0.4]}
    tdict['move'] = {'avar':'firstMovement_times','tint':[0.,0.2]}
    tdict['feedb'] = {'avar':'feedback_times','tint':[0.,0.2]}

elif tdict_type == 'IBLstyle':

    tdict = {}#this is bad because they all have different durations!!!!
    tdict['pre'] = {'avar':'start_time','tint':[-0.2,0.]}#alignment variable and time interval
    tdict['stim1'] = {'avar':'start_time','tint':[0.05,0.15]}
    tdict['stim2'] = {'avar':'start_time','tint':[0.0,0.4]}
    tdict['move'] = {'avar':'firstMovement_times','tint':[-0.05,0.2]}
    tdict['feedb'] = {'avar':'feedback_times','tint':[0.,0.15]}

else: assert 0, 'unknown tdict_type %s'%tdict_type

tvars = ['pre','stim1','stim2','move','feedb']
n_tvars = len(tvars)

durvec = np.array([np.diff(tdict[tvar]['tint'])[0] for tvar in tvars])

n_recids_plot = 15#for which to plot the random examples
recids_plot = np.random.choice(recids_sel,n_recids_plot,replace=False)

plot_info_collector = {}

for recid in recids_sel:
    #recid = recids_sel[4]
    recUs = [U for U in Usel if U.recid==recid]

    nwbfile = os.path.join(dsdir,'%s.nwb'%recid)

    print('%s Nunits:%i'%(recid,len(recUs)))
    rec_uids = np.array([U.uid for U in recUs])


    trials_dict = iblfns.extract_trials_dict_ibl(nwbfile, trials_path=trials_path)

    N_trials = len(trials_dict['start_time'])


    with h5py.File(nwbfile,'r') as hand:
        uids_nwb =hand['units']['id'][()]
        utinds = hand['units/spike_times_index'][()]
        allspiketimes = hand['units/spike_times'][()]

    tint_dict = {}
    for tvar in tvars:
        tint_dict[tvar] = trials_dict[tdict[tvar]['avar']][:, None] + np.array(tdict[tvar]['tint'])[None, :]

    if plot_examples and recid in recids_plot:
        plot_info_collector[recid] = {'trials_dict':trials_dict,'uids_nwb':uids_nwb,'utinds':utinds,'allspiketimes':allspiketimes,'tint_dict':tint_dict}

    #U = recUs[50]
    for U in recUs:
        uidx = int(np.where(uids_nwb == U.uid)[0])  # just to make sure
        r1, r0 = [utinds[uidx], [utinds[uidx - 1] if uidx > 0 else 0][0]]
        spiketimes = allspiketimes[r0:r1]

        nspikes_mat = np.vstack([np.array([len(getspikes(spiketimes, tint)) for tint in tint_dict[tvar]]) for tvar in tvars])
        rate_mat = nspikes_mat/durvec[:,None]
        pvec = np.zeros(n_tvars-1)*np.nan
        for rr,ratevec in enumerate(rate_mat[1:]):
            if not ((rate_mat[0]-ratevec)==0).all():
                pvec[rr] = wilcoxon(rate_mat[0],ratevec).pvalue
        #pvec = np.array([wilcoxon(rate_mat[0],ratevec).pvalue for ratevec in rate_mat[1:]])
        pvec_fdr = iblfns.false_discovery_control(pvec, axis=0, method='bh')
        is_signif = np.nanmin(pvec_fdr)<=pthr if np.isnan(pvec_fdr).sum()<(n_tvars-1) else np.nan
        setattr(U,'issignif',is_signif)
        setattr(U,'pvec_bh',pvec_fdr)

    #writing
    outfile = os.path.join(dstdir,'%s__%s__ccCPtuning.h5'%(recid,myrun))
    uids = np.array([U.uid for U in recUs])# save

    N_us = len(uids)
    outmat = np.zeros((N_us,n_tvars-1))*np.nan# save
    signif_vec = np.zeros(N_us)*np.nan# save
    for uu,U in enumerate(recUs):
        outmat[uu] = U.pvec_bh
        signif_vec[uu] = U.issignif


    with h5py.File(outfile,'r+') as hand:
        if 'taskresp_%s'%tdict_type in hand:
            del hand['taskresp_%s'%tdict_type]
        grp = hand.create_group('taskresp_%s'%tdict_type)
        grp.attrs['signif_thr'] = pthr
        grp.attrs['ref_tvar'] = tvars[0]
        grp.create_dataset('uids',data=uids,dtype='i')
        grp.create_dataset('pmat_bh',data=outmat,dtype='f')
        grp.create_dataset('signif',data=signif_vec,dtype='f')
        asciiList = [mystr.encode("ascii", "ignore") for mystr in tvars[1:]]
        grp.create_dataset('tvars_p', (len(asciiList), 1), 'S21', asciiList)

if plot_examples:
    stylepath = os.path.join(pathdict['plotting']['style'])
    plt.style.use(stylepath)
    Usel_plot = [U for U in Usel if U.recid in recids_plot]

    showtint = [-0.5,5]
    mark_coldict = {'move':'firebrick','feedback':'orange','trialend':'steelblue'}


    nanUs =[U for U in Usel_plot if np.isnan(U.issignif)]#todo later: do not consider those
    signifUs = [U for U in Usel_plot if U.issignif==True]
    nonsignifUs = [U for U in Usel_plot if U.issignif==False]
    assert (len(nanUs) + len(signifUs) + len(nonsignifUs))==len(Usel_plot), 'check unit assignments to signif types!'
    print('Nsignif:%i Nnan: %i - fracsignif: %1.2f'%(len(signifUs),len(nanUs),len(signifUs)/(len(Usel_plot)-len(nanUs))))

    showUs = [subel for el in [list(np.random.choice(upop,np.min([n_ex,len(upop)]),replace=False)) for upop in [signifUs,nonsignifUs,nanUs]] for subel in el]

    for U in showUs:
        #U = showUs[10]
        subd = plot_info_collector[U.recid]

        uids_nwb,utinds,allspiketimes = [subd[key] for key in ['uids_nwb','utinds','allspiketimes']]
        uidx = int(np.where(uids_nwb==U.uid)[0])#just to make sure

        r1, r0 = [utinds[uidx], [utinds[uidx - 1] if uidx > 0 else 0][0]]
        spiketimes = allspiketimes[r0:r1]
        nspikes_mat = np.vstack([np.array([len(getspikes(spiketimes, tint)) for tint in subd['tint_dict'][tvar]]) for tvar in tvars])
        rate_mat = nspikes_mat / durvec[:, None]

        if U in signifUs:
            signif_tag = 'signif'
        elif U in nonsignifUs:
            signif_tag = 'nosignif'
        else:
            signif_tag = 'nansignif'

        f,axarr = plt.subplots(1,2,figsize=(7,3))
        ax1,ax2 = axarr
        ax1.boxplot([vals for vals in rate_mat],showfliers=False)
        for vv in np.arange(1,len(tvars)):
            ax1.text(vv+1,np.median(rate_mat[vv]),'%1.3f'%(U.pvec_bh[vv-1]),ha='center',va='top')
        ax1.set_xticks(np.arange(n_tvars)+1)
        ax1.set_xticklabels(tvars)
        ax1.set_ylabel('firing rate [Hz]')
        for vv in np.arange(len(tvars)):
            ax2.plot(np.ones_like(rate_mat[vv])*vv,rate_mat[vv],'k.',alpha=0.1)
        ax2.set_xticks(np.arange(len(tvars)))
        ax2.set_xticklabels(tvars)
        ax2.set_ylabel('firing rate [Hz]')
        f.suptitle('%s %i %s'%(recid,U.uid, U.region))
        f.text(0.99,0.98,signif_tag,color='grey',ha='right',va='top',fontweight='bold',fontsize=12)
        f.tight_layout()
        figsaver(f,'%s/ratedistro__%s__%s'%(signif_tag,U.id,signif_tag))


        align_times = subd['trials_dict']['start_time']
        showtints = align_times[:, None] + np.array(showtint)[None, :]
        mark_dict = {'move':subd['trials_dict']['firstMovement_times']-align_times,\
                     'feedback':subd['trials_dict']['feedback_times']-align_times,\
                     'trialend':subd['trials_dict']['stop_time']-align_times}

        fire_list = [getspikes(spiketimes,tint)-align_times[tt] for tt,tint in enumerate(showtints)]
        if len(np.hstack(fire_list))>0:
            rastermat = np.hstack([[stimes,tt*np.ones(len(stimes))] for tt,stimes in enumerate(fire_list) if len(stimes)>0]).T

            f,ax = plt.subplots(figsize=(10,6))
            ax.plot(rastermat[:,0],rastermat[:,1],'.k',markersize=1)
            for mm,marktag in enumerate(mark_coldict.keys()):
                vals = mark_dict[marktag]
                ax.plot(vals,np.arange(len(vals)),'|',mec=mark_coldict[marktag],mew=1)
                ax.text(1.01,0.9-0.04*mm,marktag,color=mark_coldict[marktag],fontweight='bold',transform=ax.transAxes)
            ax.axvline(0.,color='silver')
            ax.set_xlim(showtint)
            ax.set_ylim([0,len(showtints)])
            ax.set_ylabel('trials')
            ax.set_xlabel('time [s]')
            f.text(0.99,0.98,signif_tag,color='grey',ha='right',va='top',fontweight='bold',fontsize=12)
            f.tight_layout()
            figsaver(f,'%s/rastergen__%s__%s'%(signif_tag,U.id,signif_tag))

        #ok now for each interval
        firing_collection = {tvar:[getspikes(spiketimes,tint)-subd['trials_dict'][tdict[tvar]['avar']][tt]\
                                   for tt,tint in enumerate(subd['tint_dict'][tvar])] for tvar in tvars}
        N_trials = len(subd['trials_dict']['start_time'])
        for tvar in tvars[1:]: #apart from pre, zero all!
            firing_collection[tvar] = [el-tdict[tvar]['tint'][0] for el in firing_collection[tvar]]
        f,axarr = plt.subplots(n_tvars-1,1,figsize=(8,6))
        f.subplots_adjust(hspace=0.01,left=0.08)
        for vv,tvar in enumerate(tvars[1:]):
            ax = axarr[vv]
            for tt in np.arange(N_trials):
                base_vals = firing_collection['pre'][tt]
                var_vals =  firing_collection[tvar][tt]
                for vals in [base_vals,var_vals]:
                    ax.plot(vals,np.ones(len(vals))*tt,'.k',markersize=1)
            ax.set_xlim([-0.2,0.2])
            ax.axvline(0.,color='silver')
            ax.set_ylabel('trial')
            N_spikes = np.hstack(firing_collection[tvar]).shape[0]
            fw = 'bold' if U.pvec_bh[vv] <pthr else 'normal'
            ax.text(1.01,0.5,'%s\n%1.4f\nN=%i'%(tvar,U.pvec_bh[vv],N_spikes),transform=ax.transAxes,ha='left',va='center',fontweight=fw)
        axarr[-1].set_xlabel('time [s]')
        axarr[0].text(-0.1,N_trials*1.05,'pre stim ref (Nspikes=%i)'%(np.hstack(firing_collection['pre']).shape[0]),ha='center',va='bottom')
        for ax in axarr[:-1]:
            ax.set_xticks([])
        f.suptitle('%s %i %s'%(recid,U.uid, U.region))
        f.text(0.99,0.98,signif_tag,color='grey',ha='right',va='top',fontweight='bold',fontsize=12)
        f.tight_layout()
        figsaver(f,'%s/rasterAlign__%s__%s'%(signif_tag,U.id,signif_tag))

