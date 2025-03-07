import os
import numpy as np
import h5py
import yaml
import sys
import matplotlib.pyplot as plt


pathpath = 'PATHS/filepaths_IBLTuning.yml'#
myrun =  'runIBLTuning_utypeP'
dset = 'IBL'


dsdir = 'IBLPATH'

dstdir = '/results/paper/tuningIBL/pfcOnly/%s'%myrun
if not os.path.isdir(dstdir): os.makedirs(dstdir)
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap_paper.utils import IBLtask_funcs as iblfns
from pfcmap_paper.utils import unitloader as uloader
from pfcmap_paper import settings as S


figdir_mother = 'FIGURES/tuning_examples/%s_tuning_examples'%dset

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
print('N recs with PFC data: %i'%(len(recids_sel)))
trials_path = '/intervals/trials'
deltat_range_choice = [0.08,2.]
nshuff = 2000
pthr = 0.05 #just for example plotting
pthr_naive = 0.001
cccpthr = 0.52#to have at least some effect... avoids the super-low raters!
plot_examples = True
n_ex = 10 #per signif/nonsignif | choice/stim/feedback



param_dict = {}
param_dict['choice'] = {'var':'choice','align_type':'firstMovement_times','vals':[-1,1],'analysis_win':[-0.1,0.],\
                        'fix_dict':{'stimdir':'stimdirection'},'tbounds_show':[-0.4,0.3],'deltat_range':[0.08,2.]}
param_dict['stimulus'] = {'var':'stimdirection','align_type':'start_time','vals':[-1,1],'analysis_win':[0.,0.1],\
                        'fix_dict':{'choice':'choice'},'tbounds_show':[-0.3,0.4]}
param_dict['feedback'] = {'var':'feedbackType','align_type':'feedback_times','vals':[-1,1],'analysis_win':[0,0.2],\
                        'fix_dict':{'choice':'choice'},'tbounds_show':[-0.3,0.4]}#cannot do both choice and stim here as fixed as then feedback is a natural given...

#for saving
cond_keys = ['blks_detail','blks_summary','naive']
featnames = ['cccp','pval','pref']
vartags = list(param_dict.keys())
dim_dict = {1:['vartags',vartags],2:['conditions',cond_keys],3:['features',featnames]}# save

n_recids_plot = 15#for which to plot the random examples
recids_plot = np.random.choice(recids_sel,n_recids_plot,replace=False)

plot_info_collector = {}
for recid in recids_sel:
    recUs = [U for U in Usel if U.recid==recid]
    for U in recUs:
        setattr(U,'tunings',{})
    nwbfile = os.path.join(dsdir,'%s.nwb'%recid)

    print('%s Nunits:%i'%(recid,len(recUs)))
    rec_uids = np.array([U.uid for U in recUs])

    trials_dict = iblfns.extract_trials_dict_ibl(nwbfile, trials_path=trials_path)

    N_trials = len(trials_dict['start_time'])

    with h5py.File(nwbfile,'r') as hand:
        uids_nwb =hand['units']['id'][()]
        utinds = hand['units/spike_times_index'][()]
        allspiketimes = hand['units/spike_times'][()]

    if plot_examples and recid in recids_plot:
        print('storing plotinfo for %s'%recid)
        plot_info_collector[recid] = {'trials_dict':trials_dict,'uids_nwb':uids_nwb,'utinds':utinds,'allspiketimes':allspiketimes,'vardata':{}}

    for vartag,params in param_dict.items():
        if vartag == 'choice':
            #filter for suitable delta start_time firstMovement_time#todo only for choice! --> put this into a function
            delta_on_mov = trials_dict['firstMovement_times']-trials_dict['start_time']
            cond_deltat = (delta_on_mov>=params['deltat_range'][0]) & (delta_on_mov<=params['deltat_range'][1])
            N_trials_avail = np.sum(cond_deltat)
            frac_deltat_excl = (N_trials-N_trials_avail)/N_trials
            print(recid,'N trials avail: %i, frac deltat excluded: %1.2f'%(N_trials_avail,frac_deltat_excl))
        else:
            cond_deltat = np.ones(N_trials,dtype=bool)



        const_dict = {key:trials_dict[trialdictkey] for key,trialdictkey in params['fix_dict'].items()}


        cond_vec_blks,cond_dict_blks = iblfns.get_const_conditions(trials_dict,const_dict,summarize_blocks=False,\
                                                                    exclude_bool=~cond_deltat)
        cond_vec,cond_dict = iblfns.get_const_conditions(trials_dict,const_dict,summarize_blocks=True,\
                                                                    exclude_bool=~cond_deltat)
        cond_vec_simple = np.ones(N_trials)
        cond_vec_simple[~cond_deltat] = -1

        conditions_collected = {'blks_detail':cond_vec_blks,'blks_summary':cond_vec,'naive':cond_vec_simple}

        align_times = trials_dict[params['align_type']]
        tints = align_times[:, None] + np.array(params['analysis_win'])[None, :]

        if plot_examples and recid in recids_plot:
            plot_info_collector[recid]['vardata'][vartag] = {'cond_deltat':cond_deltat,'align_times':align_times,'conds_collected':conditions_collected}

        for U in recUs:
            U.tunings[vartag] = {}
            #U = recUs[19]#recUs[10]
            uidx = int(np.where(uids_nwb==U.uid)[0])#just to make sure
            r1, r0 = [utinds[uidx], [utinds[uidx - 1] if uidx > 0 else 0][0]]
            spiketimes = allspiketimes[r0:r1]
            nspikes_vec = np.array([len(getspikes(spiketimes,tint)) for tint in tints])
            # loop over cond_vec and choice tags here
            for condition_tag,condition_vec in conditions_collected.items():
                U.tunings[vartag][condition_tag] = {}
                cccp,p_val = iblfns.compute_ccCP(nspikes_vec[cond_deltat], trials_dict[params['var']][cond_deltat], condition_vec[cond_deltat],\
                                                  choicevals=params['vals'],n_shuffles=nshuff)

                #print(cccp,p_val,U.uid)
                if cccp<0.5:
                    result = [1-cccp,1-p_val,params['vals'][1]]
                else:
                    result = [cccp,p_val,params['vals'][0]]
                for featname,val in zip(['cccp','pval','pref'],result):
                    U.tunings[vartag][condition_tag][featname] = val
                    #setattr(U,'%s_%s'%(choicetag,featname),val)

    #now saving!
    uids = np.array([U.uid for U in recUs])# save

    N_us = len(uids)
    outmat = np.zeros((N_us,*[len(dim_dict[jj][1]) for jj in np.arange(len(dim_dict))+1]))*np.nan# save
    for uu,U in enumerate(recUs):
        for vv,vartag in enumerate(dim_dict[1][1]):
            for cc,condition_tag in enumerate(dim_dict[2][1]):
                for ff,featname in enumerate(dim_dict[3][1]):
                    outmat[uu,vv,cc,ff] = U.tunings[vartag][condition_tag][featname]

    outfile = os.path.join(dstdir,'%s__%s__ccCPtuning.h5'%(recid,myrun))

    with h5py.File(outfile,'w') as hand:
        grp = hand.create_group('ccCP')
        hand.attrs['recid'] = recid.encode()
        grp.create_dataset('uids',data=uids,dtype='i')
        grp.create_dataset('cccpmat',data=outmat,dtype='f')
        dgrp = grp.create_group('cccpmat_dimensions')
        for dkey,strlist in dim_dict.items():
            asciiList = [mystr.encode("ascii", "ignore") for mystr in [strlist[0]]+strlist[1]]
            dgrp.create_dataset('d%i'%dkey, (len(asciiList), 1), 'S21', asciiList)


if plot_examples:
    condflav = 'detail'
    Usel_plot = [U for U in Usel if U.recid in recids_plot]

    cccp_rec_dict, dimdict = iblfns.load_cccp_data(recids_plot, dstdir, myrun)
    iblfns.assing_tunings(Usel_plot, cccp_rec_dict, dim_dict, get_significance=True, pthr=pthr, pthr_naive=pthr_naive,
                          cccpthr=cccpthr)
    stylepath = os.path.join(pathdict['plotting']['style'])
    plt.style.use(stylepath)


    for vartag in vartags:
        params = param_dict[vartag]

        signifUs = [U for U in Usel_plot if U.tsignif[vartag][condflav]]
        nonsignifUs = [U for U in Usel_plot if not U in signifUs]
        print('%s - Nsignif:%i - fracsignif: %1.2f'%(vartag,len(signifUs),len(signifUs)/len(Usel_plot)))
        showUs = [subel for el in [list(np.random.choice(upop,np.min([n_ex,len(upop)]),replace=False)) for upop in [signifUs,nonsignifUs]] for subel in el]
        if vartag == 'choice':
            col_var = 'stimdirection'#the variable according to which spike trains get colored
        elif vartag == 'stimulus':
            col_var = 'choice'
        elif vartag == 'feedback':
            col_var = 'choice'



        for U in showUs:

            subd = plot_info_collector[U.recid]  # = {'trials_dict':trials_dict,'uids_nwb':uids_nwb,'utinds':utinds,'allspiketimes':allspiketimes,'vardata':{}}

            align_times = subd['vardata'][vartag]['align_times']  # ['trials_dict'][params['align_type']]
            tints = align_times[:, None] + np.array(params['tbounds_show'])[None, :]
            #U = [U for U in recUs if U.uid==251][0]#256 - this is a good one! 289 a very high pval!
            uids_nwb,utinds,allspiketimes = [subd[key] for key in ['uids_nwb','utinds','allspiketimes']]
            uidx = int(np.where(uids_nwb==U.uid)[0])#just to make sure
            r1, r0 = [utinds[uidx], [utinds[uidx - 1] if uidx > 0 else 0][0]]
            spiketimes = allspiketimes[r0:r1]
            fire_dict = {tt:getspikes(spiketimes,tint)-align_times[tt] for tt,tint in enumerate(tints)}
            #right_coices only

            mycccp,mypval,mypref = [U.tunings[vartag]['blks_%s'%condflav][subkey] for subkey in ['cccp','pval','pref']]
            mypval_naive = U.tunings[vartag]['naive']['pval']
            var = params['var']


            f, axarr = plt.subplots(2,1,figsize=(5,8))
            f.suptitle('uid:%i %s ccCP:%1.2f' % (U.uid, U.region,mycccp))

            for choiceidx,choicecol,ax in zip(params['vals'],['b','r'],axarr):

                idx_filter_fn = lambda myidx: (subd['trials_dict'][var][myidx]==choiceidx) & (subd['vardata'][vartag]['cond_deltat'][myidx])
                show_inds = np.array([tt for tt in np.arange(len(tints)) if idx_filter_fn(tt)])
                rasterlist = [fire_dict[tt] for tt in show_inds]# and len(fire_dict[tt])>0
                rastermat = np.hstack([[stimes,jj*np.ones(len(stimes))] for jj,stimes in enumerate(rasterlist)]).T

                col_sel = np.hstack([subd['trials_dict'][col_var][tt]*np.ones(len(fire_dict[tt])) for tt in show_inds])

                for jj,[selval,colval] in enumerate(zip([-1,1],['b','r'])):
                    ax.plot(rastermat[col_sel==selval,0],rastermat[col_sel==selval,1],'.%s'%colval,markersize=1)
                    f.text(0.01,0.98-0.02*jj,'%s:%i'%(col_var[:7],selval),color=colval,ha='left',va='top',fontweight='bold',fontsize=10)

                #ax.plot(rastermat[:,0],rastermat[:,1],'.k',markersize=3)
                if vartag == 'choice':
                    marklist = subd['trials_dict']['start_time'][show_inds]-align_times[show_inds]
                elif vartag == 'stimulus':
                    marklist = subd['trials_dict']['firstMovement_times'][show_inds]-align_times[show_inds]
                elif vartag == 'feedback':
                    marklist = subd['trials_dict']['firstMovement_times'][show_inds]-align_times[show_inds]

                ax.plot(marklist,np.arange(len(marklist)),'k|',mew=2)

                if choiceidx == mypref:
                    titlestr = '%s: %i (pref, p1:%1.3f, p2:%1.3f)'%(vartag,choiceidx,mypval,mypval_naive)
                else:
                    titlestr = '%s: %i'%(vartag,choiceidx)
                ax.set_title(titlestr,color='k')
                for mytime in params['analysis_win']:
                    ax.axvline(mytime,color='silver')
                #ax.set_xlim(tbounds)
                ax.set_xlim(params['tbounds_show'])
                ax.set_ylim([0,400])
                ax.set_ylabel('trials')
            axarr[-1].set_xlabel('time [s]')
            signiftag = 'signif' if U.tsignif[vartag][condflav] else 'nosignif'
            f.text(0.99,0.98,signiftag,color='grey',ha='right',va='top',fontweight='bold',fontsize=12)
            f.tight_layout()
            figsaver(f,'%s/%s/%s__%s_%s'%(vartag,signiftag,U.id,vartag,signiftag))



# use these loading functions later:
'''cccp_rec_dict,dimdict = iblfns.load_cccp_data(recids_sel,dstdir,myrun)
iblfns.assing_tunings(Usel,cccp_rec_dict,dim_dict,get_significance=True,pthr=pthr,pthr_naive=pthr_naive,cccpthr=cccpthr)'''