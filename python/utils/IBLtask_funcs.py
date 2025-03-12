import numpy as np
from scipy.stats import rankdata
import h5py
import os

def mann_whitney_u_shuf(x, y, shuf_labels):
    """
    Compute Mann-Whitney U statistic efficiently using shuffled labels.

    Parameters:
    - x: array-like, spike counts for choice A  (# shape: (nx,))
    - y: array-like, spike counts for choice B  (# shape: (ny,))
    - shuf_labels: np.array, shuffled indices  (# shape: (nx, n_shuffles + 1))

    Returns:
    - numer: np.array of U-statistics (# shape: (n_shuffles + 1,))
    """
    nx = len(x)  # Number of trials for choice A

    # Compute ranks for the combined dataset (x + y)
    t = rankdata(np.concatenate([x, y]))  # shape: (nx + ny,)

    # Apply the shuffled labels to x's ranks (indexed from shuf_labels)
    t = t[shuf_labels]  # shape: (nx, n_shuffles + 1)

    # Compute the U-statistic: sum of ranks in x minus expected sum under null hypothesis
    numer = np.sum(t, axis=0) - nx * (nx + 1) / 2  # shape: (n_shuffles + 1,)

    return numer  # First value = true U-stat, rest = shuffled U-stats


def compute_ccCP(spike_counts, choices, conditions, choicevals=[-1,1],n_shuffles=2000):
    """
    Compute combined-conditions Choice Probability (ccCP).

    Parameters:
    - spike_counts: array of spike rates per trial (# shape: (n_trials,))
    - choices: binary array of trial choices (0 = A, 1 = B) (# shape: (n_trials,))
    - conditions: array of conditionulus conditions per trial (# shape: (n_trials,))
    - n_shuffles: number of label shuffles (default = 2000)

    Returns:
    - ccCP[0]: Actual choice probability (non-shuffled)
    - p_value: p-value for significance testing
    """
    unique_conditions = np.unique(conditions)  # Unique conditions  (# shape: (n_conditions,))
    n_total = np.zeros(1 + n_shuffles)  # Array to store U-stat sums  (# shape: (n_shuffles + 1,))
    d_total = 0  # Total number of comparisons (denominator for U-stat)

    #print(unique_conditions)
    for condition in unique_conditions:
        # Find all trials for the current condition
        idx = np.where(conditions == condition)[0]  # shape: (n_trials_per_condition,)

        # Skip if there's only one choice type (no comparisons possible)
        if not set(choices[idx]).issuperset(set(choicevals)):#nans get also counted!
            #print('xxx choices',choices[idx])
            #print('xxx monochoice in',condition)
            continue
        #print(condition)
        # Split trials into choice A (0) and choice B (1)
        chA = idx[choices[idx] == choicevals[0]]  # Indices for choice A  (# shape: (nA,))
        chB = idx[choices[idx] == choicevals[1]]  # Indices for choice B  (# shape: (nB,))
        nA, nB = len(chA), len(chB)  # Number of trials per choice

        # generate shuffled labels for this conditionulus condition (only once)
        q = np.array([np.random.permutation(nA + nB)[:nA] for _ in range(n_shuffles)]).T
        # shape: (nA, n_shuffles), each col is a random subset of nA indices
        shuf_labels = np.column_stack([np.arange(nA), q])
        # shape: (nA, n_shuffles + 1), first col = identity (original order)

        # Compute Mann-Whitney U statistic for this conditionulus
        n = mann_whitney_u_shuf(spike_counts[chA], spike_counts[chB], shuf_labels)
        # shape: (n_shuffles + 1,)

        # Aggregate U-statistics across all conditions
        n_total += n  # Sum of U-statistics across conditions
        d_total += nA * nB  # Total comparisons made
        #print (condition,nA,nB)

    # Compute final ccCP score (true + shuffled)
    if d_total > 0:
        ccCP = n_total / d_total # shape: (n_shuffles + 1,)
    else: return [np.nan,np.nan]
    #print(ccCP[0],ccCP[1:10])

    # Compute rank of the true ccCP relative to shuffles
    #t = rankdata(ccCP)  # shape: (n_shuffles + 1,)
    #p_value = t[0] / (1 + n_shuffles)  # Fraction of values that are greater or equal to true ccCP
    #print(p_value,np.sum(ccCP[0]>=ccCP[1:])/n_shuffles)
    #print (ccCP)
    p_value = np.sum(ccCP[0]<=ccCP[1:])/n_shuffles
    return ccCP[0], p_value  # Return true ccCP and its p-value


def extract_trials_dict_ibl(nwbfile,trials_path='/intervals/trials'):
    with h5py.File(nwbfile,'r') as hand:
        trials_dict = {key:hand['%s/%s'%(trials_path,key)][()] for key in hand[trials_path]}
        stimdirection = np.zeros_like(trials_dict['contrastLeft'])-1
        stimdirection[~np.isnan(trials_dict['contrastLeft'])] = 1
        stimcontrast = np.zeros_like(trials_dict['contrastLeft'])*np.nan
        for contrast_tag in ['contrastLeft','contrastRight']:
            stimcontrast[~np.isnan(trials_dict[contrast_tag])] = trials_dict[contrast_tag][~np.isnan(trials_dict[contrast_tag])]
        trials_dict['stimcontrast'] = stimcontrast
        trials_dict['stimdirection'] = stimdirection

    cond_temp = trials_dict['stimdirection'] == trials_dict['choice']
    assert(trials_dict['feedbackType'][cond_temp]).min()==1, 'mismatching feedback<->choice/stimdir'
    assert(trials_dict['feedbackType'][cond_temp==False]).max()==-1, 'mismatching feedback<->choice/stimdir'

    N_trials = len(trials_dict['start_time'])

    blockvec = np.zeros(N_trials)*np.nan#
    shift_inds = np.r_[0,np.where(np.diff(trials_dict['probabilityLeft'])!=0)[0]+1,N_trials]
    for jj in np.arange(len(shift_inds)-1):
        blockvec[shift_inds[jj]:shift_inds[jj+1]] = jj
    trials_dict['blockvec'] = blockvec.astype(int)

    return trials_dict


def get_const_conditions(trials_dict,keep_fixed_dict,summarize_blocks=True,**kwargs):#~cond_deltat
    N_trials = len(trials_dict['start_time'])

    cond_vec = np.zeros(N_trials) * np.nan

    if summarize_blocks:
        myblocks = trials_dict['probabilityLeft']
    else:
        myblocks = trials_dict['blockvec']

    #u_blocktypes = np.unique(myblocks)
    var_dict = {key:val for key,val in keep_fixed_dict.items()}
    var_dict['blk'] = myblocks
    cond_vec,cond_dict = assign_condition_numbers(var_dict)
    if 'exclude_bool' in kwargs:
        cond_vec[kwargs['exclude_bool']] = -1  # this is the excluded ones
    for key in cond_dict.keys():
        cond_dict[key]['blk'] = int(cond_dict[key]['blk'])
    return cond_vec,cond_dict


def assign_condition_numbers(var_dict):
    keys = list(var_dict.keys())
    values = np.column_stack([var_dict[k] for k in keys])

    unique_rows, inverse_indices = np.unique(values, axis=0, return_inverse=True)

    outdict = {ii + 1: dict(zip(keys, row)) for ii, row in enumerate(unique_rows)}

    # Create the output vector
    output_vector = inverse_indices + 1  # +1 to start numbering from 1

    return output_vector, outdict


# todo put this into a reader fn
def load_cccp_data(recids,srcdir,runname):
    cccp_rec_dict = {}
    for recid in recids:
        srcfile = os.path.join(srcdir,'%s__%s__ccCPtuning.h5'%(recid,runname))
        with h5py.File(srcfile,'r') as hand:
            cccpmat,uids = [hand['ccCP/%s'%subkey][()] for subkey in ['cccpmat','uids']]
            if recid == recids[0]:#just do this one
                dimdict = {}
                dgrp = hand['ccCP/cccpmat_dimensions']
                for key in dgrp.keys():
                    dimdict[int(key.replace('d',''))] = [dgrp[key][0][0].decode()]+[[el[0].decode() for el in dgrp[key][1:]]]

        cccp_rec_dict[recid] = {'cccpmat':cccpmat,'uids':uids}
    return cccp_rec_dict,dimdict


def assing_tunings(Units,cccp_rec_dict,dim_dict,get_significance=True,pthr=0.05,pthr_naive=0.001,cccpthr=0.52):
    #dim_dict = {1:['vartags',vartags],2:['conditions',cond_keys],3:['features',featnames]}
    vartags = dim_dict[1][1]
    cond_keys = dim_dict[2][1]
    for U in Units:
        subdict = cccp_rec_dict[U.recid]
        uidx = int(np.where(subdict['uids']==U.uid)[0])
        setattr(U,'tunings',{vartag:{condkey:{} for condkey in cond_keys} for vartag in vartags})
        for vv,vartag in enumerate(vartags):
            for cc,condkey in enumerate(cond_keys):
                for ff,featname in enumerate(dim_dict[3][1]):
                    myval = subdict['cccpmat'][uidx,vv,cc,ff]
                    U.tunings[vartag][condkey][featname] = int(myval) if featname == 'pref' else myval
        #DeepDiff(U.tunings, U.tunings2)
    if get_significance:
        for U in Units:
            setattr(U,'tsignif',{vartag:{} for vartag in vartags})
            for vartag in vartags:
                for blk_cond in ['blks_summary','blks_detail']:
                    cond1 = U.tunings[vartag][blk_cond]['pval'] <= pthr
                    cond2 = U.tunings[vartag]['naive']['pval'] <= pthr_naive
                    cond3 = U.tunings[vartag][blk_cond]['pref'] == U.tunings[vartag]['naive']['pref']
                    cond4 = U.tunings[vartag][blk_cond]['cccp'] >= cccpthr
                    if cond1 & cond2 & cond3 & cond4:
                        signif = True
                    elif np.isnan(U.tunings[vartag][blk_cond]['cccp']):
                        signif = np.nan
                    else:
                        signif = False
                    U.tsignif[vartag][blk_cond.replace('blks_','')] = signif

def false_discovery_control(pvals,axis=0,method='bh'):
    '''taken and abbreviated from scipy.stats'''
    from copy import deepcopy
    ps = deepcopy(pvals)
    ps = np.asarray(ps)
    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]


    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)


def load_taskresp_data(recids,srcdir,runname,tdict_type='equaldur'):
    rec_dict = {}
    for recid in recids:
        srcfile = os.path.join(srcdir,'%s__%s__ccCPtuning.h5'%(recid,runname))
        with h5py.File(srcfile,'r') as hand:
            grp = hand['taskresp_%s' % tdict_type]
            pmat,signif_vec,uids = [grp[subkey][()] for subkey in ['pmat_bh','signif','uids']]
        rec_dict[recid] = {'pmat_bh':pmat,'signif':signif_vec,'uids':uids}
    return rec_dict


def assing_taskresp(Units, result_dict, pthr=0.001):
    # dim_dict = {1:['vartags',vartags],2:['conditions',cond_keys],3:['features',featnames]}
    for U in Units:
        subdict = result_dict[U.recid]
        uidx = int(np.where(subdict['uids'] == U.uid)[0])
        pvec = subdict['pmat_bh'][uidx]

        is_signif = np.nanmin(pvec) <= pthr if np.isnan(pvec).sum() < (len(pvec)) else np.nan
        setattr(U, 'issignif_task', is_signif)
        setattr(U, 'task_pvecFDR', pvec)
