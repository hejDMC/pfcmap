import h5py
import numpy as np
import os
import matplotlib as mpl
from pfcmap.python import settings as S
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh

nmin_per_block = 3
def get_units(detailfile,metricsfile,somfeats,split_dict={},chop_to_blocks=False,nperblock=20
              ):
    has_blocks = False
    with h5py.File(detailfile,'r') as hand:
        if 'blockwise' in hand:
            blockmat = np.array([hand['blockwise/%s'%var][()] for var in somfeats])
            has_blocks = True
            blocknames = np.array([el.decode() for el in hand['block_info/blocknames'][()].flatten()])
            if 'block_per_tint' in hand['block_info']:
                blockvec = hand['block_info/block_per_tint'][()]
            else:
                blockvec = None

        else:
            blocknames = np.array(['block'])
            blockvec = None


        tintmat = np.array([hand['tintwise/%s'%var.replace('_mean','')][()] for var in somfeats])
        ntints = tintmat.shape[1]

        if ntints<nperblock:
            print('WARNING - number of available time intervals is smaller than desired number of intervals per block, returning EMPTY')
            return [],np.array([]),0,np.array([]),[]

        uids = hand['uids'][()]

        #tintmat = np.zeros_like(tintmat0)
        #for vv,var in enumerate(somfeats):
        #    if var in S.featfndict:
        #        tintmat[vv] = S.featfndict[var](tintmat0[vv])
        #    else:
        #        tintmat[vv] = tintmat0[vv]


    if has_blocks:
        #print(blockmat.shape)
        bllist = [bl for bl in blockmat.transpose(2,0,1) if not np.isnan(bl).all()]
        blocknames = np.array([blname for blname,bl in zip(blocknames,blockmat.transpose(2,0,1)) if not np.isnan(bl).all()])
        for bb,bl in enumerate(blockmat.transpose(2,0,1)):
            if np.isnan(bl).all():
                blockvec[np.where(blockvec==bb)[0]] = -1
                print('deleting blockidx',bb)

        blockmat = np.array(bllist).transpose(1,2,0)


        #print ('blockmat shape',blockmat.shape)

        if np.size(split_dict)>0:
            splitkeys = np.sort([key for key in split_dict.keys()])[::-1]
            if len(splitkeys) > 1: print('WARNING - multiple splittings not tested, check for indext errors!!!!!!!')
            for block_ind in splitkeys:
                split_fac = split_dict[block_ind]

                myinds = np.where(blockvec == block_ind)[0]
                subblock_inds = np.array_split(myinds, split_fac)
                newblocks = np.zeros((len(somfeats), len(uids), split_fac)) * np.nan

                for ii, sbl_inds in enumerate(subblock_inds):
                    sbl_inds = subblock_inds[ii]
                    sub_tints = tintmat[:, sbl_inds, :]
                    for uu in np.arange(len(uids)):
                        for vv in np.arange(len(somfeats)):
                            featvec = sub_tints[vv, :, uu]
                            cond = (~np.isnan(featvec)) & (~np.isinf(featvec))  # to counteract the numpy bug
                            if np.sum(cond) >= nmin_per_block:
                                newblocks[vv, uu, ii] = np.mean(featvec[cond])
                blockmat0 = np.delete(blockmat, block_ind, axis=2)
                blockmat = np.insert(blockmat0, [block_ind], newblocks, axis=2)
                blocknames = np.insert(blocknames,block_ind,blocknames[block_ind])

                blockvec[myinds[-1] + 1:] += split_fac - 1
                subblock_inds = np.array_split(myinds, split_fac)
                for ii, subinds in enumerate(subblock_inds):
                    blockvec[subinds] = ii + block_ind

    if chop_to_blocks and not has_blocks:
        ntints = tintmat.shape[1]
        split_fac = int(ntints / nperblock)
        tintmat_blocks = np.array_split(tintmat, split_fac, axis=1)
        blockmat = np.zeros((len(somfeats), len(uids), split_fac)) * np.nan
        nblocks = split_fac
        for bb, sub_tints in enumerate(tintmat_blocks):
            for uu in np.arange(len(uids)):
                for vv,var in enumerate(somfeats):
                    featvec = sub_tints[vv, :, uu]
                    cond = (~np.isnan(featvec)) & (~np.isinf(featvec))  # to counteract the numpy bug
                    if np.sum(cond) >= nmin_per_block:
                        if var in S.featfndict:
                            myfeatvec = S.featfndict[var]['fn'](featvec[cond])
                        else:
                            myfeatvec = featvec[cond]
                        blockmat[vv, uu, bb] = np.mean(myfeatvec)
        blocknames = np.array(['sect' for bb in np.arange(nblocks)])
        blockvec = np.hstack([bb*np.ones((tintmat_blocks[bb].shape[1])) for bb in np.arange(nblocks)]).astype(int)
        has_blocks = True
    elif chop_to_blocks and has_blocks:
        assert 0, 'cannot chop existing block structure'


    #setting anatomical stuff
    anat_feats = list(S.anat_fdict.keys())

    collection_dict = {}
    with h5py.File(metricsfile,'r') as hand:
        for feat in anat_feats:
            if feat == 'roi':
                collection_dict[feat] = hand['anatomy/%s' % feat][()].astype(int)
            elif feat == 'location':
                 collection_dict[feat] = np.array([el.decode() for el in hand['anatomy/%s' % feat][()]])
                 #print(collection_dict[feat])
            else:
                collection_dict[feat] = hand['anatomy/%s' % feat][()]
        for feat in S.unittype_options:
            collection_dict[feat] =  hand['unit_type/%s'%(feat)][()]


    Units = []
    for uu,uid in enumerate(uids):
        U = uloader.SortedUnit()
        U.set_feature('uid', uid)
        uloader.set_various_features(U, uu, [anat_feats], [S.anat_fdict], collection_dict)
        U.get_area_and_layer()
        for vv,var in enumerate(somfeats):

            if var in S.featfndict:
                featvals = S.featfndict[var]['fn'](tintmat[vv,:,uu])
            else:
                featvals = tintmat[vv,:,uu]
            U.set_feature('%s_vec'%var,featvals)
            if has_blocks:
                U.set_feature('%s_blocks'%var,blockmat[vv,uu])
        uloader.set_boolstr(U, uu, S.unittype_options, collection_dict, featname='utype', na_tag='na')
        Units += [U]
    nblocks = 0 if not has_blocks else blockmat.shape[2]
    return Units,uids,nblocks,blockvec,blocknames



def get_exp_group_aversion(recid,infofiledir='NWBEXPORTPATH/info_export'):
    infofile = os.path.join(infofiledir, '%s__info.h5' % recid.replace('-', '_'))
    with h5py.File(infofile, 'r') as hand:
        genotype = hand['general/subject/genotype'][()].decode()
    if genotype.count('NPY') or genotype.count('C57BL') or genotype.count('Rbp4'):
        exp_group = 'ctrl'
    elif genotype.count('Esr1') or genotype.count('Vglut2'):
        exp_group = 'stim'
    else:
        assert 0, 'genotype %s not assigned to any exp_group' % genotype
    return exp_group


def filter_units(Units,checkfn_list):

    nchecks = len(checkfn_list)

    return [U for U in Units if np.sum([checkfn(U) for checkfn in checkfn_list]) == nchecks]

def get_set_general_feats(Units,somfeats):
    #give them their features by averaging over tints
    for U in Units:
        #feat_mat = np.array([getattr(U, sfeat+'_vec') for sfeat in somfeats])
        for sfeat in somfeats:
            featvec = getattr(U, sfeat+'_vec')
            cond =  (~np.isnan(featvec)) & (~np.isinf(featvec)) #to counteract the numpy bug
            if np.sum(cond) == 0:
                val = np.nan
            else: val = np.mean(featvec[cond])
            U.set_feature(sfeat,val)


def classify_blocks(Units,somdict,labels):
    somfeats = somdict['features']
    refmean,refstd = somdict['refmeanstd']
    featureweights = somdict['featureweights']
    weights = somdict['weights']

    get_features_blocks = lambda uobj: np.array([getattr(uobj, sfeat+'_blocks') for sfeat in somfeats])

    dmat = np.array([get_features_blocks(elobj) for elobj in Units])
    wmat_list = [uloader.reference_data(subdmat,refmean,refstd,featureweights) for subdmat in dmat.transpose(2,0,1)]#len == n blocks
    wmat = np.array(wmat_list).transpose(1,2,0)
    #nblocks = wmat.shape[2]

    #set BMU for each
    for uu,U in enumerate(Units):

        blockvals = wmat[uu]
        blockbools = np.isnan(np.sum(blockvals,axis=0))
        block_bmus = somh.get_bmus(blockvals.T,weights)
        block_bmus = np.ma.array(block_bmus,mask=blockbools)
        block_cats = np.ma.array(labels[block_bmus],mask=blockbools)

        U.set_feature('bmu_blocks',block_bmus)
        U.set_feature('clust_blocks',block_cats)

        # classify unit itself as a whole if possible
        feats_gen = np.array([getattr(U,feat) if hasattr(U,feat) else np.nan for feat in somfeats])
        if ~np.isnan(np.sum(feats_gen)):
            feats_reffed = uloader.reference_data(feats_gen, refmean, refstd, featureweights)
            bmu = int(somh.get_bmus(feats_reffed, weights))
            U.set_feature('bmu', bmu)
            U.set_feature('clust', labels[bmu])

def mask_to_val(vec,fillval=-1):
    vec2 = np.zeros_like(vec)*np.nan
    vec2[vec.mask] = fillval
    vec2[~vec.mask] = vec[~vec.mask]
    return vec2

def color_ticklabs(ncl,cdict_clust,tlabs):
    for cc in np.arange(ncl):
        col = cdict_clust[cc]
        tlabs[cc].set_color(col)

def set_color_ticklabs(ax,ncl,cdict_clust,which='x',flav='num',fs=8):
    if flav == 'num': clustvec = np.arange(ncl)+1
    else: clustvec = [flav]*ncl

    if which in ['x','both']:
        ax.set_xticks(np.arange(ncl))
        tlabs = ax.set_xticklabels(clustvec, fontweight='bold',fontsize=fs)
        color_ticklabs(ncl,cdict_clust,tlabs)
    if which in ['y','both']:
        ax.set_yticks(np.arange(ncl))
        tlabs = ax.set_yticklabels(clustvec, fontweight='bold',fontsize=fs)
        color_ticklabs(ncl,cdict_clust,tlabs)

def get_transprobmat(transcountmat):
    '''srcs x targets'''

    src_sums = transcountmat.sum(axis=1)
    return transcountmat/src_sums[:,None]

def get_expected_transcounts(transcountmat):
    '''srcs x targets'''
    target_sum = transcountmat.sum(axis=0)
    target_frac = target_sum/target_sum.sum()
    src_sum = transcountmat.sum(axis=1)
    return src_sum[:,None]*target_frac

def get_origByExpected_transcounts(transcountmat):
    return transcountmat/get_expected_transcounts(transcountmat)

def get_percIncrease_OrigOverExpected_transcounts(transcountmat):
    expected_transcounts = get_expected_transcounts(transcountmat)
    return (transcountmat-expected_transcounts)/expected_transcounts * 100


def get_cocoeff_mat(transcountmat):
    counts_1, counts_2 = [transcountmat.sum(axis=aa) for aa in [0, 1]]
    maxmat = np.zeros(transcountmat.shape)
    for cc1, c1 in enumerate(counts_1):
        for cc2, c2 in enumerate(counts_2):
            maxmat[cc1, cc2] = np.min([c1, c2])

    # ).min(axis=0)
    expmatch = get_expected_transcounts(transcountmat)
    coeff_matA = (transcountmat - expmatch) / (maxmat - expmatch)
    coeff_matB = (transcountmat-expmatch)/expmatch
    cocomat = np.zeros(transcountmat.shape)
    cocomat[transcountmat>=expmatch] = coeff_matA[transcountmat>=expmatch]
    cocomat[transcountmat<expmatch] = coeff_matB[transcountmat<expmatch]
    return cocomat


def make_statstitle(ax,plmat):
    stabi = np.nansum(np.diag(plmat))/np.nansum(plmat)
    ax.set_title('SI: %1.2f - %i'%(stabi,np.nansum(plmat.sum)),fontsize=6,y=0.9)

def matplotfn_counts(ax,tcmat,cmap='jet',titlefn=make_statstitle,**kwargs):
    vminmax = kwargs['vminmax'] if 'vminmax' in kwargs else [tcmat.min(),tcmat.max()]
    titlefn(ax,tcmat)
    return ax.imshow(tcmat.T,origin='lower',cmap=cmap,aspect='auto',vmin=vminmax[0],vmax=vminmax[1])

def matplotfn_probs(ax,tcmat,cmap='inferno',titlefn=make_statstitle,**kwargs):
    titlefn(ax,tcmat)
    plmat = get_transprobmat(tcmat)
    vminmax = kwargs['vminmax'] if 'vminmax' in kwargs else [plmat.min(),plmat.max()]
    return ax.imshow(plmat.T,origin='lower',cmap=cmap,aspect='auto',vmin=vminmax[0],vmax=vminmax[1])

def matplotfn_cocoeff(ax,tcmat,cmap='inferno',titlefn=make_statstitle,**kwargs):
    titlefn(ax,tcmat)
    plmat = get_cocoeff_mat(tcmat)
    vminmax = kwargs['vminmax'] if 'vminmax' in kwargs else [plmat.min(),plmat.max()]
    return ax.imshow(plmat.T,origin='lower',cmap=cmap,aspect='auto',vmin=vminmax[0],vmax=vminmax[1])

def matplotfn_orig_by_expected(ax,tcmat,cmap='bwr',titlefn=make_statstitle,**kwargs):
    titlefn(ax,tcmat)
    rat_mat = get_origByExpected_transcounts(tcmat)
    divnorm = kwargs['divnorm'] if 'divnorm' in kwargs else mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=rat_mat.max())
    return ax.imshow(rat_mat.T,origin='lower',cmap=cmap,aspect='auto',norm=divnorm)


def get_namevec(blockvec,blocknames,**kwargs):
    if blocknames[0] in ['on','off']:
        mode = 'context'
    elif blocknames[0] == 'sect':
        mode = 'default'
    elif 'tone' in blocknames and 'air' in blocknames:
        mode = 'aversion'
    else:
        assert 0, 'ungessable mode from blocknames %s'%(str(blocknames))

    if mode == 'default':
        return np.array(['B%i'%ii for ii in np.arange(len(blocknames))+1])

    if mode == 'aversion':
        blnums = np.unique(blockvec[~(blockvec == -1)])
        label_mapper = kwargs['label_mapper']
        return np.array([label_mapper[blnum] for blnum in blnums])




    if mode == 'context':

        blnums = np.unique(blockvec[~(blockvec == -1)])
        offinds = np.mod(blnums, 2) == 0
        oninds = np.mod(blnums, 2) == 1
        offnums = blnums[offinds] / 2 + 1
        onnums = (blnums[oninds] - 1) / 2 + 1
        uoff = np.unique(blocknames[offinds])
        uon = np.unique(blocknames[offinds])

        assert len(uoff) == 1, 'mixed offinds'
        assert len(uon) == 1, 'mixed oninds'

        offtag = blocknames[np.where(offinds)[0][0]]
        ontag = blocknames[np.where(oninds)[0][0]]
        #print(offtag,ontag)

        namevec = np.zeros(len(blocknames), dtype='<U8')
        namevec[offinds] = np.array(['%s%i' %(offtag,num) for num in offnums])
        namevec[oninds] = np.array(['%s%i' %(ontag,num) for num in onnums])
    return namevec

def check_context_pair(str1,str2):
    diff_bool = str1[:3]!=str2[:3]#different states
    if str1.count('on'):
        onstr,offstr = str1,str2
    else:
        offstr,onstr = str1,str2
    on_number =  int(''.join(filter(str.isdigit, onstr)))
    off_number = int(''.join(filter(str.isdigit, offstr)))
    numb_bool = off_number-on_number in [0,1]#on run must be younger or matched to off-number as an off always precedes an on

    return diff_bool and numb_bool



def plot_block_orig_matchmat(ax,match_mat,cdict_clust,cmap='Greys',grid_col = 'k',write_col = 'm',vminmax = [0, 1],write_data=False,write_counts=True,labelfs=8):

    match_sum = match_mat.sum(axis=1)
    match_mat_frac = match_mat / match_sum[:, None]
    nrows, ncols = match_mat_frac.shape
    ncl = len(match_sum)
    ax.imshow(match_mat_frac, cmap=cmap, origin='lower', vmin=vminmax[0], vmax=vminmax[1])
    set_color_ticklabs(ax, ncl, cdict_clust, which='both', flav='num', fs=labelfs)
    ax.hlines(y=np.arange(ncols) - 0.5, xmin=np.full(ncols, 0) - 0.5, xmax=np.full(ncols, ncols) - 0.5, color=grid_col)
    if write_data:
        for rr in np.arange(nrows):
            for cc in np.arange(ncols):
                ax.text(cc, rr, '%i' % (match_mat_frac[rr, cc] * 100), ha='center', va='center', color=write_col)
    ax.set_xlim([-0.5, ncols - 0.5])
    ax.set_ylim([-0.5, ncl - 0.5])
    if write_counts:
        for cc in np.arange(ncl):
            ax.text(ncols - 0.4, cc, '%i' % (match_sum[cc]), ha='left', va='center', color=cdict_clust[cc])
    if ncols > ncl:
        ax.axvline(ncl - 0.5, color=grid_col, lw=2)


def plot_cmap(cmapstr,vminmax):
    f, ax = mpl.pyplot.subplots(figsize=(0.8, 2.))
    f.subplots_adjust(left=0.02, right=0.3)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.pyplot.get_cmap(cmapstr),
                                   norm=mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1]), orientation='vertical')
    return f,ax


def plot_clustbars(ax,heights,cdict_clust,show_mean=True,**kwargs):
    ncl = len(heights)
    meanval = np.mean(heights)
    blist = ax.bar(np.arange(ncl), heights, color='k')
    ax.set_xticks(np.arange(ncl))
    if 'ticklabs' in kwargs:
        tlabs = ax.set_xticklabels(kwargs['ticklabs'], fontweight='bold')
        for cc,regname in enumerate(kwargs['ticklabs']):
            col = cdict_clust[regname]
            tlabs[cc].set_color(col)
            blist[cc].set_color(col)
    else:
        tlabs = ax.set_xticklabels(np.arange(ncl) + 1, fontweight='bold')
        for cc in np.arange(ncl):
            col = cdict_clust[cc]
            tlabs[cc].set_color(col)
            blist[cc].set_color(col)
    if show_mean:
        ax.axhline(meanval, color='k', linestyle='--')
        ax.text(0.99, meanval, '%1.2f' % meanval, transform=mpl.transforms.blended_transform_factory(
            ax.transAxes, ax.transData), ha='right', va='bottom', color='k')

