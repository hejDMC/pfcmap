import sys
import yaml
import os
import numpy as np
import h5py

pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

"""
pathpath = 'PATHS/filepaths_carlen.yml'
myrun =  'runC00dMP3_brain'
cmethod = 'ward'
ncluststr = '8'#
"""


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import evol_helpers as ehelpers
from pfcmap.python import settings as S

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['datapath_metricsextraction'] +'/quantities_all_meanvar/Carlen_quietactive_detail'
statstag = 'TSELprestim3__STATEactive__all'
savepath_evol = os.path.join(pathdict['savepath_gen'].replace('SOMs','category_evolutions'))


tints_per_block = 50
nperblock_min = 5 #for transmat, a unit needs nperblock_min values in a block to be considered

endtag = '%s_quantities_detail.h5'%statstag



rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)
somfile = uloader.get_somfile(rundict,myrun,pathdict['savepath_gen'])
somdict = uloader.load_som(somfile)
somfeats = somdict['features']


clustdict = uloader.extract_clusterlabeldict(somfile,cmethod,get_resorted=True)
ncl = int(ncluststr)
labels = clustdict[ncl]

ufilterfn_list = [lambda U: True if rundict['utypes'] == 'all' else U.utype in rundict['utypes']]
##units

#to only get the units that feature in the actual SOM
metricsfiles = S.get_metricsfiles_auto(rundict,pathdict['src_dirs']['metrics'],tablepath=pathdict['tablepath'])
Units_allowed,_,_ =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)#rois from path = False  gets us the Gao rois


#recid = '128514_20191215-probe0'
recids = np.unique([U.recid for U in Units_allowed])#S.get_allowed_recids_from_table(tablepath=pathdict['tablepath'],sheet='sheet0',not_allowed=['-', '?'])
for recid in recids:#
    print('PROCESSING %s'%recid)



    outfile = os.path.join(savepath_evol,'%s__%s__%s%i__%s__catevols.h5'%(recid,myrun,cmethod,ncl,statstag))


    srcfile = os.path.join(srcdir,'%s__%s'%(recid,endtag))
    metricsfile = os.path.join(pathdict['src_dirs']['metrics'],'metrics_%s.h5'%recid)
    with h5py.File(metricsfile,'r') as hand:
        exptype0 = hand['Task'][()].decode()
        if exptype0 in S.task_replacer:
            exptype = S.task_replacer[exptype0]
        else:
            exptype = str(exptype0)
        #print(exptype0,exptype)
        dataset = hand['Dataset'][()].decode()
        uids_temp_metr = hand['uids'][()]


    with h5py.File(srcfile,'r') as hand:
        #recid = hand.attrs['recid']
        has_blocks = 'blockwise' in hand
        uids_temp_src = hand['uids'][()]

    assert (uids_temp_src==uids_temp_metr).all(),'mismatching uids in metrics and details file'




    #ok, the thing to do is getting artificial blocks from this file!

    if exptype == 'Aversion':
        split_dict = {1: 2}
        stimtag = ehelpers.get_exp_group_aversion(recid, infofiledir='NWBEXPORTPATH/info_export')
        full_taskname = '%s_%s'%(exptype,stimtag)

    else:
        split_dict = {}
        full_taskname = str(exptype)




    Units,_,nblocks,blockvec,blocknames = ehelpers.get_units(srcfile,metricsfile,somfeats,split_dict,chop_to_blocks=not has_blocks,nperblock=tints_per_block)
    #Units = ehelpers.filter_units(Units,ufilterfn_list)
    uids_allowed = [U.uid for U in Units_allowed if U.recid==recid]
    Units = [U for U  in Units if U.uid in uids_allowed]
    if len(Units)>1:
        ehelpers.get_set_general_feats(Units,somfeats)
        Units = [U for U in Units if ~np.isnan(np.mean([getattr(U,feat) for feat in somfeats]))]#control that they are not nan overall!


        ehelpers.classify_blocks(Units, somdict, labels)#categorized

        #np.unique([U.area for U in Units])
        #--> save blockvec,blocknames,nblocks,exptype,full_taskname
        blocknums = np.unique(blockvec[blockvec>-1])
        trials_per_block = np.array([np.sum(blockvec==bb) for bb in blocknums])


        #get the numbers
        get_all_vecs_unit = lambda Uobj: np.vstack([getattr(Uobj,'%s_vec'%feat) for feat in somfeats]).T
        nU =  len(Units)
        trials_per_block_mat = np.zeros((nblocks,nU)).astype(int)
        for uu,U in enumerate(Units):
            vmat = get_all_vecs_unit(U)
            for bb,blocknum in enumerate(blocknums):
                cond = (~np.isnan(vmat.max(axis=1))) & (blockvec == blocknum)
                trials_per_block_mat[bb,uu] = np.sum(cond)
        #--> save trials_per_block_mat
        #--> save trials_per_block

        my_uids = np.array([U.uid for U in Units])
        #--> save my_uids --> uids

        bmu_vec,clust_vec = np.array([[U.bmu,U.clust] for U in Units]).T.astype(int)
        bmu_blocks,clust_blocks = np.array([[ehelpers.mask_to_val(U.bmu_blocks),ehelpers.mask_to_val(U.clust_blocks)] for U in Units]).transpose(1,2,0).astype(int)
        #--> save bmu_blocks and clust_blocks, bmu_vec and clust_vec

        #now only consider transitions with enough N in the thing

        n_trans = nblocks-1

        trans_collection = np.zeros((n_trans,ncl,ncl),dtype=int)#ntrans x srcs x target
        for block_src_idx in np.arange(nblocks-1):
            block_target_idx = block_src_idx + 1
            transmat = trans_collection[block_src_idx]

            n_cond = (trials_per_block_mat[block_src_idx]>=nperblock_min) & (trials_per_block_mat[block_target_idx]>=nperblock_min)
            csrc = clust_blocks[block_src_idx,n_cond]
            ctarget = clust_blocks[block_target_idx,n_cond]
            for cc_src in np.arange(ncl):
                transmat[cc_src] = np.array([np.sum(ctarget[csrc==cc_src]==cc_target) for cc_target in np.arange(ncl)])

        #--> save transmat_collection,nperblock_min(as params)

        save_dict = {'blockvec':blockvec,\
                     'blocknames':blocknames,\
                     'nblocks':nblocks,\
                     'task_orig':exptype0,\
                     'task':exptype,\
                     'task_full':full_taskname,\
                    'trials_per_block_units':trials_per_block_mat,\
                      'trials_per_block':trials_per_block,
                      'uids':my_uids,\
                     'bmu_vec':bmu_vec,\
                     'clust_vec':clust_vec,\
                     'bmu_blocks':bmu_blocks,\
                     'clust_blocks':clust_blocks,\
                     'transmats':trans_collection,\
                     'recid':recid,\
                     'nperblock_min':nperblock_min,\
                     'statstag':statstag,\
                     'srcfile':srcfile,\
                     'metricsfile':metricsfile}

        uloader.save_dict_to_hdf5(save_dict, outfile,strtype='S10')
    else:
        print('too few units in rec %s (N=%i)'%(recid,len(Units)))


'''
f,ax = plt.subplots()
ax.imshow(transmat.T,origin='lower',cmap='jet',aspect='auto')
ax.set_xlabel('from')
ax.set_ylabel('to')

overalltm = np.sum(trans_collection,axis=0)
f,ax = plt.subplots()
ax.imshow(overalltm.T,origin='lower',cmap='jet',aspect='auto')

#normalize by expected

target_sums = overalltm.sum(axis=1)
expected_target_fracs = target_sums/np.sum(target_sums)
target_fracs = overalltm/target_sums[:,None]#column sum(targets)==1; compare eg overalltm[3]/np.sum(overalltm[3]) np.sum(target_fracs[3])==1


f,ax = plt.subplots()
ax.imshow(target_fracs.T,origin='lower',cmap='jet',aspect='auto')

f,ax = plt.subplots()
ax.plot(expected_target_fracs,'r')
ax.plot(target_fracs[1],color='k')

diff_perc = target_fracs/expected_target_fracs
from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=diff_perc.max())
f,axarr = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,0.1]})
ax,cax = axarr
im = ax.imshow(diff_perc.T,origin='lower',cmap='bwr',aspect='auto',norm=divnorm)
f.colorbar(im,cax=cax)
'''