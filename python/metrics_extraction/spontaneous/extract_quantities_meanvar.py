import sys
from matplotlib import pyplot as plt
import yaml
import os
import numpy as np
import h5py
import git

n_tints_min = 6 #minimum number of tints needed to calculate mean and std
nmin_per_block = 3 #minimal number of tints per block to calculate quantity
mininput_ifn = 4 #min_nspike_per_trial = mininput_ifn+2#n-spikes = n_intervals +1; if intervals are paired you need +1 to make the pairing; so with 3 mininput you have at least 5 spikes/trial
full_estimate_trials_only = False


plot_on = True
save_on = True


plotvars = ['rate','B','LvR','M']
varselection = ['M', 'B', 'Lv', 'LvR', 'CV2', 'IR', 'Rho']
allvarnames = varselection + ['rate']

figpath = 'FIGDIR/preprocessing/meanvar_evolution'
stimplotdir =  'config/stimplotstyles'
recretrieval_paths = 'PATHS/general_paths.yml'
stylepath = 'config/presentation.mplstyle'


filename,pathpath,ttag,dstpath,tintfile,logfile,exptype = sys.argv[1:]

savename_stub = os.path.basename(tintfile).split('.h5')[0]

genfigpath = os.path.join(figpath,'%s__meanvar'%savename_stub)
with open(pathpath) as yfile: pdict = yaml.safe_load(yfile)

sys.path.append(pdict['code'])

from utils import data_classes as dc
from utils import data_handling as dh
from utils import interval_fns as ifn
from utils import file_helpers as fhs
from utils import stimfuncs as sfn
from utils import accesstools as act


plt.style.use(stylepath)


aSetting = act.Setting(recretrieval_paths)


fn_dict = ifn.intfn_dict

#if varselection == 'all':
#    varselection = list(fn_dict.keys())


plotvars = [var for var in plotvars if var in allvarnames]

logger = fhs.make_my_logger(pdict['logger']['config_file'],logfile)


cfgpath,dspath = dc.get_paths('Passive',pdict)#N.B.: we do not need specific configs for the other types of datasets

aRec = dc.RecPassive(filename,cfgpath,dspath)

smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
stimdict = smod.get_stimdict(aRec)
plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple,horizontal=False)


logger.info('#############START Processing %s'%aRec.id)
uids = aRec.unit_ids.astype(int)

logger.info('Nunits: %i'%len(uids))

if len(uids) == 0:
    sys.exit('No units- stopping here')




unitdict = {unitid:np.unique(dh.grab_spikes_unit(aRec, unitid)) for unitid in uids}
genrates = np.array([len(unitdict[uid]) for uid in uids])/aRec.dur

#getting the time intervals


#tintfile = os.path.join(tsel_path,'%s__TSEL%s.h5'%(recid,ttag))
with h5py.File(tintfile,'r') as hand: tints = hand['tints'][()]

tintdurs = np.diff(tints)[:,0]

if len(tints)==0:
    logger.info('NO TINTS AVAILABLE')
    exit(1)



dtints = tints[-1,-1]-tints[0,0]

smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
blockdict = smod.get_blockdict(aRec,stimdict)
# cut out where the block is 0 for aversion
# allow only tints that are in the blockdict --> thereby effectively cutting out the block is 0 for aversion
block_intervals = np.array([vals[1] for vals in blockdict.values()])
tints = np.array([tint for tint in tints if np.sum([dh.check_olap(block_int,tint) for block_int in block_intervals])>0])

#now for all units
datadict = {}
n_isis = np.zeros((len(uids),len(tints)),dtype='int')
tintdurs = np.diff(tints)[:,0]
for uu,uid in enumerate(uids):
    spiketimes = unitdict[uid]
    nspikevec = np.array([len(ifn.getspikes(spiketimes, tint)) for tint in tints])
    isi_list = ifn.collect_intervals_per_tint(tints, spiketimes)

    pairs_list = [np.vstack([isis[:-1], isis[1:]]) for isis in isi_list]
    rates_temp = np.ma.log10(nspikevec/tintdurs)
    subdict = {'rate':rates_temp.filled(np.nan)}
    for varname in varselection:
        subdict[varname] = ifn.calc_quant_segwise(isi_list, pairs_list, varname, fn_dict, mininputsize=mininput_ifn)#at least 3 isis! needed i.e. for paired like mem at least 5 needed!

    if full_estimate_trials_only:

        #getting all vars nan-ed where one var is nan to ensure same number of trials everywhere
        subdict_stack = np.vstack([subdict[varname] for varname in varselection])
        nancond =  np.isnan(subdict_stack.mean(axis=0))
        for varname in varselection+['rate']:
            subdict[varname][nancond] = np.nan


    datadict[uid] = subdict
    if len(isi_list)>0:
        isis = np.hstack(isi_list)
        nsavail = np.array([len(vec) for vec in isi_list])
        n_isis[uu] = nsavail

#varnames = ['B','Rho','LvR','rate']
gendict = {uid:{var:np.array([np.nan,np.nan]) for var in allvarnames} for uid in uids}
for uu,uid in enumerate(uids):
    for var in allvarnames:
        vals_temp = datadict[uid][var]
        cond = (~np.isnan(vals_temp)) & (~np.isinf(vals_temp))
        if np.sum(cond) >= n_tints_min:
            vals = vals_temp[cond]
            gendict[uid][var] = np.array([np.mean(vals),np.std(vals)])

blockkeys = np.sort(list(blockdict.keys()))
tint_blocks = [blocknum for blocknum in blockkeys if len([tt for tt,tint in enumerate(tints) if dh.check_olap(blockdict[blocknum][1],tint)])>0 ]

nvars,nblocks = len(allvarnames),len(tint_blocks)

if nblocks>1:
    block_per_tint = np.array([ int(np.array(tint_blocks)[np.array([dh.check_olap(blockdict[blnum][1],tint) for blnum in tint_blocks])][0]) for tint in tints])
    assert len(block_per_tint) == len(datadict[uid][var]),'len-mismatch blockpertintvec and actual data!'
    datad_blwise = {uid:{var:{} for var in allvarnames} for uid in uids}
    datad_blwise = {uid:np.zeros((nblocks,nvars,2))+np.nan for uid in uids}

    for bb,blocknum in enumerate(tint_blocks):
        blname,bltint = blockdict[blocknum]
        tint_inds = np.array([tt for tt,tint in enumerate(tints) if dh.check_olap(bltint,tint)])
        if len(tint_inds)>0:
            for uid in uids:
                for vv,varname in enumerate(allvarnames):
                    vals_temp = datadict[uid][varname][tint_inds]
                    cond = (~np.isnan(vals_temp)) & (~np.isinf(vals_temp))
                    if np.sum(cond) >= nmin_per_block:
                        datad_blwise[uid][bb,vv] = np.array([np.mean(vals_temp[cond]),np.std(vals_temp[cond])])
else:
    print('only one block!')

#saving stuff

logger.info('Finished Processing %s'%aRec.id)
if save_on:
    savename_stub = os.path.basename(tintfile).split('.h5')[0]
    #savename = os.path.join(dstpath,'%s__TSEL%s_quantities.h5'%(aRec.id,ttag))
    savename = os.path.join(dstpath,'%s_quantities_meanvar.h5'%savename_stub)
    with h5py.File(savename,'w') as hand:
        sgrp = hand.create_group('seg')
        for var in allvarnames:
            meanvec,stdvec = np.array([gendict[uid][var] for uid in uids]).T
            sgrp.create_dataset('%s_mean'%var,data=meanvec,dtype='f')
            sgrp.create_dataset('%s_std'%var,data=stdvec,dtype='f')

        hand.create_dataset('uids',data=uids)
        hand.create_dataset('n_isis',data=n_isis)
        hand.attrs['recid'] = aRec.id
        hand.attrs['ttag'] = ttag
        hand.attrs['n_tints'] = len(tints)
        hand.attrs['tfile'] = tintfile
        hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
        hand.attrs['srcfile'] = __file__

    logger.info('SAVED %s at %s################'%(aRec.id,savename))

    dstpath2 = dstpath+'_detail'
    savename2 = os.path.join(dstpath2,'%s_quantities_detail.h5'%savename_stub)
    with h5py.File(savename2,'w') as hand:
        if len(tint_blocks) > 1:

            blgrp = hand.create_group('blockwise')
            for vv,var in enumerate(allvarnames):
                varmat_mean,varmat_std = np.array([datad_blwise[uid][:,vv] for uid in uids]).transpose(-1,0,1)
                blgrp.create_dataset('%s_mean'%var,data=varmat_mean,dtype='f')
                blgrp.create_dataset('%s_std'%var,data=varmat_std,dtype='f')

            # include blockinfo
            binf = hand.create_group('block_info')
            binf.create_dataset('blocknums',data = np.array(tint_blocks),dtype='i')
            strList = [blockdict[bb][0] for bb in tint_blocks]
            asciiList = [n.encode("ascii", "ignore") for n in strList]
            binf.create_dataset('blocknames', (len(asciiList), 1), 'S10', asciiList)
            binf.create_dataset('blocktimes', data= np.array([blockdict[bb][1] for bb in tint_blocks]),dtype='f')
            binf.create_dataset('block_per_tint',data=block_per_tint,dtype='i')

        grp = hand.create_group('tintwise')
        for var in allvarnames:
            data = np.vstack([datadict[uid][var] for uid in uids]).T
            grp.create_dataset(var,data=data,dtype='f')
        hand.create_dataset('uids',data=uids)
        hand.attrs['recid'] = aRec.id
        hand.attrs['ttag'] = ttag
        hand.attrs['n_tints'] = len(tints)
        hand.attrs['tfile'] = tintfile
        hand.attrs['githash'] =  git.Repo(search_parent_directories=True).head.object.hexsha
        hand.attrs['srcfile'] = __file__

    logger.info('SAVED DETAILS %s at %s################'%(aRec.id,savename2))




if plot_on:
    logger.info('plotting meanvar %s --> %s\n'%(aRec.id,genfigpath))

    from utils import plotting_basics as pb
    from scipy.stats import scoreatpercentile as sap
    perclim = 1
    perclim_blocks = 3

    sortinds = np.argsort(aRec.unit_electrodes)#sorting unit ids so that they are sequential in terms of their electrode location
    uids_plot = uids[sortinds]
    elecs_sorted = aRec.unit_electrodes[sortinds]
    uid_locs = aRec.ellocs_all[elecs_sorted]

    def figsaver(fig, nametag, closeit=True):
        figname = os.path.join(genfigpath, '%s__%s.png'%(savename_stub,nametag))
        figdir = os.path.dirname(figname)
        if not os.path.isdir(figdir): os.makedirs(figdir)
        fig.savefig(figname)
        if closeit: plt.close(fig)



    if len(tint_blocks)>0:
        blockdict_tints = {blnum:[blockdict[blnum][0], \
                                  np.array([tt for tt,tint in enumerate(tints) \
                                            if dh.check_olap(blockdict[blnum][1],tint)])[np.array([0,-1])]] \
                           for blnum in tint_blocks}
        plot_blocks = sfn.make_blockplotfn(blockdict_tints, smod.styledict_blocks, horizontal=False)

    mymat2 = np.array([np.array([datadict[uid][var] for var in plotvars]) for uid in uids_plot])
    for vv,var in enumerate(plotvars):
        f,axarr = plt.subplots(2,3,figsize=(0.07*len(uids_plot)+2,3),gridspec_kw={'height_ratios':[1,0.02],'width_ratios':[0.05,1,0.02]})
        f.subplots_adjust(left=0.1,right=0.91,bottom=0.2,wspace=0.05,hspace=0.02)
        stax,ax,cbax = axarr[0]
        noax0,locax,noax1 = axarr[1]
        for myax in [noax0,noax1]: myax.set_axis_off()
        ax.set_facecolor('white')
        showmat = mymat2[:,vv,:].T
        cond = (~np.isnan(showmat)) & (~np.isinf(showmat))
        im = ax.imshow(showmat,cmap='jet',origin='lower',aspect='auto',interpolation='nearest',vmin=sap(showmat[cond],perclim),\
                       vmax=sap(showmat[cond],100-perclim))
        #ax.set_ylabel('block')
        ax.set_xticks([])
        pb.make_locax(locax,uid_locs, cols=['k', 'lightgray'], linecol='grey',ori='horizontal',textoff=-6,rotation=45)
        ax.get_shared_x_axes().join(ax, locax)
        f.colorbar(im,cax=cbax,extend='both')
        #cbax.text(0.5,-0.05,varnames[vv],transform=cbax.transAxes,ha='center',va='top')
        cbax.set_title(var)
        f.suptitle('%s %s'%(savename_stub,exptype))
        if len(tint_blocks)>1:
            plot_blocks([stax])
            ax.set_yticks([])
            stax.set_ylabel('trial')
            ax.get_shared_y_axes().join(ax, stax)
            stax.set_xticks([])
        else:
            ax.set_ylabel('trial')
            stax.set_axis_off()
        ax.set_ylim([-0.5,len(showmat)+0.5])
        figsaver(f,'trials_%s'%(var))

    #plotting vars
    print('XXXXXX',len(tint_blocks))
    print(filename)
    if len(tint_blocks)>1:
        print('PLOTTING BLOCKS!')
        mymat = np.array([datad_blwise[uid] for uid in uids_plot])
        yticklist = ['%s%i'%(blockdict[blnum][0],blnum) for blnum in tint_blocks]
        for vv, var in enumerate(plotvars):
            for mm,mtag in enumerate(['m','s']):

                f,axarr = plt.subplots(2,2,figsize=(len(uids_plot)*0.07+1,3),gridspec_kw={'height_ratios':[1,0.02],'width_ratios':[1,0.02]})
                f.subplots_adjust(left=0.08,right=0.91,bottom=0.2,wspace=0.05,hspace=0.02)
                ax,cbax = axarr[0]
                locax,noax = axarr[1]
                noax.set_axis_off()
                ax.set_facecolor('white')
                showmat = mymat[:,:,vv,mm].T
                cond = (~np.isnan(showmat)) & (~np.isinf(showmat))
                im = ax.imshow(showmat,cmap='jet',origin='lower',aspect='auto',vmin=sap(showmat[cond],perclim_blocks),\
                       vmax=sap(showmat[cond],100-perclim_blocks))
                #ax.set_ylabel('block')
                ax.set_yticks(np.arange(len(tint_blocks)))
                ax.set_yticklabels(yticklist)
                ax.set_xticks([])
                pb.make_locax(locax,uid_locs, cols=['k', 'lightgray'], linecol='grey',ori='horizontal',textoff=-6,rotation=45)
                ax.get_shared_x_axes().join(ax, locax)
                f.colorbar(im,cax=cbax,extend='both')
                #cbax.text(0.5,-0.05,varnames[vv],transform=cbax.transAxes,ha='center',va='top')
                cbax.set_title('%s_%s'%(var,mtag))
                f.suptitle('%s %s'%(savename_stub,exptype))
                figsaver(f,'blocks_%s_%s'%(var,mtag))



