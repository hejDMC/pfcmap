import numpy as np
import sys
import os
import yaml
import h5py
import time
import multiprocessing as mp# original multiprocessing Pool much faster than threadPool!
from datetime import timedelta

pathpath,recfilename,chanflav, regstr = sys.argv[1:]

regs_of_interest = [bla.strip() for bla in regstr.strip('[]').replace("'",'').split(',')]
check_region = lambda locname: len([1 for reg in regs_of_interest if locname.count(reg)])>0

#pfc_regs = ['ACA','ILA','MOs','ORB','PL']

pathpath = 'PATHS/transient_paths.yml'
badtimesdir = 'ZENODOPATH/preprocessing/bad_times_manual'


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])





from pfcmap.python.utils import burst_analyses as ba

from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import file_helpers as fhs
from SPOOCs.utils import helpers as hf
from SPOOCs.utils import detection as det
from pfcmap.python.utils import transient_detection as tdet
from pfcmap.python.utils import statdet_helpers as shf
logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])
logger.info('####Starting %s'%(os.path.basename(recfilename)))

#
calc_nvsthr = True
ditch_lonely_offperiods = True
maxdist_lonelies = 1.5
opto_buff = np.array([-0.006,0.015])
maxdist_opto = 0.3

aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=True)
recid = os.path.basename(recfilename).split('__')[0]

aRec.get_freetimes() #aRec.freetimes now has the free times

exptype = (aRec.h_info['exptype'][()]).decode()



configfile = pathdict['configs']['functions']
defaultdictpath = pathdict['configs']['defaultdict']
outpath_temp = pathdict['save_dirs']['outpath_temp']
collection_dir = pathdict['save_dirs']['collection_dir']
for savedir in [collection_dir,outpath_temp]:
    if not os.path.isdir(savedir):
        os.makedirs(savedir)



outfile_pattern = '%s__chCCC__transients.h5'%(aRec.id)

#chan = aRec.el_ids[121]#selecting a cute example channel
#chanloc = aRec.ellocs_all[chan]


pmod = fhs.retrieve_module_by_path(configfile)#functions are in here!
kword_changes = {}#
P = pmod.Paramgetter(aRec.sr,aRec.dur,defaultdictpath,**kword_changes)#


more_blocks = []




#blocking offstates if explicitly 'yes' in offstatefiles!
regard_offstates = False
offstatefile = os.path.join(pathdict['offstate_dir'], '%s__offstates.h5' % recid)
with h5py.File(offstatefile, 'r') as fhand:
    osrc = fhand.attrs['offdet_src']#with too few units we dont have it!

if osrc == 'self':
    myoffstatefile = str(offstatefile)
    regard_offstates = True
elif osrc.count('from-other'):
    regard_offstates = True
    refprobe = osrc.replace('from-other:','')
    thisprobe = recid.split('_')[-1]
    myoffstatefile = offstatefile.replace(thisprobe,refprobe)



if regard_offstates:
    with h5py.File(myoffstatefile, 'r') as fhand:
        quiet_pts = fhand['results/borderpts'][()]
        bw = fhand['results'].attrs['bw']

        if ditch_lonely_offperiods:
            offpts = fhand['results/offpts'][()]
            lonelies = np.array([[q0,q1] for q0,q1 in quiet_pts if np.sum((offpts[:,0]<q1) & (offpts[:,1]>q0))==1])
            #durs_lonelies = np.diff(lonelies)[:,0]*bw-0.6#thats from the param file the border added
            get_prev_neigh_diff = lambda l0: l0-quiet_pts[quiet_pts[:,1]<l0,1][-1] if not l0 == np.min(quiet_pts[:,0]) else np.nan
            get_next_neigh_diff = lambda l1: quiet_pts[quiet_pts[:,0]>l1,0][0]-l1 if not l1==np.max(quiet_pts[:,1]) else np.nan
            dist_lonelies = np.array([np.min([get_prev_neigh_diff(l0),get_next_neigh_diff(l1)]) for l0,l1 in lonelies])*bw
            kept_lonelies = lonelies[dist_lonelies<=maxdist_lonelies]
            quiet_pts = np.r_[np.array([quiet_pt for quiet_pt in quiet_pts if not quiet_pt in lonelies]),kept_lonelies]

        logger.info('blocking offstates from %s'%myoffstatefile)
        quiet_times = quiet_pts * bw
        more_blocks += [quiet_times.T]


#exclude global bad times
badint_file = os.path.join(badtimesdir,'%s__badtimes.h5'%(aRec.id.replace('-','_')))
if os.path.isfile(badint_file):
    with h5py.File(badint_file, 'r') as fhand:
        badtimes = fhand['badtints'][()]
    more_blocks += [badtimes.T]

#blocking licking
if np.size(aRec.lickmat)>0:
    logger.info('blocking lick-times')
    more_blocks += [aRec.lickmat.T]


#blocking opto
smod = fhs.retrieve_module_by_path(os.path.join(pathdict['plotting']['stimdir'],exptype.lower()+'.py'))
stimdict = smod.get_stimdict(aRec)

if 'opto' in stimdict:
    logger.info('blocking opto')
    optopts = stimdict['opto']
    bdict = ba.find_bursts(optopts, maxdist_opto)
    opto_block = np.vstack([bdict['start'], bdict['stop']]).T + opto_buff[None,:]
    if np.size(opto_block)>0:
        singlets = np.array([optopt for optopt in optopts if not np.sum((optopt>=opto_block[:,0]) & (optopt<=opto_block[:,1]))])
    else: singlets = optopts
    singleblocks = singlets[:,None]+opto_buff[None,:]
    more_blocks += [np.r_[opto_block,singleblocks].T]





threshtints = P.get_dynnorm_lim_ints(aRec,exptype, aRec.h_info,more_blocks=more_blocks)
threshpints = (threshtints * aRec.sr).astype(int)



free_tints0 = dh.mergeblocks([aRec.artblockmat]+more_blocks, output='free', t_start=0, \
                                            t_stop=aRec.dur).T
free_tints = np.vstack([free_t for free_t in free_tints0 if np.diff(free_t)>P.freedur_min_s])
#print ('freetints shape',free_tints.shape)
freepints = (free_tints*aRec.sr).astype(int)#pint=point interval

maxpint = int(P.maxdur_threshtint*aRec.sr)
pints = det.checkcut_intervals(freepints,0,maxpint,maxdur_tot=-1,mode='extend_last').astype(int)

roipts = (np.array(P.roiint)*aRec.sr).astype(int)
roipts_rd = (np.array(P.roiint_rd)*aRec.sr).astype(int)

mindist_pts = int(P.mindist_s * aRec.sr)



#for chan,chanloc in zip(chans,locs):
mychans = shf.select_chans(chanflav, aRec, regionchecker=check_region)


P.set_spparams(P.detscales)
P.get_fndict(aRec.h_lfp)

outfile_gen = os.path.join(outpath_temp,'%s__chanXXX__transients.h5'%(recid))

logger.info('Writing files to %s'%outfile_gen)


#reload(tdet)
def run_detection(chan):
    sigextrfn = lambda mypint: P.fn_dict['sigextract_pts'](chan, mypint)
    rawspectfn = lambda mypint: P.fn_dict['spgrmfn'](sigextrfn(mypint))

    reflow, refhigh, lowmat, highmat = tdet.get_dynref(threshpints, rawspectfn, P.ndetfreqs, P.percmargs)
    spectextrfn = lambda mysignal: hf.apply_dynnorm(P.fn_dict['spgrmfn'](mysignal), reflow, refhigh)

    bandmeanlist = tdet.get_bandmeanlist_simple(pints,sigextrfn,spectextrfn,freepints)



    transdict0 = tdet.get_transdict_simple(freepints,bandmeanlist,P.dethr,sigextrfn,roipts,roipts_rd,\
                                         P.maxheightfac,mindist_pts,filtfn = P.filtfn_snips)

    wholetrace = P.fn_dict['sigextract_pts'](chan, [0, int(aRec.dur * aRec.sr)])
    transdict = tdet.ampclean_transdict(transdict0, wholetrace, aRec.sr, ampthr=5, filtbord=P.detband)

    if calc_nvsthr:
        nvec = tdet.get_NvsThr(P.nvsthrvec, bandmeanlist, mindist=mindist_pts)

    outfile = outfile_gen.replace('XXX',str(chan).zfill(3))
    with h5py.File(outfile, 'w') as fdest:
        fdest.attrs['recid'] = aRec.id
        fdest.attrs['chan'] = chan

        transgroup = fdest.create_group('transdict')
        for key in transdict.keys():
            mydtype = 'i' if key in ['borders', 'sigextrema_pts'] else 'f'
            transgroup.create_dataset(key, data=transdict[key], dtype=mydtype)

        if calc_nvsthr:
            fdest.create_dataset('nvec', data=nvec.astype('int'), dtype='i')

        fdest.create_dataset('reflowhigh', data=np.vstack([reflow, refhigh]), dtype='f')
    return 0

#run_detection(mychans[-1])

tstart = time.time()
with mp.Pool(mp.cpu_count()) as pool:
    outputs = pool.map(run_detection,mychans)
logger.info('Finished MP, dur: %s'%(str(timedelta(seconds=time.time()-tstart))))


collection_file = os.path.join(collection_dir,'%s__transients.h5'%recid)

outfile_list = [outfile_gen.replace('XXX',str(chan).zfill(3)) for chan in mychans]

logger.info('Merging %i channelfiles'%len(outfile_list))

with h5py.File(collection_file, 'w') as fdest:

    fdest.attrs['recid'] = aRec.id
    fdest.attrs['offdet_src'] = osrc
    fdest.attrs['usable_offstates'] = 'y' if regard_offstates else 'n'
    mgroup = fdest.create_group('method')
    mgroup.attrs['configfile'] = configfile
    mgroup.attrs['paramdict'] = defaultdictpath
    mgroup.attrs['script'] = str(__file__)
    mgroup.attrs['git_hash'] = fhs.get_githash()

    mgroup.attrs['sr'] = aRec.sr

    mgroup.create_dataset('nvsthr_params', data=np.array(P.nvsthr_params), dtype='f')
    mgroup.create_dataset('percmargs', data=np.array(P.percmargs), dtype='f')
    mgroup.create_dataset('freepints', data=freepints, dtype='i')

    changroup = fdest.create_group('channels')
    for chanfile in outfile_list:
        with h5py.File(chanfile,'r') as chand:
            channame = chand.attrs['chan']
            thisgroup = changroup.create_group(str(channame).zfill(3))
            chand.copy(chand['transdict'], thisgroup, 'transdict')
            if calc_nvsthr:
                chand.copy(chand['nvec'],thisgroup,'nvec')
            chand.copy(chand['reflowhigh'],thisgroup,'reflowhigh')

logger.info('Deleting channelfiles')

for myfile in outfile_list:
    os.remove(myfile)

logger.info('Done %s'%aRec.id)

