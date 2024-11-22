import os
import h5py
import yaml
import matplotlib.pyplot as plt
import sys
import numpy as np
from copy import deepcopy

lfp_rangeHz = [0.5, 100]
trans_fac = 1.5 #stds from mean applied on #(transients) to include/exclude transients
ditch_lonely_offperiods = False
maxdist_lonelies = 2.
ww_smooth_trans_s = 0.1
thr2_perc = 50 #to detect network transient bursts, percentage of eligible channels showing a transient
thr1_perc = 5 #rolldown of thr 2
stim_range_s = np.array([0.02,0.15])
mindur_superblock = 0.5
superburst_thr_s = 0.5
trans_borders = 0.2
mindur_active = 1.
block_stims = ['air','tone','bluen','10kHz', '2kHz', '5kHz','bluenoise','sound1','sound2']


#plotexamples:
plot_examples_on = True

N_events_block = 3
N_events_rand = 3
windur = 15



recid,pathpath,regstr,qsrc = sys.argv[1:]
'''
recid = 'PL079_20230620_probe0'
regstr = str(['ACA', 'ILA', 'MOs', 'ORB', 'PL'])
pathpath = 'PATHS/merge_quiet_paths.yml'
qsrc = 'self'
'''
recfilename = 'NWBEXPORTPATH/XXX_export/%s__XXX.h5'%(recid)
#anat_rois = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']
regs_of_interest = [bla.strip() for bla in regstr.strip('[]').replace("'",'').split(',')]
check_region = lambda locname: len([1 for reg in regs_of_interest if locname.count(reg)]) > 0

#parampaths = ['PATHS/offstate_paths.yml','PATHS/transient_paths.yml']
def get_pdict(mypath):
    with open(mypath, 'r') as myfile:
        return yaml.safe_load(myfile)

pathdict = get_pdict(pathpath)
pathdictO,pathdictT = [get_pdict(ppath) for ppath in [pathdict['configs']['offpath'],pathdict['configs']['transpath']]]
figsavedir = pathdict['figsavedir']
outdir = pathdict['outdir']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import transient_detection as tdet
from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import filtering as filt
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import statedet_helpers as shf
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python.utils import stimfuncs as sfn

logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])

logger.info('Doing %s'%recid)

aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=True)
recid = os.path.basename(recfilename).split('__')[0]#aRec.id.replace('-', '_')
exptype = (aRec.h_info['exptype'][()]).decode()
aRec.get_freetimes()


stimplotdir = pathdict['plotting']['stimdir']

smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
stimdict = smod.get_stimdict(aRec)

plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple)

usable_str = 'no'
offstatefile = os.path.join(pathdictO['outdir'], '%s__offstates.h5' % recid)
if qsrc.count('from-other'):
    refprobe = qsrc.replace('from-other:', '')
    thisprobe = recid.split('_')[-1]
    myoffstatefile = offstatefile.replace(thisprobe, refprobe)
    print('getting offstates from-other: %s'%myoffstatefile)
else:
    myoffstatefile = str(offstatefile)

if os.path.isfile(myoffstatefile):
    with h5py.File(myoffstatefile,'r') as fhand:
        #recid_ = fhand.attrs['recid']
        usable_str = fhand.attrs['usable']
        if usable_str == 'yes':

            rhand = fhand['results']
            bw,g_mean,thr = [rhand.attrs[akey] for akey in ['bw','g_mean','thr']]
            borderpts,burstpts,offpts,ratevec,offpts0 = [rhand[dskey][()] for dskey in ['borderpts','burstpts','offpts','ratevec','offpts_nodurlim']]

            bordertimes = borderpts*bw
            offtimes = offpts*bw
            offburstimes = burstpts*bw
if usable_str == 'no':
    bordertimes = np.empty((0,2))
    offtimes = np.empty((0,2))
    offburstimes = np.empty((0,2))



# load transients and get transient consensus on several channels(as for SPOOC-burstdet)
transientfile = os.path.join(pathdictT['save_dirs']['collection_dir'],'%s__transients.h5'%recid)


if qsrc.count('from-other'):
    assert 0, 'from-other currently disabled - fix the region checker and in the mean time copy the quiet directly'
    mytransientfile = transientfile.replace(thisprobe, refprobe)
    recfilename_ref = recfilename.replace(thisprobe, refprobe)
    aRec2 = aSetting.loadRecObj(recfilename_ref, get_eois_col=True)
    transchans = np.array([chan for chan in aRec2.eois if check_region(aRec2.ellocs_all[chan])])
    detvec = np.zeros(int(aRec2.dur*aRec2.sr),dtype=int)
    print('getting transients from-other: %s'%mytransientfile)

else:
    mytransientfile = str(transientfile)


    transchans = np.array([chan for chan in aRec.eois if check_region(aRec.ellocs_all[chan])])
    detvec = np.zeros(int(aRec.dur*aRec.sr),dtype=int)




# get pfc chans! --> this could be sensitive to broken pfc chans --> check it out!
tdetdict = tdet.read_collectionfile(mytransientfile,transchans,dskeys=['borders','sigextrema_pts'])

nvec = np.array([len(tdetdict[chan]['sigextrema_pts']) for chan in transchans])
mean_n,std_n = [np.mean(nvec),np.std(nvec)]


genfigpath = os.path.join(figsavedir,recid)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(genfigpath, nametag + '.png')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


f,ax = plt.subplots(figsize=(5,8))
f.subplots_adjust(left=0.17,bottom=0.07,top=0.95)
f.suptitle('%s src:%s'%(aRec.id,qsrc))
plots.prepare_locpanel(f,ax,aRec,ellocs=aRec.ellocs_all[transchans])
ax.plot(nvec,np.arange(len(nvec)),'k')#interesting!
for myval in [mean_n-std_n*trans_fac,mean_n+std_n*trans_fac]:
    ax.axvline(myval,color='firebrick',linestyle='dotted',zorder=-1)
ax.axvline(mean_n,color='firebrick',linestyle='dashed')
ax.set_xlabel('N transients')
figsaver(f, '%s__n_transients'%recid, False)

thrint_nchans = np.array([mean_n-std_n*trans_fac,mean_n+std_n*trans_fac])
chans_kept = np.array([chan for chan,myn in zip(transchans,nvec) if (myn<=thrint_nchans[1]) & (myn>=thrint_nchans[0])])

for chan in chans_kept:
    detvec[tdetdict[chan]['sigextrema_pts']] +=1
N = int(ww_smooth_trans_s*aRec.sr)//2*2
fac = 1/len(chans_kept)*100
stim_range = (stim_range_s*aRec.sr).astype(int)
stimpts = np.hstack([stimdict[key]*aRec.sr for key in block_stims if key in stimdict]).astype(int)

stimmask = stim_range[None,:]+stimpts[:,None]
detvec_masked = deepcopy(detvec)
for m0,m1 in stimmask:
    detvec_masked[m0:m1] = 0
movavg = np.convolve(detvec_masked, np.ones(N)/N, mode='valid') #gaussian_filter(detvec,sigma=int(0.05*aRec.sr))







#axes wanted: stims,overall quiet assignment, transients (with anatax as y), LFP midchans, offepisodes (going through spikes),actual spikes,

from scipy.stats import scoreatpercentile
tidbit = 0.001 #to avoid == 0 which is super annoying with zero-crossings
dvec2 = movavg*fac
dthr1 = thr1_perc/N+tidbit
dthr2 = thr2_perc/N+tidbit

starts0 = np.where(np.diff(np.sign(dvec2 - dthr1)) > 0)[-1]  # start of a potential trough interval interval
stops0 = np.where(np.diff(np.sign(dvec2 - dthr1)) < 0)[0]  # stop of a potential trough interval

stops = stops0[stops0 >= starts0[0]]  # excluding the first if it is an open interval being finished
starts = starts0[starts0 < stops0[-1]]  # excluding the last if it the interval remains to be finished
# print(len(starts), len(stops))
cross_ints = np.vstack([starts, stops]).T
peaks_temp = np.array([np.max(dvec2[p0:p1]) for p0, p1 in cross_ints])
cond = np.greater_equal(peaks_temp, dthr2)
transbursts = cross_ints[cond]+N//2
#transbursts,transpeaks = shf.get_halfwaves(dvec2,dthr1,-dthr2,mindur_pts=0,mode='trough')


# get superbursts of transbursts and quiet episodes
transtimes = transbursts/aRec.sr
trans_and_off = dh.mergeblocks([bordertimes.T,transtimes.T],output='block',t_start=0,t_stop=aRec.dur).T
superblocks0 = shf.get_superbursts(trans_and_off,superburst_thr_s)

durs = np.diff(superblocks0)
#f,ax = plt.subplots()
#ax.hist(np.log10(durs),color='k',histtype='step')#,np.arange(0.05,3,0.02)
#ax.axvline(np.log10(durthresh),color='firebrick')
superblocks = superblocks0[np.diff(superblocks0).flatten()>=mindur_superblock]
#and remove the ones that correspond to a single block

if ditch_lonely_offperiods and usable_str:
    lonelies = np.array([[q0, q1] for q0, q1 in superblocks if np.sum((offtimes[:, 0] < q1) & (offtimes[:, 1] > q0)) == 1 \
                         and np.sum((transtimes[:, 0] < q1) & (transtimes[:, 1] > q0)) < 2 ])
    if np.size(lonelies) >0:
        get_prev_neigh_diff = lambda l0: l0 - superblocks[superblocks[:, 1] < l0, 1][-1] if not l0 == np.min(
            superblocks[:, 0]) else np.nan
        get_next_neigh_diff = lambda l1: superblocks[superblocks[:, 0] > l1, 0][0] - l1 if not l1 == np.max(
            superblocks[:, 1]) else np.nan
        dist_lonelies = np.array([np.min([get_prev_neigh_diff(l0), get_next_neigh_diff(l1)]) for l0, l1 in lonelies])
        kept_lonelies = lonelies[dist_lonelies <= maxdist_lonelies]
        print(lonelies.shape,kept_lonelies.shape,superblocks.shape)
        superblocks = np.sort(np.r_[np.array([quiet_pt for quiet_pt in superblocks if not quiet_pt in lonelies]), kept_lonelies],0)
    else:
        print('No lonelies')

    #superblocks = np.array([sblock for sblock in superblocks if not sblock in lonelies])


# check wether the superepisode starts or stops with a trans, then add trans border
starterinds = np.array([ii for ii,sstart in enumerate(superblocks[:,0]) if sstart in transtimes[:,0]])
stopperinds =  np.array([ii for ii,sstop in enumerate(superblocks[:,1]) if sstop in transtimes[:,1]])
if len(starterinds)>0: superblocks[starterinds,0] -= trans_borders
if len(stopperinds)>0: superblocks[stopperinds,1] += trans_borders




# save the interesting things that are plotted, that is most importantly the superblocks, but also maybe the transienttrace!
# superblocks
# detvec*fac
# movavg*fac

outfile = os.path.join(outdir,'%s__transAndOff.h5'%(recid))

with h5py.File(outfile, 'w') as fdest:

    fdest.attrs['recid'] = aRec.id
    fdest.attrs['off_used'] = usable_str


    mgroup = fdest.create_group('method')

    mgroup.attrs['pathpath'] = pathpath
    mgroup.attrs['script'] = str(__file__)
    mgroup.attrs['git_hash'] = fhs.get_githash()
    mgroup.attrs['srcfile'] = recfilename
    mgroup.attrs['transientfile'] = transientfile
    mgroup.attrs['offstatefile'] = offstatefile

    rgroup = fdest.create_group('results')
    rgroup.attrs['sr'] = aRec.sr
    rgroup.attrs['thr1'] = dthr1
    rgroup.attrs['thr2'] = dthr2
    for dsname,datavec in zip(['quiet_merge','detvec1_trans','detvec2_trans'],[superblocks,detvec*fac,movavg*fac]):
        rgroup.create_dataset(dsname,data=datavec,dtype='f')



