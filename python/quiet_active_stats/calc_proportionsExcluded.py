import os
import h5py
import yaml
import sys
import numpy as np

from copy import deepcopy


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

statedir = 'ZENODOPATH/quiet_active_detection/merged_quiet'
dstdir = 'ZENODOPATH/quiet_active_detection/quiet_active_stats'


recid,pathpath,regstr,qsrc = sys.argv[1:]



outfile = os.path.join(dstdir, '%s__quietActiveStats.h5' %recid)

regs_of_interest = [bla.strip() for bla in regstr.strip('[]').replace("'",'').split(',')]
check_region = lambda locname: len([1 for reg in regs_of_interest if locname.count(reg)]) > 0

recfilename = 'NWBEXPORTPATH/XXX_export/%s__XXX.h5'%(recid)

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from state_detection.utils import transient_detection as tdet
from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import statedet_helpers as shf

def get_pdict(mypath):
    with open(mypath, 'r') as myfile:
        return yaml.safe_load(myfile)

pathdictO,pathdictT = [get_pdict(ppath) for ppath in [pathdict['configs']['offpath'],pathdict['configs']['transpath']]]

aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=True)
recid = os.path.basename(recfilename).split('__')[0]#aRec.id.replace('-', '_')
exptype = (aRec.h_info['exptype'][()]).decode()
aRec.get_freetimes()

###getting the overall quiet times
statefile = os.path.join(statedir, '%s__transAndOff.h5' % (recid.replace('-', '_')))
with h5py.File(statefile, 'r') as fhand:
    quietmat = fhand['results/quiet_merge'][()]
    offstatefile = fhand['method'].attrs['offstatefile']
    transientfile = fhand['method'].attrs['transientfile']

    usable_spikeoff_str = fhand.attrs['off_used']
    assert usable_spikeoff_str in ['no', 'yes'], 'inadmissible string spikeoff detection: %s' % usable_spikeoff_str



usable_str = 'no'
offstatefile = os.path.join(pathdictO['outdir'], '%s__offstates.h5' % recid)
if qsrc.count('from-other'):
    refprobe = qsrc.replace('from-other:', '')
    thisprobe = recid.split('_')[-1]
    myoffstatefile = offstatefile.replace(thisprobe, refprobe)
    print('getting offstates from-other: %s'%myoffstatefile)
else:
    myoffstatefile = str(offstatefile)



###getting the offstates
if os.path.isfile(myoffstatefile):
    with h5py.File(myoffstatefile,'r') as fhand:
        #recid_ = .attrs['recid']
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



#ok now get the transient times
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




tdetdict = tdet.read_collectionfile(mytransientfile,transchans,dskeys=['borders','sigextrema_pts'])

nvec = np.array([len(tdetdict[chan]['sigextrema_pts']) for chan in transchans])
mean_n,std_n = [np.mean(nvec),np.std(nvec)]

thrint_nchans = np.array([mean_n-std_n*trans_fac,mean_n+std_n*trans_fac])
chans_kept = np.array([chan for chan,myn in zip(transchans,nvec) if (myn<=thrint_nchans[1]) & (myn>=thrint_nchans[0])])

for chan in chans_kept:
    detvec[tdetdict[chan]['sigextrema_pts']] +=1
N = int(ww_smooth_trans_s*aRec.sr)//2*2

if len(chans_kept) >0:
    fac = 1/len(chans_kept)*100

    # set detvec to zero in the stim range
    stimplotdir = pathdict['plotting']['stimdir']
    smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
    stim_range = (stim_range_s*aRec.sr).astype(int)
    stimdict = smod.get_stimdict(aRec)
    stimpts = np.hstack([stimdict[key]*aRec.sr for key in block_stims if key in stimdict]).astype(int)

    stimmask = stim_range[None,:]+stimpts[:,None]
    detvec_masked = deepcopy(detvec)
    for m0,m1 in stimmask:
        detvec_masked[m0:m1] = 0
    movavg = np.convolve(detvec_masked, np.ones(N)/N, mode='valid') #gaussian_filter(detvec,sigma=int(0.05*aRec.sr))


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


    transtimes = transbursts/aRec.sr
else:
    transtimes = np.empty((0,2))
'''
### plot bordertimes, transtimes check whethter they overlap and write their total duration
print('transdur [s]',np.sum(np.diff(transtimes)))

print('offdur [s]',np.sum(np.diff(bordertimes)))


f,ax = plt.subplots()
ax.plot(aRec.tvec,detvec,'k')
ax.hlines(np.ones_like(quietmat[:,0])*80,quietmat[:,0],quietmat[:,1],color='skyblue',alpha=1,lw=10)
ax.hlines(np.ones_like(bordertimes[:,0])*90,bordertimes[:,0],bordertimes[:,1],color='firebrick',alpha=1,lw=10)
ax.hlines(np.ones_like(transtimes[:,0])*95,transtimes[:,0],transtimes[:,1],color='orange',alpha=1,lw=10)
'''
#transblocks = tdet.get_transient_networkbursts(tdetdict,aRec.sr,aRec.dur,durdiff_max_s = 0.5,bmin=2, chanburst_frac = 0.5)


# trace back the ones that need to be deleted
trans_and_off = dh.mergeblocks([bordertimes.T,transtimes.T],output='block',t_start=0,t_stop=aRec.dur).T
superblocks0 = shf.get_superbursts(trans_and_off,superburst_thr_s)
durs = np.diff(superblocks0)
superblocks = superblocks0[np.diff(superblocks0).flatten()>=mindur_superblock]

dur_quiet = np.sum(np.diff(quietmat))
dur_blocked = np.sum(np.diff(superblocks))
dur_off = np.sum(np.diff(bordertimes))
dur_trans = np.sum(np.diff(transtimes))
offlist = [myoff for myoff in bordertimes if np.sum([dh.check_olap(myoff,superblock) for superblock in superblocks ])>0 ]
translist = [myoff for myoff in transtimes if np.sum([dh.check_olap(myoff,superblock) for superblock in superblocks ])>0]
if len(offlist) == 0:
    offs_retained = bordertimes
else: offs_retained = np.vstack(offlist)

if len(translist) == 0:
    trans_retained = transtimes
else:
    trans_retained = np.vstack(translist)

dur_offretained = np.sum(np.diff(offs_retained))
dur_transretained = np.sum(np.diff(trans_retained))

print ('REC: %s'%aRec.id)
print('quietdur [s]',dur_quiet)

print('blockdur [s]',dur_blocked)#roughly the same!
print('offdur [s]',dur_off)#roughly the same!
print('transdur [s]',dur_trans)#roughly the same!
print('retained offdur [s]',dur_offretained)#roughly the same!
print('retained transdur [s]',dur_transretained)#roughly the same

'''
f,ax = plt.subplots()
ax.plot(aRec.tvec,detvec,'k')
ax.hlines(np.ones_like(superblocks[:,0])*80,superblocks[:,0],superblocks[:,1],color='skyblue',alpha=1,lw=10)
ax.hlines(np.ones_like(bordertimes[:,0])*90,bordertimes[:,0],bordertimes[:,1],color='firebrick',alpha=0.5,lw=10)
ax.hlines(np.ones_like(offs_retained[:,0])*90,offs_retained[:,0],offs_retained[:,1],color='firebrick',alpha=1,lw=10)
ax.hlines(np.ones_like(transtimes[:,0])*95,transtimes[:,0],transtimes[:,1],color='orange',alpha=0.5,lw=10)
ax.hlines(np.ones_like(trans_retained[:,0])*95,trans_retained[:,0],trans_retained[:,1],color='orange',alpha=1,lw=10)
'''
# proportion of time in superblocks (subtracting the artifacts)
artdur = np.sum(np.diff(aRec.artblockmat.T))


with h5py.File(outfile,'w') as hand:
    hand.attrs['recid'] = aRec.id
    hand.attrs['off_used'] = usable_str
    mgroup = hand.create_group('method')
    mgroup.attrs['transientfile'] = transientfile
    mgroup.attrs['offstatefile'] = offstatefile
    for dsname,mydata in zip( ['recdur','artdur','quietdur','transdur','offdur'],\
                            [aRec.dur,artdur,dur_blocked,dur_transretained,dur_offretained]):
        hand.create_dataset(dsname,data=np.array([mydata]),dtype='f')
    for dsname,mydata in zip( ['offtimes','transtimes','quiettimes'],\
                        [offs_retained,trans_retained,superblocks]):
        hand.create_dataset(dsname,data=mydata,dtype='f')

#with h5py.File(outfile,'r') as hand:
#    print(hand.keys())
'''
#readouts
dur_analyzed = aRec.dur-artdur
frac_quiet = dur_blocked/dur_analyzed
frac_off = dur_offretained/dur_analyzed
frac_trans = dur_transretained/dur_analyzed
off_trans_ratio = frac_off/frac_trans
'''