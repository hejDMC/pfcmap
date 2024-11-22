import numpy as np
from pfcmap.python.utils import data_handling as dh
from scipy.stats import gmean
from scipy.stats import scoreatpercentile as sap
import matplotlib as mpl
from matplotlib import pyplot as plt

def get_halfwaves(vals,rolldownthr,minpeakheight,mindur_pts=0,mode='trough'):
    starts0 = np.where(np.diff(np.sign(vals-rolldownthr))<0)[0]#start of a potential trough interval interval
    stops0 = np.where(np.diff(np.sign(vals-rolldownthr))>0)[0]#stop of a potential trough interval

    stops = stops0[stops0>starts0[0]] #excluding the first if it is an open interval being finished
    starts = starts0[starts0<stops0[-1]]#excluding the last if it the interval remains to be finished
    #print(len(starts), len(stops))
    cross_ints = np.vstack([starts,stops]).T
    myfn,compfn = [np.min,np.less_equal] if mode == 'trough' else [np.max,np.greater_equal]
    peaks_temp = np.array([myfn(vals[p0:p1]) for p0,p1 in cross_ints])
    cond = compfn(peaks_temp,minpeakheight)
    cond2 = np.diff(cross_ints)[:,0] > mindur_pts
    return cross_ints[cond & cond2],peaks_temp[cond & cond2]

def make_spikehist(spikes,bw,tint,**kwargs):
    spikebins = np.arange(tint[0],tint[1],bw)
    spikehist,_ = np.histogram(spikes,spikebins)
    spikebins_plot = spikebins[:-1] +bw*0.5
    if 'n_units' in kwargs:
        ratevec = spikehist/(bw*kwargs['n_units'])
    else:
        ratevec = spikehist/bw
    return ratevec,spikebins_plot


def get_superbursts(startstoppts,maxint):
    """returns superburst postions in pts"""

    if maxint is None or np.size(startstoppts) <= 2: return startstoppts
    else:
        starts, stops = startstoppts.T
        intervals = starts[1:] - stops[:-1]
        if intervals.min() >=  maxint:
            return startstoppts
        else:
            startinds = np.r_[0, np.where(intervals >= maxint)[0] + 1]
            stopinds = np.r_[np.where(intervals >= maxint)[0], len(stops) - 1]
            newstarts = starts[startinds]
            newstops = stops[stopinds]
            return np.vstack([newstarts, newstops]).T  # thats the larger sleep episodes

def get_events_per_super(startstoppts,superbursts):
    return np.array([np.sum((startstoppts[:, 0] <= superburst[1]) & (startstoppts[:, 1] >= superburst[0])) for superburst in
              superbursts])

def make_bordermat(superbursts,n_event_vec,tintpts,nmin_super=3,border_super=np.array([-10,10]),border_nonsuper=np.array([-5,5]),minpts_free=0):
    """returns startstoppts"""

    bordermat0 = np.zeros_like(superbursts)
    cond = n_event_vec>=nmin_super
    bordermat0[cond] = superbursts[cond]+border_super[None,:]
    bordermat0[np.invert(cond)] = superbursts[np.invert(cond)]+border_nonsuper[None,:]

    bordermat = dh.mergeblocks([bordermat0.T], output='block', t_start=tintpts[0], t_stop=tintpts[1]).T

    if minpts_free>0:
        freepts = dh.mergeblocks([bordermat.T], output='free', t_start=tintpts[0], t_stop=tintpts[1]).T
        frees_long = freepts[np.diff(freepts)[:, 0] > minpts_free]
        blstarts = frees_long[:-1, 1]
        blstops = frees_long[1:, 0]
        if frees_long[0, 0] > tintpts[0]: blstarts, blstops = np.r_[tintpts[0], blstarts], np.r_[frees_long[0, 0], blstops]
        if frees_long[-1, 1] < tintpts[1]:  blstarts, blstops = np.r_[blstarts, frees_long[-1, 1]], np.r_[
            blstops, tintpts[1]]
        bordermat = np.vstack([blstarts, blstops]).T

    return (bordermat).astype(int)


get_zeroline = lambda ratevals: gmean(ratevals[ratevals>0])

def get_detdict(rvec,recdur,ratethr_frac=0.2,mindur_trough_pts=5,maxint_between_off=0,bw=0.01, \
                nmin_super=3, marg_super=0, marg_nonsuper=0, minpts_free=1, maxpts_off=100,print_on=False):
    tintpts = (np.array([0, recdur]) / bw).astype(int)

    rate_gmean = get_zeroline(rvec)
    rate_thr = ratethr_frac * rate_gmean

    troughpints0, peakamps = get_halfwaves(rvec, rate_gmean, rate_thr, mindur_pts=mindur_trough_pts, mode='trough')

    troughpints = troughpints0[(np.diff(troughpints0) <= maxpts_off).flatten()]


    if print_on:
        troughtimes = troughpints*bw

        print('off/min', troughtimes.shape[0] / recdur * 60.)
        print('% spent in off', np.sum(np.diff(troughtimes)) / recdur * 100)


    # superbursts
    superbursts = get_superbursts(troughpints, maxint_between_off)
    events_per_super = get_events_per_super(troughpints, superbursts)
    if print_on:
        print('% spent in off episodes (merged):', np.sum(np.diff(superbursts)) * bw / recdur * 100)
    # print('% spent in off episodes (superbursts):', np.sum(np.diff(spikebins_plot[superbursts]))/aRec.dur*100)#the exact same

    # make safetyborders around superbursts
    bordermat = make_bordermat(superbursts, events_per_super, tintpts, nmin_super=nmin_super, \
                                   border_super=marg_super, border_nonsuper=marg_nonsuper, minpts_free=minpts_free)
    if print_on:
        print('% of data excluded:', np.sum(np.diff(bordermat)) / recdur * bw * 100)
    return {'g_mean': rate_gmean, 'thr': rate_thr, 'trtimes': troughpints, 'sbursts': superbursts,
                    'nps': events_per_super, 'borderpts': bordermat,'trtimes0':troughpints0}


def get_lfpartfree_quiet_and_awake_times(aRec,bordertimes):
    artmat = dh.mergeblocks([aRec.freetimes.T], output='free', t_start=0, t_stop=aRec.dur).T
    #bordertimes = borderpts * bw

    awake_times = dh.mergeblocks([artmat.T, bordertimes.T], output='free', t_start=0, t_stop=aRec.dur).T
    quiet_times = dh.mergeblocks([artmat.T, awake_times.T], output='free', t_start=0, t_stop=aRec.dur).T
    return quiet_times,awake_times



#datasnips_overall = [tracedict[chan][p0:p1] for p0,p1 in (aRec.freetimes*aRec.sr).astype(int)]




def get_spectrum_from_episodes_fft(episode_dict,sr,ww=256*2*2*2,step=10,scorepercs=[20,50,80],return_dur_ana=True):
    hann = np.hanning(ww)
    fftfn = lambda dsnip: np.abs(np.fft.rfft (dsnip*hann, ww)) ** 2. / ww


    fftdict = {}
    durdict = {}
    for tag,datasnips in episode_dict.items():
        mysnips = [snip for snip in datasnips if len(snip)>ww]
        if return_dur_ana: durdict[tag] = np.sum([len(snip) for snip in mysnips])/sr
        if len(mysnips) > 0:
            fft_spgrm = np.vstack([np.array([fftfn(snip[pstart:pstart+ww]) for pstart in np.arange(0,len(snip)-ww,step)]) for snip in mysnips])
            fftdict[tag] = [sap(fft_spgrm,perc,axis=0) for perc in scorepercs]#np.median(fft_spgrm,axis=0)
        else:
            fftdict[tag] = [np.zeros((ww//2+1))-1 for perc in scorepercs]
    if return_dur_ana:
        return fftdict,np.fft.rfftfreq(ww,1/sr),durdict
    else: return fftdict,np.fft.rfftfreq(ww,1/sr)


def get_spectrum_from_episodes_superlets(episode_dict,sr,freqvec,omax=5,c1=3):
    from SPOOCs.utils import superlets

    scales = superlets.scale_from_period(1 / freqvec)
    superletfn = lambda signal: abs(superlets.FASLT(signal,sr, scales, \
                                                    omax, order_min=1, c_1=c1)) ** 2


    spectdict = {}
    for tag, datasnips in episode_dict.items():
        spgrm = np.vstack([superletfn(snip).T for snip in datasnips])
        spectdict[tag] = [sap(spgrm, perc, axis=0) for perc in [20, 50, 80]]  # np.median(fft_spgrm,axis=0)
    return spectdict

def get_ampvardict(episode_dict):
    ampvardict = {}
    for tag, datasnips in episode_dict.items():
        ampvardict[tag] = np.vstack(
            [np.array([fn(np.abs(datasnip)) for datasnip in datasnips]) for fn in [np.mean, np.std]])
    return ampvardict


def get_midchans(aRec,regionchecker=None):

    my_inds = dh.get_middle_locations(aRec.el_locs)
    if type(regionchecker) != type(None):   xx = np.array([idx for idx in my_inds if regionchecker(aRec.el_locs[idx])])
    else: xx = my_inds
    return aRec.eois[xx],aRec.el_locs[xx]

def get_colordict(chans,cmap=plt.cm.nipy_spectral):
    norm = mpl.colors.Normalize(vmin=0.5, vmax=len(chans) - 0.5)
    return {chan: cmap(norm(cc)) for cc, chan in enumerate(chans)}
def get_midchans_and_coloring(aRec,regionchecker=None,cmap=mpl.cm.nipy_spectral,verbose_output=True):
    chans,locs = get_midchans(aRec,regionchecker=regionchecker)
    cdict = get_colordict(chans,cmap=cmap)
    locdict = {chan:loc for chan,loc in zip(chans,locs)}
    if verbose_output:
        clist = [cdict[chan] for chan in chans]
        return chans,locdict,cdict,clist
    else:
        return locdict,cdict

def get_upstates_after_offstates(ratevec,offpts):
    starts0 = np.where(np.diff(np.sign(ratevec)) > 0)[0]
    stops0 = np.where(np.diff(np.sign(ratevec)) < 0)[0]
    ustops = stops0[stops0 > starts0[0]]  # excluding the first if it is an open interval being finished
    ustarts = starts0[starts0 < stops0[-1]]  # excluding the last if it the interval remains to be finished
    uppts_temp = np.vstack([ustarts, ustops]).T
    uppts = np.vstack([uppts_temp[uppts_temp[:, 0] == offpt[1]] for offpt in offpts])

    return uppts


def select_chans(chanflav, aRec, regionchecker=None):
    if chanflav == 'midchans':
        mychans = get_midchans(aRec, regionchecker=regionchecker)


    elif chanflav == 'regionchans':

        mychans = [chan for chan in aRec.eois if regionchecker(aRec.ellocs_all[chan])]

    elif chanflav == 'all':
        mychans = aRec.eois[:]

    else:
        assert 0, 'undefined chanflav %s' % chanflav

    return mychans


