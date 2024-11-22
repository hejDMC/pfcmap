import h5py
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os

pathpath,recfilename,regstr = sys.argv[1:]
regs_of_interest = [bla.strip() for bla in regstr.strip('[]').replace("'",'').split(',')]

recid = os.path.basename(recfilename).split('__')[0]


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import statedet_plot as plots
from pfcmap.python.utils import statedet_helpers as shf
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import filtering as filt
from pfcmap.python.utils import stimfuncs as sfn

logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])
fnfile = pathdict['configs']['functions']
defaultdictpath = pathdict['configs']['defaultdict']
stimplotdir = pathdict['plotting']['stimdir']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

cdict_states = {'awake': 'k', 'quiet': 'purple'}
episode_tags = ['awake', 'quiet']

#settings for lfp analysis only
ww_fft = 2048
step_fft = ww_fft//4
freqvec_spl = np.linspace(1,20,40)#only if superlet plotting is on

lfp_rangeHz = [0.5, 100]
lfp_ww_rms_sec = 0.5

#settings for examples
nex_small = 3
nex_big = 5
ex_windur = 10
ex_outlap = 2

#settings for example plots
plot_lfp = True
plot_emg = True
plot_stats = True
plot_examples = True
plot_superlets = False

lfpfilename = recfilename.replace('XXX','lfp')
if not os.path.isfile(lfpfilename):
    plot_lfp = False
    logger.warning('LFPfile %s does not exist - no lfp plotting'%lfpfilename)




aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=plot_lfp)
exptype = (aRec.h_info['exptype'][()]).decode()
logger.info('######starting %s'%aRec.id)

if not os.path.isfile(lfpfilename):
    aRec.set_recstoptime(aRec.h_units['units/spike_times'][()].max())  # in case you dont have an lfp ready yet
    logger.info('setting recstoptime to last spike')

aRec.get_freetimes()
artblockmat = aRec.artblockmat

smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
stimdict = smod.get_stimdict(aRec)



plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple)



offstatefile = os.path.join(pathdict['outdir'],'%s__offstates.h5'%recid)

#extract the offstates
with h5py.File(offstatefile,'r') as fhand:
    recid_ = fhand.attrs['recid']
    usable_str = fhand.attrs['usable']

    rhand = fhand['results']
    bw,g_mean,thr = [rhand.attrs[akey] for akey in ['bw','g_mean','thr']]
    borderpts,burstpts,offpts,ratevec,offpts0 = [rhand[dskey][()] for dskey in ['borderpts','burstpts','offpts','ratevec','offpts_nodurlim']]

bordertimes = borderpts*bw
offtimes = offpts*bw


genfigpath = os.path.join(pathdict['figsavedir'],recid)
logger.info('saving figures in %s'%genfigpath)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(genfigpath, nametag + '.png')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


if plot_stats:
    logger.info('plotting stats')

    f,ax = plots.plot_hist(np.diff(bordertimes),bins=50,col=cdict_states['quiet'],htype='count',xlab='quiet duration [s]')
    ax.set_yscale('log')
    ax.set_xlim([0,ax.get_xlim()[1]])
    f.suptitle('%s'%(recid_.split('-')[0]))
    figsaver(f,'stats/durhist_quiet')

    starts,stops = offtimes.T
    intervals = starts[1:]-stops[:-1]
    f,ax = plots.plot_hist(np.log10(intervals),bins=50,col=cdict_states['quiet'],htype='count',xlab='log10(inter-off-interval [s])')
    f.suptitle('%s'%(recid_.split('-')[0]))
    figsaver(f,'stats/inter_off_intervals')


    # duration hist of off episodes
    dur_off = np.diff(offtimes)[:, 0]
    dur_off0 = np.diff(offpts0 * bw)[:, 0]  # the full length ones
    if len(dur_off)>500: nbins = 50
    else:nbins = int(len(dur_off) / 10)
    if nbins == 0: nbins = 3

    f, ax, mybins = plots.plot_hist(dur_off, bins=nbins, col=cdict_states['quiet'], htype='count', xlab='off duration [s]',
                                    return_bins=True,hlinecol='darkgrey')
    f.suptitle('%s n=%i' % (recid_.split('-')[0], len(dur_off)))
    figsaver(f, 'stats/off_dur',closeit=False)

    _ = ax.hist(dur_off0, bins=np.arange(dur_off0.min(), dur_off0.max(), np.diff(mybins)[0]), color=cdict_states['quiet'], \
                histtype='step', linewidth=1, alpha=0.4)
    figsaver(f, 'stats/off_dur_includingShorts')


    # overall plot through recording

    try:
        blockdict = smod.get_blockdict(aRec,stimdict)
        plot_blocks = sfn.make_blockplotfn(blockdict, smod.styledict_blocks)
    except:
        logger.warning('no blockdict available, plotting stims')

        plot_blocks = plot_stims

    f, ax, mystr = plots.plot_timecourse_recording(bordertimes, offpts.shape[0], plot_blocks, aRec.dur,
                                                   artmat=artblockmat.T)
    f.suptitle('%s %s' % (aRec.id, mystr), fontsize=10)
    figsaver(f, 'stats/timecourse_recording')


if plot_emg or plot_lfp:

    quiet_times,awake_times = shf.get_lfpartfree_quiet_and_awake_times(aRec, bordertimes)
    extract_datasnips = lambda data, pt_intervals: [data[p0:p1] for p0, p1 in pt_intervals]


if plot_emg:
    logger.info('plotting emg')
    emg_snipdict = {etag: extract_datasnips(aRec.emg_data,(mytimes*aRec.emg_sr).astype(int)) for etag,mytimes in zip(episode_tags,[awake_times,quiet_times])}

    emg_dict = {tag: np.array([np.mean(np.abs(datasnip)) for datasnip in datasnips]) for tag,datasnips in emg_snipdict.items()}

    f,ax = plots.plotcompare_hist(emg_dict,\
                                  episode_tags,cdict_states,bins=50,xlab = 'rms(emg)',htype='probability')
    f.suptitle('%s'%(aRec.id.split('-')[0]))
    figsaver(f,'stats/emg_power')


if plot_lfp:
    logger.info('plotting LFP')

    lfp_ww_rms = int(lfp_ww_rms_sec * aRec.sr)

    filtfn_lfp = lambda datatrace: filt.butter_bandpass_filter(datatrace, lfp_rangeHz[0], lfp_rangeHz[1], aRec.sr)
    get_lfp_pts = lambda mychan, tint: dh.grab_filter_dataslice(aRec.h_lfp,tint,mychan,filtfn=filtfn_lfp)

    check_region = lambda locname: len([1 for reg in regs_of_interest if locname.count(reg)]) > 0
    chans,locdict,cdict,clist = shf.get_midchans_and_coloring(aRec,regionchecker=check_region)
    calc_rms = lambda signal,ww: np.sqrt(np.convolve(signal**2, np.ones(ww)/ww, 'valid'))

    tracedict = {}
    for chan in chans:
        tracedict[chan] = get_lfp_pts(chan, [0, int(aRec.dur * aRec.sr)])

    rms_lfp = {}
    for chan in chans:
        rms_lfp[chan] = calc_rms(tracedict[chan], lfp_ww_rms)

    episode_dict = {chan:{etag: extract_datasnips(tracedict[chan],(mytimes*aRec.sr).astype(int)) for etag,mytimes in zip(episode_tags,[awake_times,quiet_times])}\
                        for chan in chans}


    fftdict = {}
    durdict_fft = {}
    for chan in chans:
        fftdict[chan],freqvec,durdict_fft[chan] = shf.get_spectrum_from_episodes_fft(episode_dict[chan],aRec.sr,ww=ww_fft,step=step_fft,scorepercs=[20,50,80])

    #for chan in chans:
    for chan in chans:
        f,ax = plots.plot_spectrum(freqvec,episode_tags,fftdict[chan],cdict_states,xlim=[1,20],durdict=durdict_fft[chan])
        f.suptitle('%s %s'%(aRec.id.split('-')[0],locdict[chan]))
        figsaver(f,'lfp/fftpows/fft_%s'%locdict[chan].replace('/','_'))

    if plot_superlets:
        logger.info('plotting LFP: superlets')

        spdict = {}

        for chan in chans:
            spdict[chan] = shf.get_spectrum_from_episodes_superlets(episode_dict[chan],aRec.sr,freqvec_spl,omax=5,c1=3)

        durdict_all = {chan: {tag: np.sum([len(snip) for snip in episode_dict[chan][tag]])/aRec.sr for tag in episode_tags} for chan in
                    chans}

        for chan in chans:
            f,ax = plots.plot_spectrum(freqvec_spl,episode_tags,spdict[chan],cdict_states,xlim=[1,20],durdict=durdict_all[chan])
            f.suptitle('%s %s'%(aRec.id.split('-')[0],locdict[chan]))
            figsaver(f,'lfp/superletpows/superl_%s'%locdict[chan].replace('/','_'))


    for chan in chans:
        ampvardict = shf.get_ampvardict(episode_dict[chan])
        fano_dict = {etag:(ampvardict[etag][1]**2)/ampvardict[etag][0] for etag in episode_tags}
        f, ax = plots.plotcompare_hist(fano_dict, \
                                       episode_tags, cdict_states, bins=50, xlab='fano rms [muV]', htype='probability')
        f.suptitle('               %s %s' % (aRec.id.split('-')[0], locdict[chan]))
        figsaver(f,'lfp/ampvar/fanofactor/fano_%s'%locdict[chan].replace('/','_'))


        #histogram for this
        for idx,xlab in zip([0,1],['mean','std']):
            f,ax = plots.plotcompare_hist({etag: ampvardict[etag][idx] for etag in episode_tags},\
                                          episode_tags,cdict_states,bins=50,xlab = '%s rms [muV]'%xlab,htype='probability')
            f.suptitle('               %s %s'%(aRec.id.split('-')[0],locdict[chan]))
            figsaver(f,'lfp/ampvar/%s_hist/ampvar_%s_%s'%(xlab,xlab,locdict[chan].replace('/','_')))



if plot_examples:
    logger.info('plotting examples with spikes')

    spikebins = np.arange(len(ratevec))*bw
    allspikes = aRec.h_units['units/spike_times'][()]

    off_per_burst =  shf.get_events_per_super(offpts, burstpts)
    toidict = plots.randchoice_intervals_of_interest(burstpts*bw,off_per_burst,aRec.dur,n_ex_big=nex_big,n_ex_small=nex_small,n_bigsmall_thr=3,\
                                         windur=ex_windur,outlap_dur=ex_outlap)

    fmod = fhs.retrieve_module_by_path(fnfile)

    F = fmod.DataFuncs(pathdict['code']['recretrieval'])

    P = fmod.Paramgetter(defaultdictpath)  #
    rvec_smoothed = F.smooth_ratevec(ratevec, P, style='savgol')

    detdict = {'trtimes':offpts,'sbursts':burstpts,'borderpts':borderpts,'g_mean':g_mean,'thr':thr,'ratevec':rvec_smoothed}

    if not plot_lfp: tracedict,cdict,locdict = {},{},{}

    f,axarr = plots.multiplot_detection(aRec,tracedict,spikebins,ratevec,detdict,cdict,locdict,plot_stims,wildcard=True,bw=bw)
    plots.prepare_locpanel(f,axarr[2],aRec)
    adaptaxlist,ybuffs = [[axarr[1]],[-20]] if len(tracedict)>0 else [[],[]]
    tcounter = 0
    for seltype,tsels in toidict.items():
        for tsel in tsels:

            plots.plot_rasterpanel(axarr[2], aRec, allspikes, tsel, yscale_axes=adaptaxlist, yscale_buffers=ybuffs)
            f.suptitle('%s %s ex:%i - usable:%s'%(aRec.id,seltype,tcounter+1,usable_str))
            figsaver(f, 'spike_examples/spikeexample_%i'%(tcounter+1),False)
            tcounter += 1
    plt.close(f)
