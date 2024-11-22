# python 02_run_lickextraction_fromtrace.py PL077_20230407-probe0 -ipimin 0.015 -npmin 1 -pthr 3.
from scipy.signal import find_peaks
from argparse import ArgumentParser
import sys
import h5py
import numpy as np
import yaml
import os
from scipy.interpolate import interp1d
from copy import deepcopy
from matplotlib.widgets import Button

from matplotlib.widgets import RectangleSelector
import time
import matplotlib.pyplot as plt

###################################
###!!!! dont use the caller - use it as a direct command line tool
###################################



pathpath = 'PATHS/artifact_paths.yml'


parser = ArgumentParser()
parser.add_argument('recid0')
parser.add_argument('-ipimin',default=0.07)#inter peak interval in acorr of licksnippet
parser.add_argument('-ipimax',default=0.2)
parser.add_argument('-pthr',default=1.)#inter peak interval in acorr of licksnippet
parser.add_argument('-crthr',default=0.5)#inter peak interval in acorr of licksnippet
parser.add_argument('-npmin',default=2)#inter peak interval in acorr of licksnippet
parser.add_argument('-int_start',default=9999.)
parser.add_argument('-int_stop',default=9999.)

args = parser.parse_args()

recid0 = args.recid0
ipi_min_sec = float(args.ipimin)
ipi_max_sec = float(args.ipimax)
peakthr = float(args.pthr)
cross_thr = float(args.crthr)
npeaks_min = float(args.npmin)
int_start = float(args.int_start)
int_stop = float(args.int_stop)

print('IPI bounds [%1.4f,%1.4f] s'%(ipi_min_sec,ipi_max_sec))
print('thresholds [%1.2f,%1.2f] s'%(peakthr,cross_thr))

'''
recid0 = 'PL077_20230407-probe0'
ipi_min_sec = 0.015
ipi_max_sec = 0.2
peakthr = 1#2 #z
cross_thr = 0.5#1 #z
'''
recid = recid0.replace('-','_')
recfilename = 'NWBEXPORTPATH/XXX_export/%s__XXX.h5'%(recid)

maxdist_sec = 0.11 #between crossings to merge lickarts
mindur = 0.05#minimal duration of lick-episode

buff_pts = 30#for the cutouts when calculating at acorr


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import filtering as filt
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python.utils import corrfuncs as cfns
from pfcmap.python.utils import statedet_helpers as shf
from pfcmap.python.utils import stimfuncs as sfn

logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])

plot_on = True
load_on = True #only off for interactive use
save_on = True
interactive = True

artblockfillval = -10

#npeaks_min = 2 #number
#for plotting params see below

aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=True)
recid = aRec.id.replace('-', '_')
#recid0 = (os.path.basename(recfilename).split('__')[0]).replace('_probe','-probe')
#setattr(aRec,'id',recid0) #because we have this annoying _1_ _2_ in some files as ids
exptype = (aRec.h_info['exptype'][()]).decode()
outfile = os.path.join(pathdict['outpaths']['data'],'%s__artifactsICA.h5'%(recid0))
logger.info('## Starting lickdetection %s'%recid0)

if load_on:
    with h5py.File(outfile,'r') as hand:
        artvec = hand['ica/artvec'][()]
        artblocks = hand['methods/artblocks'][()]

npts_tot = len(artvec)
zartvec = np.abs((artvec-np.nanmean(artvec))/np.nanstd(artvec))
peaks,_ = find_peaks(zartvec,peakthr)


# f,ax = plt.subplots()
# ax.plot(aRec.tvec,zartvec,'k')
# ax.plot(aRec.tvec[peaks],zartvec[peaks],'o',mfc='none',mec='r')

logger.info('interpolating between peaks')

z_interp = deepcopy(zartvec)
for b0,b1 in artblocks: z_interp[b0:b1] = artblockfillval
allpeaks,_ = find_peaks(z_interp)
upper = interp1d(np.r_[0,allpeaks,npts_tot-1],np.r_[z_interp[0],z_interp[allpeaks],z_interp[-1]], kind = 'linear',\
                 bounds_error = False, fill_value=0.0)
envelope = upper(np.arange(npts_tot))

#getting the crossings
starts0 = np.where(np.diff(np.sign(envelope-cross_thr))>0)[0]
stops0 = np.where(np.diff(np.sign(envelope-cross_thr))<0)[0]
stops = stops0[stops0>starts0[0]] #excluding the first if it is an open interval being finished
starts = starts0[starts0<stops0[-1]]#
crossings = np.vstack([starts,stops]).T
#mindur = int(0.02*aRec.sr)
#h_crossings = np.array([[c0,c1] for c0,c1 in crossings if np.max(zartvec[c0:c1])>thr and (c1-c0)>=mindur])

#filtering the crossigs for containg at least npeaks_min:
h_crossingx = np.array([[c0,c1] for c0,c1 in crossings if np.sum((peaks>=c0) & (peaks<=c1))>=npeaks_min])


maxdist = int((maxdist_sec)*aRec.sr)
dists = h_crossingx[1:,0]-h_crossingx[:-1,1]
mergepts = np.where(dists<=maxdist)[0]
n_mergpts = np.where(np.diff(mergepts)>1)[0]
mergelist = np.split(mergepts,n_mergpts+1)
merged_crossings0 = np.vstack([[h_crossingx[el[0],0],h_crossingx[el[-1]+1,1]] for el in mergelist])
singles = np.where(dists>maxdist)[0]
c_singles = h_crossingx[singles]
merged_crossings = np.sort(np.vstack([c_singles,merged_crossings0]),axis=0)

merged_crossings = (dh.mergeblocks([merged_crossings.T],output='block',t_start=0).T).astype(int)
merged_crossings = merged_crossings[np.diff(merged_crossings)[:,0]>mindur*aRec.sr]



ipi_min = int(ipi_min_sec*aRec.sr)
ipi_max = int(ipi_max_sec*aRec.sr)

merged_crossingsTemp = (merged_crossings+np.array([-buff_pts,buff_pts])[None,:]).clip(0,int(aRec.sr*aRec.dur))
boolvec = np.zeros(len(merged_crossings),dtype=bool)
for cc,cpts in enumerate(merged_crossingsTemp):
    event = artvec[cpts[0]:cpts[1]]

    corr = cfns.myCorr(event,event,len(event)/2)
    peaks,props = find_peaks(corr,height=0.1)#,distance=0.2*aRec.sr
    ipis = np.diff(peaks)
    cond1 = len(peaks)>=npeaks_min
    n_sizecrit = np.sum((ipis>=ipi_min) & (ipis<=ipi_max))
    cond2 = (n_sizecrit >=npeaks_min-1) & (n_sizecrit >= len(ipis)*0.4)
    #print(cond1,cond2)
    #if cond1:
    #    cond2 = (ipis.max()-ipis.min())<20#np.sum((ipis>=ipi_min) & (ipis<=ipi_max))>=2
    boolvec[cc] = cond1 & cond2

my_crossings = merged_crossings[boolvec]

if int_start!=9999.:
    print('selecting interval')
    boolvec[(merged_crossings[:,0]/aRec.sr<int_start) | (merged_crossings[:,0]/aRec.sr>int_stop)] = 0


if save_on:
    logger.info('saving automatically detect licktimes')
    with h5py.File(outfile,'r+') as fdest:
        if 'detection' in fdest:
            del fdest['detection']
        dgroup = fdest.create_group('detection')
        for name,vals in zip(['startstops'],[my_crossings]):
            dgroup.create_dataset(name,data=vals,dtype='i')

#h_crossings = np.array([np.max(zartvec[c0:c1]) for c0,c1 in crossings])
#from state_detection.utils import helpers as hf
#h_crossings2 = hf.get_superbursts(h_crossings,int(aRec.sr*0.02))

#h_crossingx[:,idx]


if plot_on or interactive:
    # plot examples showing in one plot (stim,detected lick episodes,artsrc + envelope + peaks, PFC centerchans)
    logger.info('plotting')


    stimplotdir = pathdict['plotting']['stimdir']

    n_ex = 6
    env_color = 'silver'
    peak_col = 'darkorange'
    cross_col = 'grey'
    detcol = 'firebrick'
    dur_ex = np.array([5,10,20])
    anat_rois = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']#only for LFP, this is where the middlechannels get selected from
    lfp_rangeHz = [0.5, 100]

    stylepath = os.path.join(pathdict['plotting']['style'])
    plt.style.use(stylepath)
    figdir = os.path.join(pathdict['outpaths']['figures'],recid0)
    if not os.path.isdir(figdir): os.makedirs(figdir)


    filtfn_lfp = lambda datatrace: filt.butter_bandpass_filter(datatrace, lfp_rangeHz[0], lfp_rangeHz[1], aRec.sr)
    get_lfp_pts = lambda mychan, tint: dh.grab_filter_dataslice(aRec.h_lfp,tint,mychan,filtfn=filtfn_lfp)

    check_region = lambda locname: True
    chans,locdict,cdict,clist = shf.get_midchans_and_coloring(aRec,regionchecker=check_region)

    tracedict = {}
    for chan in chans:
        tracedict[chan] = get_lfp_pts(chan, [0, int(aRec.dur * aRec.sr)])


    smod = fhs.retrieve_module_by_path(os.path.join(stimplotdir,exptype.lower()+'.py'))
    stimdict = smod.get_stimdict(aRec)
    stimdict['lick'] = np.array([])
    stimdict['LICK'] = my_crossings/aRec.sr

    plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple)

    # select n_ex random intervals of exdur duration to save
    dur_snips = np.random.choice(dur_ex,n_ex)
    exstarts = np.sort(np.random.uniform(0,aRec.dur-dur_ex.max(),n_ex))
    seltimes = np.vstack([exstarts,exstarts+dur_snips]).T

if not interactive:
    f,axarr = plt.subplots(4,1,figsize=(16,8),gridspec_kw={'height_ratios':[0.1,0.07,1,1]},sharex=True)
    f.subplots_adjust(left=0.07, right=0.93,bottom=0.07,top=0.95,hspace=0.05)
    stax,ax1,ax,lax = axarr
    plot_stims([stax])
    ax.plot(aRec.tvec,zartvec,'k')
    ax.plot(aRec.tvec[peaks],zartvec[peaks],'o',mfc='none',mec=peak_col)
    ax.axhline(peakthr,color=peak_col,linestyle='--',zorder=-10)
    ax.plot(aRec.tvec,envelope,color=env_color)
    ax.axhline(cross_thr,color=cross_col,linestyle='--',zorder=-10)
    ax1.hlines(np.zeros(crossings.shape[0]),crossings[:,0]/aRec.sr,crossings[:,1]/aRec.sr,color=cross_col,lw=3)
    #ax1.hlines(np.zeros(h_crossings.shape[0])-0.75,h_crossings[:,0]/aRec.sr,h_crossings[:,1]/aRec.sr,color='navy',lw=3)
    #ax1.hlines(np.zeros(h_crossings2.shape[0])-0.5,h_crossings2[:,0]/aRec.sr,h_crossings2[:,1]/aRec.sr,color='firebrick',lw=3)
    ax1.hlines(np.zeros(my_crossings.shape[0])+0.3,my_crossings[:,0]/aRec.sr,my_crossings[:,1]/aRec.sr,color=detcol,lw=3)
    #ax1.hlines(np.zeros(bpts.shape[0]),bpts[:,0]/aRec.sr,bpts[:,1]/aRec.sr,color='orange',lw=3)
    #ax1.set_ylim([-1.1,0.1])
    ax1.set_ylim([-0.1,0.4])
    for myax in [stax,ax1]: myax.set_axis_off()
    for cc,chan in enumerate(chans):
        offset_trace = tracedict[chan]+cc*60
        lax.plot(aRec.tvec,offset_trace,color=cdict[chan])
    for a0,a1 in artblocks/aRec.sr:
        lax.axvspan(a0,a1,color='khaki',lw=0.,alpha=0.4)

    for myax in [ax1,lax,ax]:
        myyl = myax.get_ylim()
        for idx in [0,1]:
            myax.vlines(my_crossings[:,idx]/aRec.sr,myyl[0],myyl[1],color=detcol,zorder=-10,lw=0.5)
        myax.set_ylim(myyl)
    N = len(locdict)
    legend_anchor = lax.get_position().y1
    for cc, [chan, chanloc] in enumerate(locdict.items()):
        chancol = cdict[chan]
        f.text(0.99, legend_anchor+ (cc - N) * 0.03, chanloc, ha='right', va='top', color=chancol,
               fontweight='bold', fontsize=10)
    f.suptitle('%s (%s)'%(recid0,exptype))
    ax.set_ylabel('abs(art) [z]')
    lax.set_ylabel('LFP [muV]')
    axarr[-1].set_xlabel('time [s]')
    if save_on:
        for tt,selt in enumerate(seltimes):
            ax.set_xlim(selt)
            f.savefig(os.path.join(figdir,'%s__ex%i.png'%(recid0,tt+1)))
        plt.close(f)
    logger.info('#### DONE automatic lickdetection %s'%recid0)


if interactive:
    logger.info('interactive lick detection')

    level_0 = 8.5
    level_1 = 8.7
    level_2 = 9.2
    level_edit = 9
    del_button = 3
    add_button = 1


    addlist = []
    removers = []

    fig_on = True
    def append_adders(x1,x2):
        global addlist
        addlist = addlist + [[x1,x2,time.time()]]

    def append_removers(x1,x2):
        global removers
        removers = removers + [[x1,x2,time.time()]]



    def line_select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")

        #print(f" The buttons you used were: {eclick.button} {erelease.button}")
        if eclick.button == add_button:
            plt.gca().hlines(level_edit,x1,x2,colors='m',lw=10.)
            print('adding [%1.2f %1.2f]'%(x1,x2))
            #adders += [(x1,x2)]
            append_adders(x1,x2)
        elif eclick.button == del_button:
            plt.gca().hlines(level_edit,x1,x2,colors='cornflowerblue',lw=10.)
            print('removing arts in [%1.2f %1.2f]'%(x1,x2))
            append_removers(x1,x2)
        plt.draw()

    def toggle_selector(event):
        if event.key == 't':
            if toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            else:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

    def donefn(event):
        global fig_on
        fig_on = False
        plt.close('all')


    artvec_normed = (artvec-np.nanmean(artvec))/np.nanstd(artvec)

    stimdict = smod.get_stimdict(aRec)

    plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple)

    f,axarr = plt.subplots(4,1,figsize=(16,8),gridspec_kw={'height_ratios':[0.15,1,1,0.4]},sharex=True)
    f.subplots_adjust(left=0.07, right=0.93,bottom=0.07,top=0.95,hspace=0.05)
    doneax = f.add_axes([0.95, 0.02, 0.04, 0.05])
    bdone = Button(doneax, 'DONE', color='b', hovercolor='c')
    bdone.label.set_color('w')
    bdone.label.set_fontweight('bold')
    bdone.on_clicked(donefn)

    stax,lax,ax,emax = axarr
    plot_stims([stax])
    stax.set_yticks([])

    emax.plot(aRec.emg_tvec, aRec.emg_data, color='k')
    emax.set_ylim([-30, 30])
    emax.set_ylabel('emg')

    for cc, chan in enumerate(chans):
        offset_trace = tracedict[chan] + cc * 60
        lax.plot(aRec.tvec, offset_trace, color=cdict[chan])
        lax.text(1.01, 0.99+(-len(chans)+cc)*0.1, '%s' % (locdict[chan]), color=cdict[chan], transform=lax.transAxes, \
                ha='left', va='top', fontweight='bold')
    lax.set_ylim([-8*np.std(tracedict[chans[0]]),cc*60+8*np.std(tracedict[chan])])

    for a0, a1 in artblocks / aRec.sr:
        lax.axvspan(a0, a1, color='khaki', lw=0., alpha=0.4)
    #ax2.plot(aRec.tvec,tracedict[ch],color='c')
    ax.plot(aRec.tvec,artvec_normed,'k')
    ax.hlines(np.zeros_like(h_crossingx[:,0])+level_0,h_crossingx[:,0]/aRec.sr,h_crossingx[:,1]/aRec.sr,color='grey',lw=4)
    ax.hlines(np.zeros_like(merged_crossings[:,0])+level_1,merged_crossings[:,0]/aRec.sr,merged_crossings[:,1]/aRec.sr,color='orange',lw=4)
    ax.hlines(np.zeros_like(merged_crossings[boolvec,0])+level_2,merged_crossings[boolvec,0]/aRec.sr,merged_crossings[boolvec,1]/aRec.sr,color='r',lw=4)
    for a0,a1 in merged_crossings[boolvec]:
        ax.axvline((a0+(a1-a0)/2)/aRec.sr,color='r',lw=2,alpha=0.3)
        lax.axvline((a0+(a1-a0)/2)/aRec.sr,color='r',lw=2,alpha=0.3)
    def line_select_callback(eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")

        #print(f" The buttons you used were: {eclick.button} {erelease.button}")
        if eclick.button == add_button:
            ax.hlines(level_edit,x1,x2,colors='m',lw=10.)
            print('adding [%1.2f %1.2f]'%(x1,x2))
            #adders += [(x1,x2)]
            append_adders(x1,x2)
        elif eclick.button == del_button:
            ax.hlines(level_edit,x1,x2,colors='cornflowerblue',lw=10.)
            print('removing arts in [%1.2f %1.2f]'%(x1,x2))
            append_removers(x1,x2)
        plt.draw()
    #for cc,mcr in enumerate(merged_crossings):
    #    ax.text(mcr[0]/aRec.sr,level_1,'%i'%cc,va='bottom',ha='left',color='orange')
    #myylim = ax.get_ylim()
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[add_button, del_button],  # disable middle button
                                           minspanx=0.01, minspany=0.2,
                                           spancoords='data',
                                           interactive=True)
    #ax.set_ylim(myylim)
    ax.set_xlim([0,aRec.dur])
    f.suptitle('%s (%s) del_button:%i; add_button:%i'%(aRec.id,exptype,del_button,add_button))


    ax.set_ylabel('ica-art [z]')
    lax.set_ylabel('LFP + offset [muV]')
    axarr[-1].set_xlabel('time [s]')
    ax.set_ylim([-12,14])
    cid = f.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()
    plt.draw()
    #fignum = f.number


if fig_on == False:
    #while plt.fignum_exists(fignum):
    #   time.sleep(1)

    if interactive:
        logger.info('sorting and saving interactive lick detection')
        # clean up among addlist and removers according to timestamp
        # delete/add to/from the lickdata according to addlist,removers

        pre_edited = merged_crossings[boolvec]/aRec.sr

        addmat = np.vstack(addlist) if len(addlist)>0 else np.empty((0,3))
        remmat = np.vstack(removers) if len(removers)>0 else np.empty((0,3))
        addmat2 = np.hstack([addmat,np.ones(len(addmat))[:,None]])
        remmat2 = np.hstack([remmat,np.zeros(len(remmat))[:,None]])
        dmat = np.hstack([pre_edited,np.zeros(len(pre_edited))[:,None],np.ones(len(pre_edited))[:,None]])

        collective = np.vstack([dmat,addmat2,remmat2])#start,stop,timestamp,add/removebool
        sortinds = np.argsort(collective[:,2])
        collective = collective[sortinds]
        resultmat = np.empty((0,4))
        for element in collective:
            if element[3] == 1:
                resultmat = np.vstack([resultmat,element])
            elif element[3] == 0:
                del_start,del_stop = element[:2]
                del_inds = np.where((resultmat[:,0]<del_stop) & (resultmat[:,1]>del_start))[0]
                #print('Del inds',del_inds)
                if len(del_inds)>0:
                    resultmat = np.delete(resultmat,del_inds,axis=0)


        resulmat = np.unique(resultmat,axis=0)
        resultmat = resultmat[np.argsort(resultmat[:,0])]
        deleted0 = [el for el in dmat if len(np.where((resultmat == el).all(axis=1))[0])==0]
        deleted = np.vstack(deleted0)[:,:2] if len(deleted0)>0 else np.empty((0,2))
        inserted_inds = np.sort(np.array([int(np.where((resultmat == el).all(axis=1))[0]) for el in resultmat \
                                          if len(np.where((dmat == el).all(axis=1))[0])==0]))

        lickarttimes = resultmat[:,:2]
        inserted_times = lickarttimes[inserted_inds]
        lickarttimes = dh.mergeblocks([lickarttimes.T], output='block', t_start=0).T#to avoid double dets

        # save lickarttimes, index of inserted, deleted times (resultmat[:,:2])
        if save_on:
            logger.info('saving manually edited licktimes')
            with h5py.File(outfile,'r+') as fdest:
                if 'detection_edited' in fdest:
                    answer = input("do you really want to overwrite detection_edited (yes)?")
                    if answer.lower() == 'yes':
                        del fdest['detection_edited']
                    else:
                        print('answer: %s, exiting script'%answer)
                        exit()
                dgroup = fdest.create_group('detection_edited')
                for name,vals,dtype in zip(['startstoptimes','deleted','inserted_times'],[lickarttimes,deleted,inserted_times],['f','f','f']):
                    dgroup.create_dataset(name,data=vals,dtype=dtype)

    if interactive and plot_on:
        logger.info('saving interactive lick detection example plots')

        #load saved times
        with h5py.File(outfile,'r+') as fsrc:
            print('loading')
            lickarttimes,deleted,inserted_times = [fsrc['detection_edited/%s'%key][()] \
                                                   for key in ['startstoptimes','deleted','inserted_times']]


        #plot example with the final decision, selected and de-selected marked


        stimdict = smod.get_stimdict(aRec)
        stimdict['LICK'] = lickarttimes

        plot_stims = sfn.make_stimfunc_lines(stimdict, smod.styledict_simple)

        f,axarr = plt.subplots(5,1,figsize=(16,8),gridspec_kw={'height_ratios':[0.1,0.07,1,1.2,0.4]},sharex=True)
        f.subplots_adjust(left=0.07, right=0.93,bottom=0.07,top=0.95,hspace=0.05)
        stax,ax1,ax,lax,emax = axarr
        plot_stims([stax])
        for myax in [stax,ax1]:
            myax.set_yticks([])

        for l0,l1 in lickarttimes:
            ax1.axvspan(l0,l1,color='r',alpha=0.5,lw=0)
        for l0,l1 in inserted_times[:,:2]:
            ax1.axvspan(l0,l1,hatch='+',edgecolor='r',facecolor='none',lw=3)
        for l0,l1 in deleted:
            ax1.axvspan(l0,l1,hatch='x',edgecolor='k',facecolor='grey',lw=1,alpha=0.5)
        ax.plot(aRec.tvec,artvec_normed,'k')
        for a0, a1 in artblocks / aRec.sr:
            lax.axvspan(a0, a1, color='khaki', lw=0., alpha=0.4)
        for cc,chan in enumerate(chans):
            offset_trace = tracedict[chan]+cc*60
            lax.plot(aRec.tvec,offset_trace,color=cdict[chan])
            lax.text(1.01, 0.99+(-len(chans)+cc)*0.1, '%s' % (locdict[chan]), color=cdict[chan], transform=lax.transAxes, \
                    ha='left', va='top', fontweight='bold')

        emax.plot(aRec.emg_tvec,aRec.emg_data,color='k')
        emax.set_ylim([-30,30])
        emax.set_ylabel('emg')
        lax.set_ylim([-8*np.std(tracedict[chans[0]]),cc*60+8*np.std(tracedict[chan])])
        ax.set_ylim([-10,10])
        f.suptitle('%s (%s)'%(aRec.id,exptype))
        ax.set_ylabel('ica-art [z]')
        lax.set_ylabel('LFP + offset [muV]')
        axarr[-1].set_xlabel('time [s]')
        axarr[-1].set_xlim([0.,aRec.dur])

        if save_on:
            for tt,selt in enumerate(seltimes):
                ax.set_xlim(selt)
                f.savefig(os.path.join(figdir,'%s__edited_ex%i.png'%(recid,tt+1)))
            plt.close(f)






'''
ch = list(tracedict.keys())[-1]

f,ax = plt.subplots(figsize=(16,5))
ax.plot(aRec.tvec,artvec,'k')
ax2 = ax.twinx()
ax2.plot(aRec.tvec,tracedict[ch],color='r')
ax.hlines(np.zeros_like(h_crossingx[:,0])+0.02,h_crossingx[:,0]/aRec.sr,h_crossingx[:,1]/aRec.sr,color='grey',lw=4)
ax.hlines(np.zeros_like(merged_crossings[:,0])+0.03,merged_crossings[:,0]/aRec.sr,merged_crossings[:,1]/aRec.sr,color='orange',lw=4)
#autocorr


#ipi_min = int(0.11*aRec.sr)
#ipi_max = int(0.2*aRec.sr)


cpts = merged_crossings[ii]
#event = artvec[cpts[0]:cpts[1]]
event = artvec[cpts[0] - buff:cpts[1] + buff]
ant.higuchi_fd(event)
myfn = lambda x: ant.spectral_entropy(x, sf=aRec.sr, method='fft', normalize=True)
#myfn = lambda x: ant.perm_entropy(x, normalize=True)
#myfn = lambda x: ant.higuchi_fd(x)


fd_vec = np.array([myfn(artvec[cpts[0]:cpts[1]]) for cpts in merged_crossings])
f,ax = plt.subplots()
ax.hist(fd_vec,100)

fd_thr = 1.2
myinds = np.where(fd_vec>fd_thr)[0]

f,ax = plt.subplots(figsize=(5,12))
for mm,myind in enumerate(np.random.choice(myinds,50)):
    cpts = merged_crossings[myind]
    event = artvec[cpts[0] - buff:cpts[1] + buff]
    ax.plot(event+mm*0.01)
ax.set_xlim(0,1200)



f,ax = plt.subplots()
ax.plot(event,color='k')
corr = cfns.myCorr(event,event,len(event)/2)
f,ax = plt.subplots()
ax.plot(corr)


f,ax = plt.subplots()
myfft = np.abs(np.fft.rfft(event))**2
ax.plot(myfft)

import antropy as ant

f,ax = plt.subplots()
ax.hist(np.diff(merged_crossings)/aRec.sr,100)

XXX
'''


'''
delta = (h_crossingx[:,1]-h_crossingx[:,0])/aRec.sr

f,ax = plt.subplots()
ax.hist(delta,100)
'''

'''
my_crossings = my_crossings[np.diff(my_crossings)[:,0]>0.04*aRec.sr]
fd_thr = 0.35#1.2
#myfn = lambda x: ant.higuchi_fd(x)
myfn = lambda x: ant.spectral_entropy(x, sf=aRec.sr, method='fft', normalize=True)

fd_vec = np.array([myfn(artvec[cpts[0]:cpts[1]]) for cpts in my_crossings])
myinds = np.where(fd_vec<=fd_thr)[0]
my_crossings = my_crossings[myinds,:]
'''

'''

# ipi = np.log10(np.diff(peaks/aRec.sr))
# f,ax = plt.subplots()
# ax.hist(ipi,100)
#
# durdiff_max = int(0.05*aRec.sr)
# bdict = ba.find_bursts(peaks, durdiff_max)
# bpts = np.vstack([bdict['start'],bdict['stop']]).T
'''