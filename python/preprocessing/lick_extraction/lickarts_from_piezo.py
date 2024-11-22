import os
import sys
import scipy.io
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as ss
import yaml


split_biglicks = True
##PARAMETERS
cross_thr = 0.1 #for hp-filtered envelope [z]
min_npts = 2
merge_dur = 0.015 #<=mergedur detections get merged
hp_freq = 150
sg_params = [51,3]#savgol smoothing
lp_freq = 30
min_peak = 0.15 #peak in hp-filtered envelope [z]
lfmax_thr = 1. #protection against getting removed
peakthr = 1.5 #protection against getting removed
chunk_dur = 300  #for low-ram solution
#########

####plotting params
#stacked snippets:
n_ex_snips = 20
n_ex_pan = 3
buff_t = 0.05
#on trace:
det_col_cross = 'firebrick'
det_col_super =  'darkviolet'
dur_ex = 5
n_ex = 10
####



lickfile,pathpath = sys.argv[1:]


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import filtering as filt
from pfcmap.python.utils import statedet_helpers as hf
from pfcmap.python.utils import file_helpers as fhs

figdir_gen = pathdict['outpaths']['figures']
dstdir = pathdict['outpaths']['data']
stylepath = pathdict['plotting']['style']
plt.style.use(stylepath)


an_id = os.path.basename(lickfile).split('.')[0]
figdir_an = os.path.join(figdir_gen,an_id)

def figsaver(fig, nametag, closeit=True):
    #print(an_id)
    figname = os.path.join(figdir_an, '%s__%s'%(an_id,nametag) + '.png')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)




mat = scipy.io.loadmat(lickfile)

licktrace = mat['lick'][0][0][0][0]
sr = float(mat['lick'][0][0][1][0])
dur = len(licktrace)/sr





merge_thresh = merge_dur*sr
chunk_pts = int(chunk_dur*sr)
window_vec = np.r_[np.arange(0,len(licktrace),chunk_pts),len(licktrace)]
n_win = len(window_vec)
h_envel_filt = np.zeros_like(licktrace)
l_envel = np.zeros_like(licktrace)
for ii in np.arange(n_win-1):
    print('windowing envelope (hp) %i/%i'%(ii+1,n_win))
    #getting the crossings
    w0,w1 = window_vec[ii:ii+2]
    snip = licktrace[w0:w1]
    highfilt = filt.butter_highpass_filter(snip,hp_freq,fs=int(sr))
    h_envel = np.abs(ss.hilbert(highfilt))
    h_envel_filt[w0:w1] = filt.savitzky_golay(h_envel,sg_params[0],sg_params[1])
    lowfilt = filt.butter_lowpass_filter(snip, lp_freq, int(sr))
    l_envel[w0:w1]= np.abs(ss.hilbert(lowfilt))

envelz = (h_envel_filt-h_envel_filt.mean())/h_envel_filt.std()
lenvelz = (l_envel-l_envel.mean())/l_envel.std()



starts0 = np.where(np.diff(np.sign(envelz-cross_thr))>0)[0]
stops0 = np.where(np.diff(np.sign(envelz-cross_thr))<0)[0]
stops = stops0[stops0>starts0[0]] #excluding the first if it is an open interval being finished
starts = starts0[starts0<stops0[-1]]#
crossings = np.vstack([starts,stops]).T
superpts0 = hf.get_superbursts(crossings,merge_thresh)
superpts0 = np.vstack([el for el in superpts0 if np.diff(el)>min_npts])

areas0 = np.array([np.sum(envelz[p0:p1]-0) for p0,p1 in superpts0])/sr
peaks0 = np.array([np.max(envelz[p0:p1]) for p0,p1 in superpts0])
lfmax0 =  np.array([np.max(lenvelz[p0:p1]) for p0,p1 in superpts0])
sel_inds0 = np.where((peaks0>min_peak) & (areas0>0))[0]
areas0 = areas0[sel_inds0]
peaks0 = peaks0[sel_inds0]
lfmax0 = lfmax0[sel_inds0]
superpts0 = superpts0[sel_inds0]


area_bw = 0.01
areab = np.arange(areas0.min(),5,area_bw)

areah,_ = np.histogram(areas0,areab)




thr_areah = np.mean(areah)+4*np.std(areah)
area_crossinds = np.where(areah>thr_areah)[0]
if 0 in area_crossinds:
    pdiff_big = np.where(np.diff(area_crossinds)>1)[0]
    if np.size(pdiff_big) == 0:
        area_idx = area_crossinds[0]
    else:
        area_idx = pdiff_big[0]
    areathr = areab[area_idx+1]
else:
    areathr = areas0.min()


cump = np.cumsum(areah[::-1])
f,ax = plt.subplots()
ax.plot(areab[:-1]+0.5*area_bw,cump[::-1],'k')
ax.axvline(areathr,color='r',zorder=-5)
ax.set_ylabel('counts')
ax.set_xlabel('areathr')
figsaver(f,'areathr_cumhist')


sel_inds = np.where((areas0>areathr) | (lfmax0>lfmax_thr) | (peaks0>peakthr))[0]


dur = (len(licktrace)/sr)

superpts = superpts0[sel_inds]

areas = areas0[sel_inds]
peaks = peaks0[sel_inds]
lfmax = lfmax0[sel_inds]

bdiff_max = 0.3*sr

big_inds = np.where(peaks>=3)[0]
bigpts = superpts[big_inds]
bigcenters = bigpts[:,0]+np.diff(bigpts)[:,0]/2
small_inds = np.where(peaks<1.)[0]
smallpts = superpts[small_inds]
smallcenters = smallpts[:,0]+np.diff(smallpts)[:,0]/2

deleteinds = []
for ii in np.arange(len(small_inds)):
    sidx = small_inds[ii]
    scent = smallcenters[ii]
    bdiff = np.min(np.abs(bigcenters-scent))
    if bdiff>bdiff_max:
        deleteinds += [sidx]

del_inds = np.array(deleteinds)

deleted = superpts[deleteinds]

if len(deleteinds)>0:
    areas = np.delete(areas,del_inds)
    peaks = np.delete(peaks,del_inds)
    superpts = np.delete(superpts,del_inds,axis=0)

print('## N deleted %i'%len(deleteinds))


if split_biglicks:
    from scipy.signal import find_peaks
    lp_peak_splitthr = 5
    minpdist = 0.015*sr
    distbuff = int(0.01*sr)
    splitdict = {}
    for ii in np.arange(len(superpts)):
        spt0,spt1 = superpts[ii]
        snipl = lenvelz[spt0:spt1]
        sniph = envelz[spt0:spt1]
        peaksl = find_peaks(snipl,height=lp_peak_splitthr)[0]
        peaksh = find_peaks(sniph,height=15,prominence=15.,distance=minpdist)[0]
        if len(peaksl)>1 and len(peaksh)>1:
            splitpts = []
            midpeaks = []
            for jj in np.arange(len(peaksh)-1):
                selh = sniph[peaksh[jj]+distbuff*2:peaksh[jj+1]-distbuff]
                if len(selh)>0:
                    midps,output = find_peaks(selh,height=2.)
                    if len(midps)>0:
                        phts = output['peak_heights']
                        midp = midps[np.argmax(phts)]
                    else:
                        midp = len(selh)
                    splitpt = np.argmin(selh[:midp])+peaksh[jj]+distbuff*2
                    splitpts += [splitpt]
                    midpeaks += [midp+peaksh[jj]+distbuff*2]
            splitpts = np.array(splitpts)
            splitdict[ii] = splitpts
            '''snip = licktrace[spt0:spt1]
            f,axarr = plt.subplots(3,1,sharex=True)
            ax,ax2,axr = axarr
            ax.plot(snipl,'k')
            ax2.plot(sniph,'g')
            ax2.plot(peaksh,sniph[peaksh],'ob')
            if len(splitpts)>0:
                ax2.plot(splitpts,sniph[splitpts],'om')
                ax2.plot(midpeaks,sniph[midpeaks],'o',color='pink')
            axr.plot(snip,'grey')'''

    print ('splitting %i big artifacts'%(len(splitdict)))


    new_superpts = np.empty((0,2),dtype=int)
    for ii,spts in enumerate(superpts):
        if ii in splitdict.keys():
            splitvec = splitdict[ii]
            #spts = superpts[ii]
            newstarts = np.r_[spts[0],spts[0]+splitvec]
            newstops = np.r_[spts[0]+splitvec,spts[1]]
            newpts = np.vstack([newstarts,newstops]).T
            new_superpts = np.vstack([new_superpts,newpts])
        else:
            new_superpts = np.vstack([new_superpts,spts])

    superpts = (new_superpts).astype(int)
    areas = np.array([np.sum(envelz[p0:p1]-0) for p0,p1 in superpts])/sr
    peaks = np.array([np.max(envelz[p0:p1]) for p0,p1 in superpts])
    lfmax =  np.array([np.max(lenvelz[p0:p1]) for p0,p1 in superpts])



f,ax = plt.subplots()
ax.hist(areas,int(len(areas)/5),color='k')
ax.set_xlabel('env area [z*s]')
ax.set_ylabel('count')
ax.set_title('%s, N: %i (%1.4f/s)'%(an_id,len(areas),len(areas)/dur))
ax.set_xlim([0,ax.get_xlim()[1]])
figsaver(f,'area_hist')

f,ax = plt.subplots()
ax.hist(peaks,int(len(peaks)/5),color='k')
ax.set_xlabel('env peak')
ax.set_ylabel('count')
ax.set_title('%s, N: %i (%1.4f/s)'%(an_id,len(areas),len(areas)/dur))
ax.set_xlim([0,ax.get_xlim()[1]])
figsaver(f,'peak_hist')

f,ax = plt.subplots()
ax.hist(peaks,int(len(peaks)/5),color='k')
ax.set_xlabel('lfmax env')
ax.set_ylabel('count')
ax.set_title('%s, N: %i (%1.4f/s)'%(an_id,len(areas),len(areas)/dur))
ax.set_xlim([0,ax.get_xlim()[1]])
figsaver(f,'lfmax hist')



#saving data
outfile = os.path.join(dstdir,'%s__piezoLicks.h5'%(an_id))
with h5py.File(outfile, 'w') as fdest:
    dgroup = fdest.create_group('detection')
    for name, vals, dtype in zip(['startstoppts','crosspts', 'areas', 'peaks','lfmax','sr'], [superpts,crossings,areas,peaks,lfmax,np.array(sr)],
                                 ['i', 'i', 'f','f','f','f']):
        dgroup.create_dataset(name, data=vals, dtype=dtype)
    methods = fdest.create_group('methods')
    methods.attrs['min_peak'] = min_peak
    methods.attrs['areathr'] = areathr
    methods.attrs['hp_freq'] = hp_freq
    methods.attrs['cross_thr'] = cross_thr
    methods.attrs['merge_dur'] = merge_dur
    methods.attrs['sg_params'] = str(sg_params)
    methods.attrs['lp_freq'] = lp_freq
    methods.attrs['lfmax_thr'] = lfmax_thr
    methods.attrs['peakthr'] = peakthr
    methods.attrs['git_hash'] = fhs.get_githash()


print('SAVED LICKS %s'%an_id)

print('PLOTTING EXAMPLES %s'%an_id)



buffpts = int(buff_t*sr)
l_std = np.std(licktrace)

for jj in np.arange(n_ex_pan):
    ex_super = np.random.choice(np.arange(len(superpts)),n_ex_snips,replace=False)

    f,axarr = plt.subplots(1,2,figsize=(10,12))
    f.subplots_adjust(left=0.07,right=0.98,top=0.92,bottom=0.07,wspace=0.3)
    ax,ax2 = axarr
    for ii,idx in enumerate(ex_super):
        p0,p1 = superpts[idx]+np.array([-buffpts,buffpts])
        thist= np.linspace(0,p1-p0,p1-p0)/sr-buff_t
        e_off = ii*5
        o_off = ii*l_std*14
        etrace = envelz[p0:p1]+e_off
        otrace = licktrace[p0:p1]+o_off
        ax.plot(thist,etrace)
        ax2.plot(thist,otrace)
        for myax,myy,myoff in zip(axarr,[etrace,otrace],[e_off,o_off]):
            myax.plot(thist[-buffpts],myy[-buffpts],'o',mec='none',mfc='grey',ms=8)
            myax.axhline(myoff,alpha=0.5,color='grey',zorder=-5)
    for myax in axarr:
        myax.axvline(0.,color='grey',zorder=-10)
        myax.set_xlabel('time [s]')
    ax.set_ylabel('amp [z]')
    ax2.set_ylabel('amp [muV]')
    ax.set_title('hf envelope')
    ax2.set_title('orig.')
    f.suptitle('%s - lick examples %i'%(an_id,jj+1))
    figsaver(f, 'detexamples%i'%(jj+1))



#tint = [tint_start,tint_start+20]
ex_sels = np.sort(np.random.choice(np.arange(len(superpts)),n_ex))
exstarts = superpts[ex_sels,1]/sr - 0.5*dur_ex
exseltimes = np.vstack([exstarts, exstarts + dur_ex]).T

for tt,tint in enumerate(exseltimes):

    #tint = [250 ,275]
    #tint = [180,250]
    w0,w1 = (np.array(tint)*sr).astype(int)
    snip = licktrace[w0:w1]
    highfilt = filt.butter_highpass_filter(snip, hp_freq, fs=int(sr))
    h_envel = np.abs(ss.hilbert(highfilt))
    lowfilt = filt.butter_lowpass_filter(snip,30,int(sr))
    lowfilz = (lowfilt-lowfilt.mean())/lowfilt.std()
    l_envel = np.abs(ss.hilbert(lowfilz))


    supertimes = superpts[(superpts[:,1]>w0) & (superpts[:,0]<w1)]/sr
    deletedtimes = deleted[(deleted[:,1]>w0) & (deleted[:,0]<w1)]/sr


    tcrossings = crossings[(crossings[:,1]>w0) & (crossings[:,0]<w1)]/sr
    tvec = np.linspace(tint[0],tint[1],len(snip))

    f,axarr = plt.subplots(5,figsize=(16,6),sharex=True,gridspec_kw={'height_ratios':[0.1,1,1,1,1]})
    f.subplots_adjust(left=0.05,right=0.99)
    dax,rax,hax,eax,lax = axarr

    dax.hlines(np.zeros(len(tcrossings))+0.5,tcrossings[:,0],tcrossings[:,1],color=det_col_cross,lw=4)
    dax.hlines(np.zeros(len(supertimes))+0.5,supertimes[:,0],supertimes[:,1],color=det_col_super,lw=2)

    dax.set_ylim([0,1])
    dax.set_yticks([])
    rax.plot(tvec,snip,color='k')

    hax.plot(tvec,highfilt,color='grey')
    #hax.plot(tvec,h_envel,color='orange')
    hax.plot(tvec,h_envel_filt[w0:w1],color=det_col_cross)

    eax.plot(tvec,envelz[w0:w1],color=det_col_cross,alpha=1)
    eax.axhline(cross_thr,color='k')
    lax.plot(tvec,lowfilz,color='b')
    lax.plot(tvec,l_envel,color='c')


    for myax in axarr[1:]:
        for s0,s1 in supertimes:
            myax.axvspan(s0,s1,color=det_col_super,alpha=0.5,lw=0,zorder=-5)
        for s0,s1 in deletedtimes:
            myax.axvspan(s0,s1,color='grey',alpha=0.5,lw=0,zorder=-5)

    for myax in [rax,hax]:
        myax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
    axarr[-1].set_xlabel('time [s]')
    rax.set_ylabel('orig.')
    hax.set_ylabel('hp.',color='grey')
    lax.set_ylabel('lp. [z]',color='grey')

    eax.set_ylabel('env. [z]',color=det_col_cross)
    axarr[-1].set_xlim(tint)
    f.suptitle('%s example %i'%(an_id,tt+1))
    figsaver(f, 'tracexamples%i'%(tt+1))

