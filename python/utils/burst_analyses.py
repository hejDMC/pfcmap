import numpy as np
from . import data_handling as dh
grab_filter_datacube,check_olap = dh.grab_filter_datacube, dh.check_olap
from . import plotting_basics as pb
import matplotlib as mpl
from copy import deepcopy
from scipy.interpolate import interp1d
from skimage import measure
from . import corrfuncs as cfns

def findmerge_bursts(spiketimes, interval_threshold,maxint=None):
    isis = np.diff(spiketimes)
    spike_parts = np.where(isis < interval_threshold)[0]
    spike_starts = np.append(spike_parts[0], spike_parts[np.where(np.diff(spike_parts) > 1)[0] + 1])
    spike_stops = np.append(spike_parts[np.where(np.diff(spike_parts) > 1)] + 1, spike_parts[-1] + 1)
    bstarts = spiketimes[spike_starts]
    bstops = spiketimes[spike_stops]
    startstop_array = np.vstack([bstarts, bstops]).T
    if maxint is None or np.size(startstop_array) <= 2: return startstop_array
    else:
        starts = startstop_array[:, 0]
        stops = startstop_array[:, 1]
        intervals = starts[1:] - stops[:-1]

        startinds = np.r_[0, np.where(intervals >= maxint)[0] + 1]
        stopinds = np.r_[np.where(intervals >= maxint)[0], len(stops) - 1]

        newstarts = starts[startinds]
        newstops = stops[stopinds]
        return np.vstack([newstarts, newstops]).T

def find_bursts(spiketimes, maxdist):
    '''input: times when blips occured, threshold for maximum ibi
    returns dictionary with start-times and stop-times of bursts, and number of blips within bursts
    '''

    isis = np.diff(spiketimes)
    burst_parts = np.where(isis < maxdist)[0]
    if len(burst_parts)>0:
        burst_starts = np.append(burst_parts[0], burst_parts[np.where(np.diff(burst_parts) > 1)[0] + 1])
        burst_stops = np.append(burst_parts[np.where(np.diff(burst_parts) > 1)] + 1, burst_parts[-1] + 1)

        burst_dict = {}
        burst_dict['start'] = spiketimes[burst_starts]
        burst_dict['stop'] = spiketimes[burst_stops]
        burst_dict['blips'] = burst_stops - burst_starts + 1
    else:
        burst_dict = {key:np.array([]) for key in ['start','stop','blips']}
    return burst_dict

def get_nwburst_dict(bmat,spikemat,d_chan=4,min_pop=4,max_gap=3):
    # min_pop: minimum size of population
    # max_gap: electrodes allowed to be left out in a population
    bdict = {}
    get_neighbors = lambda chan: chans[(chans >= chan - max_gap * d_chan) & (chans <= chan + max_gap * d_chan)]
    counter = 0
    for b1,b2 in bmat:
        spikes, chans = spikemat[:, b1:b2]
        #print(ii)
        coreinds = np.array([ii for ii,chan in enumerate(chans) if len(get_neighbors(chan)) >= min_pop])
        if len(coreinds) > 0:
            allinds = coreinds[:]
            if len(coreinds) <len(chans):
                newadded = 1
                while newadded > 0:
                    side_inds = np.array(
                        [ii for ii, chan in enumerate(chans) if
                         not chan in chans[allinds] and np.min(np.abs(chan - chans[allinds])) <= (d_chan * max_gap)]).astype(int)
                    allinds = np.r_[allinds,side_inds]
                    newadded = len(side_inds)

            allchans = chans[allinds]
            sortinds = np.argsort(allchans)
            bdict[counter] = [spikes[allinds][sortinds],allchans[sortinds]]
            counter += 1
    return bdict

def get_burstpatches(B,maxchandiff,n_min=4):
    Bsub = {}
    counter = 0
    for ii in B.keys():
        sortinds = np.argsort(B[ii][1])
        spikes, chanvals = B[ii][0][sortinds], B[ii][1][sortinds]
        borders = np.where(np.diff(chanvals) > maxchandiff)[0]
        inds1 = np.r_[0, borders + 1]
        inds2 = np.r_[borders + 1, len(spikes)]
        for i1, i2 in zip(inds1, inds2):
            if (i2-i1) >= n_min:
                Bsub[counter] = np.vstack([spikes[i1:i2], chanvals[i1:i2]])
                counter += 1
    return Bsub

def bdetect(spikes,maxdiff,n_min=2,mode='spikeinds'):
    borders = np.where(np.diff(spikes)>maxdiff)[0]
    inds1 = np.r_[0,borders+1]
    inds2 = np.r_[borders+1,len(spikes)-1]
    temp = np.vstack([inds1, inds2]).T
    spikeinds = temp[np.diff(temp)[:,0]>=n_min,:]
    if mode == 'spikeinds':
        return spikeinds
    elif mode == 'burstborders':
        return np.vstack([spikeinds[:,0],spikeinds[:,1]-1]).T


def get_fofthresh_allchans(aRec,freepts,filtfn,threshvec):

    fofthr_dict = {}
    for el_id in aRec.el_ids:
        # el_id = aRec.el_ids[10]
        trace = np.squeeze(
            grab_filter_datacube(aRec.h_lfp, np.array([0, aRec.dur])[None, :], np.array([el_id]), filtfn,
                                    prepost=[0, 0]))
        maxima = np.array([], dtype=int)
        for pt1, pt2 in freepts:
            T = np.abs(trace[pt1:pt2])
            maxima_ = np.where(np.r_[False, T[1:] > T[:-1]] & np.r_[T[:-1] > T[1:], False])[0] + pt1
            maxima = np.r_[maxima, maxima_]
        z_trace = (trace - np.mean(trace)) / np.std(trace)
        amps = np.abs(z_trace[maxima])

        hst, bns = np.histogram(amps, threshvec)
        fofthr_dict[el_id] = len(amps) - np.r_[0, np.cumsum(hst)]  # exactly the same as the classicaf fofthr
    return fofthr_dict

def spikethresh_from_frac(fofthr_dict,threshvec,thr_frac):
    fofthr_avg = np.array(list(fofthr_dict.values())).mean(axis=0)
    fofthr_norm = fofthr_avg / fofthr_avg.max()
    return threshvec[np.argmin(np.abs(fofthr_norm - (1 - thr_frac)))]


def get_spikemats(aRec,freepts,spikethr,filtfn,verbose=False):
    spikedict = {}
    ampdict = {}

    for el_id in aRec.el_ids:
        #el_id = aRec.el_ids[10]
        trace = np.squeeze(grab_filter_datacube(aRec.h_lfp,np.array([0,aRec.dur])[None,:],np.array([el_id]),filtfn,prepost=[0,0]))
        maxima = np.array([],dtype=int)
        for pt1,pt2 in freepts:
            T = np.abs(trace[pt1:pt2])
            maxima_ = np.where(np.r_[False, T[1:] > T[:-1]] & np.r_[T[:-1] > T[1:], False])[0]+pt1
            maxima = np.r_[maxima,maxima_]
        z_trace = (trace - np.mean(trace)) / np.std(trace)
        amps = np.abs(z_trace[maxima])
        spikepts = maxima[amps>spikethr]
        spikedict[el_id],ampdict[el_id] = spikepts,trace[spikepts]

    #get it into a nice plottable matrix
    spikemats = {}
    for tag,operator in zip(['p','n'],[np.greater,np.less]):
        spikedict_np = {el_id:np.array([spikept for ii,spikept in enumerate(spikedict[el_id]) if operator(ampdict[el_id][ii],0)]) for el_id in spikedict.keys()}
        spikemat_ = np.hstack([[vals,np.ones_like(vals)*el_id] for el_id,vals in spikedict_np.items()])
        spikemats[tag] = spikemat_[:,spikemat_[0,:].argsort()]

    if verbose:
        return spikemats,spikedict,ampdict
    else:
        return spikemats


def plot_fofthr(aRec,els,fofthr_dict,threshvec,cmapstr='jet',**kwargs):
    N = len(fofthr_dict)
    el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])
    fofthr_avg = np.array(list(fofthr_dict.values())).mean(axis=0)
    fofthr_norm = fofthr_avg / fofthr_avg.max()

    colors = pb.get_colors(cmapstr, N)
    f, axarr = mpl.pyplot.subplots(1,3,gridspec_kw={'width_ratios':[1,0.1,0.07]})
    ax,cax,locax = axarr
    for ii, el_id in enumerate(fofthr_dict.keys()):
        ax.plot(threshvec, fofthr_dict[el_id] / fofthr_dict[el_id].max(), color=colors[ii])
    ax.plot(threshvec, fofthr_norm, color='k')
    if 'spikethr' in kwargs:
        ax.axvline(kwargs['spikethr'], color='grey')

    if 'thr_frac' in kwargs:
        ax.axhline(1 - kwargs['thr_frac'], color='grey')
    ax.set_xlim([threshvec.min(), threshvec.max()])
    ax.set_xlabel('threshold [z]')
    ax.set_ylabel('norm. peak count')
    #im = cax.imshow(np.vstack([np.ones(N),np.arange(N)]),cmap=cmapstr)
    cax.imshow(els[:, None], cmap=cmapstr, origin='lower', aspect='auto')
    # cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
    #                                cmap=mpl.cm.get_cmap(cmapstr),vmin=0,vmax=96)
    cax.set_axis_off()
    #cax.set_axis_off()
    pb.make_locax(locax, aRec.el_locs[el_inds], linecol='k', tha='left', textoff=1.1, cols=['k', 'lightgray'],
                  boundary_axes=[cax])

    return f





def plot_spikes(ax,aRec,spikemats,tint,pn_cols=['orange', 'steelblue'],mfc='var',mec='k', mew=0, ms=2,**kwargs):
    pint = (np.array(tint) * aRec.sr).astype(int) if not 'pint' in kwargs else kwargs['pint']
    isvar_mfc = True if mfc=='var' else False
    isvar_mec = True if mec=='var' else False
    for tag, col in zip(['p', 'n'], pn_cols):
        spikemat = deepcopy(spikemats[tag])
        cond = (spikemat[0] >= pint[0]) & (spikemat[0] <= pint[1])
        if isvar_mfc:mfc = str(col)
        if isvar_mec:mec = str(col)
        #print(mfc,str(col))
        ax.plot(spikemat[0, cond] / aRec.sr, spikemat[1, cond], 'o', mfc=mfc,mec=mec,mew=mew,ms=ms)
    ax.set_ylim([aRec.el_ids[0]-1, aRec.el_ids[-1]+1])

def plot_burstsdet(ax,aRec,bmats,spikemats,tint,coldict={'p':['goldenrod','saddlebrown'],'n':['b','c']},**kwargs):
    pint = (np.array(tint) * aRec.sr).astype(int) if not 'pint' in kwargs else kwargs['pint']
    for tag in ['p','n']:
        bmat = deepcopy(bmats[tag])
        #print (sum(cond))
        cols = coldict[tag]
        cond = (spikemats[tag][0, bmat[:,0]]>=pint[0])&(spikemats[tag][0, bmat[:,0]]<=pint[1])
        for ii, [b1, b2] in enumerate(bmat[cond]):
            col = cols[np.mod(ii, 2)]
            spikes, chans = spikemats[tag][:,b1:b2]

            if ii == 10: print((spikes/aRec.sr).mean())
            ax.plot(spikes/aRec.sr, chans, 'o', mec=col, mfc='none', ms=8)

def plot_newtworkbursts(ax,aRec,NBmat,tint,pn_cols=['orange', 'steelblue'],**kwargs):
    pint = (np.array(tint) * aRec.sr).astype(int) if not 'pint' in kwargs else kwargs['pint']
    for tag,col in zip(['p','n'],pn_cols):
        B = deepcopy(NBmat[tag])
        keys = [ii for ii, vals in B.items() if (vals[0].max() >= pint[0]) & (vals[0].min() <= pint[1])]
        for key in keys:
            spikes,chans = B[key]
            ax.plot(spikes / aRec.sr, chans, 'o-', color=col, mec=col, mfc='none',**kwargs)

def plot_data(ax,aRec,dmat,tint,yoff=1,kwargs_plot={},**kwargs):
    nchans = dmat.shape[0]
    chanvec = kwargs['chanvec'] if 'chanvec' in kwargs else np.arange(nchans)
    tvec = (np.arange(dmat.shape[1]) / aRec.sr) + tint[0]
    for ii, trace in zip(chanvec,dmat):
        ax.plot(tvec, trace * 0.1*np.diff(chanvec)[0] +ii, 'k-',**kwargs_plot)
    ax.set_ylim([chanvec.min()-yoff, chanvec.max()+yoff])

def plot_spikes_on_data(ax,aRec,spikemats,tint,els,dmat,pn_cols=['orange', 'steelblue'],yoff=1,spike_kwargs={},data_kwargs={}):
    nchans = dmat.shape[0]
    pint = (np.array(tint) * aRec.sr).astype(int)
    tvec = (np.arange(dmat.shape[1])/aRec.sr)+tint[0]
    for ii,trace in enumerate(dmat):
        el_id = els[ii]
        ax.plot(tvec, trace * 0.1 + ii, 'ko-',**data_kwargs)
        for tag, col in zip(['p', 'n'], pn_cols):
            spikemat = spikemats[tag]


            myspks_ = spikemat[:,spikemat[1]==el_id][0]
            cond = (myspks_ > pint[0]) & (myspks_ < pint[1])
            myspks = myspks_[cond]
            ax.plot(myspks/aRec.sr,trace[myspks-pint[0]]*0.1+ii,'o',mfc='none',mec=col,**spike_kwargs)
    ax.set_ylim([0 - yoff, nchans - 1 + yoff])

def get_isis(spikemat,freepts):
    isis_ = np.array([])
    for pt1, pt2 in freepts:
        spikes = spikemat[0][(spikemat[0] >= pt1) & (spikemat[0] <= pt2)]
        isis_ = np.r_[isis_, np.diff(spikes)]
    return isis_

def plot_isis(dlist,collist,nbins=100,xlab='ISI',logy=True,showstyle='hist'):
    allisis = np.hstack(dlist)
    bins = np.linspace(np.unique(allisis)[1], allisis.max(), nbins)
    f,ax = mpl.pyplot.subplots()
    for isis,col in zip(dlist,collist):
        if showstyle=='hist':ax.hist(isis,bins,histtype='step',color=col)
        elif showstyle=='line':
            hist, bins = np.histogram(isis, bins)
            ax.plot(bins[:-1],hist,color=col)
    ax.set_ylabel('count')
    ax.set_xlabel(xlab)
    ax.set_xlim([bins.min(),bins.max()])
    if logy:ax.set_yscale('log')
    return f

def plotmake_hist2d(ax,x,y,binsx,binsy,tfn=lambda x:x,cmap='jet',**kwargs):
    palette = mpl.cm.get_cmap(cmap).copy()
    palette.set_bad(alpha=0.0)
    H, xedges, yedges = np.histogram2d(x, y, bins=(binsx, binsy),**kwargs)

    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, tfn(H.T), cmap=palette)
    return H,im

def project_hist2d(ax,refax,H,axis,mybins,ylab='counts',xlab='',tfn=lambda x:x,logy=False,flipxy=False):

    yvals = tfn(H.sum(axis=axis))
    xvals = mybins[:-1]+0.5*np.diff(mybins)#np.linspace(mybins.min(),mybins.max(),len(yvals),endpoint=True)
    [x,y,lfn] = [xvals,yvals,lambda:ax.set_yscale('log')] if not flipxy else [yvals,xvals,lambda:ax.set_xscale('log')]
    ax.plot(x,y,'ko-',ms=2,mew=0,mfc='k')
    if flipxy:
        ax.set_ylim(refax.get_ylim())
        ax.set_xlabel(ylab)
        ax.set_ylabel(xlab)
    else:
        ax.set_xlim(refax.get_xlim())
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
    if logy: lfn()


def plot_spatial_spanhist(aRec,mybins,starts,stops,plot_symm=False,loctrfn=lambda x:x,cmap='jet',logc=False,lab_bottom='bottom electrode',lab_top='top electrode',**kwargs):
    labtag = ' (upper tri.)' if plot_symm else ''
    f,axarr = mpl.pyplot.subplots(3,4,figsize = (10,10),gridspec_kw={'width_ratios':[0.1,0.5,2,0.1],'height_ratios':[0.1,0.5,2][::-1]})
    ax = axarr[0,-2]
    #ax.set_aspect('equal')
    axy = axarr[0,1]
    axx = axarr[1,-2]
    laxx = axarr[-1,-2]
    laxy = axarr[0,0]
    cax = axarr[0,-1]

    if logc:
        tfn = lambda x:np.ma.log10(np.ma.masked_where(x==0,x))
        clabel = 'log10(counts)'
    else:
        tfn = lambda x:x
        clabel = 'counts'
    if tfn in kwargs:
        tfn = kwargs['tfn']
    H,im = plotmake_hist2d(ax,starts,stops,mybins,mybins,tfn=tfn,cmap=cmap)
    if plot_symm: _,im = plotmake_hist2d(ax,stops,starts,mybins,mybins,tfn=tfn,cmap=cmap)#make it both

    project_hist2d(axx,ax,H,1,mybins,tfn=lambda x:x,logy=False,flipxy=False,xlab='%s%s'%(lab_bottom,labtag))
    project_hist2d(axy,ax,H,0,mybins,tfn=lambda x:x,logy=False,flipxy=True,xlab='%s%s'%(lab_top,labtag))
    axy.set_ylabel(axy.get_ylabel(),labelpad=8)
    pb.make_locax(laxx, aRec.el_locs,ori='horizontal', linecol='grey', tha='center', textoff=-0.7,tva='top', cols=['k', 'lightgray'],rotation=45,
                      boundary_axes=[axx,ax],loctrfn=loctrfn)
    pb.make_locax(laxy, aRec.el_locs,ori='vertical', linecol='grey', tha='right', textoff=-0.1,tva='center', cols=['k', 'lightgray'],
                      boundary_axes=[axy,ax],loctrfn=loctrfn)
    laxx.set_xlim([mybins.min(),mybins.max()])
    laxy.set_ylim([mybins.min(),mybins.max()])
    axy.set_yticks([])
    axx.set_xticks([])
    cb = f.colorbar(im,cax=cax)
    cb.set_label(clabel,rotation=-90, labelpad=10)
    for ii in np.arange(3):
        for jj in np.arange(4):
            if ii>0 and jj!=2:
                axarr[ii,jj].set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    return f,ax


def make_hist_with_proj(ydata,xdata,ybins,xbins,cmap='jet',ylab='',xlab='',logc=False,**kwargs):
    f, axarr = mpl.pyplot.subplots(2, 3, figsize=(10, 10), gridspec_kw={'width_ratios': [0.5, 2, 0.1],
                                                                        'height_ratios': [0.5, 2][::-1]})
    ax = axarr[0, 1]
    #ax.set_aspect('equal')
    axy = axarr[0, 0]
    axx = axarr[1, 1]
    cax = axarr[0, 2]

    if logc:
        tfn = lambda x: np.ma.log10(np.ma.masked_where(x == 0, x))
        clabel = 'log10(counts)'
    else:
        tfn = lambda x: x
        clabel = 'counts'
    if tfn in kwargs:
        tfn = kwargs['tfn']
    H, im = plotmake_hist2d(ax, xdata, ydata, xbins, ybins, tfn=tfn, cmap=cmap)

    project_hist2d(axx, ax, H, 1, xbins, tfn=lambda x: x, logy=False, flipxy=False, xlab=xlab)
    project_hist2d(axy, ax, H, 0, ybins, tfn=lambda x: x, logy=False, flipxy=True, xlab=ylab)
    axy.set_ylabel(axy.get_ylabel(), labelpad=8)

    cb = f.colorbar(im, cax=cax)
    cb.set_label(clabel, rotation=-90, labelpad=10)


    axarr[1,0].set_axis_off()
    axarr[1,2].set_axis_off()

    ax.set_xticks([])
    ax.set_yticks([])

    return f, ax,axx,axy


def plot_anat_vs_other(aRec,anatbins,xbins,anatdata,xdata,logc=False,loctrfn=lambda x:x,cmap='jet',logy=False,ylab='',xlab='',**kwargs):

    if logc:
        tfn = lambda x: np.ma.log10(np.ma.masked_where(x == 0, x))
        clabel = 'log10(counts)'
    else:
        tfn = lambda x: x
        clabel = 'counts'
    if tfn in kwargs:
        tfn = kwargs['tfn']

    f,axarr = mpl.pyplot.subplots(2,4,figsize = (10,7.5),gridspec_kw={'width_ratios':[0.1,0.5,2,0.1],'height_ratios':[2,0.5]})
    f.subplots_adjust(hspace=0.1)
    ax = axarr[0,2]
    #ax.set_aspect('equal')
    axy = axarr[0,1]#here go the anatbals
    axx = axarr[1,2]
    locaxy = axarr[0,0]
    cax = axarr[0,3]
    empties = [[1,0],[1,1],[1,3]]

    H,im = plotmake_hist2d(ax,xdata,anatdata,xbins,anatbins,tfn=tfn,cmap=cmap)

    project_hist2d(axx,ax,H,1,xbins,tfn=lambda x:x,logy=False,flipxy=False,xlab=xlab)
    project_hist2d(axy,ax,H,0,anatbins,tfn=lambda x:x,logy=False,flipxy=True,xlab=ylab)
    if logy:
        for myax in [ax,axy]:myax.set_yscale('log')
        axy.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%i"))
        axy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
    axy.set_ylabel(axy.get_ylabel(),labelpad=90)
    axy.set_yticks([])
    pb.make_locax(locaxy, aRec.el_locs,ori='vertical', linecol='grey', tha='right', textoff=-0.1,tva='center', cols=['k', 'lightgray'],
                      boundary_axes=[axy,ax],loctrfn=loctrfn)
    locaxy.set_ylim([anatbins.min(),anatbins.max()])
    #axy.set_yticks([])
    axy.set_xticks([])
    cb = f.colorbar(im,cax=cax)
    cb.set_label(clabel,rotation=-90, labelpad=10)
    for ept in empties:
        axarr[ept[0],ept[1]].set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    return f,ax

def plot_size_vs_center(aRec,centbins,sizebins,cents,sizes,loctrfn=lambda x:x,cmap='jet',logy=False):
    f,axarr = mpl.pyplot.subplots(3,3,figsize = (10,10),gridspec_kw={'width_ratios':[0.5,2,0.1],'height_ratios':[0.1,0.5,2][::-1]})
    ax = axarr[0,-2]
    #ax.set_aspect('equal')
    axy = axarr[0,0]
    axx = axarr[1,-2]
    laxx = axarr[-1,-2]
    cax = axarr[0,-1]

    H,im = plotmake_hist2d(ax,cents,sizes,centbins,sizebins,tfn=lambda x:np.ma.log10(np.ma.masked_where(x==0,x)),cmap=cmap)

    project_hist2d(axx,ax,H,1,centbins,tfn=lambda x:x,logy=False,flipxy=False,xlab='center electrode')
    project_hist2d(axy,ax,H,0,sizebins,tfn=lambda x:x,logy=False,flipxy=True,xlab='event size')
    if logy:
        for myax in [ax,axy]:myax.set_yscale('log')
        axy.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%i"))
        axy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
    axy.set_ylabel(axy.get_ylabel(),labelpad=14)
    pb.make_locax(laxx, aRec.el_locs,ori='horizontal', linecol='grey', tha='center', textoff=-0.7,tva='top', cols=['k', 'lightgray'],rotation=45,
                      boundary_axes=[axx,ax],loctrfn=loctrfn)
    laxx.set_xlim([centbins.min(),centbins.max()])
    #axy.set_yticks([])
    axx.set_xticks([])
    cb = f.colorbar(im,cax=cax)
    cb.set_label('log10(counts)',rotation=-90, labelpad=10)
    for ii in np.arange(3):
        for jj in np.arange(3):
            if ii>0 and jj!=1:
                axarr[ii,jj].set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

    return f,ax

def find_next_inds(ii,SBjoint,ghost_times,maxdiff,searchwin=10):
    G1 = SBjoint[ii]
    c_inds = np.arange(1, searchwin) + ii
    cand_times = ghost_times[c_inds]
    close_enough = (cand_times - ghost_times[ii]) <= maxdiff
    is_opposite = np.array([SBjoint[cind][1] != G1[1] for cind in c_inds])
    is_connected = np.array(
        [check_olap(SBjoint[cind][0][1, np.array([0, -1])], G1[0][1, np.array([0, -1])]) for cind in c_inds])
    return c_inds[is_opposite & is_connected & close_enough]


def eval_interp(data,xnew,int_mode='continuous',trfn=lambda x:x,eval_fn=lambda x: np.sqrt(np.mean(x**2)),\
                          interfn=lambda x,y:interp1d(x, y,kind = 'linear', fill_value="extrapolate"),
                          verbose=False):

    trace = trfn(data)
    z_cross = np.where(np.diff(np.sign(trace)))[0]

    vals = np.zeros(len(z_cross) - 1)
    for zz in np.arange(len(z_cross) - 1):
        z1, z2 = z_cross[zz], z_cross[zz + 1]
        vals[zz] = eval_fn(trace[z1:z2])

    if int_mode == 'continuous':
        bvec = (z_cross[:-1] + np.diff(z_cross) / 2).astype(int)
        yvec = vals[:]
    elif int_mode == 'hist':
        bvec = np.r_[z_cross[0],z_cross[1:-1].repeat(2,0),z_cross[-1]]
        yvec = vals.repeat(2, 0)

    ifn = interfn(bvec, yvec)  # opting for continuous mode here...
    if verbose:
        return [z_cross,bvec,yvec,ifn(xnew)]
    else: return ifn(xnew)

def calc_centroid(contour):
    M = measure.moments_coords(contour)
    return (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

def get_centroids(contours):
    return np.array([calc_centroid(cont) for cont in contours])

def flip_select_contours(contours,minsize=0):
    return [contour[:,::-1] for contour in contours if contour[:,0].max()-contour[:,0].min()>= minsize]


def contours_to_rects(contours):
    #output n_conts x 2 x 2 [2nd dim 0: x-coords, dim1: y-coords]
    return np.vstack([np.vstack([cont.min(axis=0), cont.max(axis=0)]).T[None, :, :] for cont in contours])

def get_centroids_rect(rects):
    return rects.mean(axis=2)

def plot_fofthresh(threshvec,countvec,ylab='count',xlab='threshold [z]'):
    f, ax = mpl.pyplot.subplots(figsize=(4, 3))
    f.subplots_adjust(left=0.17,bottom=0.15)
    ax.plot(threshvec, countvec, 'k')
    ax.set_xlim([threshvec.min(), threshvec.max()])
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    f.tight_layout()
    return f


def plot_xcorr(ax,corrmat,lags,sr,elim,xlab='',cmap='jet',**kwargs):
    xext = np.array([lags.min(),lags.max()])/sr
    ax.imshow(corrmat, origin='lower', aspect='auto', extent=[*xext, *elim], cmap=cmap)
    ax.axvline(0,color='w',alpha=1)
    if 'refchan' in kwargs: ax.axhline(kwargs['refchan'],color='w',alpha=1)
    ax.set_xlabel(xlab)
    ax.set_ylim(elim)
    ax.set_xlim(xext)

def plot_cutout(ax,dmat,elim,tlim,tint,sr,col='k',speccol='r',ttype='realtime',**kwargs):
    chanvec = np.arange(elim[0], elim[1])
    if ttype == 'realtime':tvec = np.arange(tlim[0], tlim[1]) / sr +tint[0]
    elif ttype == 'zeroed': tvec = np.linspace(0,dmat.shape[1]/sr,dmat.shape[1])

    if 'refchan' in kwargs:
        cols = [col if chan!=kwargs['refchan'] else speccol for chan in chanvec]
    else:
        cols = len(chanvec)*[col]
    #print(dmat.shape,len(chanvec),len(cols))
    for ii, [chan,trace] in enumerate(zip(chanvec, dmat)):
        ax.plot(tvec, trace * 0.1 + chan, 'o-', ms=1.2,color=cols[ii] )
    ax.set_xlim([tvec.min(), tvec.max()])

def plotget_acorr(ax,trace,sr,fac=1,xlab='',plotdict={},**kwargs):
    acorr, lags = cfns.xcorr(trace, trace, maxlag=None)
    lags_ = lags/sr*fac
    ax.plot(lags_,acorr.real,**plotdict)
    if 'freqborders' in kwargs:
        flower,fupper = kwargs['freqborders']
        for xmin,xmax in zip([1/fupper,-1/flower],[1/flower,-1/fupper]):
            ax.axvspan(xmin*fac,xmax*fac,color='khaki',alpha=0.5,linewidth=0)
    ax.set_xlim([lags_.min(),lags_.max()])

    ax.set_xlabel(xlab)

def plot_analyze_patch(aRec,patch,dmat,dmat_bb,sr_ratio,tint,freqbounds):
    rawcol, rawselcol = 'grey', 'orange'
    filtcol, filtselcol = 'k', 'firebrick'

    tlim = (patch[0] * sr_ratio).astype(int)
    elim = (np.ceil(patch[1])).astype(int)
    bbmat = dmat_bb[elim[0]:elim[1], tlim[0]:tlim[1]]
    filtmat = dmat[elim[0]:elim[1], tlim[0]:tlim[1]]

    flower, fupper = freqbounds

    midchan_idx = int(np.diff(elim) / 2)
    midchan = elim[0] + midchan_idx

    reftrace = filtmat[midchan_idx]
    corrmat = np.zeros((filtmat.shape[0], filtmat.shape[1] * 2 - 1))
    for ii, data in enumerate(filtmat):
        ctemp, lags = cfns.xcorr(data, reftrace, maxlag=None)
        corrmat[ii] = ctemp.real

    f = mpl.pyplot.figure(figsize=(14, 9))

    f.subplots_adjust(left=0.1,right=0.99,bottom=0.05)

    gs = f.add_gridspec(2, 4)
    rax = f.add_subplot(gs[:, 0])
    fax = f.add_subplot(gs[:, 1])
    arax = f.add_subplot(gs[0, 2])
    afax = f.add_subplot(gs[1, 2])
    aax = f.add_subplot(gs[:, -1])
    plot_cutout(rax, bbmat, elim, tlim, tint, aRec.sr, col=rawcol, speccol=rawselcol,ttype='zeroed', refchan=midchan)
    rax.autoscale(enable=True, axis='y', tight=True)
    plot_cutout(fax, filtmat, elim, tlim, tint, aRec.sr, col=filtcol, speccol=filtselcol,ttype='zeroed', refchan=midchan)
    fax.set_ylim(rax.get_ylim())
    plotget_acorr(arax, bbmat[midchan_idx], aRec.sr, fac=1000, xlab='', plotdict=dict(color=rawselcol),
                     freqborders=[flower, fupper])
    plotget_acorr(afax, filtmat[midchan_idx], aRec.sr, fac=1000, xlab='lag [ms]', plotdict=dict(color=filtselcol),
                     freqborders=[flower, fupper])
    plot_xcorr(aax, corrmat, lags * 1000, aRec.sr, elim, xlab='lag [ms]', cmap='jet')

    for titlestr,myax in zip(['raw','filtered','acorr','','crosscorr'],[rax,fax,arax,afax,aax]):
        myax.set_title(titlestr)

    pos = rax.get_position()
    newwidth = 0.03
    locax = f.add_axes([pos.x0 - newwidth*2, pos.y0, newwidth, pos.height])  # l,b,w,h
    pb.make_locax(locax, aRec.el_locs[elim[0]:elim[1]], linecol='k', tha='right', textoff=-0.1, cols=['k', 'lightgray'],
                  boundary_axes=[aax])
    for myax in [rax,fax,aax]:
        myax.set_yticks([])
    for myax in [rax,fax]: myax.set_xlabel('time [s]')

    return f