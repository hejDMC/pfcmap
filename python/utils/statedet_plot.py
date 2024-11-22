from matplotlib import pyplot as plt
from pfcmap.python.utils import plotting_basics as pb
from SPOOCs.utils import helpers as sph
from scipy.stats import spearmanr
import numpy as np

def plotcompare_detmethods(detdict,tags,xvec,recdur,**kwargs):

    npans = len(detdict)+1 if 'emg' in kwargs else len(detdict)

    f,axarr = plt.subplots(npans,1,figsize=(16,6),sharex=True)
    f.subplots_adjust(left=0.07, right=0.93)
    for tag,ax in zip(tags,axarr[:len(detdict)]):
        ax.text(0.01,0.98,tag,ha='left',va='top',transform=ax.transAxes,fontsize=12)
        ax.set_ylabel('spikes [Hz]')

        ax.plot(xvec, detdict[tag]['ratevec'], 'k')
        ax.axhline(detdict[tag]['g_mean'], color='grey', alpha=0.5, zorder=-10)
        ax.axhline(detdict[tag]['thr'], color='grey', alpha=1, zorder=-10, linestyle='--')

        troughtimes = xvec[detdict[tag]['trtimes']]
        supertimes = xvec[detdict[tag]['sbursts']]#*bw
        bordertimes = xvec[detdict[tag]['borderpts']]#*bw

        ax.hlines(np.zeros(troughtimes.shape[0]) + detdict[tag]['g_mean'], troughtimes[:, 0], troughtimes[:, 1], color='r', lw=2)
        for trtime in troughtimes:
            ax.axvspan(trtime[0],trtime[1],color='purple',alpha=0.3,linewidth=0,zorder=-10)
        for supertime in supertimes:
            ax.axvspan(supertime[0],supertime[1],color='orchid',alpha=0.2,linewidth=0,zorder=-12)
        for bordert in bordertimes:
            ax.axvspan(bordert[0],bordert[1],color='grey',alpha=0.3,linewidth=0,zorder=-13)

        offrate,p_excl = troughtimes.shape[0]/recdur*60.,np.sum(np.diff(bordertimes))/recdur*100
        mystr = 'off rate: %i/min, excluded: %i%%'%(offrate,p_excl)
        ax.text(0.5,1.01,mystr,ha='center',va='bottom',transform=ax.transAxes,fontsize=10,color='k')
        if 'artmat' in kwargs:
            artmat = kwargs['artmat']
            ax.hlines(np.zeros(artmat.shape[0])-2,artmat[:,0],artmat[:,1],color='darkorange',linewidth=4)
    ax.set_xlim([0,recdur])
    if 'emg' in kwargs:
        emg_tvec,emg_data = [kwargs['emg'][key] for key in ['tvec','data']]
        emax = axarr[-1]
        emax.plot(emg_tvec,emg_data,color='k')
        emax.set_ylabel('emg')
    axarr[-1].set_xlabel('time [s]')
    return f,axarr

def plot_spectrum(freqvec,tags,spectdict,cdict,**kwargs):
    f, ax = plt.subplots(figsize=(4,3))
    f.subplots_adjust(left=0.25,bottom=0.2)
    if 'xlim' in kwargs:
        myx = kwargs['xlim']
        cond = (freqvec>=myx[0]) & (freqvec<=myx[1])
    else:
        cond = np.ones(len(freqvec),dtype=int)

    for tt,tag in enumerate(tags):
        col = cdict[tag]
        ax.fill_between(freqvec[cond], spectdict[tag][0][cond], spectdict[tag][2][cond], color=col, alpha=0.4, linewidth=0)
        ax.plot(freqvec[cond], spectdict[tag][1][cond], color=col, linewidth=2)
        if 'durdict' in kwargs:
            mystr = '%s %1.1fmin'%(tag,kwargs['durdict'][tag]/60.)
        else:
            mystr = str(tag)
        ax.text(0.99,0.98-tt*0.1,mystr,color=col,va='top',ha='right',transform=ax.transAxes,fontsize=10)

    # ax.set_xlim([1,12])
    ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power [muV**2/Hz]')
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    return f,ax

def plot_lfp_ampfeatures(tags,ampvardict,cdict,xlab='mean rms [muV]',ylab='std rms [muV]'):
    f, ax = plt.subplots(figsize=(3,3))
    f.subplots_adjust(left=0.2,bottom=0.2)
    for tt, tag in enumerate(tags):
        col = cdict[tag]
        ax.text(0.99,0.98-tt*0.1,tag,color=col,va='top',ha='right',transform=ax.transAxes,fontsize=10)
        ax.plot(ampvardict[tag][0], ampvardict[tag][1], 'o', mec=col, alpha=0.3, mfc='none')
        ax.axvline(np.median(ampvardict[tag][0]), color=col, alpha=0.5,zorder=-10)
        ax.axhline(np.median(ampvardict[tag][1]), color=col, alpha=0.5,zorder=-11)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return f,ax

def plotcompare_hist(valdict,tags,cdict,bins=50,xlab = 'values',htype='count'):

    f, ax = plt.subplots(figsize=(4,3))
    f.subplots_adjust(left=0.22,bottom=0.22)
    for tt, tag in enumerate(tags):
        col = cdict[tag]
        ax.text(0.99,0.98-tt*0.1,tag,color=col,va='top',ha='right',transform=ax.transAxes,fontsize=10)

        ax.set_ylabel(htype)
        ax.set_xlabel(xlab)
        ax.axvline(np.mean(valdict[tag]), color=col)

        if htype == 'probability':
            weights = np.ones_like(valdict[tag]) / len(valdict[tag])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        else:
            weights = None
        ax.hist(valdict[tag], bins, color=col, histtype='step',
                            weights=weights, linewidth=2)

        #if htype == 'probability':
        #    ax.get_yaxis().get_offset_text().set_position((ax.get_xlim()[0],ax.get_ylim()[1]))

    return f,ax

def plot_hist(vals,bins=50,col='k',htype='count',xlab='values',return_bins=False,hlinecol='same'):
    f, ax = plt.subplots(figsize=(4,3))
    f.subplots_adjust(left=0.22,bottom=0.22)


    ax.set_ylabel(htype)
    ax.set_xlabel(xlab)

    if hlinecol == 'same':
        c_hline = str(col)
    else: c_hline = hlinecol
    ax.axvline(np.mean(vals), color=c_hline,zorder=-20)

    if htype == 'probability':
        weights = np.ones_like(vals) / len(vals)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    else:
        weights = None
    _,mybins,_ = ax.hist(vals, bins, color=col, histtype='step',
                        weights=weights, linewidth=2,alpha=0.8)
    if return_bins:
        return f,ax,mybins
    else:
        return f,ax


def plot_timecourse_recording(bordertimes,n_troughs,stimplotfn,recdur,**kwargs):
    fac = 1 / 60.
    offrate, p_excl = n_troughs / recdur * 60., np.sum(np.diff(bordertimes)) / recdur* 100
    mystr = 'off-events: %i/min, excluded: %i%%' % (offrate, p_excl)
    f, axarr = plt.subplots(2, 1, figsize=(6.5, 1.5), gridspec_kw={'height_ratios': [0.3, 1]})
    f.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.4, hspace=0.1)
    stax, ax = axarr

    stimplotfn([stax])
    stax.set_xticks([])
    stax.set_xlim([0, recdur])

    for bordert in bordertimes * fac:
        ax.axvspan(bordert[0], bordert[1], color='purple', alpha=0.5, linewidth=0, zorder=2)

    if 'artmat' in kwargs:
        #aRec.artblockmat.T
        if np.size(kwargs['artmat'])>0:
            for myart in kwargs['artmat'] * fac:
                ax.axvspan(myart[0], myart[1], color='grey', alpha=0.3, zorder=-10, linewidth=0)
    ax.set_xlabel('time [min]')
    ax.set_xlim([0, recdur * fac])
    for myax in [stax, ax]: myax.set_yticks([])
    return f,axarr,mystr



def randchoice_intervals_of_interest(supertimes,n_per_super,recdur,n_ex_big=5,n_ex_small=3,n_bigsmall_thr=3,\
                                     windur=10.,outlap_dur=2):


    n_bigs = np.min([np.sum(n_per_super > n_bigsmall_thr), n_ex_big])
    n_small = np.min([np.sum(n_per_super <= n_bigsmall_thr), n_ex_small])

    inds_bigs = np.sort(
        np.random.choice(np.arange(len(supertimes))[n_per_super > n_bigsmall_thr], n_bigs, replace=False))
    bigpos_idx = np.random.choice([0, 1], n_bigs)  # whether to look at beginning or end
    inds_smalls = np.sort(
        np.random.choice(np.arange(len(supertimes))[n_per_super <= n_bigsmall_thr], n_small, replace=False))


    bigbords = supertimes[inds_bigs, bigpos_idx]
    big_starts = (bigbords - np.array([-outlap_dur, -(windur - outlap_dur)])[bigpos_idx]).clip(0, recdur - windur)

    small_starts = supertimes[inds_smalls, 0] + np.diff(supertimes[inds_smalls])[:,
                                                0] / 2. - windur / 2  # centering on smalls

    return {'quiet_long': np.vstack([big_starts, big_starts + windur]).T, \
                'quiet_short': np.vstack([small_starts, small_starts + windur]).T}


def multiplot_detection(aRec,tracedict,tvec_rates,raw_ratevec,detdict,cdict,locdict,stimplotfn,wildcard=True,\
                        ratecol='cornflowerblue',thrcol='grey',detcol='purple',\
                        sburstcol='orchid',bordcol='thistle',**kwargs):

    if 'bw' in kwargs:
        bw = kwargs['bw']
        troughtimes = detdict['trtimes']*bw
        supertimes = detdict['sbursts']*bw
        bordertimes = detdict['borderpts']*bw

    else:
        troughtimes = tvec_rates[detdict['trtimes']]
        supertimes = tvec_rates[detdict['sbursts']]
        bordertimes = tvec_rates[detdict['borderpts']]


    npans,hrats = [5,[0.1,1,1,0.5,0.5]] if wildcard else [4,[0.1,1,0.5,0.5]]
    f,axarr = plt.subplots(npans,figsize=(16,10),gridspec_kw={'height_ratios':hrats},sharex=True)
    f.subplots_adjust(left=0.07, right=0.93,bottom=0.06,top=0.95,hspace=0.05)
    stax,lax = axarr[:2]
    muax, emax = axarr[-2:]
    #stimulus
    stimplotfn([stax])
    stax.set_yticks([])
    #lfp
    if len(tracedict)>0:
        for cc, chan in enumerate(tracedict.keys()):
            color = cdict[chan]
            offset = cc * 110
            lax.plot(aRec.tvec, tracedict[chan] + offset, color=color)
            #rms_ax.plot(tvec_lfp_rms, rms_lfp[chan] + offset, color=color)
            #rms_ax.plot(aRec.tvec, slowband_lfp[chan] + offset, color=color)

        lax.hlines(np.zeros(aRec.artmat.shape[0])-200,aRec.artmat[:,0],aRec.artmat[:,1],color='k',linewidth=4)
        lax.set_ylabel('lfp [muV]')

    muax.plot(tvec_rates,raw_ratevec,'k')
    muax.axhline(detdict['g_mean'], color=thrcol, alpha=0.5, zorder=-10)
    muax.axhline(detdict['thr'], color=thrcol, alpha=1, zorder=-10, linestyle='--')
    if 'ratevec' in detdict: muax.plot(tvec_rates, detdict['ratevec'], ratecol)#thats the filtered one!
    muax.hlines(np.zeros(troughtimes.shape[0]) + detdict['g_mean'], troughtimes[:, 0], troughtimes[:, 1], color=detcol, lw=2,zorder=30)

    muax.set_ylabel('spikes [Hz]')

    #emg
    emax.plot(aRec.emg_tvec, aRec.emg_data, color='k')
    emax.set_ylabel('emg')
    emax.set_xlabel('time [s]')
    emax.set_xlim([0, aRec.dur])


    for myax in [lax,muax,emax]:
        for trtime in troughtimes:
            myax.axvspan(trtime[0],trtime[1],color=detcol,alpha=0.3,linewidth=0,zorder=-10)
        for supertime in supertimes:
            myax.axvspan(supertime[0],supertime[1],color=sburstcol,alpha=0.2,linewidth=0,zorder=-12)
        for bordert in bordertimes:
            myax.axvspan(bordert[0],bordert[1],color=bordcol,alpha=0.3,linewidth=0,zorder=-13)

    N = len(locdict)
    for cc, [chan, chanloc] in enumerate(locdict.items()):
        chancol = cdict[chan]
        f.text(0.99, 0.85 + (cc - N) * 0.03, chanloc, ha='right', va='top', color=chancol,
               fontweight='bold', fontsize=10)

    return f,axarr



def prepare_locpanel(f,ax,aRec,**kwargs):
    # spax = axarr[-2]
    myellocs = kwargs['ellocs'] if 'ellocs' in kwargs else aRec.ellocs_all
    pos = ax.get_position()
    locax = f.add_axes([pos.x0 - 0.015, pos.y0, 0.01, pos.height])
    ax.set_yticks([])
    ax.set_ylim([0 - 0.5, len(myellocs) - 0.5])
    pb.make_locax(locax,myellocs, cols=['k', 'lightgray'], linecol='grey')
    ax.get_shared_y_axes().join(ax, locax)


def plot_rasterpanel(ax,aRec,allspikes,tsel,yscale_axes=[],yscale_buffers=[]):
    rmat = sph.get_rastermat_splitchans(aRec, allspikes, tsel, pos_adj_fac=1)
    rhand = ax.plot(rmat[:, 0], rmat[:, 1], '.k', ms=1.5)

    if len(yscale_axes)>0:
        t0, t1 = (np.array(tsel) * aRec.sr).astype(int)
        maxs, mins = 0, 0
        for myax,scalebuff in zip(yscale_axes,yscale_buffers):
            for myline in myax.get_lines():
                try:
                    showsnip = myline.get_data()[1][t0:t1]
                    maxs = np.max([maxs, showsnip.max()])
                    mins = np.min([mins, showsnip.min()])
                except:
                    pass
            myax.set_ylim([mins - scalebuff, maxs + scalebuff])

    ax.set_xlim(tsel)


def make_corrstrs(xvals,yvals):
    corrc = np.corrcoef(xvals,yvals)[0,1]
    spear = spearmanr(xvals, yvals).correlation
    return 'CC: %1.2f, Spear: %1.3f, N=%i'%(corrc,spear,len(xvals))


def plot_corrpanel(xvals,yvals,xlab,ylab,xcol='k',ycol='k'):
    f,ax = plt.subplots(figsize=(3.2,3.2))
    f.subplots_adjust(left=0.22,top=0.87,bottom=0.17,right=0.95)
    ax.set_title(make_corrstrs(xvals,yvals),fontsize=10)
    ax.plot(xvals,yvals,'o',mec='k',mfc='none',alpha=0.5)
    ax.set_xlabel(xlab,color=xcol)
    ax.set_ylabel(ylab,color=ycol)
    if not ycol == 'k':
        [t.set_color(ycol) for t in ax.yaxis.get_ticklines()]
        [t.set_color(ycol) for t in ax.yaxis.get_ticklabels()]
    if not xcol == 'k':
        [t.set_color(xcol) for t in ax.xaxis.get_ticklines()]
        [t.set_color(xcol) for t in ax.xaxis.get_ticklabels()]
    return f,ax

def plot_corrpanel_from_vardict(vardict,var_x,var_y,col_on=True,**kwargs):
    if 'cond' in kwargs:
        xvals,yvals = [vardict[var]['vals'][kwargs['cond']] for var in [var_x,var_y]]
    else:
        xvals,yvals = [vardict[var]['vals'] for var in [var_x,var_y]]

    xlab,ylab = [vardict[var]['lab'] for var in [var_x,var_y]]
    if col_on:
        xcol,ycol = [vardict[var]['color'] for var in [var_x,var_y]]
    else:
        xcol,ycol = 'k','k'
    f,ax = plot_corrpanel(xvals,yvals,xlab,ylab,xcol=xcol,ycol=ycol)
    return f,ax