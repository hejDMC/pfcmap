#CORRELATION FUNCTIONS
#1) MATLAB-STYLE CALCULATION of XCORR
#2) CORRSHIFT - ANALYSIS
#3) PLOTTING
#4) TRCCs
###--> please refer to doof>29_saveTheThetaShift.py for usage instructions

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


  
def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))
  
  
def myCorr(x,y, maxlag, plot=False):
    """This function takes ndimensional *data* array, computes the cross-correlation in the frequency domain
    and returns the cross-correlation function between [-*maxlag*:*maxlag*].
    !add a line on the +++++----- to -----++++++
  
    Parameters
    ----------
    data : numpy.ndarray
        This array contains the fft of each timeseries to be cross-correlated.
    maxlag : int
        This number defines the number of samples (N=2*maxlag + 1) of the CCF that will be returned.
  
    Returns
    -------
    CCF : numpy.ndarray
        The cross-correlation function between [-maxlag:maxlag]
"""
  
    data = np.vstack([scipy.fftpack.fft(x),scipy.fftpack.fft(y)])
    normalized = True
    allCpl = False
  
    maxlag = np.round(maxlag)
    #~ print "np.shape(data)",np.shape(data)
    if np.shape(data)[0] == 2:
        #~ print "2 matrix to correlate"
        if allCpl:
            # Skipped this unused part
            pass
        else:
            K = np.shape(data)[0]
            #couples de stations
            couples = np.concatenate((np.arange(0, K), K + np.arange(0, K)))
  
    Nt = np.shape(data)[1]
    Nc = 2 * Nt - 1
    Nfft = 2 ** nextpow2(Nc)
  
    # corr = scipy.fftpack.fft(data,int(Nfft),axis=1)
    corr = data
  
    if plot:
            plt.subplot(211)
            plt.plot(np.arange(len(corr[0])) * 0.05, np.abs(corr[0]))
            plt.subplot(212)
            plt.plot(np.arange(len(corr[1])) * 0.05, np.abs(corr[1]))
  
    corr = np.conj(corr[couples[0]]) * corr[couples[1]]
    corr = np.real(scipy.fftpack.ifft(corr)) / Nt
    corr = np.concatenate((corr[-Nt + 1:], corr[:Nt + 1]))
  
    if plot:
        plt.figure()
        plt.plot(corr)
  
    E = np.sqrt(np.mean(scipy.fftpack.ifft(data, axis=1) ** 2, axis=1))
    normFact = E[0] * E[1]
  
    if normalized:
        corr /= np.real(normFact)
  
    if maxlag != Nt:
        tcorr = np.arange(-Nt + 1, Nt)
        dN = np.where(np.abs(tcorr) <= maxlag)[0]
        corr = corr[dN]
  
    del data
    return corr


def xcorr_unnormed(X, Y, NFFT):
    '''fft-based cross-correlation, works like matlab xcorr with norm: coeff'''

    return np.fft.fftshift(np.fft.ifft(np.fft.fft(X, NFFT) * np.conj(np.fft.fft(Y, NFFT))))

def xcorr(X,Y,maxlag = None):
    '''fft-based cross-correlation, works like matlab xcorr with norm: coeff'''
    
    corrLength=len(X)+len(Y)-1;
    c=np.fft.fftshift(np.fft.ifft(np.fft.fft(X,corrLength)*np.conj(np.fft.fft(Y,corrLength))));

    cnorm = c/(np.linalg.norm(X)*np.linalg.norm(Y))#the coeff-type normalisation
    
    N = len(X)
    if maxlag == None:
        maxlag = N-1
    else:
        assert maxlag < N
        
    
    #return cnorm
    csnip = cnorm[int(len(cnorm)/2-maxlag):int((len(cnorm)+1)/2+maxlag)]
    lags = np.arange(-maxlag, maxlag+1)

    return csnip,lags
    

def xcorr_slow(x, y=None, maxlags=None, norm='biased'):
    
    """Cross-correlation using numpy.correlate
  
    Estimates the cross-correlation (and autocorrelation) sequence of a random
    process of length N. By default, there is no normalisation and the output
    sequence of the cross-correlation has a length 2*N+1.
  
    :param array x: first data array of length N
    :param array y: second data array of length N. If not specified, computes the
        autocorrelation.
    :param int maxlags: compute cross correlation between [-maxlags:maxlags]
        when maxlags is not specified, the range of lags is [-N+1:N-1].
    :param str option: normalisation in ['biased', 'unbiased', None, 'coeff']
  
    The true cross-correlation sequence is
  
    .. math:: r_{xy}[m] = E(x[n+m].y^*[n]) = E(x[n].y^*[n-m])
  
    However, in practice, only a finite segment of one realization of the
    infinite-length random process is available.
  
    The correlation is estimated using numpy.correlate(x,y,'full').
    Normalisation is handled by this function using the following cases:
  
        * 'biased': Biased estimate of the cross-correlation function
        * 'unbiased': Unbiased estimate of the cross-correlation function
        * 'coeff': Normalizes the sequence so the autocorrelations at zero
           lag is 1.0.
  
    :return:
        * a numpy.array containing the cross-correlation sequence (length 2*N-1)
        * lags vector
  
    .. note:: If x and y are not the same length, the shorter vector is
        zero-padded to the length of the longer vector.
  
    .. rubric:: Examples
  
    .. doctest::
        #
        # >>> from spectrum import *
        # >>> x = [1,2,3,4,5]
        # >>> c, l = xcorr(x,x, maxlags=0, norm='biased')
        # >>> c
        array([ 11.])
  
    .. seealso:: :func:`CORRELATION`. 
    """
    from pylab import rms_flat
    import numpy
    from numpy import arange
    
    N = len(x)
    if y == None:
        y = x
    assert len(x) == len(y), 'x and y must have the same length. Add zeros if needed'
    assert maxlags <= N, 'maxlags must be less than data length'
  
    if maxlags == None:
        maxlags = N-1
        lags = numpy.arange(0, 2*N-1)
    else:
        assert maxlags < N
        lags = numpy.arange(N-maxlags-1, N+maxlags)
  
    res = numpy.correlate(x, y, mode='full')
  
    if norm == 'biased':
        res = res[lags] / float(N)    # do not use /= !!
    elif norm == 'unbiased':
        res = res[lags] / (float(N)-abs(arange(-N+1, N)))[lags]
    elif norm == 'coeff':       
        rms = rms_flat(x) * rms_flat(y)
        res = res[lags] / rms / float(N)
    else:
        res = res[lags]
  
    lags = arange(-maxlags, maxlags+1)       
    return res, lags


def interpolate_trace(trace,sr_old=2.,sr_interp=1000.,kind='cubic'):

    from scipy.interpolate import interp1d

    tvec_old = np.linspace(0.,len(trace)/sr_old,len(trace))
    tvec_new = np.linspace(0.,len(trace)/sr_old,len(trace)*sr_interp/sr_old)
    f_interp = interp1d(tvec_old,trace, kind=kind)
    trace_interp = f_interp(tvec_new)
    return trace_interp,tvec_new

sinewave = lambda t,freq,shiftdeg: np.sin(2*np.pi*freq*t +shiftdeg*np.pi/180.)

def sinewave_freqmod(f1,f2,tvec,changemode='log',both=False):

    #t = np.linspace(0,1,n)
    t = tvec[:]
    dt = t[1] - t[0] # needed for integration
    # define desired logarithmic frequency sweep
    if changemode == 'log':f_inst = np.logspace(np.log10(f1), np.log10(f2), len(t))
    elif changemode == 'lin': f_inst = np.linspace(f1, f2, len(t))
    phi = 2 * np.pi * np.cumsum(f_inst) * dt # integrate to get phase
    if both:
        return np.sin(phi),f_inst
    else:
        return np.sin(phi)

def mod_sinewave(tvec,**kwargs):
    
    dt = tvec[1]-tvec[0]
    if kwargs.has_key('freq'):
        phi = 2*np.pi*kwargs['freq']*tvec 
        
    if kwargs.has_key('freqmod'):
        phi = 2*np.pi*np.cumsum(kwargs['freqmod'])*dt
        
    if kwargs.has_key('shift'):
        phi = phi+kwargs['shift']*np.pi/180.
        
    if kwargs.has_key('ampmod'):
        return np.sin(phi)*kwargs['ampmod']
    else:
        return np.sin(phi)

getMinima = lambda a: np.r_[False, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], False]
getMaxima = lambda a: np.r_[a[1:] < a[:-1],False] & np.r_[False,a[:-1] < a[1:]]

def get_shift(corr,lags,sr,mode='max',shiftwrap=False,both=True):
    if mode=='max': myfn = getMaxima
    elif mode=='min': myfn = getMinima
        
    extrema = myfn(corr)
    extr_shifts = lags[extrema]/sr
    if shiftwrap == True: 
        myshift = extr_shifts[np.abs(extr_shifts)==np.min(np.abs(extr_shifts))][0]
        
    else:    
        myshift = extr_shifts[np.where(extrema)[0]-(len(corr)/2.-1)>=0][0]
        
    if both == True:
        return myshift, np.abs(corr[lags/sr==myshift])[0]
    else:
        return myshift



def get_shiftmat(dg_freqs,dg_shifts,EC,maxlag,timevec,both = True):
    #timeshift_vec = np.zeros((len(dg_shifts)))
    shiftmat = np.zeros((len(dg_freqs),len(dg_shifts)))
    if both == True: ampmat = np.zeros((len(dg_freqs),len(dg_shifts)))
    sr = 1./(timevec[1]-timevec[0])
    for ff,freq in enumerate(dg_freqs):
        for ss,shift in enumerate(dg_shifts):
            DG = sinewave(timevec,freq,shift)
            [corr,lags] = xcorr(EC, DG, maxlag=int(maxlag))
            if both==True : shiftmat[ff,ss],ampmat[ff,ss] = get_shift(corr,lags,sr,mode='max',shiftwrap=True,both=True)
            else: shiftmat[ff,ss] = get_shift(corr,lags,sr,mode='max',shiftwrap=True,both=False)
    if both==True:
        return shiftmat,ampmat
    else:return shiftmat

def plot_shiftmat(shiftmat,dg_freqs,dg_shifts,refreq,mode='shift'):
    import matplotlib as mpl
    f = mpl.pyplot.figure(facecolor='w',figsize=(10,6))
    
    imax = f.add_axes([0.1,0.1,0.75,0.8])
    cax = f.add_axes([0.87,0.1,0.02,0.8])
    #imax = f.add_subplot(111)
    if mode=='shift': 
        myline,mylabel = 'k','Timeshift [ms]'
        im = imax.imshow(shiftmat*1000.,cmap='hsv',origin='lower',aspect='auto'\
                         ,interpolation= 'none',filternorm=None,alpha=0.5)
        
        
    elif mode=='amp': 
        myline,mylabel = 'r','Correlation at Timeshift'
    
        cmap=mpl.pyplot.get_cmap('gray')

        myticks = np.flipud(np.array([1.,0.5,0.1,0.05]))
        vmin = np.log10(myticks[0]) #Lower value
        vmax = np.log10(myticks[-1]) #Upper value
        bounds = np.linspace(vmin,vmax,100)
        
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
        im = imax.imshow(np.log10(shiftmat),cmap=cmap,origin='lower',aspect='auto'\
                         ,interpolation= 'none',filternorm=None,vmin=vmin,vmax=vmax)#scaleFactor
    
    im.set_extent([dg_shifts.min(),dg_shifts.max(),dg_freqs.min(),dg_freqs.max()])
    imax.vlines(0.,dg_freqs.min(),dg_freqs.max(),myline,linestyle='--',linewidth=1)
    imax.hlines(refreq,dg_shifts.min(),dg_shifts.max(),myline,linestyle='--',linewidth=1)
    
    if mode=='shift': cbar = plt.colorbar(im,cax=cax)
    elif mode=='amp':
        cbar = mpl.colorbar.ColorbarBase(cax, norm=norm,
                                   extend='min',
                                   cmap=cmap,ticks=np.log10(myticks))
        cbar.ax.set_yticklabels(myticks)
    
    cbar.ax.set_ylabel(mylabel,rotation=-90,labelpad=20)
    
    imax.set_xlabel('DG-Shift [Deg]')
    imax.set_ylabel('DG-Frequency [Hz]')
    imax.set_xlim([dg_shifts.min(),dg_shifts.max()])
    imax.set_ylim([dg_freqs.min(),dg_freqs.max()])
    return f

def diagnose_correlation_plot(DG,EC,timevec,maxlag,sr,shiftwrap=True):
    from matplotlib.pyplot import figure
    f = figure(facecolor='w',figsize=(8,8))
    f.subplots_adjust(left=0.13,right=0.97,hspace=0.4)
    ax = f.add_subplot(211)
    l1 = ax.plot(timevec,EC,'grey',lw=2,label='EC',alpha=0.75)
    l2 = ax.plot(timevec,DG,'DarkOrange',lw=2,label='DG',alpha=0.75)
    ax.legend()
    ax.set_xlim([0.,2.])
    ax.set_ylim([-1.2,1.2])
    ax.set_yticks([1,0,1])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    
    ax2 = f.add_subplot(212)
    
    [corr,lags] = xcorr(EC, DG, maxlag=int(maxlag))
    corrshift = get_shift(corr,lags,sr,mode='max',shiftwrap=True,both=False)
    ax2.set_title('Correlation-Shift: %d ms'%(corrshift*1000.),fontsize=15)
    maxima = getMaxima(corr)
    minima = getMinima(corr)
    
    ax2.plot(lags/sr*1000, corr,'k',linewidth=2)
    ax2.plot(lags[maxima]/sr*1000.,corr[maxima],'ro',markeredgecolor='none',markersize=10)
    ax2.plot(lags[minima]/sr*1000.,corr[minima],'bo',markeredgecolor='none',markersize=10)
    corramp = np.abs(np.max(corr)-np.min(corr))
    myylim = [np.min(corr)-0.1*corramp,np.max(np.max(corr)+0.1*corramp)]#ax2.get_ylim()
    ax2.vlines(corrshift*1000.,myylim[0],myylim[1],'r',linewidth=2)
    ax2.set_xlim([-maxlag/sr*1000-20,maxlag/sr*1000+20])
    ax2.set_ylim(myylim)
    ax2.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
    ax2.set_xlabel('Lag [ms]')
    ax2.set_ylabel('Correlation')
    return f


def calc_trcc(EC,DG,ww,step,maxlag,sr):
    wincents = np.arange(ww/2.,len(DG)-ww/2.+0.1,step)
    trcc_mat = np.zeros((len(wincents),int(2*maxlag+1)))
    shift_amp_vec = np.zeros((len(wincents),2))
    for ii,cent in enumerate(wincents):
        dg = DG[int(cent-ww/2.):int(cent+ww/2.)]
        ec = EC[int(cent-ww/2.):int(cent+ww/2.)]
        [corr,lags] = xcorr(ec,dg, maxlag=int(maxlag))
        shift_amp_vec[ii] = get_shift(corr,lags,sr,mode='max',shiftwrap=True)
        trcc_mat[ii] = np.real(corr)
    return trcc_mat,shift_amp_vec


def plot_trcc(trcc,wincents_s,lags_s,shiftvec,**kwargs):
    from matplotlib.pyplot import figure
    
    kwarglist = kwargs.keys()
    if ('ampmod' in kwarglist) and ('freqmod' in kwarglist):
        f = figure(facecolor='w',figsize=(11,6))
        startx,starty,leny = 0.08,0.1,0.8
        wside,wtab = 0.1,0.01
        imax = f.add_axes([startx+2*(wside+wtab),starty,0.6,leny])
        #cax = 
        fmax = f.add_axes([startx,starty,wside,leny]) 
        amax = f.add_axes([startx+wside+wtab,starty,wside,leny])        
        
        
    elif ('ampmod' in kwarglist) and (not 'freqmod' in kwarglist): 
        f = figure(facecolor='w',figsize=(9,6))
        startx,starty,leny = 0.1,0.1,0.8
        wside,wtab = 0.1,0.01
        imax = f.add_axes([startx+(wside+wtab),starty,0.66,leny])
        amax = f.add_axes([startx,starty,wside,leny]) 
       
    elif ('freqmod' in kwarglist) and (not 'ampmod' in kwarglist):
        f = figure(facecolor='w',figsize=(9,6))
        startx,starty,leny = 0.1,0.1,0.8
        wside,wtab = 0.1,0.01
        imax = f.add_axes([startx+(wside+wtab),starty,0.66,leny])
        fmax = f.add_axes([startx,starty,wside,leny]) 
        
    else:
        f = figure(facecolor='w',figsize=(8,6))
        imax = f.add_axes([0.1,0.1,0.75,0.8])
    
    if 'shift' in kwarglist: f.text(0.3,0.92,r'$\Delta(\phi):\hspace{0.5} %d^{\circ} $' %(kwargs['shift']),fontsize=20)
       
    im = imax.imshow(trcc,cmap='jet',origin='lower',aspect='auto'\
                             ,interpolation= 'none',filternorm=None)
    imax.plot(shiftvec,wincents_s,'w',lw=2)
    imax.vlines(0.,wincents_s.min(),wincents_s.max(),color='grey',lw=2,linestyle='--')
    
    im.set_extent([lags_s.min(),lags_s.max(),wincents_s.min(),wincents_s.max()])
    imax.set_xlabel('Lag [s]')
    if (not 'ampmod' in kwarglist) and (not 'freqmod' in kwarglist): imax.set_ylabel('Time [s]')
    
    
    ax2 = imax.twinx()
    meanmat = np.mean(trcc,axis=0)
    stdmat = np.std(trcc,axis=0)
    ax2.fill_between(lags_s, meanmat-stdmat, meanmat+stdmat,color='w',alpha=0.25)
    ax2.plot(lags_s,meanmat,'k',lw=2)
    ax2.plot(lags_s,meanmat-stdmat,'grey',alpha=0.7)
    ax2.plot(lags_s,meanmat+stdmat,'grey',alpha=0.7)
    ax2.set_xlim([lags_s.min(),lags_s.max()])
    ax2.set_ylabel(r'$\bar{\rho}$',fontsize=30,rotation=-90.,labelpad=20)
    ax2.set_ylim([-1.,1.])
    
    if 'freqmod' in kwarglist:
        timevec,freqmod = kwargs['timevec'], kwargs['freqmod']
        fmax.plot(freqmod,timevec,'k',lw=2)
        fmax.set_ylabel('Time [s]')
        fmax.set_title('FM [Hz]',fontsize=16,fontweight='bold')
        imax.set_yticks([])
        fmax.spines['right'].set_visible(False)
        fmax.spines['top'].set_visible(False)
        fmax.yaxis.set_ticks_position('left')
        fmax.xaxis.set_ticks_position('bottom')
        if freqmod.min()==freqmod.max():fmax.set_xticks([freqmod.min()])
        else: fmax.set_xticks([freqmod.min(),freqmod.max()])
        
        
    if 'ampmod' in kwarglist:
        timevec,ampmod = kwargs['timevec'],kwargs['ampmod']
        amax.plot(ampmod,timevec,'k',lw=2)
        amax.set_title('AM',fontsize=16,fontweight='bold')
        imax.set_yticks([])
        amax.spines['right'].set_visible(False)
        amax.spines['top'].set_visible(False)
        amax.xaxis.set_ticks_position('bottom')
        amax.set_xlim([ampmod.min()-0.1,ampmod.max()+0.1])
        if ampmod.min()==ampmod.max():amax.set_xticks([ampmod.min()])
        else: amax.set_xticks([ampmod.min(),ampmod.max()])
        
        if not 'freqmod' in kwarglist: 
            amax.set_ylabel('Time [s]')
            amax.yaxis.set_ticks_position('left')
        else:
            amax.spines['left'].set_visible(False)
            amax.set_yticks([])

    return f




