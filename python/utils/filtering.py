import numpy as np
from scipy.signal import resample
from scipy.signal import butter, filtfilt,lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, np.squeeze(data))
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):

    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


get_pow2 = lambda myint: 2**myint.bit_length()

def smartwins(datalen,winpts,overfrac=1/6.,pow2=True):
    ww = 2**winpts.bit_length() if pow2 else int(winpts)
    overest = int(ww*overfrac)
    overlap = overest+1 if np.mod(overest,2) else int(overest)#make the overlap divisible by 2
    nwins = (datalen-ww)//(ww-overlap)+1
    winstarts = np.arange(0,nwins*ww,ww)
    winstarts = winstarts[:] if overfrac == 0 else winstarts-np.arange(0.,overlap*nwins,overlap)
    winarray = np.vstack([winstarts,winstarts+ww])
    lastdiff = datalen-winarray[1][-1]
    last_ww = 2*ww if lastdiff>(ww-overfrac) else int(ww)
    lastwin = np.array([datalen-last_ww,datalen])
    winarray = np.hstack([winarray,lastwin[:,None]]).astype(int)
    if (winarray[0][-1]-winarray[1][-2])>0:print ('WARNING: last window doesnt overlap')
    return winarray.T


def resample_portions(data, winarray, sr, new_sr):
	'''winarray is n x 2: start,stop of datacutout in points'''

	rate_ratio = sr / new_sr

	# tlist = []
	logger.debug('resampling subwindows')
	resampled_list = []
	for start, stop in winarray:
		# print start,stop
		snip = data[start:stop]
		sniplen = stop - start
		resampled = resample(snip, int(sniplen / rate_ratio))
		resampled_list.append(np.squeeze(resampled))
	# tlist.append(np.linspace(start/sr,stop/sr,len(resampled)))

	# now fuse together
	logger.debug('fusing overlapping windows')
	overlap = winarray[0][1] - winarray[1][0]
	ww = np.diff(winarray[0])
	logger.debug('ww: %1.2f, overlap: %1.2f ' % (ww, overlap))
	nwins = winarray.shape[0]
	firstind = np.array([0, float(ww - overlap * 0.5)])
	indbody = np.tile(np.array([float(0.5 * overlap), float(ww - 0.5 * overlap)]), (nwins - 2, 1))
	inds = (np.vstack([firstind, indbody]) * new_sr / sr).astype(int)
	fused0 = np.hstack([resampled_list[ii][start:stop] for ii, [start, stop] in enumerate(inds)])
	missing_pts = int(winarray[-1][1] * new_sr / sr - len(fused0))
	fused = np.r_[fused0, resampled_list[-1][-missing_pts:]]

	# check whether durations match
	datadur = len(data) / sr
	fuseddur = len(fused) / new_sr
	if np.isclose(datadur, fuseddur, atol=0.1):
		logger.debug('durations match %1.2f' % (datadur))
	else:
		logger.error('duration mismatch raw:%1.2f vs resampled:%1.2f' % (datadur, fuseddur))

	return fused