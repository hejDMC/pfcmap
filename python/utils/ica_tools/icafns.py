from .. import plotting_basics as pb
from .. import data_classes as dc
from .. import data_handling as dh
from sklearn.decomposition import FastICA
from scipy.stats import scoreatpercentile
import numpy as np
import matplotlib as mpl


def storedprop(fn):
	attr_name = '_' + fn.__name__

	@property
	def _lazyprop(self):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, fn(self))
		return getattr(self, attr_name)
	return _lazyprop

class ICA(object):
	def __init__(self,datamat,id='',**kwargs):
		'''datamat is in ptsxchans'''
		self.X = datamat
		self.id = id
		self.ncomps = kwargs['ncomps'] if 'ncomps' in kwargs else self.X.shape[1]
		self.rs = kwargs['rs'] if 'rs' in kwargs else self.X.shape[1]

	@storedprop
	def Xwhite(self):
		return (self.X - self.X.mean(axis=0))/self.X.std(axis=0)

	@storedprop
	def sources(self):
		print ('Transforming ...')
		if not hasattr(self,'ica'):
			self.make()
		return self.ica.fit_transform(self.Xwhite)

	def make(self,rs=None):
		self.ica = FastICA(n_components=self.ncomps,whiten=True,random_state=rs) #init X.shape[1]


	def set_filtinds(self,filtinds):
		self.filtinds = filtinds

	@property
	def recon(self):
		if not hasattr(self,'_sources'):self.sources
		if hasattr(self,'filtinds'):
			print('filtering before reconstructing')
			reduced = np.zeros_like(self.ica.mixing_)
			reduced[:,self.filtinds] = self.ica.mixing_[:,self.filtinds]
			return  np.dot(self.sources,reduced.T)
		else:
			return np.dot(self.sources,self.ica.mixing_.T)

	@property
	def reconS(self):
		return self.recon*self.X.std(axis=0)+self.X.mean(axis=0)

class Splitter(object):
	def __init__(self,datalist,timesplits,sr,tbound,timeaxis=1):
		#self.datalist = datalist
		self.timesplits = timesplits
		self.sr = sr
		self.tbound = tbound
		self.splitinds = np.cumsum([data.shape[timeaxis] for data in datalist])[:-1]

	def maskmerge(self,dlist,tax=1):
		return dh.maskmerge_datalist(dlist,self.tbound,self.timesplits,tax=tax,sr=self.sr)

	def split_mat(self,data,tax=1):
		return np.split(data,self.splitinds,axis=tax)


	def splitmerge(self,data,tax=1):
		dlist = self.split_mat(data,tax=tax)
		return self.maskmerge(dlist,tax=tax)



def apply_trialwise(aRec,ttypes,datamat,fn,margin=[0.,0.]):
	powdict = {}
	for my_ttype in ttypes:
		aTC = dc.TrialCollection(my_ttype,parentobj=aRec)
		tmat = aTC.ttmat + np.array(margin)
		tpts = (tmat*aRec.sr).astype(int)
		srcstack = np.array([datamat[:,tpt[0]:tpt[1]] for tpt in tpts])
		powdict[my_ttype] = fn(srcstack,2)
	return powdict


def plot_singlesource_vals(valvec,cmapstr='tab10',ylab='vals',**kwargs):
	ncomps = len(valvec)
	ica_cols = pb.get_colors(cmapstr, ncomps, **kwargs)

	f, ax = mpl.pyplot.subplots(figsize=(5, 4))
	f.subplots_adjust(left=0.17, bottom=0.17)
	colors = ica_cols[:, :3]
	xvec = np.arange(ncomps) + 1
	ax.scatter(xvec, valvec, c=colors)
	ax.set_xticks(xvec)
	ax.set_xticklabels(list(xvec.astype(str)))
	ax.set_xlabel('source')
	ax.set_ylabel(ylab)
	for xtick, color in zip(ax.get_xticklabels(), colors):
		xtick.set_color(color)
		xtick.set_fontweight('bold')
	return f

def plot_ica_and_data(aRec, data, els,tminmax,stimplotfn=None,cmapstr='Dark2_r',mode='input',reconcol='k',**kwargs):
	'''data is chans (resp. icas)xtime '''
	cmap =  mpl.cm.get_cmap(cmapstr).colors

	cond = (aRec.tvec <= tminmax[1]) & (aRec.tvec >= tminmax[0])
	cond2 = (aRec.emg_tvec <= tminmax[1]) & (aRec.emg_tvec >= tminmax[0])

	plmat = data[:, cond]

	if 'recon_mat' in kwargs:
		recon_mat = kwargs['recon_mat'][:,cond]
		#return recon_mat

	alpha = 1
	spacefac = plmat.std()*5
	ylim_emg = [-20, 20]

	if mode == 'input':
		cdict = pb.get_colordict_electrodes(aRec,els,cmapstr)

	elif mode == 'ica':
		mycmap = pb.concat_cmaps(cmap,data.shape[0])

	f, axarr = mpl.pyplot.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [0.1, 1, 0.15]},
							sharex=True)
	f.subplots_adjust(top=0.95, bottom=0.05, left=0.05, hspace=0.1)
	stax, dax, emax = axarr
	dax.set_xlim(tminmax)
	trans = mpl.transforms.blended_transform_factory(dax.transAxes, dax.transData)

	lastloc = 'na'
	for cc in np.arange(len(els)):
		# alpha = 1 if cc in my_eois else 0.5
		myel = els[cc]
		yplace = cc
		if mode == 'input':
			loc = np.array(aRec.el_locs)[myel == aRec.el_ids][0]
			col = cdict[loc]
		else:
			yplace = len(els)-cc
			col = mycmap[cc]
			loc = 'src%i'%(cc+1)
		dax.plot(aRec.tvec[cond], plmat[cc] + yplace * spacefac, color=col, alpha=alpha)
		if 'recon_mat' in locals():
			#print ('jey')
			#print(recon_mat[cc])
			dax.plot(aRec.tvec[cond], recon_mat[cc] + yplace * spacefac, color=reconcol, alpha=0.8,lw=0.5,zorder=5)
		if loc != lastloc:
			dax.text(1.01, yplace * spacefac, loc, color=col, ha='left', va='center', fontweight='bold', transform=trans)
		lastloc = loc

	if not mode == 'ica':
		pb.plot_scalebar(dax,200,mode='y',unit='mV',ha='left',va='bottom',ta='right')

	#scalebar = ScaleBar( 2000, "cm", rotation="vertical", scale_loc="bottom")
	pb.make_invisible(dax)
	if not isinstance(stimplotfn,type(None)):	stimplotfn(stax)
	emax.plot(aRec.emg_tvec[cond2], aRec.emg_data[cond2], 'k')
	emax.text(0.02, 0.8, 'EMG', color='k', ha='left', va='top', fontweight='bold',
			  bbox={'facecolor': 'w', 'linewidth': 0}, transform=emax.transAxes)
	emax.set_ylim(ylim_emg)


	axarr[-1].set_xlabel('Time [s]')
	return f

def plot_singlesource_recon(aRec,els,tminmax,A,src_mat,inds,cmapstr='Dark2_r',**kwargs):
	toff = kwargs['toff'] if 'toff' in kwargs else 0

	pmin,pmax = (np.array(tminmax)*aRec.sr).astype(int)
	pmin_emg,pmax_emg = (np.array(tminmax)*aRec.emg_sr).astype(int)
	tvec = np.arange(0.,(pmax-pmin)/aRec.sr,1/aRec.sr)+toff
	tvec_emg = np.arange(0.,(pmax_emg-pmin_emg)/aRec.emg_sr,1/aRec.emg_sr)+toff
	ylim_emg = [-20, 20]

	N = len(inds)
	if 'labelints' in kwargs:
		labels = ['src %i'%lab for lab in kwargs['labelints']]
	else:
		labels = ['src %i'%(lab+1) for lab in np.arange(N)]

	cdict = pb.get_colordict_electrodes(aRec,els,cmapstr)
	#f,axarr = mpl.pyplot.subplots(figsize = (2+N*2.5,10))
	f = mpl.pyplot.figure(figsize=(2+N*2.5,10))
	gs1 = f.add_gridspec(nrows=1, ncols=N, left=0.05,right=0.93,top=0.93, bottom=0.21, hspace=0.05)
	gs2 = f.add_gridspec(nrows=1, ncols=N, left=0.05,right=0.93, top=0.21, bottom=0.07, hspace=0.05)

	recon_list = [np.dot(A[:,idx][:,None],src_mat[idx][None,pmin:pmax]) for idx in inds]

	spacefac = np.hstack(recon_list).std()*5
	#print(tvec.shape)
	lastloc = 'na'
	for ii,idx in enumerate(inds):
		Xrecon = recon_list[ii]

		ax = f.add_subplot(gs1[0, ii])
		ax.set_title(labels[ii],pad=-5)

		for cc in np.arange(len(els)):
			myel = els[cc]
			loc = np.array(aRec.el_locs)[myel == aRec.el_ids][0]
			col = cdict[loc]
			#print(Xrecon.shape)
			ax.plot(tvec, Xrecon[cc] + cc * spacefac, color=col)
			if ii==len(inds)-1:
				if loc != lastloc:
					trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
					ax.text(1.03, cc * spacefac, loc, color=col, ha='left', va='center', fontweight='bold',
							 transform=trans)
				lastloc = loc
		if 'stimbar' in kwargs:
			y = (cc+2)*spacefac
			x = kwargs['stimbar']
			ax.plot(x,[y,y],'k',lw=4)
		ax.set_xlim([tvec.min(), tvec.max()])
		pb.make_invisible(ax)
		ax2 = f.add_subplot(gs2[0, ii])
		ax2.plot(tvec_emg, aRec.emg_data[pmin_emg:pmax_emg], 'k')
		ax2.set_xlabel('Time [s]')
		ax2.set_xlim([tvec_emg.min(),tvec_emg.max()])
		ax2.set_ylim(ylim_emg)
		if ii ==0:
			ax2.set_ylabel('EMG [mV]')
			# ax2.text(0.02, 0.8, 'EMG', color='k', ha='left', va='top', fontweight='bold',
			# 	  bbox={'facecolor': 'w', 'linewidth': 0}, transform=ax2.transAxes)
		else:
			ax2.set_yticklabels([])


	return f



def plot_ICA_loadings(aRec,mixmat,els,cmapstr='tab10',labels=True,lab='loading [A.U.]',tag='src',**kwargs):
	'''mixmat is chans x component sources'''

	nchans,ncomps = mixmat.shape
	lw = 4 if 'indiv' in kwargs else 2

	el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])
	ylim = [0, nchans - 1]

	ica_cols = pb.get_colors(cmapstr,ncomps,**kwargs)


	f, axarr = mpl.pyplot.subplots(1, 2, figsize=(6, 7), gridspec_kw={'width_ratios': [0.05, 1]})
	f.subplots_adjust(top=0.95)
	locax, ax = axarr
	pb.make_locax(locax, aRec.el_locs[el_inds], cols=['k', 'lightgray'], boundary_axes=[ax], lim=ylim)
	for ii in np.arange(ncomps):
		col = ica_cols[ii]
		comp = mixmat[:, ii]
		ax.plot(comp, np.arange(nchans),color=col,lw=lw,zorder=100-ii)
		if 'indiv' in kwargs:
			subset = kwargs['indiv'][ii]
			for subcomp in subset:
				ax.plot(subcomp, np.arange(nchans),color=col,lw=2,zorder=-ii,alpha=0.7)
		if 'bounds' in kwargs:
			lower,upper = kwargs['bounds'][ii]
			ax.fill_betweenx(np.arange(nchans),lower,upper,alpha=0.5,lw=0,color=col,zorder=-ii)
		if labels:ax.text(1.01, 0.99 - ii * 0.05, '%s%i' % (tag,ii + 1), color=col, ha='left', va='center',
				fontweight='bold', transform=ax.transAxes)
	ax.ticklabel_format(axis='x', style='sci', scilimits=(-1, 2))
	ax.axvline(0., color='grey', alpha=0.5)
	ax.set_ylim(ylim)
	ax.set_xlabel(lab)
	ax.yaxis.set_visible(False)
	return f


def plot_trialaligned_sources(aRec,ttype_vec,dlist,cmapstr='tab10',**kwargs):
	toff = kwargs['toff'] if 'toff' in kwargs else 0
	unique_types = np.unique(ttype_vec)

	nsrcs,npts = dlist[0].shape

	if 'labelints' in kwargs:
		labels = ['src %i'%lab for lab in kwargs['labelints']]
	else:
		labels = ['src %i'%(lab+1) for lab in np.arange(nsrcs)]


	npts = dlist[0].shape[1]
	ica_cols = pb.get_colors(cmapstr, nsrcs, **kwargs)
	tvec = np.linspace(0, npts / aRec.sr, npts)+toff

	ylims = np.vstack([fn(np.hstack(dlist), axis=1) for fn in [np.min, np.max]]).T

	###
	N_types = len(unique_types)
	f, axarr = mpl.pyplot.subplots(nsrcs, N_types, figsize=(2+N_types*2.3, 1.8+nsrcs*1.5), sharex=True)# sharey=True,
	f.subplots_adjust(left=0.09, right=0.98)
	for tt, ttype in enumerate(unique_types):
		cond = ttype_vec == ttype

		for ss in np.arange(nsrcs):
			ax = axarr[ss, tt] if N_types>1 else axarr[ss]
			mysrc = np.vstack([ddlist[ss] for ddlist in dlist])[cond]
			col = ica_cols[ss]
			ax.plot(tvec, mysrc.T, color='grey', lw=0.2)
			ax.plot(tvec, mysrc.mean(axis=0), color=col, lw=3)
			if ss == nsrcs - 1: ax.set_xlabel('Time [s]')
			# else: ax.set_xticklabels([])
			if ss == 0: ax.set_title('\n '.join(aRec.ttype_to_str(ttype)))
			if tt == 0:
				ax.set_ylabel(labels[ss], color=col, fontweight='bold')
			else:
				ax.set_yticklabels([])
			ax.set_ylim(ylims[ss])
		ax.set_xlim([tvec.min(), tvec.max()])
	#f.tight_layout()
	return f


def plot_stimlayout(omnidict,plotfn,**kwargs):

	omnikeys = kwargs['omnikeys'] if 'omnikeys' in kwargs else omnidict.keys() #omnikeys is just specify the plot-order
	col_ext = 1 if 'cmapfn' or 'baselinefn' in kwargs else 0
	fsize = kwargs['fsize'] if 'fsize' in kwargs else (10,8)
	fcounter = 0

	ampvec = np.unique(np.hstack([omnidict[soundcat]['amps'] for soundcat in omnidict.keys()]))
	ncols = np.sum([len(omnidict[soundcat]['fvec']) for soundcat in omnidict.keys()]) +col_ext
	nrows =  len(ampvec)

	f, axarr = mpl.pyplot.subplots(nrows, ncols, figsize=fsize, sharey=True)
	for soundcat in omnikeys:
		subdict = omnidict[soundcat]
		for ttype, amp, freq in zip(subdict['ttypes'], subdict['amps'], subdict['freqs']):
			fidx, aidx = np.where(subdict['fvec'] == freq)[0][0], np.where(subdict['avec'] == amp)[0][0]
			ax = axarr[len(ampvec) - aidx - 1, fidx + fcounter]
			plotfn(ax,subdict,ttype)
			## ax.text(0.99,0.99,'A%i,F%i'%(aidx,fidx),transform=ax.transAxes,color='k')
			if 'bg_colfn' in kwargs:
				print
				kwargs['bg_colfn'](ax,ttype)
			# stats
			if 'statsfn' in kwargs:
				kwargs['statsfn'](ax,subdict,ttype)
			if aidx == 0:
				ax.set_xticks([0])
				if soundcat.count('pure'):
					ax.set_xticklabels(['freq %i kHz' % int(freq/1000)])
				else:
					ax.set_xticklabels([soundcat])

			else:
				ax.set_xticks([])

			if fidx + fcounter == 0:
				# print(aidx,amp,freq)
				ax.set_ylabel('amp %1.1f V' % amp)
		fcounter += len(subdict['fvec'])

	if 'limfn' in kwargs:
		 kwargs['limfn'](axarr)

	if 'baselinefn' in kwargs:
		ax = axarr[0, ncols - 1]
		kwargs['baselinefn'](ax)
		for ax in axarr[1:, ncols - 1]: ax.set_axis_off()


	# colorbar
	if 'cmapfn' in kwargs:
		pos_t = axarr[1, ncols - 1].get_position()  # top position
		pos_b = axarr[nrows - 1, ncols - 1].get_position()  # bottom position
		cbarheight = pos_t.ymax - pos_b.ymin  # - pos_t.height*0.4
		cbax = f.add_axes([pos_t.x0 + 0.3 * pos_t.width, pos_b.y0, pos_t.width * 0.3, cbarheight])
		kwargs['cmapfn'](cbax)
	return f

def plot_powdict_vals(ax,subdict,ttype,myfn,mec='m'):
	vals = myfn(subdict['powdict'][ttype])
	ax.plot(np.zeros_like(vals), vals, 'o', mec=mec, mfc='none')
	ax.yaxis.grid(color='lightgrey', linestyle=':', zorder=-10)
	ax.axhline(1., color='lightgrey', linestyle='-', zorder=-10)
	ax.text(0.97, 0.97, 'N:%i' %len(vals), transform=ax.transAxes, ha='right', va='top',color=mec)

	ax.set_xlim([-0.1, 0.1])

def plot_pval(ax,subdict,ttype,myfn):
	pval = myfn(subdict,ttype)
	if pval < 0.05:
		ax.text(0.97, 0.03, 'p:%1.4f' %pval, transform=ax.transAxes, ha='right', va='bottom', color='k',
				fontweight='normal',bbox=dict(facecolor='w', alpha=0.5, edgecolor='none',pad=0.1))
		for pos in ['top', 'bottom', 'left', 'right']: ax.spines[pos].set_linewidth(3)

def plot_baselinehist(ax,vals,N):
	bins = np.linspace(scoreatpercentile(vals,0.5),scoreatpercentile(vals,99.5),20) #to avoid extreme vals distorting the range
	ax.hist(vals, bins=bins, orientation='horizontal', histtype='stepfilled',
			color='gray')
	ax.set_xticks([])
	ax.text(0.5, 0.95, 'N:%i' %N, transform=ax.transAxes, ha='center', va='top')
	ax.set_title('baseline')

def plot_cmap(cbax,cmap,cnorm,clab):
	cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, orientation='vertical', norm=cnorm)
	cb.set_label(clab, rotation=-90, labelpad=15)
	cb.ax.axhline(1., color='lightgrey', linestyle='-')


def set_axlim(axarr,vals,axtype='y'):
	#print(axarr)
	valmin,valmax = vals.min(),vals.max()
	buff  = 0.05*(valmax-valmin)
	lim = [valmin-buff,valmax+buff]
	for ax in axarr.flatten():
		if axtype == 'y': ax.set_ylim(lim)
		elif axtype == 'x':ax.set_xlim(lim)


make_z = lambda vals: (vals-vals.mean(axis=1)[:,None])/vals.std(axis=1)[:,None]


def eval_labeling(mode,data,labs):
	from sklearn.metrics import (davies_bouldin_score, silhouette_score)
	if mode.count('sil'):
		return silhouette_score(data, labs)
	elif mode.count('db'):
		return davies_bouldin_score(data, labs)



def get_clustevalfn(mode,**kwargs):
	quality_dict = {}
	if mode.count('fuzzy'):
		import skfuzzy
		eval_fn = lambda data,ncents: skfuzzy.cluster.cmeans(data, ncents, **kwargs)[-1]

	elif mode.count('gmm'):

		def eval_fn(data,ncents):

			gmm = GaussianMixture(ncents).fit(data)
			labs = np.argmax(gmm.predict_proba(data),axis=1)
			return eval_labeling(mode,data,labs)
	return eval_fn

def run_clustevalfn(clustevalfn,data,nclustvec,niter=50,percs=[20,50,80]):
	N_models = len(nclustvec)
	fpcvec = np.zeros((niter,N_models)).astype(float)
	for ii in np.arange(niter):
		for nn, ncents in enumerate(nclustvec):
			fpcvec[ii,nn] = clustevalfn(data,ncents)#, ncents, 2, error=0.005, maxiter=1000,init=None)
	fpc_percs = np.vstack([scoreatpercentile(fpcvec,perc,axis=0) for perc in percs])
	return fpc_percs

def plot_nclust_quality(quality_vec,nclustvec,ylab='',col='k',**kwargs):
	if not 'ax' in kwargs:
		f, ax =  mpl.pyplot.subplots(figsize=(5,4))
		f.subplots_adjust(left=0.15,bottom=0.15)
	else:
		ax = kwargs['ax']

	N_models = len(nclustvec)
	#print(N_models)
	for ii in np.arange(N_models):
		y = [quality_vec[0,ii],quality_vec[2,ii]]
		x = [nclustvec[ii],nclustvec[ii]]
		ax.plot(x,y,color='grey')
	ax.plot(nclustvec, quality_vec[1], 'o-',color=col)
	ax.set_xlabel('N clusters')
	ax.set_ylabel(ylab)#fuzzy partition coefficient
	if not 'ax' in kwargs:
		return f

def gmm_cluster(data,ncents):
	from sklearn.mixture import GaussianMixture
	gmm = GaussianMixture(ncents).fit(data)
	labs = np.argmax(gmm.predict_proba(data),axis=1)
	conf = np.max(gmm.predict_proba(data),axis=1)
	return labs,conf

def fuzzyc_cluster(data,ncents,**kwargs):
	'''returns cluster labels and confidence'''
	import skfuzzy as fuzz
	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, ncents, **kwargs)
	return np.argmax(u, axis=0), np.max(u, axis=0)

def plot_anatomical_func_borders(aRec,conf_func,labs_func,conf_anat,labs_anat,els,colorstr='Set3',anat_cols = ['grey','silver'],assert_match=False,**kwargs):
	nchans = len(els)

	ncents = len(np.unique(labs_func))
	ylim = [0, nchans - 1]
	chanvec = np.arange(nchans)
	el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])

	if assert_match: assert (labs_anat == aRec.el_locs[el_inds]).all(), 'mismatching location assignments!'

	loclist, Nperloc = dh.get_loclist_nlist(aRec.el_locs[el_inds])

	cols_anat =  anat_cols*int(np.ceil(len(loclist)/2.))
	cols_func = pb.get_colors(colorstr, ncents,**kwargs)

	borders_func = np.where(np.diff(labs_func) != 0)[0] + 0.5

	f, axarr = mpl.pyplot.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [0.1, 1, 1]}, figsize=(5.8, 7.3))
	f.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.98, hspace=0.2, wspace=0.2)
	locax, ax2, ax = axarr
	ax.plot(conf_func, chanvec, 'ok-', ms=4)
	ax2.plot(conf_anat, chanvec, 'ok-', ms=4)
	for bord in borders_func:
		ax.axhline(bord, color='grey', linestyle=':')


	#fill the clusters 1) func 2)anat
	for lab, col in zip(np.unique(labs_func), cols_func):
		ax.fill_betweenx(chanvec, np.zeros_like(conf_func), conf_func, where=labs_func == lab, color=col)
	for lab, col in zip(loclist, cols_anat):
		ax2.fill_betweenx(chanvec, np.zeros_like(conf_anat), conf_anat, where=labs_anat == lab, color=col)

	pb.make_locax(locax, aRec.el_locs[el_inds], cols=anat_cols, boundary_axes=[ax2], lim=ylim)

	ax2.set_xlim([10, 141 + 2])
	ax.set_xlim([0., 1.01])

	ax2.set_title('spatial')
	ax.set_title('functional')
	ax2.set_xlabel('dist next [mum]')
	ax.set_xlabel('cluster conf.')

	for myax in axarr:
		myax.set_yticks([])
		myax.set_ylim([chanvec.min(), chanvec.max()])
	return f


def plot_tsne_proj(aRec,els,proj,cmap='jet',plotstyle='continuous',labels=None,**kwargs):
    f,axarr = mpl.pyplot.subplots(1,3,gridspec_kw={'width_ratios':[15,1,1]})
    el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])
    ax,cax,locax = axarr
    if plotstyle == 'continuous':
        sc = ax.scatter(proj[:,0],proj[:,1], c=np.arange(proj.shape[0]), cmap=cmap, vmin=0, vmax=proj.shape[0] - 1)
        f.colorbar(sc, cax=cax, orientation='vertical')
    elif plotstyle == 'labeled':
        cols = pb.get_colors(cmap, len(np.unique(labels)),**kwargs)

        for lab, col in zip(np.unique(labels), cols):
            cond = labels==lab
            sc = ax.scatter(proj[cond,0],proj[cond,1], color=col)

            cax.fill_betweenx(np.arange(proj.shape[0]), np.zeros_like(labels), np.ones_like(labels), where=labels == lab, color=col)
        cax.set_ylim([0,proj.shape[0]])
        cax.set_xlim([0,1])
    pb.make_locax(locax, aRec.el_locs[el_inds], linecol='k', tha='left', textoff=1.1, cols=['k', 'lightgray'],
              boundary_axes=[cax])

    cax.set_axis_off()
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim2')

    return f