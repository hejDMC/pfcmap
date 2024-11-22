import numpy as np
import matplotlib.transforms as transforms
import matplotlib as mpl
from . import data_handling as dh
import os

art_colors = {'art':'firebrick','buff':'darkorange','free0':'skyblue','free':'navy'}


def get_datalim(datalist,tvec,tlim,ampfrac=0.05):
	datasnip = np.hstack([data[(tvec>=tlim[0]) & (tvec<=tlim[1])] for data in datalist])#concatenated when e.g. plotting multiple lines in one panel
	amp = datasnip.max()-datasnip.min()
	buff = ampfrac*amp
	return [datasnip.min()-buff,datasnip.max()+buff]


def plot_spgrm(ax,spgrm,tvec,freqvec,label='dynnorm [0,1]',ylab=True):
	im = ax.imshow(spgrm.T, cmap='inferno', origin='lower', aspect='auto', \
					interpolation='none', filternorm=None, vmin=0, vmax=1)
	im.set_extent([tvec.min(), tvec.max(), freqvec.min(), freqvec.max()])
	if ylab: ax.set_ylabel('Freq [Hz]')
	if label: ax.text(1.01, 0.99, label, transform=ax.transAxes, ha='left', va='top')

def plot_arts(ax,data_list,keys_list=['art','buff','free0','free'],invisibleax=True):
	'''data_list must contain data in order of keys_list'''
	for ii,[key,tmat] in enumerate(zip(keys_list,data_list)):
		col = art_colors[key]
		ax.hlines(np.ones(len(tmat)) * (4 - ii), tmat[:, 0], tmat[:, 1], color=col, alpha=1, lw=10)
		ax.text(1.01, 1 - ii * 0.2, key, color=col, transform=ax.transAxes, va='top', ha='left')
	if invisibleax: make_invisible(ax)

def make_invisible(ax):
	for pos in ['bottom', 'top', 'left', 'right']: ax.spines[pos].set_visible(False)
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)



def plot_simuli_fndict(ax,ds_handle,stimplotfns,invisibleax=True,**kwargs):

    stimtypes_show = kwargs['stimtypes'] if 'stimtypes' in kwargs else list(stimplotfns.keys())
    myylim = kwargs['ylim'] if 'ylim' in kwargs else [0,1]

    for ii,stimtype in enumerate(stimtypes_show):
        timestamps = ds_handle[stimtype]['timestamps'][()]
        stimplotfns[stimtype](timestamps,ax)
        if 'labels' in kwargs:
            lab,col = [kwargs['labels'][stimtype][tag] for tag in ['label','color']]
            ax.text(1.01, 1 - ii * 0.2, lab, color=col, transform=ax.transAxes, va='top', ha='left')
    ax.set_ylim(myylim)
    if invisibleax: make_invisible(ax)

def plot_stimuli(ax,RecObj,colordict,invisibleax=True,extension_fac_nonb=0,maxnorm=True):
	'''colordict needs an entry for every trial params in .bool_params and .nonbinary_params'''
	nonb_ticks = np.unique(np.hstack([vals for vals in RecObj.nonbinary_inddict.values()]))

	for cc, cond in enumerate(RecObj.bool_params):
		condarray = RecObj.trialset[:, RecObj.trial_params.index(cond)]
		cond_times = RecObj.ttmat[condarray == 1, :]
		# xarr[1].hlines(np.ones(len(cond_times)),cond_times[:,0],cond_times[:,1],color=colordict[cond],alpha=1,lw=20)#*cc
		ax.text(1.01, 1 - cc * 0.2, cond, color=colordict[cond], transform=ax.transAxes, va='top', ha='left')
		for ct in cond_times:
			ax.axvspan(ct[0], ct[1], color=colordict[cond], alpha=0.75, zorder=2)
		# äaxarr[1].add_patch(patches.Rectangle((cond_times[,0],0),self.dur*ufac,1,color=self.color,linewidth=0))

	# axarr[1].set_ylim([0.98,1.02])
	# axarr[1].set_yticks([])
	for ccc, cond in enumerate(RecObj.nonbinary_params):
		yvals = RecObj.nonbinary_inddict[cond]
		if maxnorm: yvals = yvals/yvals.max()
		if cond.count('freq'):
			zorder, lw = 10, 4
		else:
			zorder, lw = 11, 2
		# tstarts,tstops = ttmat[:,0],ttmat[:,0]+trialdiffs/2.
		# else:tstarts,tstops = ttmat[:,0]+trialdiffs/2.,ttmat[:,1]
		# axarr[2].hlines(yvals,tstarts,tstops,color=colordict[cond],alpha=1,lw=1)
		ax.hlines(yvals, RecObj.ttmat[:, 0]-0.5*extension_fac_nonb*RecObj.trial_dur, RecObj.ttmat[:, 1]+0.5*extension_fac_nonb*RecObj.trial_dur, color=colordict[cond], alpha=0.9, lw=lw, zorder=zorder)
		ax.text(1.01, 1 - (ccc + cc + 1) * 0.2, cond.replace('sound_', ''), color=colordict[cond],
				  transform=ax.transAxes, va='top', ha='left')
	# axarr[2].set_yticks(nonb_ticks)
	#ax.hlines(nonb_ticks, np.zeros(len(nonb_ticks)), np.zeros(len(nonb_ticks)) + RecObj.dur, ls=':', color='grey', zorder=5)
	if maxnorm: ax.set_ylim(-0.1,1.1)
	else: ax.set_ylim([-0.5, nonb_ticks[-1] + 0.5])
	if invisibleax: make_invisible(ax)


def plot_stimuliCtxt(ax,aRec,params=['lick','valve','wn','blocks'],stimcoldict={'lick':'c','valve':'b'},block_colors={0:'grey',1:'k'},wn_colors={True:'lightpink',False:'deeppink'},invisibleax=True,valvedur=0.050):
	blockdict = aRec.get_blocks(prepost_adj=True)

	ii = 0
	ytop = 2
	for tag,vals in zip(['lick','valve'],[aRec.licktimes,aRec.vopen_times]):
		if tag in params:
			col = stimcoldict[tag]
			ax.vlines(vals, ytop-1, ytop, color=col)
			if tag == 'valve':
				ax.hlines(np.zeros_like(vals)+ytop-0.5,vals,vals+valvedur,color=col)
			ax.text(1.01, 1 - ii * 0.2, tag, color=col, transform=ax.transAxes, va='top', ha='left')
			ii += 1
			ytop -= 1

	if 'wn' in params:
		uniq_wn = np.unique(aRec.wn_bool)
		for B in uniq_wn:
			booltimes = aRec.wn_times[ aRec.wn_bool== B]
			ax.vlines(booltimes,  ytop-1, ytop, color=wn_colors[B])
			if B == 1:
				inds = np.where(aRec.wn_bool==B)[0]
				plotvec = np.vstack([aRec.wn_times[inds],aRec.wn_times[inds+1]])
				ax.hlines(np.zeros_like(plotvec[0])+ytop-0.5,plotvec[0],plotvec[1],color=wn_colors[B],linewidth=3)

		wnstrs = ['wn%i'%int(B) for B in uniq_wn]
		wncols = [wn_colors[B] for B in uniq_wn]
		multicol_legend(ax, [1.01,1-ii*0.2], wnstrs, wncols)
		ii += 1
		ytop -= 1

	if 'blocks' in params:
		uniq_blks = np.unique(blockdict['id'])
		for blockid in uniq_blks:
			cond = blockdict['id'] == blockid
			col = block_colors[blockid]
			ax.hlines(np.ones(np.sum(cond)) * ytop, blockdict['starts'][cond], blockdict['stops'][cond],
					  color=col, lw=4)
		blstrs = ['bl%i'%idx for idx in uniq_blks]
		blcols =  [block_colors[idx] for idx in uniq_blks]
		multicol_legend(ax,[1.01,1-ii*0.2],blstrs,blcols)

	ax.set_ylim([-1.3,2.])

	if invisibleax: make_invisible(ax)

def plot_trialbools(ax,aRec,params=['lick','valve'],markattr='ttmat',colordict={'lick':'c','valve':'b'}):
	ax.text(1.01, 1 , 'bools', color='k', transform=ax.transAxes, va='top', ha='left',fontstyle='italic')
	for cc, cond in enumerate(params[::-1]):
		condarray = aRec.trialset[:, aRec.trial_params.index(cond)]
		cond_times = getattr(aRec,markattr)[condarray == 1, :]
		ax.hlines(np.ones(cond_times.shape[0])*(cc*0.3),cond_times[:,0],cond_times[:,1],linewidth=2,color=colordict[cond],alpha=1)
		if cond == 'sound':
			ax.text(1.03, 1-0.6 , cond, color=colordict[cond], transform=ax.transAxes, va='top', ha='left')
	ax.set_ylim([-0.1,cc*0.3+0.1])
	make_invisible(ax)

def multicol_legend(ax,startpos,strlist,collist,spacefac=0.12):
	renderer = ax.figure.canvas.get_renderer()
	transf = ax.transAxes.inverted()
	for ii,[mystr,col] in enumerate(zip(strlist,collist)):

		if ii==0:
			txt = ax.text(startpos[0], startpos[1], mystr, color=col, transform=ax.transAxes, va='top', ha='left')
			ex = txt.get_window_extent(renderer=renderer).transformed(transf)
		else:
			xpos = (ex.x1 + spacefac * ex.width)
			ypos = ex.y0
			#print(blockid, xpos, ypos)
			txt = ax.text(xpos, ypos, mystr, color=col, va='bottom', ha='left', transform=ax.transAxes)
			ex = txt.get_window_extent(renderer=renderer).transformed(transf)

def plotoverlay_trialsArtfree(ax,RecObj,collist=['skyblue'],alpha=0.2):
	for tid in RecObj.trials_artfree:
		tstart, tstop = RecObj.cutout_times[tid]
		if len(collist)==3:
			s1 = RecObj.pre_stim+tstart
			s2 = tstop-RecObj.post_stim
			for col,tpair in zip(collist,[(tstart,s1),(s1,s2),(s2,tstop)]):
				ax.axvspan(tpair[0],tpair[1],color=col, alpha=alpha, zorder=2,linewidth=0)
		elif len(collist)==1:
			col = collist[0]
			ax.axvspan(tstart, tstop, color=col, alpha=alpha, zorder=2,linewidth=0)
			ax.axvspan(tstart + RecObj.pre_stim, tstop - RecObj.post_stim, color=col, alpha=alpha, zorder=3,linewidth=0)
	if len(collist)==3:
		for cc,[col,tag] in enumerate(zip(collist,['prestim','stim','poststim'])):
			ax.text(1.01,1-cc*0.1,tag,color=col,transform=ax.transAxes,va='top',ha='left',alpha=alpha,fontweight='bold')



def plot_units(ax,spikelist,rs_bool,rs_col='grey',fs_col='b'):
	for uu,spikes in enumerate(spikelist):
		col= rs_col if rs_bool[uu] else fs_col
		ax.plot(spikes,np.zeros(len(spikes))+uu,'.',color=col)
	for tt,[tag,col] in enumerate(zip(['RS','FS'],[rs_col,fs_col])):
		 ax.text(1.01,1-tt*0.2,tag,color=col,transform=ax.transAxes,va='top',ha='left')
	ax.set_ylim([-1,len(spikelist)+1])
	ax.set_ylabel('Unit')

def plot_ozis(ax,tvec,osc_dict,ylab='filt. LFP  [muV]',alpha=0.5):
	'''eg. oszillations'''
	for ii in np.arange(len(osc_dict)):
		lab, col,data = osc_dict[ii]
		ax.plot(tvec, data, color=col, alpha=0.5)
		ax.text(1.01, 1 - ii * 0.2, lab, color=col, transform=ax.transAxes, va='top', ha='left', alpha=alpha)
	ax.set_ylabel(ylab)


def plot_traceAndPercentiles(ax,tvec,middle,upper,lower):
	ax.fill_between(tvec, lower, upper, color='grey', alpha=0.5)
	ax.plot(tvec, middle, 'k', lw=2)

def make_gridlines(axarr,tlist_main,tlist_sub=[],gridres=0.05):
	for pos in tlist_main:
		for ax in axarr: ax.axvline(pos,color='grey',linestyle=':',zorder=-2)
	for pos in np.arange(tlist_sub[0],tlist_sub[1],gridres):
		for ax in axarr: ax.axvline(pos,color='grey',linestyle=':',zorder=-2,linewidth=1,alpha=0.5)


def make_ybar(ax,ybar=200,yunit='muV',pos='left',fracx=0.01,fracy=0.05,box_on=True):
	yanch = ax.get_ylim()[0] + (fracy * np.diff(ax.get_ylim())[0])
	if pos == 'left':
		xanch = ax.get_xlim()[0] + (fracx * np.diff(ax.get_xlim())[0])
		rot = 90
		ha = 'right'
	elif pos == 'right':
		xanch = ax.get_xlim()[1] - (fracx * np.diff(ax.get_xlim())[0])
		rot = -90
		ha = 'left'
	ax.plot([xanch, xanch], [yanch, yanch + ybar], 'k-', linewidth=2,zorder=50)
	t = ax.text(xanch, yanch + 0.5 * ybar, '%i %s' % (ybar, yunit), color='k', fontweight='bold', ha=ha, va='center',
			 rotation=rot,zorder=49)
	if box_on: t.set_bbox(dict(facecolor='w', alpha=0.7, edgecolor='none'))

def assign_from_coldict(unique_elecs,refdict,default_color= 'grey'):
	colordictR = {}
	for loc in unique_elecs:
		matches = [col for key, col in refdict.items() if loc.count(key)]
		if len(matches) == 1:
			colordictR[loc] = matches[0]
		elif len([col for key, col in refdict.items() if loc.count(key)]) == 0:
			colordictR[loc] = default_color
		else:
			assert 0, 'oops'
	return colordictR


def concat_cmaps(cmap,N):
	from itertools import chain
	listoflists = [list(cmap) for ii in np.arange(np.ceil(N / len(cmap)))]
	return list(chain(*listoflists))


def get_colors(cmapstr,ncomps,**kwargs):
	if 'colors' in kwargs:
		cols = kwargs['colors']
	elif 'col' in kwargs: cols = ['k']*ncomps
	else:
		cmap = mpl.cm.get_cmap(cmapstr)
		if not hasattr(cmap,'colors'):
			cols = cmap(np.linspace(0.05,0.95,ncomps))
		else: cols = concat_cmaps(cmap.colors, ncomps)
	return cols


def plot_scalebar(ax,barlen,mode ='y',unit='',ha='left',va='bottom',ta='right',ybuff=0.05,xbuff=0.01,verbose=False):
	hidx = 0 if ha == 'left' else 1
	vidx = 0 if va == 'bottom' else 1

	yanch = ax.get_ylim()[hidx] + (ybuff * np.diff(ax.get_ylim())[0])
	xanch = ax.get_xlim()[vidx] + (xbuff * np.diff(ax.get_xlim())[0])

	if mode == 'y':
		xanch = xbuff
		xdata,ydata = [xanch,xanch],[yanch, yanch + barlen]
		tpos_x,tpos_y = xanch, yanch + 0.5 * barlen
		tha,tva = ta,'center'
		rot = 90
		trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
	elif mode == 'x':
		yanch = ybuff
		xdata,ydata = [xanch,xanch+barlen],[yanch, yanch]
		tpos_x,tpos_y = xanch+ 0.5 * barlen, yanch
		tha,tva = 'center',ta
		rot = 0
		trans = transforms.blended_transform_factory(ax.transData,ax.transAxes)

	stxt = ax.text(tpos_x,tpos_y, '%i %s' % (barlen, unit), color='k', fontweight='bold', ha=tha, va=tva, rotation=rot,transform=trans,zorder=30)
	stxt.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))
	sbar = ax.plot(xdata, ydata, 'k-', linewidth=2,transform=trans,zorder=40)

	if verbose:
		return sbar,stxt

def make_locax(ax,elec_locs,ori='vertical',cols=['k','lightgray'],linecol='gray',tha='right',tva='center',textoff=-0.01,rotation=0,**kwargs):
	'''elec_locs is array of strings specifying electrode regions, eg. aRec.el_locs'''
	lim = kwargs['lim'] if 'lim' in kwargs else [-0.5, len(elec_locs)-0.5]
	fw = kwargs['fw'] if 'fw' in kwargs else 'bold'
	zord = kwargs['zord'] if 'zord' in kwargs else 100
	alph = kwargs['alph'] if 'alph' in kwargs else 1

	loclist, Nperloc = dh.get_loclist_nlist(elec_locs)
	locblocks = np.r_[0, np.cumsum(Nperloc)] - 0.5
	if ori=='vertical':
		trans = transforms.blended_transform_factory(ax.transAxes,ax.transData)
		patchfn = lambda startval,stopval,colval:mpl.patches.Rectangle((0, startval), 1, stopval - startval, color=colval, lw=0)
		textfn = lambda startval,stopval,locstr: ax.text(textoff,startval+0.5*(stopval-startval),locstr,fontweight=fw,ha=tha,va=tva,transform=trans,rotation=rotation)
		limfn = ax.set_ylim
		linefn = lambda bax,pos: bax.axhline(pos, linestyle=':', color=linecol,zorder=zord,alpha=alph)
	elif ori=='horizontal':
		trans = transforms.blended_transform_factory(ax.transData,ax.transAxes)
		patchfn = lambda startval,stopval,colval:mpl.patches.Rectangle((startval,0), stopval - startval,1, color=colval, lw=0)
		textfn = lambda startval,stopval,locstr: ax.text(startval+0.5*(stopval-startval),textoff,locstr,fontweight=fw,\
														 ha=tha,va=tva,transform=trans,rotation=rotation)
		limfn = ax.set_xlim
		linefn = lambda bax,pos: bax.axvline(pos, linestyle=':', color=linecol,zorder=zord,alpha=alph)


	if 'loctrfn' in kwargs:
		locblocks = kwargs['loctrfn'](locblocks)
	for ii in np.arange(len(locblocks) - 1):
		start, stop = locblocks[ii:ii + 2]
		#print (start,stop)
		mycol = cols[0] if np.mod(ii, 2) == 0 else cols[1]
		ax.add_patch(patchfn(start,stop,mycol))
		textfn(start,stop,loclist[ii])
	if 'boundary_axes' in kwargs:
		for bax in kwargs['boundary_axes']:
			for locpos in locblocks[1:-1]: linefn(bax,locpos)
	make_invisible(ax)
	limfn(lim)

def get_colordict_electrodes(aRec,els,cmapstr='Dark2_r'):
	cmap =  mpl.cm.get_cmap(cmapstr).colors
	inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])
	unique_strs = ['X']
	for ll, loc in enumerate(aRec.el_locs[inds]):
		if loc != unique_strs[-1]: unique_strs += [loc]
	unique_strs = unique_strs[1:]
	mycmap = concat_cmaps(cmap, len(unique_strs))
	return {loc: mycmap[ll] for ll, loc in enumerate(unique_strs)}


def plot_chanpanel(ax,data,tvec,aRec,els,cdict,alpha=1,invisibleax=True,**kwargs):

	tminmax = kwargs['tminmax'] if 'tminmax' in kwargs else [tvec[0],tvec[-1]]

	assert data.shape[0] == len(els), 'data-electrode dimension mismatch %i vs %i'%(data.shape[0],len(els))

	trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
	cond = (tvec <= tminmax[1]) & (tvec >= tminmax[0])
	plmat = data[:, cond]
	spacefac = plmat.std()*5
	lastloc = 'na'
	for cc,myel in enumerate(els):
		loc = np.array(aRec.el_locs)[myel == aRec.el_ids][0]
		col = cdict[loc]

		ax.plot(tvec[cond], plmat[cc] + cc * spacefac, color=col, alpha=alpha)
		if loc != lastloc:
			ax.text(1.01, cc * spacefac, loc, color=col, ha='left', va='center', fontweight='bold',
					 transform=trans)
		lastloc = loc
	ax.set_xlim(tminmax)
	if invisibleax:make_invisible(ax)


def mystep(x,y, ax=None, where='post', **kwargs):
	assert where in ['post', 'pre']
	x = np.array(x)
	y = np.array(y)
	if where=='post': y_slice = y[:-1]
	if where=='pre': y_slice = y[1:]
	X = np.c_[x[:-1],x[1:],x[1:]]
	Y = np.c_[y_slice, y_slice, np.zeros_like(x[:-1])*np.nan]
	if not ax: ax=mpl.gca()
	return ax.plot(X.flatten(), Y.flatten(), **kwargs)

def plot_corrmat(corrmat,cstr='RdBu',lab='',title=''):
	N = corrmat.shape[0]
	maxval = np.abs([corrmat.min(),corrmat.max()]).max()
	corrmat[np.arange(N),np.arange(N)] = 9
	plotmat = np.ma.masked_where(corrmat==9, corrmat)
	cmap = mpl.cm.get_cmap(cstr).copy()
	cmap.set_bad('grey',1.)
	f,ax = mpl.pyplot.subplots(figsize=(3+N*0.3,2.5+N*0.3))
	#f.subplots_adjust(top=0.7)
	im = ax.imshow(plotmat,cmap=cmap,vmin=-maxval,vmax=maxval,origin='lower')
	ax.set_xticks(np.arange(N))
	ax.set_yticks(np.arange(N))
	ax.set_xticklabels(np.arange(N)+1)
	ax.set_yticklabels(np.arange(N)+1)
	for tick in ax.yaxis.get_ticklabels():tick.set_fontweight('bold')
	for tick in ax.xaxis.get_ticklabels():tick.set_fontweight('bold')
	ax.set_xlabel(lab)
	ax.set_ylabel(lab)
	#ax.set_title(title)
	f.suptitle(title)
	ax.set_aspect('equal')
	cb = f.colorbar(im)
	cb.set_label('corr.coef.',rotation=-90,labelpad=20)
	f.tight_layout()
	return f

def get_colors(cmapstr,n_elements,**kwargs):
	if 'colors' in kwargs:
		cols = kwargs['colors']
	elif 'col' in kwargs: cols = ['k']*n_elements
	else:
		cmap = mpl.cm.get_cmap(cmapstr)
		if not hasattr(cmap,'colors'):
			cols = cmap(np.linspace(0.05,0.95,n_elements))
		else: cols = concat_cmaps(cmap.colors, n_elements)
	return cols

def plotget_sceleton(axplotfn,aRec,tint,stimplotfn_gen,maskmat,ylab_off=False,loclinecol='grey',show_boundaries=True,**kwargs):

	el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in aRec.el_ids])
	ptint = (np.array(tint) * aRec.sr).astype(int)


	ylim_emg = [-20, 20]
	f, axarr = mpl.pyplot.subplots(3, 2, figsize=(16, 10),
								   gridspec_kw={'height_ratios': [0.1, 1, 0.1], 'width_ratios': [0.05, 1]},
								   sharex='col')
	f.subplots_adjust(top=0.95, bottom=0.065, left=0.05, hspace=0.1,wspace=0.01)
	locax = axarr[1, 0]
	ax = axarr[1, 1]
	emax = axarr[2, 1]
	axarr[0, 0].set_axis_off()
	axarr[2, 0].set_axis_off()

	stax = axarr[0, 1]
	stimplotfn_gen(stax, aRec)
	nchans = len(aRec.el_ids)
	#ylim = [0-yoff, nchans - 1+yoff]
	axplotfn(ax)
	ylim = ax.get_ylim()
	baxes = [ax] if show_boundaries else []
	#print(baxes)
	make_locax(locax, aRec.el_locs[el_inds], cols=['k', 'lightgray'],linecol=loclinecol, boundary_axes=baxes, lim=ylim,**kwargs)  #
	for M in maskmat:
		ax.axvspan(M[0], M[1], color='khaki', zorder=-10, alpha=0.3)
	ax.set_xlim(tint)
	# ax.set_ylim([aRec.el_ids.min()-yoff,aRec.el_ids.max()+yoff])

	cond2 = (aRec.emg_tvec <= tint[1]) & (aRec.emg_tvec >= tint[0])
	emax.plot(aRec.emg_tvec[cond2], aRec.emg_data[cond2], 'k')
	emax.text(0.02, 0.8, 'EMG', color='k', ha='left', va='top', fontweight='bold',
			  bbox={'facecolor': 'w', 'linewidth': 0}, transform=emax.transAxes)
	emax.set_ylim(ylim_emg)
	emax.set_xlabel('time [s]')

	if ylab_off:
		ax.set_yticks([])


	return f,ax


def find_extend(vmin, vmax, datamin, datamax):
	#extend{'neither', 'both', 'min', 'max'}
	if datamin >= vmin:
		if datamax <= vmax:
			extend="neither"
		else:
			extend="max"
	else:
		if datamax <= vmax:
			extend="min"
		else:
			extend="both"
	return extend

def get_cbar_extend(showmat,**kwargs):
	min_ext, max_ext = False, False
	if 'vmin' in kwargs and kwargs['vmin'] > np.nanmin(showmat): min_ext = True
	if 'vmax' in kwargs and kwargs['vmax'] < np.nanmax(showmat): max_ext = True
	if min_ext and max_ext: return 'both'
	elif min_ext and not max_ext: return 'min'
	elif max_ext and not min_ext: return 'max'
	else: return 'neither'

def imshow_on_sceleton(ax,showmat,tint,cmap='jet',clab='',show_cbar=True,widthfac=1/4.,cbar_ext=True,**kwargs):
	yext = kwargs['yext'] if 'yext' in kwargs else [-0.5, showmat.shape[0]-0.5]
	im = ax.imshow(showmat, aspect='auto', cmap=cmap, origin='lower', extent=[*tint,*yext ], **kwargs)
	if show_cbar:

		c_extend = get_cbar_extend(showmat,**kwargs) if cbar_ext else 'neither'

		pos = ax.get_position()
		newwidth = (1 - pos.x1) *widthfac
		f = ax.get_figure()
		cax = f.add_axes([pos.x1 + newwidth, pos.y0, newwidth, pos.height])  # l,b,w,h
		cb = f.colorbar(im, cax=cax,extend=c_extend)
		cb.set_label(clab,rotation=-90,labelpad=20)
	return im

def plot_contours(ax,contours,tint,sr_adjust=1,centroids=None,mode='outline',kwargs_cont={},kwargs_cent={}):
	if mode == 'outline':
		for contour in contours:
			ax.plot(contour[:, 0] / sr_adjust + tint[0], contour[:, 1],**kwargs_cont)
	elif mode == 'filled':
		for contour in contours:
			ax.fill(contour[:,0]/sr_adjust+tint[0],contour[:,1],**kwargs_cont)
	if type(centroids)!= type(None):
		for cent in centroids:
			ax.plot(cent[0]/sr_adjust+tint[0], cent[1],'o',**kwargs_cent)

def plot_rects(ax,rects,tint,sr_adjust=1,centroids=None,mode='outline',kwargs_cont={},kwargs_cent={}):
	if mode == 'outline':
		for rect in rects:
			[x0, x1], [y0, y1] = rect[0]/sr_adjust+tint[0], rect[1]
			ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0],**kwargs_cont)

	elif mode == 'filled':
		for rect in rects:
			[x0, x1], [y0, y1] = rect[0]/sr_adjust+tint[0], rect[1]
			ax.fill([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0],**kwargs_cont)

	if type(centroids)!= type(None):
		for cent in centroids:
			ax.plot(cent[0]/sr_adjust+tint[0], cent[1],'o',**kwargs_cent)

def make_figsaver(figdir,setting_id,id_as_sup=True,create_figdir=False,dpi=100):

	if create_figdir:
		if not os.path.isdir(figdir): os.makedirs(figdir)

	def figsaver(f,figtag):

		if id_as_sup:
			if f._suptitle==None: f.suptitle('%s'%setting_id)
			else: f.suptitle(f._suptitle.get_text()+'    %s'%setting_id,fontsize=8)
			#print ('setting suptitle')

		f.savefig(os.path.join(figdir, '%s__%s.png' % (figtag, setting_id)),dpi=dpi)
		mpl.pyplot.close(f)
	return figsaver


def magnify_labeling():
	mpl.rcParams['']


def gradient_image(ax, extent, cmap, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* kwarg.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, cmap=cmap, interpolation='bicubic',
                   vmin=0, vmax=1, **kwargs)
    return im

def loggify_y(ax):
	ax2 = ax.twinx()
	ax2.set_ylim(10**np.array(ax.get_ylim()))
	ax2.set_yscale('log')
	for pos in ['right','left']:ax.spines[pos].set_visible(False)
	ax2.yaxis.set_label_position('left')
	ax2.yaxis.set_ticks_position('left')
	ax2.set_ylabel(ax.get_ylabel())
	ax.set_ylabel('')
	ax.set_yticks([])
	return ax2

def loggify_x(ax):
	ax2 = ax.twiny()
	ax2.set_xlim(10**np.array(ax.get_xlim()))
	ax2.set_xscale('log')
	for pos in ['bottom','top']:ax.spines[pos].set_visible(False)
	ax2.xaxis.set_label_position('bottom')
	ax2.xaxis.set_ticks_position('bottom')
	ax2.set_xlabel(ax.get_xlabel())
	ax.set_xlabel('')
	ax.set_xticks([])
	return ax2
