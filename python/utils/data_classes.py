from . import data_handling as dh

import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
import io
import os


def get_paths(exptype,pathdict,key_pref='code_dataaccess',key_default='code'):
	mykey = key_pref if key_pref in pathdict else key_default
	cfgpath = os.path.join(pathdict[mykey], 'config/rec_config%s.yml' % exptype)
	dspath = os.path.join(pathdict[mykey], 'config/dspaths.yml')
	return cfgpath,dspath



def storedprop(fn):
    attr_name = '_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop


class Period(object):
	'''Something that is extended in time (has a start, stop and duration)'''
	def __init__(self,start,stop):
		self.start = start
		self.stop = stop

	@property
	def dur(self):
		return self.stop-self.start

	@property
	def roi(self):
		return [self.start,self.stop]


class DataPeriod(Period):

	def __init__(self, start, stop, parentobj=None):
		self.start = start
		self.stop = stop
		self.type = 'DataPeriod'
		if parentobj: self.parent = parentobj


class Rec(DataPeriod):

	def __init__(self,filepath,cfgpath,dspath,replace_stimparams=True):
		#print (filepath,cfgpath)
		#print (id)
		#self.my_id = id
		#self._id = my_id
		self.filepath = filepath
		self.replace_stimparams = replace_stimparams

		with io.open(cfgpath, 'r') as ymlfile: self.cfg = yaml.safe_load(ymlfile)

		with io.open(dspath, 'r') as ymlfile: dspath_temp = yaml.safe_load(ymlfile)


		if filepath.count('.nwb') and not self.filepath.count('XXX'):
			self.filesource = 'orig'
			self.dsdict = dspath_temp['nwb']
		else:
			self.filesource = 'export'
			self.dsdict = dspath_temp['h5export']

		self.get_filehands()

		self.start = 0

	@storedprop
	def id(self):
		return self.h_info['identifier'][()].decode('utf-8')

	def set_id(self,my_id):
		self._id = my_id

	def get_filehands(self):
		if self.filesource == 'orig':self.get_fhands_nwb()
		elif self.filesource == 'export': self.get_fhands_export()

	def get_fhands_nwb(self):
		self.h_info =  h5py.File(self.filepath, 'r')
		self._h_units = self.h_info
		self._h_lfp = self.h_info


	def get_fhands_export(self):
		self.h_info =  h5py.File(self.filepath.replace('XXX', 'info'), 'r')

	
	
	@property
	def h_lfp(self):
		if not hasattr(self,'_h_lfp'):
			self._h_lfp = h5py.File(self.filepath.replace('XXX', 'lfp'), 'r')
		return self._h_lfp

	@property
	def h_units(self):
		if not hasattr(self,'_h_units'):
			self._h_units = h5py.File(self.filepath.replace('XXX', 'units'), 'r')
		return self._h_units

	@storedprop
	def coord_labs(self):
		return ['AP','DV','ML']


	def get_elecs(self,eoi_only=False):

		subhand = self.h_info[self.dsdict['electrodes']]
		
		if eoi_only: myinds = self.eois[:]
		else: myinds = np.arange(0,subhand['id'].shape[0])

		self.el_coords = np.array([subhand[myc][myinds] for myc in self.coord_labs])
		self.el_locs = subhand['location'][myinds].astype('<U10')
		self.el_ids = subhand['id'][myinds]

	@property
	def ellocs_all(self):
		if not hasattr(self,'_ellocs_all'):
			self._ellocs_all =  self.h_info[self.dsdict['electrodes']]['location'][()].astype('<U10')
		return self._ellocs_all
		#return  self.h_info[self.dsdict['electrodes']]['location'][()].astype('<U10')

	def set_ellocs_all(self,ellocs_all):
		#NB: this is to handle the id-labeling in the IBL dataset
		self._ellocs_all = ellocs_all
	@property
	def el_columns(self):
		try:
			path = self.dsdict['chan_bools']
			colbools = self.h_info[path][()]
			cols = colbools[self.el_ids]
		except:
			cols = 'NA'
		return cols

	@property
	def sr(self):
		if not hasattr(self,'_sr'):
			self._sr = self.h_lfp[self.dsdict['sr_path']].attrs['rate']
		return self._sr

	def set_sr(self,sr):
		self._sr = sr



	@storedprop
	def datalen_pts(self):
		if self.dsdict['LFP_data'] in self.h_lfp:
			return self.h_lfp[self.dsdict['LFP_data']].shape[0]
		else:
			return None




	@property
	def stop(self):
		if not hasattr(self,'_stop'):
			'''thats equal to dur'''
			if type(self.datalen_pts) == type(None):#when there is no lfp in the nwb
				maxspiketime = np.max(self.h_units['units/spike_times'][()])
				print('getting dur from units, maxspiketime: %imin'%(maxspiketime/60.))

				self._stop =  maxspiketime
			else: self._stop = self.datalen_pts/self.sr
		return self._stop

	def set_recstoptime(self,recstoptime):
		#useful when you have no lfp handle, in seconds
		self._stop = recstoptime

	# @storedprop
	# def ref_elecs(self):
	# 	path = self.dsdict['chan_inds']
	# 	chaninds = self.h_lfp[path][()]
	# 	return np.array([chan for chan in self.cfg['ref_elecs'] if chan in chaninds])

	@storedprop
	def elecs_avail(self):
		if	self.filesource == 'orig':
			subhand = self.h_info['/general/extracellular_ephys/electrodes']
			return np.arange(0, subhand['id'].shape[0])
		else:
			path = self.dsdict['chan_inds']
			chaninds = self.h_lfp[path][()]
			chaninds = chaninds[chaninds!=-1]
			return chaninds

	def get_column_chans(self,col_bool=1,channel_positions=None):
		'''channel_positions: n x 2 only needed if format nwb'''
		if self.filesource == 'export':
			path = self.dsdict['chan_bools']
			colbools = self.h_info[path][()]
			thiscolinds = np.where(colbools == col_bool)[0]
			channels = np.array([el for el in self.elecs_avail if el in thiscolinds])
		elif self.filesource == 'nbw':
			assert type(channel_positions)!=type(None), 'need to insert file with channel postions for nwb'
			hpos, vpos = channel_positions.T
			horval_selected = np.unique(hpos)[col_bool]
			sel_inds = np.where(hpos==horval_selected)[0]
			channels = np.zeros(len(hpos),dtype=int)
			channels[sel_inds] = 1
		return channels


	def set_eois(self,eois):
		'''Electrodes of interest, must be electrode id!'''
		self.eois = eois
		self.eoi_inds = np.array([np.where(self.elecs_avail == eoi)[0][0] for eoi in self.eois])


	def get_datamat(self):
		'''this is of the whole trial'''
		#self.datamat = dh.get_datamat(self._h_lfp,self.eois)
		path = self.dsdict['LFP_data']
		return self.h_lfp[path][:,self.eoi_inds]

	@storedprop
	def tvec(self):
		return np.linspace(self.start, self.stop, self.datalen_pts)

	def get_artparams(self):
		for dkey,attrname in zip(['mindur_free','pre','post'],['mindur_free','artpre','artpost']):
			setattr(self,attrname,self.cfg['artifacts'][dkey])

	@storedprop
	def artmat(self):
		dspath = self.dsdict['LFP_artifacts']
		if 'artifacts' in self.h_lfp:
			myartmat = self.h_lfp['artifacts'][()]#the self-edited artifacts
			print('retrieving edited artifacts from export file')
			return myartmat
		else:
			return np.vstack([self.h_info[dspath.replace('XXX',key)][()] for key in ['start', 'stop']]).T

	@property
	def lickmat(self):
		if not hasattr(self,'_lickmat'):
			try:
				lickpath = 'processing/behavior/LickYYY/lick_yyy_data/timestamps'
				lick_starts, lick_stops = [self.h_info[lickpath.replace('YYY', tag.capitalize()).replace('yyy', tag)][()] for
										   tag in ['onset', 'offset']]
				self._lickmat = np.vstack([lick_starts, lick_stops]).T
				print('extracted lickmat')
			except:
				self._lickmat = np.empty((0,2))
		return self._lickmat

	@property
	def spikesatmat(self):
		if not hasattr(self,'_spikesatmat'):
			temp = np.vstack(
				[self.h_info['analysis/spike saturation %s/timestamps' % (mystr)][()] for mystr in ['start', 'stop']]).T
			hasnan = np.isnan(temp.min())
			if np.size(temp)==2 and hasnan:
				self._spikesatmat = np.empty((0,2))
			elif hasnan and np.size(temp)>2:
				assert 0, 'strange spikesatmat %s'%(str(temp))
			else:
				self._spikesatmat = temp
		return self._spikesatmat


	def get_freetimes(self):
		if np.isnan(self.artmat).all():
			self.freetimes = np.array([0,self.dur])[None,:]
			self.artblockmat = np.empty((2,0))
			self.artbuffered = np.empty((0,2))
			self.freetimes0 = np.empty((0,2))

		else:
			if not hasattr(self,'artpre'): self.get_artparams()
			self.artbuffered = np.vstack([self.artmat[:,0]-self.artpre,self.artmat[:,1]+self.artpost]).T
			self.artblockmat, self.freetimes0 = dh.mergeblocks([self.artbuffered.T], output='both', t_start=0., t_stop=self.dur)
			self.freetimes0 = self.freetimes0.T
			self.freetimes = np.vstack([free for free in self.freetimes0 if np.diff(free)>self.mindur_free])

	def get_baseline(self,prestim_marg=0.,poststim_marg=1.):
		tmat = self.ttmat + np.array([prestim_marg,poststim_marg])
		shortfrees = np.vstack([free for free in self.freetimes0 if np.diff(free)<self.mindur_free])
		_,blinetimes = dh.mergeblocks([self.artbuffered.T,tmat.T,shortfrees.T], output='both', t_start=0., t_stop=self.dur)
		self.baselinetimes = blinetimes.T

	@storedprop
	def interttmat(self):
		#NB this also includes the time intervals before the first and after the last trial
		return np.vstack([np.r_[0, self.ttmat[:, 1]], np.r_[self.ttmat[:, 0], self.dur]]).T

	@storedprop
	def trial_ids(self):
		return self.h_info[self.dsdict['trialids']][()]

	@storedprop
	def ttmat(self):
		'''matrix of trial times'''
		if dh.keys_exists(self.cfg, 'stimulus', 'dur') and self.replace_stimparams:
			path = self.dsdict['trialXXX_time'].replace('XXX','start')
			self._trial_dur = self.cfg['stimulus']['dur']
			starts = self.h_info[path][()]
			return np.vstack([starts, starts + self._trial_dur]).T
		else:
			path = self.dsdict['trialXXX_time']
			return np.vstack([self.h_info[path.replace('XXX',key)][()] for key in ['start','stop']]).T

	@storedprop
	def ttype_vec(self):
		allttypes = np.array(['not_available'] * self.ttmat.shape[0])
		for ttype in self.stimdict.keys():
			inds = self.stimdict[ttype][1]
			allttypes[np.array(inds).astype(int)] = ttype
		return allttypes

	@storedprop
	def trial_dur(self):
		return np.unique(np.diff(self.ttmat))[0]

	@storedprop
	def trialparams_orig(self):
		return [key for key in self.h_info['intervals/trials'].keys() if not key in self.cfg['boring_trialkeys']]

	@storedprop
	def trial_params(self):
		if dh.keys_exists(self.cfg, 'stimulus', 'name_replacedict'):
			#renaming stimuli here
			repldict = self.cfg['stimulus']['name_replacedict']
			return [[mystr.replace(key,val) for mystr in self.trialparams_orig] for key, val in repldict.items()][0]
		else: return self.trialparams_orig

	@storedprop
	def bool_params(self):
		return [tag for tt, tag in enumerate(self.trial_params) if
				  ((self.trialset[:, tt] == 0) | (self.trialset[:, tt] == 1)).all() and not (self.trialset[:, tt] == 1).all()]#everyting==1 is boring

	@storedprop
	def nonbinary_params(self):
		return [tag for tag in self.trial_params if not tag in self.bool_params if not (self.trialset[:,self.trial_params.index(tag)]==1).all()]

	@storedprop
	def nonbinary_inddict(self):
		'''useful for plotting'''
		nonb_inddict = {}
		for cond in self.nonbinary_params:
			condarray = self.trialset[:, self.trial_params.index(cond)]
			uniques = np.unique(condarray)
			nonb_inddict[cond] = np.array([int(np.argwhere(uniques == val)) for val in condarray])
		return nonb_inddict

	@storedprop
	def trialset_orig(self):
		return np.vstack([self.h_info['intervals/trials/%s'%key][()] for key in self.trialparams_orig]).T

	@storedprop
	def trialset(self):
		from copy import copy
		if dh.keys_exists(self.cfg,'stimulus','param_replacedict'):
			trialset = copy(self.trialset_orig)
			repl_dict = self.cfg['stimulus']['param_replacedict']
			for stimparam,subdict in repl_dict.items():
				idx = self.trial_params.index(stimparam)
				vals = trialset[:,idx]
				for oldval,newval in subdict.items():vals[vals==oldval] = newval
				trialset[:,idx] = vals
			return trialset
		else:
			return self.trialset_orig

	@storedprop
	def trial_types(self):
		return np.unique(self.trialset,axis=0)

	@storedprop
	def stimdict0(self):
		'''before removal of artifact-overlapping trials'''
		return {'ttype_%s'%str(tt+1).zfill(2):[ttype,[ii for ii,row in enumerate(self.trialset) if (row==ttype).all()]] for tt,ttype in enumerate(self.trial_types)}

	@storedprop
	def stimdict(self):
		'''after removal of artifact-overlapping trials'''
		return {'ttype_%s'%str(tt+1).zfill(2):[ttype,[ii for ii,row in enumerate(self.trialset) if (row==ttype).all() and ii in self.trials_artfree]]\
            for tt,ttype in enumerate(self.trial_types)}

	def get_matching_ttypes(self,query_dict):
		inds, targetvals = np.vstack([[self.trial_params.index(tag), val] for tag, val in query_dict.items()]).T
		inds = inds.astype(int)
		is_querymatch = lambda ttype_array: np.sum(
			[ttype_array[idx] == targetval for idx, targetval in zip(inds, targetvals)]) == len(inds)
		return [ttype_key for ttype_key, entry in self.stimdict.items() if is_querymatch(entry[0])]


	def get_prepoststim(self):
		self.pre_stim = self.cfg['trial_cutout']['pre_stim']
		self.post_stim = self.cfg['trial_cutout']['post_stim']

	@storedprop
	def cutout_times(self):
		if not hasattr(self,'pre_stim'): self.get_prepoststim()
		return self.ttmat + np.array([-self.pre_stim, self.post_stim])

	@storedprop
	def trials_artfree(self):
		'''trials not overlapping with artifacts
		slower, more understandable version in dh.get_artfree_trials2
		--> 1-D array of trial ids outside of artifacts'''
		if not hasattr(self,'freetimes'): self.get_freetimes()
		frees_temp = np.vstack([np.array([-2, -1]), self.freetimes])
		return np.array([tt for tt, [tstart, tstop] in enumerate(self.cutout_times) if
						 tstart < frees_temp[frees_temp[:, 0] < tstart][-1][1] and tstop <
						 frees_temp[frees_temp[:, 0] < tstart][-1][1]])

	@storedprop
	def emg_data(self):
		return self.h_info[self.dsdict['emg']+'/data'][()]

	@storedprop
	def emg_sr(self):
		return self.h_info[self.dsdict['emg']+'/starting_time'].attrs['rate']

	@storedprop
	def emg_tvec(self):
		return np.linspace(0,len(self.emg_data)/self.emg_sr,len(self.emg_data))

	@storedprop
	def blink_data(self):
		return self.h_info['processing/behavior/Blink/blink/data'][()]#AU

	@storedprop
	def blink_tvec(self):
		return self.h_info['processing/behavior/Blink/blink/timestamps'][()]

	@storedprop
	def pupil_data(self):
		return self.h_info['processing/behavior/PupilTracking/eye_area/data'][()]

	@storedprop
	def pupil_tvec(self):
		return self.h_info['processing/behavior/PupilTracking/eye_area/timestamps'][()]

	@storedprop
	def face_data(self):
		return self.h_info['processing/behavior/BehavioralTimeSeries/face_motion_data/data'][()]#AU

	@storedprop
	def face_tvec(self):
		return self.h_info['processing/behavior/BehavioralTimeSeries/face_motion_data/timestamps'][()]

	@storedprop
	def unit_ids(self):
		return self.h_units['units/id'][()]

	@storedprop
	def unit_idx(self):
		return self.h_units['units/spike_times_index'][()]

	@storedprop
	def unit_electrodes(self):
		return self.h_units['units/electrodes'][()]


	@storedprop
	def rs_bools(self):
		return self.h_units['units/RS'][()]

	get_ridx = lambda self,my_unit: [self.unit_idx[my_unit], [self.unit_idx[my_unit - 1] if my_unit > 0 else 0][0]]


	@property
	def unique_locs(self):
		return dh.get_unique_locs_ordered(self.el_locs)

	@property
	def N_unique_locs(self):
		return len(self.unique_locs)

	@storedprop
	def classtype(self):
		return self.__class__.__name__

	def get_blocks(self, prepost_adj=True):
		bldata = self.trialset[:, self.trial_params.index('Block')]
		blstart_inds = np.r_[0, np.where(np.abs(np.diff(bldata)) > 0)[0] + 1]
		blstarts = self.ttmat[blstart_inds, 0]
		blstop_inds = np.r_[blstart_inds[1:] - 1, -1]
		blstops = self.ttmat[blstop_inds, 1]
		if prepost_adj:
			if not hasattr(self,'pre_stim'): self.get_prepoststim()
			blstarts -= self.pre_stim
			blstops += self.post_stim
		return {'starts': blstarts, 'stops': blstops, 'id': bldata[blstart_inds]}

	def ttype_to_str(self,ttype):
		if ttype == 'not_available': return 'NA'
		else:
			return [': '.join([tr_param.replace('sound', '').replace('_', ''), str(val)]) for tr_param, val in \
						 zip(self.trial_params, self.stimdict[ttype][0]) if not val == 0 and not tr_param == 'sound']

	def set_spatial_conf(self,spat_conf,spat_locs):
		self.spatial_conf = spat_conf
		self.spatial_loc = spat_locs


class RecPassive(Rec):

	@storedprop
	def stimkeys(self):
		return list(self.h_info['stimulus/presentation'].keys())

	@storedprop
	def _setlabs(self):
		return ['sound_puretone' if key.count('Hz') else 'sound_' + key.split('_')[1] for key in self.stimkeys]

	@storedprop
	def _ulabs(self):
		return  list(np.unique(self._setlabs))

	@storedprop
	def trial_params(self):
		return ['sound','sound_amp','sound_freq']+self._ulabs

	@storedprop
	def ttmat(self):
		genpath = self.dsdict['stimpres']
		trial_dur = self.cfg['stimulus']['dur']
		tstamps = np.array([])
		for key in self.h_info[genpath]:
			this = self.h_info[genpath][key]
			if 'timestamps' in this.keys():
				tstamps = np.hstack([tstamps, this['timestamps'][()]])
		tstarts = np.sort(tstamps)
		return np.vstack([tstarts, tstarts + trial_dur]).T


	@storedprop
	def trialset_orig(self):
		str_to_Hz = lambda mystr: int(mystr.split('_')[1].split('kHz')[0]) * 1000
		subhand = self.h_info[self.dsdict['stimpres']]
		amps0 = [subhand['%s/data' % key][()] for key in self.stimkeys]
		tpts0 = [subhand['%s/timestamps' % key][()] for key in self.stimkeys]
		freqs0 = [np.array([str_to_Hz(key)] * len(tpts0[ii])) if key.count('Hz') else np.array([0] * len(tpts0[ii])) for
				  ii, key in enumerate(self.stimkeys)]

		tonedict = {lab: np.hstack([np.array([1]*len(tpts0[ii])) if self._setlabs[ii]==lab else np.array([0]*len(tpts0[ii])) for ii,key in enumerate(self.stimkeys)]) for lab in self._ulabs}

		amps,tpts,freqs = np.hstack(amps0),np.hstack(tpts0),np.hstack(freqs0)

		sortinds = np.argsort(tpts)
		Ntrials = len(amps)
		Nparams = len(self.trial_params)
		trialset = np.zeros((Ntrials,Nparams))
		trialset[:,self.trial_params.index('sound')] = 1
		trialset[:,self.trial_params.index('sound_amp')] = amps[sortinds]
		trialset[:,self.trial_params.index('sound_freq')] = freqs[sortinds]
		for lab in self._ulabs:
			trialset[:,self.trial_params.index(lab)] = tonedict[lab][sortinds]
		return trialset



class RecCtxt(Rec):
	@storedprop
	def boring_trials_keys(self):
		return ['id', 'start_time', 'stop_time']

	@storedprop
	def licktimes(self):
		return self.h_info['processing/behavior/Lick/lick/timestamps'][()]

	@storedprop
	def wn_bool(self):
		data = self.h_info['stimulus/presentation/white_noise/data'][()]
		result = np.r_[data, data[-2:]] if len(data) == (len(self.wn_times) - 2) else data
		return result

	@storedprop
	def wn_times(self):
		return self.h_info['stimulus/presentation/white_noise/timestamps'][()]

	@storedprop
	def vopen_times(self):
		return self.h_info['stimulus/presentation/valve_opening/timestamps'][()]

class TrialCollection(object):

	def __init__(self,ttype,id='NA',**kwargs):
		self.id = id
		self.ttype = ttype
		if 'parentobj' in kwargs:
			self.parent = kwargs['parentobj']
			self.conds, self.trialids = self.parent.stimdict[self.ttype]


	def set_baddict(self,baddict):
		self.baddict = baddict

	@storedprop
	def bad_trials(self):
		bad_trials = []
		if self.parent.id in self.baddict.keys():
			for tid in self.baddict[self.parent.id]['bad_trialids']:
				if tid in self.trialids: bad_trials+=[tid]
		return np.array(bad_trials)

	def purge_badtrials(self):
		for badtrial in self.bad_trials:
			self.trialids.remove(badtrial)
			print('purged badtrial %i' % badtrial)

	@storedprop
	def descr_list(self):
		dlist = []
		for ii,key in enumerate(self.parent.trial_params):
			if key in self.parent.bool_params: mystr = '%s: %i'%(key,self.parent.stimdict[self.ttype][0][ii])
			elif key in self.parent.nonbinary_params: mystr = '%s: %1.2f'%(key,self.parent.stimdict[self.ttype][0][ii])
			if key in self.parent.bool_params or key in self.parent.nonbinary_params: dlist += [mystr]
		return dlist

	@storedprop
	def descr_str(self):
		return ', '.join(self.descr_list)

		
	@storedprop
	def ttmat(self):
		return self.parent.ttmat[self.trialids]

	@storedprop
	def cutout_times(self):
		return self.parent.cutout_times[self.trialids]


	@property
	def eois(self):
		if not hasattr(self,'_eois'):
			self._eois =  self.parent.eois[:]
		return self._eois


	def set_eois(self, eois):
		'''Electrodes of interest, must be electrode id!
		only set if different from parent!'''
		self._eois = eois
		self._eoi_inds = np.array([np.where(self.parent.elecs_avail == eoi)[0][0] for eoi in eois])

	@property
	def eoi_inds(self):
		if not hasattr(self, '_eoi_inds'):
			self._eoi_inds = np.array([np.where(self.parent.elecs_avail == eoi)[0][0] for eoi in self.eois])
		return self._eoi_inds

	@storedprop
	def cutout_dur(self):
		return np.unique(np.diff(self.cutout_times))[0]

	@storedprop
	def trial_dur(self):
		return np.unique(np.diff(self.ttmat))[0]

	@storedprop
	def cutout_pts(self):
		return (self.cutout_times * self.parent.sr).astype(int)

	@storedprop
	def trial_pts(self):
		return (self.ttmat * self.parent.sr).astype(int)

	def get_datamat(self,**kwargs):
		eoi_inds = kwargs['eoi_inds'] if 'eoi_inds' in kwargs else self.parent.eoi_inds
		hand = self.parent.h_lfp[self.parent.dsdict['LFP_data']]
		if self.cutout_pts.size == 2:
			pstart,pstop = self.cutout_pts
			return np.vstack([hand[pstart:pstop,eoi_inds]])
		else:
			return np.vstack([[hand[pstart:pstop,eoi_inds]] for pstart,pstop in self.cutout_pts]).transpose(2,0,1)# chans x trials x tpts

	@storedprop
	def tvec(self):
		return np.arange(0,self.cutout_dur,1/self.parent.sr)-self.parent.pre_stim



class Trial(TrialCollection):

	def __init__(self,id,**kwargs):
		self.id = id
		self.trialids = np.array(self.id)
		if 'collection' in kwargs:
			self.collection = kwargs['collection']
			self.ttype = self.collection.ttype
			if hasattr(self.collection,'parent'):
				self.parent = self.collection.parent

	@storedprop
	def N_before(self):
		'''number of same trials before'''
		return np.where(np.array(self.parent.stimdict0[self.collection.ttype][-1])==self.id)[0][0]


	def get_emg(self):
		self.emg_data,emgt0 = dh.get_snip(self.parent.emg_tvec,self.parent.emg_data,self.cutout_times)
		self.emg_tvec = emgt0-self.ttmat[0]




class Unit(object):
    def __init__(self,uid,parent_rec,**kwargs):
        self.id = uid
        self.rec = parent_rec
        if 'spikes' in kwargs: self.set_spikes(kwargs['spikes'])

    @storedprop
    def el_id(self):
        idx = np.where(self.rec.unit_ids==self.id)[0][0]
        return self.rec.unit_electrodes[idx]

    @storedprop
    def loc(self):
        return self.rec.ellocs_all[self.el_id]

    @property
    def infostr(self):
        return 'uid %i, elid %i, %s'%(self.id,self.el_id,self.loc)

    @property
    def spikes(self):
        if not hasattr(self,'_spikes'):
            self._spikes = dh.grab_spikes_unit(self.rec, self.id)
        return self._spikes

    def set_spikes(self,spikes):
        self._spikes = spikes