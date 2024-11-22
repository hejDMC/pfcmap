import numpy as np
auddur = 0.2
valvedur = 0.5
stimplotfns = {
    'sound1': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps], \
    'sound2': lambda timestamps, ax: [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'grey', 'alpha': 1, 'lw': 0}) \
                                      for timestamp in timestamps], \
    'valve_opening': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + valvedur, **{'color': 'cornflowerblue', 'alpha': 1, 'lw': 0}) \
                                            for timestamp in timestamps],\
    'lick': lambda timestamps,ax:  [ax.axvline(timestamp, **{'color': 'r', 'alpha': 1, 'lw': 1}) \
                                                 for timestamp in timestamps]}

labels = {
    'sound1': {'label': 'sound1','color':'k'},\
    'sound2': {'label': 'sound1','color':'grey'},\
    'valve_opening': {'label': 'valve','color':'cornflowerblue'},\
    'LICK': {'label': 'lick','color':'r'}}

simplels = '-'
styledict_simple = {'sound1': {'color': 'k', 'ls': simplels, 'alpha': 1,'lw':1}, \
                    'sound2': {'color': 'grey', 'ls': simplels, 'alpha': 1,'lw':1}, \
                    'freq1': {'color': 'blue', 'ls': simplels, 'alpha': 0.8,'lw':5,'zorder':-5}, \
                    'freq2': {'color': 'magenta', 'ls': simplels, 'alpha': 0.8,'lw':5,'zorder':-5}, \
                    'lighton': {'color': 'gold', 'ls': simplels, 'alpha': 1,'lw':2},\
                    'match_MARK': {'marker': '+', 'ms': 10, 'alpha': 1,'lw':5,'color':'k'},\
                    'mismatch_MARK': {'marker': 'x', 'ms': 10, 'alpha': 1,'lw':5,'color':'k'},\
                    'valve': {'color': 'cornflowerblue', 'ls': simplels, 'alpha': 1}, \
                    #'lick': {'color': 'r', 'ls': simplels, 'alpha': 0.7,'zorder':-9},\
                    'LICK': {'color': 'r', 'ls': simplels, 'alpha': 0.5,'zorder':-10}}


styledict_blocks = {'att':{'color':'k','linewidth':0}}


def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'AudStim_first': 'sound1','AudStim_second':'sound2','valve_opening':'valve'}
    sound_freq1,sound_freq2 = [aRec.h_info['intervals/trials/sound_freq%i'%ii][()] for ii in [1,2]]
    match_mismatch = aRec.h_info['intervals/trials/match_mismatch'][()]
    stimdict = {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}
    reffreq = sound_freq1[0]
    stimdict['freq1'] = np.sort(np.r_[stimdict['sound1'][sound_freq1==reffreq],stimdict['sound2'][sound_freq2==reffreq]])
    stimdict['freq2'] = np.sort(np.r_[stimdict['sound1'][sound_freq1!=reffreq],stimdict['sound2'][sound_freq2!=reffreq]])
    stimdict['lighton'] = aRec.h_info['intervals/trials/start_time'][()]
    stimdict['match_MARK'] = stimdict['sound2'][match_mismatch==1]
    stimdict['mismatch_MARK'] = stimdict['sound2'][match_mismatch==0]

    stimdict['valve_close'] = stimdict['valve'] + valvedur

    lickpath = 'processing/behavior/LickYYY/lick_yyy_data/timestamps'
    lick_starts,lick_stops = [aRec.h_info[lickpath.replace('YYY',tag.capitalize()).replace('yyy',tag)][()] for tag in ['onset','offset']]
    stimdict['LICK'] = np.vstack([lick_starts,lick_stops]).T


    return stimdict

def get_blockdict(aRec,stimdict):
    #stimdict = get_stimdict(aRec,ds_path = 'stimulus/presentation')
    allstims = np.sort(np.hstack([stimdict['sound1'],stimdict['sound2']]))
    blockdict = {0:['att',np.array([allstims.min(),allstims.max()])]}
    return blockdict



'''
lickpath = 'processing/behavior/BehavioralTimeSeries/lick_offset_data/timestamps'
try:
    stimdict['lick'] = aRec.h_info[lickpath][()]# I know, technically not a stim...
except:
    print('OMITTING LICK - no data at %s'%lickpath)

lickpath2 = 'processing/behavior/Lick/lick/timestamps'

try:
    stimdict['LICK'] = aRec.h_info[lickpath2][()]  # I know, technically not a stim...
except:
    print('OMITTING LICK - no data at %s' % lickpath2)
'''
