import numpy as np
auddur = 0.2
valvedur = 0.5
stimplotfns = {
    'AudStim_10kHz': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps],\
    'valve_opening': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + valvedur, **{'color': 'cornflowerblue', 'alpha': 1, 'lw': 0}) \
                                            for timestamp in timestamps],\
    'lick': lambda timestamps,ax:  [ax.axvline(timestamp, **{'color': 'r', 'alpha': 1, 'lw': 1}) \
                                                 for timestamp in timestamps]}

labels = {
    'AudStim_10kHz': {'label': '10kHz sound','color':'k'},\
    'valve_opening': {'label': 'valve','color':'cornflowerblue'},\
    'LICK': {'label': 'lick','color':'r'},\
    'valve_close': {'color': 'cornflowerblue', 'ls': ':', 'alpha': 1}}

simplels = '-'
styledict_simple = {'tone': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    'valve': {'color': 'cornflowerblue', 'ls': simplels, 'alpha': 1}, \
                    'LICK': {'color': 'r', 'ls': simplels, 'alpha': 0.5,'zorder':-10}}


styledict_blocks = {'det':{'color':'k','linewidth':0}}


def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'AudStim_10kHz': 'tone','valve_opening':'valve'}
    stimdict = {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}
    stimdict['valve_close'] = stimdict['valve'] + valvedur

    lickpath = 'processing/behavior/LickYYY/lick_yyy_data/timestamps'
    lick_starts, lick_stops = [aRec.h_info[lickpath.replace('YYY', tag.capitalize()).replace('yyy', tag)][()] for tag in
                               ['onset', 'offset']]
    stimdict['LICK'] = np.vstack([lick_starts, lick_stops]).T
    return stimdict

def get_blockdict(aRec,stimdict):
    #stimdict = get_stimdict(aRec,ds_path = 'stimulus/presentation')
    allstims = stimdict['tone']
    blockdict = {0:['det',np.array([allstims.min(),allstims.max()])]}
    return blockdict

