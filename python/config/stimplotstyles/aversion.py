import numpy as np
from pfcmap.python.utils import stimfuncs as sfn

stimplotfns = {
    'AudStim_10kHz': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps],\
    'Optogenetics': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'm', 'alpha': 1, 'lw': 0}) \
                                            for timestamp in timestamps],\
    'passive_bluenoise': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'royalblue', 'alpha': 1, 'lw': 0}) \
                                                 for timestamp in timestamps],\
    'air_puff':  lambda timestamps,ax: ax.plot(timestamps, np.zeros_like(timestamps) + 0.5, \
                                               **{'mfc': 'grey', 'marker': 'o', 'mec': 'none', 'ms': 5, 'lw': 0})}

labels = {
    'AudStim_10kHz': {'label': '10kHz sound','color':'k'},\
    'Optogenetics': {'label': 'opto','color':'m'},\
    'passive_bluenoise': {'label': 'bluenoise','color':'royalblue'},\
    'air_puff': {'label': 'airpuff','color':'grey'}}

simplels = '-'
styledict_simple = {'tone': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    'bluen': {'color': 'royalblue', 'ls': simplels, 'alpha': 1}, \
                    'air': {'color': 'grey', 'ls': simplels, 'alpha': 1},\
                    'opto':{'color': 'm', 'ls': simplels, 'alpha': 1}}

styledict_blocks = {'tone':{'color':'k','linewidth':0},\
                    'bluen':{'color':'royalblue','linewidth':0},\
                    'air':{'color': 'grey','linewidth':0},
                    'opto':{'color': 'm','linewidth':0}}


def get_blockdict(aRec,stimdict,stimkeys_ascending=['tone', 'bluen', 'opto', 'air']):
    bltimes = sfn.extract_blocktimes(aRec,exclude_zeroblocks=True)
    blocknames = sfn.label_blocks(bltimes, stimdict, stimkeys_ascending)
    return {bb: [blname,bltime] for bb,[blname,bltime] in enumerate(zip(blocknames,bltimes))}



def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'passive_bluenoise': 'bluen', 'air_puff': 'air', 'AudStim_10kHz': 'tone','OptogeneticSeries':'opto'}
    return {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}




