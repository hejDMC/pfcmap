import numpy as np
from pfcmap.python.utils import stimfuncs as sfn

stimplotfns = {
    'AudStim_10kHz': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps],\
    'Optogenetics': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'm', 'alpha': 1, 'lw': 0}) \
                                            for timestamp in timestamps]}

labels = {
    'AudStim_10kHz': {'label': '10kHz sound','color':'k'},\
    'Optogenetics': {'label': 'opto','color':'m'}}

simplels = '-'
styledict_simple = {'tone': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    'opto':{'color': 'm', 'ls': simplels, 'alpha': 1}}

def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'AudStim_10kHz': 'tone','OptogeneticSeries':'opto'}
    return {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}


styledict_blocks = {'tone':{'color':'k','linewidth':0},\
                    'opto':{'color': 'm','linewidth':0}}


def get_blockdict(aRec,stimdict,stimkeys_ascending=['opto','tone']):
    #the order of the stimkeys is reversed because the first opto-block starts too early by marking and there are no tones during opto:PL066_20200514_probe0
    bltimes = sfn.extract_blocktimes(aRec)
    blocknames = sfn.label_blocks(bltimes, stimdict, stimkeys_ascending)
    return {bb: [blname,bltime] for bb,[blname,bltime] in enumerate(zip(blocknames,bltimes))}