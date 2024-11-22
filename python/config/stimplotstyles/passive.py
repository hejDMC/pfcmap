import numpy as np
stimplotfns = {
    'AudStim_10kHz': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps],\
    'passive_bluenoise': lambda timestamps,ax:  \
        [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'royalblue', 'alpha': 1, 'lw': 0}) for timestamp in timestamps]
}

labels = {
    'passive_10kHz': {'label': '10kHz sound','color':'k'},\
    'passive_1kHz': {'label': '1kHz sound','color':'k'},\
    'passive_2kHz': {'label': '2kHz sound','color':'k'},\
    'passive_5kHz': {'label': '5kHz sound','color':'k'},\
    'passive_bluenoise': {'label': 'bluenoise','color':'royalblue'},\
    'passive_brownnoise': {'label': 'brownnoise','color':'saddlebrown'},\

}

simplels = '-'

styledict_simple = {'10kHz': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    '1kHz': {'color': 'grey', 'ls': simplels, 'alpha': 1}, \
                    '2kHz': {'color': 'grey', 'ls': simplels, 'alpha': 1},\
                    '5kHz':{'color': 'k', 'ls': simplels, 'alpha': 1},\
                    'bluenoise': {'color': 'royalblue', 'ls': simplels, 'alpha': 1},\
                    'brownnoise':{'color': 'chocolate', 'ls': simplels, 'alpha': 1},\
                    }


mykeys = ['passive_10kHz', 'passive_1kHz', 'passive_2kHz', 'passive_5kHz', 'passive_bluenoise', 'passive_brownnoise']




def get_stimdict(aRec,ds_path = 'stimulus/presentation',warn_missing=True):
    abbrev_dict = {mykey: mykey.split('_')[1] for mykey in mykeys}
    avail_keys = [mykey for mykey in mykeys if mykey in aRec.h_info[ds_path]]
    if warn_missing and len(avail_keys)<len(mykeys):
        missing_keys = [key for key in mykeys if not key in avail_keys]
        print('WARN Missing keys %s: %s'%(aRec.id,str(missing_keys)))
    return {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in avail_keys}

styledict_blocks = {'passive':{'color':'k','linewidth':0}}

def get_blockdict(aRec,stimdict):
    #stimdict = get_stimdict(aRec,ds_path = 'stimulus/presentation')
    allstims = np.hstack(list(stimdict.values()))
    blockdict = {0:['passive',np.array([allstims.min(),allstims.max()])]}
    return blockdict
