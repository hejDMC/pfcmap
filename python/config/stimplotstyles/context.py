import numpy as np

auddur = 0.2
valvedur = 0.5
stimplotfns = {
    'AudStim_10kHz': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + 0.2, **{'color': 'k', 'alpha': 1, 'lw': 0}) \
                                             for timestamp in timestamps],\
    'valve_opening': lambda timestamps,ax:  [ax.axvspan(timestamp, timestamp + valvedur, **{'color': 'cornflowerblue', 'alpha': 1, 'lw': 0}) \
                                            for timestamp in timestamps],\
    'lick': lambda timestamps,ax:  [ax.axvline(timestamp, **{'color': 'r', 'alpha': 1, 'lw': 1}) \
                                                 for timestamp in timestamps], \
    'white_noise': lambda timestamps, ax: [ax.axvline(t0,t1, **{'color': 'r', 'alpha': 0.5, 'lw': 0,'zorder':-20}) \
                                           for t0, t1 in timestamps.reshape(-1, 2)]
}

labels = {
    'AudStim_10kHz': {'label': '10kHz sound','color':'k'},\
    'valve_opening': {'label': 'valve','color':'cornflowerblue'},\
    'LICK': {'label': 'lick','color':'r'},\
    'white_noise': {'label': 'wnoise','color':'grey'}}

simplels = '-'
styledict_simple = {'tone': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    'valve': {'color': 'cornflowerblue', 'ls': simplels, 'alpha': 1}, \
                    'NOISE': {'color': 'grey', 'ls': simplels, 'alpha': 0.5,'lw':4,'zorder':-20},\
                    'LICK': {'color': 'r', 'ls': simplels, 'alpha': 0.5,'zorder':-10},\
                    'valve_close': {'color': 'cornflowerblue', 'ls': ':', 'alpha': 1}}
def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'AudStim_10kHz': 'tone','valve_opening':'valve','white_noise':'NOISE'}
    stimdict = {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}
    stimdict['valve_close'] = stimdict['valve'] + valvedur

    lickpath = 'processing/behavior/LickYYY/lick_yyy_data/timestamps'
    try:
        lick_starts, lick_stops = [aRec.h_info[lickpath.replace('YYY', tag.capitalize()).replace('yyy', tag)][()] for tag in
                                   ['onset', 'offset']]
        stimdict['LICK'] = np.vstack([lick_starts, lick_stops]).T
    except:
        print('WARNING: no licks found at %s - not plotting licks'%lickpath)
    return stimdict

def get_blockdict(aRec,stimdict):

    blocks = aRec.h_info['intervals']['trials']['Block'][()].astype(int)
    starts = aRec.h_info['intervals']['trials']['start_time'][()]
    stops = aRec.h_info['intervals']['trials']['stop_time'][()]
    #diffblocks = np.diff(blocks)
    #print('XXXXXXXXXXX',diffblocks.shape,blocks.shape)
    #print(diffblocks[:10])
    blockstartinds = np.r_[0, np.where(np.sign(np.diff(blocks)) != 0)[0]+1]
    blockstopinds = np.r_[blockstartinds[1:]-1,len(blocks)-1]
    n_blocks = len(blockstartinds)

    bltimes = np.zeros((n_blocks,2))
    for bb,[bind0,bind1] in enumerate(zip(blockstartinds,blockstopinds)):
        bltimes[bb] = np.array([starts[bind0],stops[bind1]])
    blnames = ['off' if bbool==0 else 'on' for bbool in blocks[blockstartinds] ]
    return {bb: [blname,bltime] for bb,[blname,bltime] in enumerate(zip(blnames,bltimes))}

styledict_blocks = {'off':{'color':'k','linewidth':0},\
                    'on':{'color':'silver','linewidth':0}}
