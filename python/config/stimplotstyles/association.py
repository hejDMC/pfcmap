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
    'lick': {'label': 'lick','color':'r'}}

simplels = '-'
styledict_simple = {'tone': {'color': 'k', 'ls': simplels, 'alpha': 1}, \
                    'valve': {'color': 'cornflowerblue', 'ls': simplels, 'alpha': 1}, \
                    'LICK': {'color': 'r', 'ls': simplels, 'alpha': 1},\
                    'valve_close': {'color': 'cornflowerblue', 'ls': ':', 'alpha': 1}}

def get_stimdict(aRec,ds_path = 'stimulus/presentation'):
    abbrev_dict = {'AudStim_10kHz': 'tone','valve_opening':'valve'}
    stimdict = {abbrev_dict[tag]: aRec.h_info[ds_path][tag]['timestamps'][()] for tag in abbrev_dict.keys()}
    stimdict['valve_close'] = stimdict['valve'] + valvedur
    stimdict['LICK'] = aRec.h_info['processing/behavior/Lick/lick/timestamps'][()]# I know, technically not a stim...
    return stimdict