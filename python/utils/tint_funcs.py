import numpy as np


def prestim_passive(aRec,pre_stim=1.):
    return np.vstack([aRec.ttmat[:, 0] - pre_stim, aRec.ttmat[:, 0]]).T

def stim_passive(aRec,post_stim=0.8):
    return np.vstack([aRec.ttmat[:, 0], aRec.ttmat[:, 1]+post_stim]).T

def baseline_passive(aRec,post_marg=1.):
    return aRec.interttmat[1:-1]+np.array([post_marg,0])[None,:]


def get_stim_by_amp(hand,stimnames,ampselfn=lambda amp:amp>=1.):
    if type(stimnames) == str: stimnames = [stimnames]
    tlist = []
    for stimname in stimnames:
        spath = 'stimulus/presentation/%s'%stimname
        amps = hand[spath]['data'][()]
        tstarts0 = hand[spath]['timestamps'][()]
        tlist += [tstarts0[ampselfn(amps)]]
    return np.sort(np.hstack(tlist))

def get_nostim_times(hand,stimdur=0.2,stimbuff=1,spath_gen='stimulus/presentation',**kwargs):
    if not 'stimnames' in kwargs:
        stimnames = list(hand[spath_gen].keys())
    else:
        stimnames = kwargs['stimnames']
    stimstarts = np.sort(
        np.hstack([hand['stimulus/presentation/%s/timestamps' % stimname][()] for stimname in stimnames]))
    #stops = np.r_[stimstarts]
    freestarts = np.r_[0, stimstarts[:-1] + stimdur + stimbuff]
    frees = np.vstack([freestarts,stimstarts]).T
    return frees[np.diff(frees).flatten()>0]


def get_stimtimes(hand,cutdur=1.,spath_gen='stimulus/presentation',**kwargs):
    if not 'stimnames' in kwargs:
        stimnames = list(hand[spath_gen].keys())
    else:
        stimnames = kwargs['stimnames']
    stimstarts = np.sort(
        np.hstack([hand['stimulus/presentation/%s/timestamps' % stimname][()] for stimname in stimnames]))
    #stops = np.r_[stimstarts]
    return np.vstack([stimstarts,stimstarts+cutdur]).T

def get_prestimtimes(hand,cutdur=1.,spath_gen='stimulus/presentation',**kwargs):
    if not 'stimnames' in kwargs:
        stimnames = list(hand[spath_gen].keys())
    else:
        stimnames = kwargs['stimnames']
    stimstarts = np.sort(
        np.hstack([hand['stimulus/presentation/%s/timestamps' % stimname][()] for stimname in stimnames]))
    #stops = np.r_[stimstarts]
    return np.vstack([stimstarts-cutdur,stimstarts]).T

def select_relative_to_ref(hand,refname,tstamps,refselfn=lambda mydata,refdata:mydata[mydata<=refdata[0]]):
    if refname == 'none': return tstamps
    else:
        spath = 'stimulus/presentation/%s'%refname
        refdata = hand[spath]['timestamps'][()]
        return refselfn(tstamps,refdata)

def cut_around(trefs,pre=0.,post=0.):
    return np.vstack([trefs-pre,trefs+post]).T


def select_cap_trials(tstarts,params):
    N_trials = len(tstarts)

    if params['N_trials'] == 'all':
        return tstarts

    if N_trials < params['N_trials']['Nmin']:
        return None

    elif N_trials == params['N_trials']['Nmin']:
        return tstarts

    else:
        nmax = params['N_trials']['cap']
        seltype = params['N_trials']['selection']
        if seltype == 'random':
            mystarts = np.random.choice(tstarts, nmax, replace=False)
        elif seltype == 'first':
            mystarts = tstarts[:nmax]
        else:
            assert 0, 'invalid selection type: %s' % seltype
        return mystarts

stimkeydict = {'Passive':['passive_10kHz','passive_5kHz'],\
               'Context':'AudStim_10kHz',\
               'Opto':'AudStim_10kHz',\
               'Aversion':'AudStim_10kHz',
               'Association':'AudStim_10kHz',
               'Detection':'AudStim_10kHz'}
refkeydict = {'Passive': 'none',\
              'Context': 'white_noise',\
              'Association':'valve_opening',\
              'Opto':'OptogeneticSeries',\
              'Aversion':'OptogeneticSeries',\
              'Detection': 'none',\
              'Attention': 'none'}#N.B. this is I think not needed for the Opto, as the aud will be replaced by optostim
                # for Detection it is none because it is on trained mice anyways!