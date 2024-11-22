import numpy as np
import csv
from . import plotting_basics as pb


get_n_elecs = lambda myhand: myhand['processing/LFP/data'].shape[1] #myhand['/general/extracellular_ephys/electrodes/id'].shape[0]
is_npx = lambda myhand: get_n_elecs(myhand)>150
def get_probetype(myhand,npx_thr=383,silicon_val=32):
    n_elecs = get_n_elecs(myhand)
    if  n_elecs >= npx_thr:
        return 'npx'
    elif n_elecs == silicon_val:
        return 'silicon'
    else:
        return 'unidentified_N%i'%n_elecs

def get_snip(tvec,data,tlim):
    cond = (tvec>=tlim[0]) & (tvec<=tlim[1])
    return data[cond],tvec[cond]


def get_spikelist(RecObj,my_units):
    spikelist = []
    for uu, unit in enumerate(my_units):
        r_idx1, r_idx0 = RecObj.get_ridx(unit)
        spikelist += [RecObj.h_units['units/spike_times'][r_idx0:r_idx1]]
    return spikelist

def grab_spikes_unit(aRec,unit_id,**kwargs):
    r1, r0 = aRec.get_ridx(unit_id)
    spikes0 = aRec.h_units['units/spike_times'][r0:r1]
    if 'toi' in kwargs:#time of interest
        tstart,tstop = kwargs['toi']
        return spikes0[(spikes0>=tstart) & (spikes0<=tstop)]-tstart-aRec.pre_stim
    else: return spikes0

def get_rastermat(unit_ids,grabfn):
    '''kwargs
        toi: [tstart,tstop] #list with start and stop time for time of interest
    output:
     rastermatn nx2: col1 time of spike, col2 id-counter'''
    rastermat = np.empty((0, 2), int)
    for jj,my_unit in enumerate(unit_ids):
        spikes = grabfn(my_unit)
        rastermat = np.r_[rastermat,np.vstack([spikes,np.ones_like(spikes)*jj]).T]
    return rastermat


def get_rastermat_tint(aRec,allspikes,tint,pos_adj_fac=1):
    #for jj, my_unit in enumerate(unit_ids):
    rastermat = np.empty((0, 2))
    for unit_id, unit_elid in zip(aRec.unit_ids, aRec.unit_electrodes):
        r1, r0 = aRec.get_ridx(unit_id)
        spikes0 = allspikes[r0:r1]
        spikes = spikes0[(spikes0 >= tint[0]) & (spikes0 <= tint[1])]
        rastermat = np.r_[rastermat, np.vstack([spikes, np.ones_like(spikes) * unit_elid]).T]

    return rastermat * np.array([1, pos_adj_fac])[None, :]

def get_uid_spike_dict_tint(aRec,allspikes,tint):

    uid_dict = {}
    for unit_id, unit_elid in zip(aRec.unit_ids, aRec.unit_electrodes):
        r1, r0 = aRec.get_ridx(unit_id)
        spikes0 = allspikes[r0:r1]
        uid_dict[unit_id] = spikes0[(spikes0 >= tint[0]) & (spikes0 <= tint[1])]
    return uid_dict

def get_middle_locations(el_locs):
    loclist, Nperloc = get_loclist_nlist(el_locs)
    locblocks = np.r_[0, np.cumsum(Nperloc)]
    return (locblocks[:-1] + np.diff(locblocks) // 2.).astype(int)

def get_middle_locationsOLD(electrode_locations):
    '''input: array of strings specifying electrode locations
    output: index of middle occurrence of each location type'''
    loc_list_unique = get_unique_locs_ordered(electrode_locations)
    loc_counts = np.array([np.count_nonzero(electrode_locations == loc) for loc in loc_list_unique])
    loc_inds = np.r_[0,np.cumsum(loc_counts)[:-1]]
    return loc_inds + (loc_counts*0.5).astype(int)

def get_unique_locs_ordered(electrode_locations):
    loc_list_unique = []
    for loc in electrode_locations:
        if not loc in loc_list_unique: loc_list_unique += [loc]
    return loc_list_unique

def get_loclist_nlist(electrode_locations):
    loclist = []
    N_list = []
    last = 'bla'#dummy
    counter = 0 #dummy
    for loc in electrode_locations:
        if not loc==last:
            N_list += [counter]
            counter = 1
            loclist += [loc]
            last = loc
        else:
            counter+=1
    N_array = np.array(N_list[1:]+[counter])
    return loclist,N_array




def set_middle_locs(aRec, pfc_only=True):
    isin_pfc = lambda my_loc: np.sum([my_loc.count(reg) for reg in aRec.cfg['pfc_set']]) > 0

    aRec.set_eois(aRec.elecs_avail)
    aRec.get_elecs(eoi_only=True)  # find the ones that are resampled
    my_inds = get_middle_locations(aRec.el_locs)
    if pfc_only: my_inds = np.array(
        [ind for ind in my_inds if isin_pfc(aRec.el_locs[ind])])
    my_elecs = aRec.el_ids[my_inds]
    aRec.set_eois(my_elecs)
    aRec.get_elecs(eoi_only=True)





def grab_datachunk(dhand,startstop,channels):
    chaninds = dhand['/processing/LFP/chan_indices'][:]
    coi_inds = np.array([np.where(chaninds==coi)[0][0] for coi in channels])
    sr = get_sr(dhand)
    pstart,pstop = (np.array(startstop)*sr).astype(int)
    return dhand['/processing/LFP/data'][pstart:pstop,coi_inds]

def grab_datacube(dhand,startstops,channels):
    chaninds = dhand['/processing/LFP/chan_indices'][:]
    coi_inds = np.array([np.where(chaninds==coi)[0][0] for coi in channels])
    sr = get_sr(dhand)
    time_pts = (np.array(startstops)*sr).astype(int)
    datacube = np.vstack([[dhand['/processing/LFP/data'][pstart:pstop,coi_inds]] for pstart,pstop in time_pts]).transpose(2,0,1)#chans x trials x timepts
    return datacube


def grab_emgmat(dhand,startstops,emg_sr):
    emg_path = 'processing/ecephys/EMG/ElectricalSeries/data'

    emg_pts = (np.array(startstops) * emg_sr).astype(int)
    return np.vstack([dhand[emg_path][pstart:pstop] for pstart, pstop in emg_pts])


def grab_filter_dataslice(dhand,tint,chan,filtfn=lambda x:x):
    chaninds = dhand['/processing/LFP/chan_indices'][()]
    chidx = int(np.where(chaninds == chan)[0][0])
    #marg = kwargs['safetymarg'] if 'safetymarg' in kwargs else 0
    slice = dhand['/processing/LFP/data'][tint[0]:tint[1], chidx]
    return filtfn(slice)#[marg:-marg]



def grab_filter_datacube(dhand,startstops,channels,filtfn,prepost=[1,1]):
    sr = get_sr(dhand)
    pts_pre, pts_post = (np.array(prepost) * sr).astype(int)
    pre,post = prepost
    tcutout = startstops + np.array([-pre,post])[None,:]
    datachunk = grab_datacube(dhand,tcutout,channels)
    nchans,ntrials,npts = datachunk.shape
    filtchunk = np.vstack([[np.vstack([filtfn(submat[ii,:]) for ii in np.arange(ntrials)])] for submat in datachunk])
    if np.sum(prepost)>0:
        return filtchunk[:,:,pts_pre:-pts_post]
    else:
        return filtchunk

def get_chans_avail(dhand):
    '''dhand is handle to hdf5-file'''
    chaninds = dhand['/processing/LFP/chan_indices'][:]
    return chaninds[chaninds!=-1]

def get_datamat(dhand,channels):
    chaninds = dhand['/processing/LFP/chan_indices'][:]
    coi_inds = np.array([np.where(chaninds==coi)[0][0] for coi in channels])
    return dhand['/processing/LFP/data'][:,coi_inds]

def get_sr(dhand):
    return dhand['/processing/LFP/data'].attrs['rate']

def get_arts(dhandI):
    return np.vstack([dhandI['analysis/LFP saturation %s/timestamps'%key][()] for key in ['start','stop']]).T


def get_frees(artmat,dur,artprepost=[5,5],mindur_free=0,verbose=False):
    artpre,artpost = artprepost
    artbuffered = np.vstack([artmat[:,0]-artpre,artmat[:,1]+artpost]).T
    freetimes = mergeblocks([artbuffered.T], output='free', t_start=0.,t_stop=dur).T
    frees = np.vstack([free for free in freetimes if np.diff(free)>mindur_free])
    if verbose: return artbuffered,freetimes,frees
    else: return frees


def get_artfree_trials(trialtimes,freetimes):
    '''slower, more understandable version in get_artfree_trials2
    trialtimes: ntrials x 2
    freetimes: nfreesnips x 2
    output: 1-D array of trials outside of artifacts'''
    frees_temp = np.vstack([np.array([-2,-1]),freetimes])
    return np.array([tt for tt, [tstart, tstop] in enumerate(trialtimes) if
                                 tstart < frees_temp[frees_temp[:, 0] < tstart][-1][1] and tstop <
                                 frees_temp[frees_temp[:, 0] < tstart][-1][1]])

def get_artfree_trials2(trialtimes,freetimes):
    frees_temp = np.vstack([np.array([-2,-1]),freetimes])
    permitted = []
    for tt,[tstart,tstop] in enumerate(trialtimes):
        fstart,fstop = frees_temp[frees_temp[:,0]<tstart][-1] #the free snippet of interest
        if (tstart<fstop) & (tstop<fstop):
            permitted += [tt]
    return np.array(permitted)


def mask_data(data, startstop_pts):
    startstop_pts = startstop_pts.clip(0)  # for negative point values, set 0!
    # create mask
    artmat = np.zeros(len(data))
    tvec = np.arange(len(data))
    for artstart, artstop in zip(startstop_pts[0], startstop_pts[1]):
        artmat[(tvec > artstart) & (tvec < artstop)] = 1

    if np.max(artmat) == 0:
        return data  # if only negative values were in startstop_pts: return data without mask added
    else:
        return np.ma.array(data, mask=(artmat == 1))  # mask array and return

def mergeblocks(blocklist, output='both', **kwargs):
    '''
    blocklist: items of format 2 x n, n is the number of regions blocked
                item[0]:starts, item[1]:stops
                usage example: [artifactTimes,seizureTimes]
    output: options 'both','block','free'


    EXAMPLE
        artTimes1 = np.array([[300.,400.],[550.,700.],[1000.,1100.],[5000.,5600.]]).T
        artTimes2 = np.array([[410.,460.],[690.,720.],[740.,790.],[800.,1050.],[1060.,1070.]]).T
        blocklist = [artTimes1,artTimes2]
        blockT,freeT = sa.mergeblocks(blocklist,output='both',t_start=0.,t_offset=600.,t_stop=5000.)
    '''

    t_start = kwargs['t_start'] if 't_start' in kwargs else None
    t_stop = kwargs['t_stop'] if 't_stop' in kwargs else None
    t_offset = kwargs['t_offset'] if 't_offset' in kwargs else None

    if 't_start' in kwargs and 't_offset' in kwargs:
        if t_start == t_offset:
            offsetTime = np.array([[], []])
        else:
            offsetTime = np.array([[t_start, t_offset]]).T
        blocklist.append(offsetTime)
    blocklist = [np.clip(item, t_start, t_stop) for item in blocklist]

    blocklist = [subblock[:, np.diff(subblock, axis=0).flatten() > 0] for subblock in blocklist ]#if
                #np.diff(subblock, axis=0).max() > 0




    try: blockmat = np.hstack([block for block in blocklist if np.size(block) > 0])
    except:
        blockmat, freemat = np.empty((2,0)),np.array([t_start,t_stop])[:,None]#that is, there is no art in there!
        return get_returnvals(blockmat, freemat, output)

    blockmatS = blockmat[:, np.argsort(blockmat[0])]
    cleanblock = np.array([[], []])
    nextidx = 0
    while True:
        start = blockmatS[0, nextidx]
        stop = blockmatS[1, nextidx]
        for istart, istop in blockmatS.T:
            if (istart >= start) & (istart <= stop) & (istop >= stop): stop = istop

        # print start,stop
        newvals = np.array([start, stop])[:, None]
        cleanblock = np.hstack([cleanblock, newvals])
        # print nextidx
        if np.size(np.where(blockmatS[0] > stop)) == 0: break
        nextidx = np.where(blockmatS[0] > stop)[0][0]
    all_starts, all_stops = cleanblock

    # all_starts,all_stops = all_starts[all_starts<=all_stops],all_stops[all_starts<=all_stops]
    blockmat = np.vstack([all_starts, all_stops])
    freemat = np.vstack([all_stops[:-1], all_starts[1:]])

    if 't_start' in kwargs:
        if t_start < all_starts[0]: freemat = np.hstack([np.array([[t_start], [all_starts[0]]]), freemat])

    if 't_stop' in kwargs:
        if t_stop > all_stops[-1]: freemat = np.hstack([freemat, np.array([[all_stops[-1]], [t_stop]])])

    return get_returnvals(blockmat,freemat,output)

def get_returnvals(blockmat,freemat,output):
    if output == 'both':
        return blockmat, freemat
    elif output == 'block':
        return blockmat
    elif output == 'free':
        return freemat


def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if not isinstance(element, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def apply_to_free(fn,aRec):
    return [fn(tstart,tstop) for tstart,tstop in aRec.freetimes]

def maskmerge_datalist(dlist,timeint_glob,filltimes,tax=2,sr=500):
    tpts = np.diff(np.array(timeint_glob)*sr)[0].astype(int)
    keepinds = np.delete(np.arange(dlist[0].ndim),tax)
    newshape = np.insert(np.array(dlist[0].shape)[keepinds], tax, tpts)

    temp = np.zeros(newshape) - np.nan
    #print (tpts)
    #print (temp.shape)
    for data, roi in zip(dlist, filltimes):
        pstart, pstop = (roi * sr).astype(int)
        # print(data.shape)
        # print(pstop-pstart)
        # print(temp.shape)
        temp[..., pstart:pstop] = data
    return np.ma.masked_where(np.isnan(temp), temp)

check_inlap = lambda epoc1,epoc2: epoc2[0]<=epoc1[0] and epoc2[1]>=epoc1[0]
check_outlap = lambda epoc1,epoc2:  epoc2[0]<=epoc1[1] and epoc2[1]>=epoc1[1]
check_overspan = lambda epoc1,epoc2: epoc2[0]<=epoc1[0] and epoc2[1]>=epoc1[1]
check_contained = lambda epoc1,epoc2: epoc2[0]>=epoc1[0] and epoc2[1]<=epoc1[1]
check_olap=  lambda e1,e2: check_inlap(e1,e2) or check_outlap(e1,e2) or check_overspan(e1,e2) or check_contained(e1,e2)


def load_spatialConf_csv(sc_filepath,el_inds):

    with open(sc_filepath, 'r') as file:
        file = open(sc_filepath, 'r')
        csvreader = csv.reader(file)
        vals = np.array([row for row in csvreader])

    sc_locs = vals[el_inds, 0]
    sc_conf = vals[el_inds, 1].astype(int)
    return sc_conf,sc_locs

def get_maskmat(aRec):
    maskstarts = aRec.freetimes[:-1,1]
    maskstops = aRec.freetimes[1:,0]
    if aRec.freetimes[0,0]>0: maskstarts,maskstops = np.r_[0,maskstarts],np.r_[aRec.freetimes[0,0],maskstops]
    if aRec.freetimes[-1,1]<aRec.dur: maskstarts,maskstops = np.r_[maskstarts,aRec.freetimes[-1,1]],np.r_[maskstops,aRec.dur]
    return np.vstack([maskstarts,maskstops]).T

def get_midchans(aRec,regionchecker=None):

    my_inds = get_middle_locations(aRec.el_locs)
    if type(regionchecker) != type(None):   xx = np.array([idx for idx in my_inds if regionchecker(aRec.el_locs[idx])])
    else: xx = my_inds
    return aRec.eois[xx],aRec.el_locs[xx]
