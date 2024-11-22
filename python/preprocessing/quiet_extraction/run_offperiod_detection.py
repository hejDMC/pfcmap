import sys
import yaml
import os
import h5py
import numpy as np

save_ratevec = True
smoothingstyle = 'savgol'
pathpath,recfilename,regstr = sys.argv[1:]
regs_of_interest = [bla.strip() for bla in regstr.strip('[]').replace("'",'').split(',')]

badtimesdir = 'ZENODOPATH/preprocessing/bad_times_manual'


recid0 = os.path.basename(recfilename).split('__')[0]

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python.utils import data_handling as dh



logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])
fnfile = pathdict['configs']['functions']
defaultdictpath = pathdict['configs']['defaultdict']
outdir = pathdict['outdir']

fmod = fhs.retrieve_module_by_path(fnfile)


kword_changes = {}#
F = fmod.DataFuncs(pathdict['code']['recretrieval'],my_regs=regs_of_interest)

P = fmod.Paramgetter(defaultdictpath,**kword_changes)#
logger.info('###STARTING file %s'%recfilename)

F.get_aRec_exptype(recfilename)
logger.info('recid: %s'%recid0)

#checking number of units available in pfc


F.get_allspikes()#generates .allspikes


F.get_select_spiketimes()#thats the pfc-spikes #generates .spikes and .n_units_used

if F.n_units_used<P.nunits_min:
    logger.warning('Too few units here, N=%i --> use just for illustration'%F.n_units_used)

F.aRec.set_recstoptime(F.allspikes.max())#in case you dont have an lfp ready yet

logger.info('calculating avg rate and smoothing, n_units: %i'%F.n_units_used)


ratevec,ratebins = F.get_ratevec(P.bw) #ratebins not needed here
rvec_smoothed = F.smooth_ratevec(ratevec,P,style=smoothingstyle)

badint_file = os.path.join(badtimesdir, '%s__badtimes.h5' % (recid0))
if os.path.isfile(badint_file):
    with h5py.File(badint_file, 'r') as fhand:
        badtimes = fhand['badtints'][()]

    rvec_smoothed = dh.mask_data(rvec_smoothed,  (badtimes.T/P.bw).astype(int))

    #tvec = np.arange(0,len(rvec_smoothed))*P.bw/60.
    #f,ax = plt.subplots()
    #ax.plot(tvec,rvec_smoothed)


logger.info('detecting offperiods')
detdict = F.get_detectiondict(rvec_smoothed,P)


####SAVING
outfile = os.path.join(outdir,'%s__offstates.h5'%(recid0))
logger.info('Writing files to %s'%outfile)

with h5py.File(outfile, 'w') as fdest:

    fdest.attrs['recid'] = F.aRec.id
    usable = 'no' if F.n_units_used < P.nunits_min else '?'
    fdest.attrs['usable'] = usable


    mgroup = fdest.create_group('method')
    mgroup.attrs['n_units_used'] = F.n_units_used
    mgroup.attrs['fnfile'] = fnfile
    mgroup.attrs['paramdict'] = defaultdictpath
    mgroup.attrs['script'] = str(__file__)
    mgroup.attrs['git_hash'] = fhs.get_githash()
    mgroup.attrs['smoothing'] = F.smoothingstyle
    mgroup.attrs['srcfile'] = recfilename

    rgroup = fdest.create_group('results')
    rgroup.attrs['bw'] = P.bw #basically the sr
    for dkey in ['g_mean','thr']:
        rgroup.attrs[dkey] = detdict[dkey]

    for dkey,dsname in zip(['trtimes', 'sbursts', 'borderpts', 'trtimes0'],['offpts','burstpts','borderpts','offpts_nodurlim']):
        rgroup.create_dataset(dsname,data=detdict[dkey],dtype='i')
    #_nodurlim (i.e. trtimes0) also contains <thr that were shorter than mindur off

    if save_ratevec:
        rgroup.create_dataset('ratevec',data=ratevec,dtype='f')#this is the raw one
        rgroup['ratevec'].attrs['comment'] = 'non-smoothed ratevec'

logger.info('###DONE file %s'%recid0)


