from glob import glob
import os
import numpy as np
import h5py
import git


dsdir = 'IBLPATH/IBL_Passive'
dstdir = 'ZENODOPATH/preprocessing/metrics_extraction/timeselections/IBL_Passive'#

tintdur = 3.
ttag = 'passive'
savepattern = '%s__TSEL{}{}.h5'.format(ttag,int(tintdur)) #

myfiles = glob(os.path.join(dsdir,'*.nwb'))
spont_path = '/stimulus/presentation/Spontaneous_activity/timestamps'

for datafile in myfiles:

    with h5py.File(datafile,'r') as hand:
        spont_int = hand[spont_path][()]
        recid = hand['identifier'][()].decode()

    #hack in spontaneous interval into tings
    tintstarts = np.arange(spont_int[0],spont_int[1],tintdur)
    tintmat = np.vstack([tintstarts,tintstarts+tintdur]).T[1:-1]


    savename = os.path.join(dstdir,savepattern % (recid))
    print(savename)

    with h5py.File(savename, 'w') as hand:
        ds = hand.create_dataset('tints', data=tintmat)
        ds.attrs['recid'] = recid
        ds.attrs['ttag'] = ttag

        ds.attrs['pre'] = tintdur
        ds.attrs['post'] = 0
        ds.attrs['githash'] = git.Repo(search_parent_directories=True).head.object.hexsha
        # ds.attrs['srcfile'] = __file__