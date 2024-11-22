import sys
import yaml
import os
import h5py
import pandas as pd
from scipy.stats import scoreatpercentile as sap
import numpy as np

reftag = '__all'


pathpath = 'PATHS/filepaths_carlen.yml'
runset = 'wwresp'
#runset = 'wwspont'
if runset == 'wwspont':
    spec = 'dur3'
    tsel = 'prestim'
    timeselsubfolder = 'timeselections_quietactive'
else:
    timeselsubfolder = 'timeselections_psth'

states = ['active', 'quiet', 'uncertain']

with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S

outdir = pathdict['figdir_root'] + '/quiet_active_stats'

recids = S.get_allowed_recids_from_table()




def replace_by_dict(mytxt,repldict):
    newtxt = str(mytxt)
    for rkey,rval in repldict.items():
        newtxt = newtxt.replace(rkey,rval)
    return newtxt

tintdict = {}
for recid in recids:
    tintdict[recid] = {}
    tintfilepath = os.path.join(S.timescalepath, timeselsubfolder)
    for mystate in states:
        if runset.count('spont'):
            tintfile =  os.path.join(tintfilepath, '%s__TSEL%s%s__STATE%s%s.h5' % (recid, tsel, spec.replace('dur',''),\
                                                                                       mystate,reftag))
        elif runset.count('resp'):
            repl_dict = {'RECID': recid, 'mystate': mystate, 'REFTAG': reftag}
            responsetint_tag = replace_by_dict(S.responsetint_pattern, repl_dict)
            tintfile =  os.path.join(tintfilepath, responsetint_tag)

        if os.path.isfile(tintfile):
            with h5py.File(tintfile,'r') as hand: ntints = hand['tints'][()].shape[0]
        else:
            ntints = 0
            print('%s: tintfile %s not found, setting ntints=0'%(recid,ntints))
        tintdict[recid][mystate] = ntints

# write all ntints per rec into an xlsx table
df1 = pd.DataFrame.from_dict(tintdict).transpose()

scoredict = {}
percs = [10,25,50,75,90]
for mystate in states:
    scoredict[mystate] = {}
    statevals = np.array([tintdict[recid][mystate] for recid in recids])
    for perc in percs:
        scoreval = sap(statevals,perc,interpolation_method='lower')
        scoredict[mystate][perc] = int(scoreval)
df2 = pd.DataFrame.from_dict(scoredict).transpose()

ratio_dict = {}
for recid in recids:
    ratio_dict[recid] = tintdict[recid]['quiet']/(tintdict[recid]['active']+tintdict[recid]['uncertain'])
df3 = pd.DataFrame(ratio_dict,index=[0]).transpose()

scoredict2 = {}
percs = [10,25,50,75,90]
allratios = np.array(list(ratio_dict.values()))
for perc in percs:
    scoreval = sap(allratios,perc)
    scoredict2[perc] = scoreval
df4 = pd.DataFrame(scoredict2,index=[0]).transpose()

ratio_dict2 = {}
for recid in recids:
    ratio_dict2[recid] = tintdict[recid]['quiet']/(tintdict[recid]['active']+tintdict[recid]['quiet'])
df5 = pd.DataFrame(ratio_dict2,index=[0]).transpose()

scoredict3 = {}
allratios = np.array(list(ratio_dict2.values()))
for perc in percs:
    scoreval = sap(allratios,perc)
    scoredict3[perc] = scoreval
df6 = pd.DataFrame(scoredict3,index=[0]).transpose()



overall_dict = {}
overall_dict['quiet'] =  int(np.sum([tintdict[recid]['quiet'] for recid in recids]))
overall_dict['active']  = int(np.sum([tintdict[recid]['active'] for recid in recids]))
overall_dict['uncertain']  = int(np.sum([tintdict[recid]['uncertain'] for recid in recids]))
overall_dict['q_a+u_ratio'] =  np.sum([tintdict[recid]['quiet'] for recid in recids])/np.sum([tintdict[recid]['active']+tintdict[recid]['uncertain'] for recid in recids])

df7 = pd.DataFrame(overall_dict,index=[0]).transpose()


outfile = os.path.join(outdir,'N_episodes_per_state__%s.xlsx'%(runset.replace('ww','')))
with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='states_per_rec')
    df2.to_excel(writer, sheet_name='percentiles')
    df3.to_excel(writer, sheet_name='N_quiet_active_ratio')

    df4.to_excel(writer, sheet_name='percentiles_quiet_active_ratio')
    df5.to_excel(writer, sheet_name='N_quiet_ratio - Nq by Nq+Na')
    df6.to_excel(writer, sheet_name='percentiles - Nq by Nq+Na')

    df7.to_excel(writer, sheet_name='overall')

