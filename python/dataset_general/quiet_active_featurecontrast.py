import h5py
from glob import glob
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import sem
import pandas as pd

varnames = ['B_mean','M_mean','rate_mean']


pfc_only = True
sg_on = True


pathpath = 'PATHS/filepaths_carlen.yml'
pathpath2 = 'PATHS/filepaths_IBL.yml'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
with open(pathpath2, 'r') as myfile: pathdict2 = yaml.safe_load(myfile)

srcdir = pathdict['datapath_metricsextraction'] +'/quantities_all_meanvar/Carlen_quietactive'
srcdir2 = pathdict2['datapath_metricsextraction']+ '/quantities_all_meanvar/IBL_Passive'

figdir_mother = pathdict['figdir_root'] + '/quiet_active_IBL_featurecontrast/QAI_feature_distributions'

if pfc_only:
    figdir_mother+='_PFC'
    addtag = '_PFC'
else:
    addtag = ''


metricdir = pathdict['src_dirs']['metrics']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python import settings as S
from pfcmap.python.utils import filtering as filt


#metricsfiles = glob(os.path.join(metricdir,'*.h5'))


def get_check_bools(hand,utype='all',**kwargs):
    #this should correspond to what is checked in unitloader, note that the rec itself is not checked here! if unit counts don't match this is a possible source of mismatch
    #featmat is nvars x nuints
    wquality = hand['waveform_metrics/waveform_quality'][()].astype(bool)
    qquality = hand['quality/quality'][()].astype(bool)
    if utype == 'all':
        utypebool = np.ones_like(qquality).astype(bool)
    else:
        utypebool = hand['unit_type/%s'%utype]

    boolvec = wquality & qquality & utypebool

    if 'featmat' in kwargs:
        featcomplete_bool =  ~np.isnan(featmat.mean(axis=0))
        boolvec = boolvec & featcomplete_bool

    if pfc_only:
        regs = np.array([el.decode() for el in  mhand['anatomy']['location'][()]])
        is_pfc_bool = np.array([S.check_pfc_full(reg) for reg in regs])
        boolvec = boolvec & is_pfc_bool

    return boolvec.astype(bool)


stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

statetags = ['quiet','active']
utags = ['nw','ww']
ddict = {tag:{utag:[] for utag in utags+['rec']} for tag in statetags}



for statetag in statetags:
    flav_pattern = '*TSELprestim3__STATE%s*__all*.h5'%statetag
    filepool = glob(os.path.join(srcdir,flav_pattern))

    for mvfile in filepool:
        recname = os.path.basename(mvfile).split('__')[0]
        print(recname)
        mfile = glob(os.path.join(metricdir,'*%s*.h5'%recname))
        assert len(mfile) ==1, 'not exactly one metricsfile, n=%i'%(len(mfile))
        ddict[statetag]['rec'] += [recname]
        with h5py.File(mfile[0],'r') as mhand:
            muids = mhand['uids'][()]


            with h5py.File(mvfile,'r') as hand:
                uids = hand['uids'][()]


                assert (uids==muids).all()
                for utype in utags:
                    featmat = np.vstack([hand['seg/%s'%varname][()] for varname in varnames])
                    boolvec = get_check_bools(mhand,utype=utype,featmat=featmat)
                    featmat_sel = featmat[:,boolvec]
                    ddict[statetag][utype] += [featmat_sel]

print('Number of Carlen ww %i'%(np.sum([el.shape[1] for el in ddict['active']['ww']])))#great

filepool2 = glob(os.path.join(srcdir2,'*'))
metricdir2 = pathdict2['src_dirs']['metrics']
ddict.update({'IBL':{utag:[] for utag in utags+['rec']}})

for mvfile in filepool2:
    recname = os.path.basename(mvfile).split('__')[0]
    print(recname)
    mfile = glob(os.path.join(metricdir2, '*%s*.h5' % recname))
    assert len(mfile) == 1, 'not exactly one metricsfile, n=%i' % (len(mfile))
    ddict['IBL']['rec'] += [recname]
    with h5py.File(mfile[0], 'r') as mhand:
        muids = mhand['uids'][()]
        with h5py.File(mvfile, 'r') as hand:
            uids = hand['uids'][()]
            assert (uids == muids).all()
            for utype in utags:
                featmat = np.vstack([hand['seg/%s' % varname][()] for varname in varnames])
                boolvec = get_check_bools(mhand, utype=utype, featmat=featmat)
                featmat_sel = featmat[:, boolvec]
                ddict['IBL'][utype] += [featmat_sel]

print('Number of IBL ww %i'%(np.sum([el.shape[1] for el in ddict['IBL']['ww']])))#great


statetags = ['active','quiet']
valdict = {tag:{utag:[] for utag in utags} for tag in statetags+['IBL']}
for statetag in statetags+['IBL']:
    for utype in utags:
        valmat = np.hstack(ddict[statetag][utype])
        valdict[statetag][utype] = valmat

#write down basic stats
outfile_name = os.path.join(figdir_mother,'meanstd_featdistros%s.xlsx'%addtag)

labs_short = [varname.split('_')[0] for varname in varnames]
label_list = ['N']+['mean(%s)'%lab for lab in labs_short]+['std(%s)'%lab for lab in labs_short]+['sem(%s)'%lab for lab in labs_short]
result_dict = {utag:{} for utag in utags}
for utype in utags:
    for statetag in statetags+['IBL']:
        featmat =  valdict[statetag][utype]
        means = featmat.mean(axis=1)
        stds = featmat.std(axis=1)
        sems = sem(featmat,axis=1)
        N = featmat.shape[1]
        result_dict[utype][statetag] = [N] +list(means) +list(stds) +list(sems)


with pd.ExcelWriter(outfile_name, engine='openpyxl') as writer:
    for utype in utags:
        df = pd.DataFrame.from_dict(result_dict[utype], orient='index',columns=label_list).sort_values(by=['N'], ascending=False)
        df.to_excel(writer, sheet_name=utype)

###plotting distributions

cdict = {'active':{'ww':'k','nw':'m'},'quiet':{'ww':'grey','nw':'b'},'IBL':{'ww':'orange','nw':'darkgoldenrod'}}

ranges = {'B_mean':[-0.6,0.6],\
          'M_mean':[-0.7,0.7],
          'rate_mean':[np.log10(0.34),np.log10(150)]}

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '.svg')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)

nbins = 100#int(np.sqrt(len(featvals))
filt_fn = lambda hist: filt.savitzky_golay(hist,17,5) if sg_on else hist[:]

for utype in utags:
    for ii,featname in enumerate(varnames):
        #featname = varnames[ii]
        f, ax = plt.subplots(figsize=(3, 2.5))
        f.subplots_adjust(left=0.18, bottom=0.22)
        ax.set_title(utype)
        for jj,statetag in enumerate(statetags):
            col = cdict[statetag][utype]
            featvals = valdict[statetag][utype][ii]
            fmin,fmax = ranges[featname]
            mybins = np.linspace(fmin,fmax,nbins)
            myhist,mybins = np.histogram(featvals,mybins)
            plbins = mybins[:-1]+float(np.diff(mybins)[0])
            myvals = filt_fn(myhist)
            ax.plot(plbins,myvals/np.sum(myvals),color=col,lw=2)
            ax.text(0.95,0.99-jj*0.1,'%s:%i'%(statetag,len(featvals)),color=col,transform=ax.transAxes,ha='right',va='top')
            #ax.hist(featvals, int(np.sqrt(len(featvals))), histtype='step', color=col, weights=np.ones_like(featvals) / len(featvals),linewidth=2)
        ax.set_xlabel(featname.replace('_',' ').replace('rate','log(rate)'))
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel('prob.')
        ax.set_xlim(ranges[featname])
        figsaver(f, 'states/%s_%s'%(utype,featname))

for statetag in statetags+['IBL']:
    for ii,featname in enumerate(varnames):
        #featname = varnames[ii]
        f, ax = plt.subplots(figsize=(3, 2.5))
        f.subplots_adjust(left=0.18, bottom=0.22)
        ax.set_title(statetag)
        for jj,utype in enumerate(utags):
            col = cdict[statetag][utype]
            featvals = valdict[statetag][utype][ii]
            featvals = featvals[~np.isnan(featvals)]
            fmin,fmax = ranges[featname]
            mybins = np.linspace(fmin,fmax,nbins)
            myhist,mybins = np.histogram(featvals,mybins)
            plbins = mybins[:-1]+float(np.diff(mybins)[0])
            myvals = filt_fn(myhist)
            ax.plot(plbins,myvals/np.sum(myvals),color=col,lw=2)
            ax.text(0.95,0.99-jj*0.1,'%s:%i'%(utype,len(featvals)),color=col,transform=ax.transAxes,ha='right',va='top')
            #ax.hist(featvals, int(np.sqrt(len(featvals))), histtype='step', color=col, weights=np.ones_like(featvals) / len(featvals),linewidth=2)
        ax.set_xlabel(featname.replace('_',' ').replace('rate','log(rate)'))
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xlim(ranges[featname])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel('prob.')
        figsaver(f, 'utypes/%s_%s'%(statetag,featname))

for utype in utags:
    for ii,featname in enumerate(varnames):
        f, ax = plt.subplots(figsize=(3, 2.5))
        f.subplots_adjust(left=0.18, bottom=0.22)
        ax.set_title(utype)
        for jj,statetag in enumerate(statetags+['IBL']):
            col = cdict[statetag][utype]
            featvals = valdict[statetag][utype][ii]
            featvals = featvals[~np.isnan(featvals)]
            fmin,fmax = ranges[featname]
            mybins = np.linspace(fmin,fmax,nbins)
            myhist,mybins = np.histogram(featvals,mybins)
            plbins = mybins[:-1]+float(np.diff(mybins)[0])
            myvals = filt_fn(myhist)
            ax.plot(plbins,myvals/np.sum(myvals),color=col,lw=2)
            ax.text(0.95,0.99-jj*0.1,'%s:%i'%(statetag,len(featvals)),color=col,transform=ax.transAxes,ha='right',va='top')
            #ax.hist(featvals, int(np.sqrt(len(featvals))), histtype='step', color=col, weights=np.ones_like(featvals) / len(featvals),linewidth=2)
        ax.set_xlabel(featname.replace('_',' ').replace('rate','log(rate)'))
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xlim(ranges[featname])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_ylabel('prob.')
        figsaver(f, 'states_and_IBL/%s_%s'%(utype,featname))



