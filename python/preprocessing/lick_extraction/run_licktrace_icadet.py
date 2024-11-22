import h5py
import sys
import numpy as np
import yaml
import os
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from scipy.signal import medfilt
import matplotlib.pyplot as plt



pathpath,recfilename = sys.argv[1:]

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import accesstools as act

from pfcmap.python.utils.ica_tools import icasso_fns as icf
from pfcmap.python.utils import plotting_basics as pb
from pfcmap.python.utils.ica_tools import icafns as icaf
from pfcmap.python.utils import filtering as filt
from pfcmap.python.utils import data_handling as dh
from pfcmap.python.utils import file_helpers as fhs

logger = fhs.make_my_logger(pathdict['logger']['config_file'],pathdict['logger']['out_log'])

plot_on = True
save_on = True
max_ncomps = 10
min_ncomps = 4
pca_thresh = 98.
tintdur_s = 2*60. #when projecting on not analyzed
calcdur_s = 5*60. #for getting the unmixing matrix
snipdur_calc_s = 20.# duration of randomly selected snippets, totalling calcdur_s
hp_cutoff = 2 #Hz
hp_acqu = 3

aSetting = act.Setting(pathpath=pathdict['code']['recretrieval'])
aRec = aSetting.loadRecObj(recfilename, get_eois_col=True)
recid0 = (os.path.basename(recfilename).split('__')[0]).replace('_probe','-probe')
recid = aRec.id.replace('-', '_')
exptype = (aRec.h_info['exptype'][()]).decode()
outfile = os.path.join(pathdict['outpaths']['data'],'%s__artifactsICA.h5'%(recid0))

LFP_path = recfilename.replace('XXX','lfp')
nchans = len(aRec.eois)

logger.info('## Starting %s (%s)'%(aRec.id,exptype))

acqu_filter = lambda dtrace: filt.butter_highpass_filter(dtrace,hp_acqu,aRec.sr,order=3)
def get_X(pints):
    pts_total = int(np.diff(pints).sum())
    X = np.zeros((nchans, pts_total))

    slicevec = np.zeros(len(pints), int)
    with h5py.File(LFP_path, 'r') as hand:
        chanindsL = hand['/processing/LFP/chan_indices'][()]
        pcount = 0
        for tt, pint in enumerate(pints):
            pstart,pstop = pint
            npts = pstop - pstart
            slicevec[tt] = npts
            for cc, chan in enumerate(aRec.eois):
                X[cc, pcount:pcount + npts] = acqu_filter(hand['/processing/LFP/data'][pstart:pstop,
                                              int(np.where(chanindsL == chan)[0])])
            pcount += npts
    return X, slicevec

aRec.get_freetimes()
tintdur = int(tintdur_s*aRec.sr)
calcdur = int(calcdur_s*aRec.sr)
snipdur_calc = int(snipdur_calc_s*aRec.sr)

npts_tot = int(aRec.dur*aRec.sr)

n_snips = int(calcdur/snipdur_calc)
snipstarts = np.sort(np.random.randint(0,npts_tot-snipdur_calc,n_snips))
pints0 = np.vstack([snipstarts,snipstarts+snipdur_calc]).T
freepts = (aRec.freetimes*aRec.sr).astype(int)
pints = np.vstack([pint for pint in pints0 if np.sum([(pint[0]<freeint[1]) & (pint[1]>freeint[0]) for freeint in freepts])==1])#remove the artifact ones
pints.clip(0,npts_tot-1)

X,slicevec = get_X(pints)

pcs,pca = icf.do_pca(X)#or better, do pcs,pca = icf.do_pca2(X) to do demeaning only as in ICA!
n_comps0 = icf.get_ncomp_suggestion(pca,pca_thresh)#to
n_comps = np.min([max_ncomps,n_comps0])
n_comps = np.max([n_comps,min_ncomps])
logger.info('doing ica, ncomps: %i'%n_comps)


ica_params = {'n_components':n_comps,'whiten':True}

ica = FastICA(random_state=42, **ica_params)
ica.fit(X.T)#X is features x tpts

W = ica._unmixing#thats W_hat
whitened = ica.whitening_
C0 = np.dot(W,whitened)
S0 = np.dot(C0,X)

#sort it
kurt = kurtosis(S0, axis=1)
sortinds = np.argsort(kurt)[::-1]
C = C0[sortinds]
S  = np.dot(C,X)
A = np.linalg.pinv(C)


#flip A
flipfacs = np.sign(A.sum(axis=0)).astype(int)
A = A*flipfacs[None,:] #mixing
C = C*flipfacs[:,None] #unmixing
S = S*flipfacs[:,None]



A_mfilt =  np.vstack([medfilt(trace,3) for trace in A.T ]).T#to avoid high ffs when one elec is broken
ff_comp = A_mfilt.std(axis=0)/A_mfilt.mean(axis=0)
#ff_comp = A.std(axis=0)/A.mean(axis=0)
artidx = np.argmin(ff_comp)
logger.info('extracting artsrc, artidx %i'%artidx)

artblocks = (dh.mergeblocks([freepts.T],output='free',t_start=0,t_stop=npts_tot).T).astype(int)
artvec = np.zeros(npts_tot)
for a0,a1 in artblocks:
    artvec[a0:a1] = np.nan

pintmat = np.empty((0,2),dtype=int)
for freeint in freepts:
    n_subints = np.diff(freeint)[0]//tintdur+1
    if n_subints == 1:
        pintmat0 = freeint[None,:]
    else:
        temp = np.linspace(freeint[0],freeint[1],n_subints,endpoint=True).astype(int)

        #print(temp.max()-np.diff(freeint)-freeint[0])
        pintmat0 = np.vstack([temp[:-1],temp[1:]]).T
    pintmat = np.vstack([pintmat,pintmat0])


for pint in pintmat:
    Xdata,_ = get_X(pint[None,:])
    src_proj  = np.dot(C,Xdata)
    #hp_art = filt.butter_highpass_filter(src_proj[artidx],hp_cutoff, aRec.sr) removed because now hp at loading

    artvec[pint[0]:pint[1]] = src_proj[artidx]#hp_art


if save_on:
    logger.info('writing to %s'%outfile)
    save_it = True

    #save artvec, C, sr, recid and artblocks,artidx,ff_comp

    if os.path.isfile(outfile):
        with h5py.File(outfile, 'r+') as fdest:
            if 'detection_edited' in fdest:
                answer = input("do you really want to overwrite the manually curated detection_edited (yes)?")
                if answer.lower() == 'yes':
                    save_it = True
                else:
                    print('answer: %s, exiting script' % answer)
                    save_it = False
                    exit()

    if save_it:
        with h5py.File(outfile, 'w') as fdest:

            fdest.attrs['recid'] = aRec.id
            methods = fdest.create_group('methods')
            methods.create_dataset('artblocks',data=artblocks,dtype='i')
            methods.attrs['sr'] = aRec.sr

            icagroup = fdest.create_group('ica')
            for name,vals in zip(['artvec','C','ff_comp'],[artvec,C,ff_comp]):
                icagroup.create_dataset(name,data=vals,dtype='f')
            icagroup.attrs['artidx'] = artidx


if plot_on:
    logger.info('plotting loadings')

    figdir = os.path.join(pathdict['outpaths']['figures'],recid0)
    if not os.path.isdir(figdir): os.makedirs(figdir)

    cstr = 'nipy_spectral'#for coloring components
    ica_cols = pb.get_colors(cstr,n_comps)
    labs = ['src%i'%(ii+1) for ii in np.arange(n_comps)]


    #f = icf.plot_kurt_simple(kurt,ica_cols)

    f = icaf.plot_ICA_loadings(aRec, A_mfilt, aRec.eois, cmapstr=cstr)
    ax = f.gca()
    for col,comp in zip(ica_cols,A.T):
        ax.plot(comp, np.arange(nchans),color=col,lw=0.5,alpha=0.5)
    f.suptitle('artifacts in src%i'%(artidx+1),color=ica_cols[artidx])
    if save_on:
        f.savefig(os.path.join(figdir,'%s_loadings.png'%recid))
        plt.close(f)

logger.info('##### DONE arttrace extraction: %s'%aRec.id)
