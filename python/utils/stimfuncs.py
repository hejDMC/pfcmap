import numpy as np
from . import data_handling as dh

def make_stimfunc_lines(stimdict,styledict,horizontal=True):
    if horizontal:
        spanfn = 'axvspan'
        linefn = 'axvline'
    else:
        spanfn = 'axhspan'
        linefn = 'axhline'
    def plot_stims(axarr,zord=-10,markpos=0.5):
        for key,subdict in styledict.items():
            #if not key.count('_MARK'):col,ls,alph = [subdict[subkey] for subkey in ['col','ls','alph']]
            #print(str(key))
            if key in stimdict:

                if key.isupper():#like for whitenoise in context
                    try:
                        stimmat = stimdict[key].reshape(-1, 2)

                        for start,stop in stimmat:
                            getattr(ax,spanfn)(start,stop,**subdict)#color=col,alpha=alph,zorder = subdict['zorder'],lw=0

                    except:
                        print('WARNING not plotting %s - maybe uneven stims (len=%i)?'%(key,len(stimdict[key])))
                elif key.count('_MARK'):
                    for stimt in stimdict[key]:
                        for ax in axarr:
                            if horizontal: ax.plot([stimt],[markpos],**subdict)
                            else: ax.plot([markpos],[stimt],**subdict)
                else:
                    #subdict['zorder'] = zord
                    for stimt in stimdict[key]:
                        for ax in axarr:
                            getattr(ax,linefn)(stimt,**subdict)#,color=col,linestyle=ls,alpha=alph,zorder=zord
            else:
                print('KEY <<%s>> NOT AVAILABLE IN STIMDICT'%key)
    return plot_stims


def extract_blocktimes(aRec,exclude_zeroblocks=False):

    blocks = aRec.h_info['intervals']['trials']['Block'][()]
    if exclude_zeroblocks:
        cond = blocks>0.

    else:
        cond = np.ones(len(blocks)).astype(bool)
    blocks = blocks[cond]
    starts = aRec.h_info['intervals']['trials']['start_time'][()][cond]
    stops = aRec.h_info['intervals']['trials']['stop_time'][()][cond]
    ublocks = np.unique(blocks)
    bltimes = np.zeros((len(ublocks),2))
    for bb,ublock in enumerate(ublocks):
        binds = np.where(blocks==ublock)[0]
        bltimes[bb] = np.array([starts[binds[0]],stops[binds[-1]]])
    return bltimes


def label_blocks(bltimes,stimdict,stimkeys):
    check_inblock = lambda stimkey, blockint: np.sum([dh.check_olap(blockint, [stime, stime]) for stime in stimdict[stimkey]]) > 0
    blocknames = np.empty(len(bltimes),'<U10')
    for bb,mybl in enumerate(bltimes):
        for stimkey in stimkeys:
            if check_inblock(stimkey,mybl):
                blocknames[bb] = stimkey
    return blocknames

def make_blockplotfn(blockdict,styledict,horizontal=True):
    if horizontal:
        spanfn = 'axvspan'
    else:
        spanfn = 'axhspan'
    def plot_blocks(axarr):
        for key in blockdict.keys():
            stimname,blocktimes = blockdict[key]
            for ax in axarr:
                getattr(ax,spanfn)(blocktimes[0],blocktimes[1],**styledict[stimname])
    return plot_blocks
