import numpy as np
from scipy.stats import percentileofscore as percscore
import matplotlib as mpl

vals_to_num =  lambda uvals: {val:ii for ii,val in enumerate(uvals)}
def check_convert_vals(uvals,vals):
    if type(uvals[0]) == type('bla') or type(uvals[0]) == type(np.array(['bla'])[0]):
        vdict = vals_to_num(uvals)
        vals = np.array([vdict[val] for val in vals])
        uvals = np.unique(vals)
    return uvals,vals

def get_matchmat(uvals1,uvals2,vals1,vals2,n1,n2):
    matchmat = np.zeros((n1, n2))
    #print (matchmat.shape)
    #print('u1',uvals1)
    #print('u2',uvals2)
    #print('l1',len(vals1))
    #print('l2',len(vals1))

    for u1 in uvals1:
        matchinds1 = np.where(vals1==u1)[0]
        for u2 in uvals2:
            matchmat[u1, u2] = np.sum(vals2[matchinds1]==u2)
    return matchmat

def get_matchmat_efficient(uvals1,uvals2,vals1,vals2,n1,n2):

    if n1>=n2:
        matchmat = get_matchmat(uvals1,uvals2,vals1,vals2,n1,n2)
    else:
        matchmat = get_matchmat(uvals2,uvals1,vals2,vals1,n2,n1).T
    return matchmat

def extract_vals_naive(Units,attr1,attr2):
    vals1 = np.array([getattr(U,attr1) for U in Units])
    vals2 = np.array([getattr(U,attr2) for U in Units])
    uvals1 = np.unique(vals1)
    uvals2 = np.unique(vals2)
    return uvals1,uvals2,vals1,vals2

def extract_vals(Units,attr1,attr2,return_orig_uvals=False):
    uvals_1,uvals_2,vals1,vals2 = extract_vals_naive(Units, attr1, attr2)
    uvals1,vals1 = check_convert_vals(uvals_1,vals1)
    uvals2,vals2 = check_convert_vals(uvals_2,vals2)
    n1,n2 = len(uvals1),len(uvals2)
    if return_orig_uvals:
        return uvals1,uvals2,vals1,vals2,n1,n2,uvals_1,uvals_2

    else:
        return uvals1,uvals2,vals1,vals2,n1,n2




def permute_const(vals1,vals2,uconstvec,const_vec):
    #const vec is like recids for shuffling only within a recid
    vals_shuff1 = np.empty(0)
    vals_shuff2 = np.empty(0)

    for recid in uconstvec:
        #print(recid)
        inds = np.where(const_vec == recid)[0]
        #print(len(inds))
        shuff_snip1 = np.random.permutation(vals1[inds])
        shuff_snip2 = np.random.permutation(vals2[inds])

        vals_shuff1 = np.hstack([vals_shuff1, shuff_snip1])
        vals_shuff2 = np.hstack([vals_shuff2, shuff_snip2])

    return vals_shuff1,vals_shuff2

def get_shuffledict_const(Units,attr1,attr2,nreps=1000,const_attr='recid'):
    '''
    const_attr can be a string or list of strings, in case you want a more specific nested shuffle over eg. both area and recid
    '''
    uvals1, uvals2, vals1, vals2, n1, n2 = extract_vals(Units, attr1, attr2)
    if n1>=n2:
        matchmatgetter = lambda shuffvals1,shuffvals2: get_matchmat(uvals1,uvals2,shuffvals1,shuffvals2,n1,n2)
    else:
        matchmatgetter = lambda  shuffvals1,shuffvals2:get_matchmat(uvals2,uvals1,shuffvals2,shuffvals1,n2,n1).T

    if type(const_attr) == type('bla'):
        #eg. const_attr= recid
        full_const_vals = np.array([getattr(U,const_attr) for U in Units])
        unique_const = np.unique(full_const_vals)

    elif type(const_attr) == type(['bla','blub']):
        #e.g. const_attr = ['area','recid']
        arearecid_vec = np.array([np.array([getattr(U, attrname) for attrname in const_attr]) for U in Units])
        arearecids = np.unique(arearecid_vec, axis=0)
        arearecids_num = np.arange(len(arearecids))#unique numbers are easier to handle
        arearecid_vec_num = np.array([int(np.where((arearecids == thisarearec).all(axis=1))[0]) for thisarearec in arearecid_vec])
        unique_const = arearecids_num
        full_const_vals = arearecid_vec_num


    shuff_dict = {}

    for rr in np.arange(nreps):
        #print(rr)
        shuff_vals_1,shuff_vals_2 = permute_const(vals1,vals2,unique_const,full_const_vals)
        #print(rr,len(shuff_vals_1))
        shuff_dict[rr] = matchmatgetter(shuff_vals_1,shuff_vals_2)
    return shuff_dict


def calc_percofscoremat(matchmat,shuffdict):
    shuffstack = np.array([shuffdict[rr] for rr in np.arange(len(shuffdict))]).transpose(1, 2, 0)
    n1,n2 =  shuffstack.shape[:2]
    percmat = np.zeros((n1,n2))
    for u1 in np.arange(n1):
        for u2 in np.arange(n2):
            orig = matchmat[u1, u2]
            shuff_distro = shuffstack[u1, u2]
            percmat[u1, u2] = percscore(shuff_distro, orig)
    return percmat



def get_sep_idx(mypercmat,mid=50):
    return np.mean(np.abs(mypercmat-mid)/mid)
def plot_matchmat(ax,showmat,xlab='x',ylab='y',clab='perc. of score',cmap='RdBu_r',mode='percofscore',**kwargs):
    f = ax.get_figure()
    if mode == 'percofscore':
        vmin,vmax = [0,100]
    else:
        vmin,vmax = [showmat.min(),showmat.max()]
    n1,n2 = showmat.shape
    im = ax.imshow(showmat.T, origin='lower', interpolation='nearest', aspect='auto',
                   extent=[0.5, n1 + 0.5, 0.5, n2 + 0.5], vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(n1) + 1)
    ax.set_yticks(np.arange(n2) + 1)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if mode == 'percofscore':
        f.text(0.01, 0.99, 'SI:%1.2f' % get_sep_idx(showmat), ha='left', va='top', fontweight='bold', fontsize=10)
    if not 'cax' in kwargs:
        pos = ax.get_position()
        cax = f.add_axes([pos.x1+0.015, pos.y0, 0.025, pos.height])
    else:
        cax = kwargs['cax']
    cb = f.colorbar(im, cax=cax)
    if not 'cax' in kwargs: cb.set_label(clab, rotation=-90, labelpad=15)
    else: cax.set_title(clab)
    if mode == 'counts':
        cax.ticklabel_format(axis='y', style='sci', scilimits=(-0,1))


def plot_pmat(ax,showmat,color_map,signif_levels=np.array([0.05,0.01,0.001]),xlab='x',ylab='y',write_signif=True):
    n1,n2 = showmat.shape[:2]
    im = ax.imshow(showmat.transpose(1,0,2), origin='lower', interpolation='nearest', aspect='auto',
                   extent=[0.5, n1 + 0.5, 0.5, n2 + 0.5])
    ax.set_xticks(np.arange(n1) + 1)
    ax.set_yticks(np.arange(n2) + 1)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)


    if write_signif:
        ax.text(1.01,1.01,'pvalues',color='k',transform=ax.transAxes,ha='left',va='bottom',fontsize=10,fontweight='bold')

        for ii,lev in enumerate(signif_levels):
            ax.text(1.02,0.99-ii*0.1,'<%s'%(str(lev)),color=color_map[ii+1],transform=ax.transAxes,ha='left',va='top',fontsize=10,fontweight='bold')
            #for ii,lev in np.arange(signif_levels):
            ax.text(1.02,0.7-ii*0.1,'<%s'%(str(lev)),color=color_map[-(ii+1)],transform=ax.transAxes,ha='left',va='top',fontsize=10,fontweight='bold')


def get_all_shuffs(Usel,attr1,attr2,const_attr='task',nshuff=1000,signif_levels=np.array([0.05,0.01,0.001])):
    uvals1,uvals2,vals1,vals2,n1,n2,uorig1,uorig2 = extract_vals(Usel,attr1,attr2,return_orig_uvals=True)
    matchmat = get_matchmat_efficient(uvals1,uvals2,vals1,vals2,n1,n2)
    shuffdict = get_shuffledict_const(Usel,attr1,attr2,nreps=nshuff,const_attr=const_attr)
    shuffstack = np.array([shuffdict[rr] for rr in np.arange(len(shuffdict))]).transpose(1, 2, 0)
    mean_shuff = shuffstack.mean(axis=2)
    std_shuff = shuffstack.std(axis=2)
    percmat = calc_percofscoremat(matchmat,shuffdict)
    pmat = calc_plevel_mat(percmat,signif_levels=signif_levels)
    #cmat = colorlize_pmat(pmat,color_map)
    #u_attr1 = np.unique([getattr(U,attr1) for U in Usel])
    #level_dict = {key: cmat[kk] for kk,key in enumerate(u_attr1) }
    #perc_dict= {key: percmat[kk] for kk,key in enumerate(u_attr1) }
    sep_idx = get_sep_idx(percmat)
    out_dict = {'matches':matchmat,'levels':pmat,'pofs':percmat,'SI':sep_idx,'meanshuff':mean_shuff,'stdshuff':std_shuff,\
                'avals1':uorig1,'avals2':uorig2,'a1':attr1,'a2':attr2,'cattr':const_attr}
    return out_dict

def plot_all_stats(statsdict,plotsel='full',zlim=7,**kwargs):
    mean_shuff,std_shuff = [statsdict[key] for key in ['meanshuff','stdshuff']]
    zmat = (statsdict['matches']-mean_shuff)/std_shuff
    if 'zmat_zerolims' in kwargs:
        zlower,zupper = kwargs['zmat']
        zmat[(zmat<=zupper) & (zmat>=zlower)] = np.nan


    ny,ncl = zmat.shape
    if 'sortinds' in kwargs:
        #N.B.: sortinds can actually be selection inds!
        mysortinds = kwargs['sortinds']
    else:
        mysortinds = np.arange(ny)

    if plotsel == 'full':
        plotkeys = ['mean shuff counts', 'std shuff counts', 'fano shuff', 'orig data counts', 'z from shuff',
                 'perc of score', 'p-levels']
    else:
        plotkeys = plotsel[:]

    mega_plotdict ={}
    for tag,myplotmat in zip(['mean shuff counts', 'std shuff counts', 'fano shuff', 'orig data counts', 'z from shuff',
                 'perc of score', 'p-levels'], \
                [mean_shuff,std_shuff,  std_shuff ** 2 /mean_shuff, statsdict['matches'], zmat, statsdict['pofs'], statsdict['levels']]):
        mega_plotdict[tag] = myplotmat[mysortinds]

    if zlim == 'free':   zlim,extend = np.max(np.abs([mega_plotdict['z from shuff'].min(),mega_plotdict['z from shuff'].max()])),False
    else: extend = True


    n_plots = len(plotkeys)
    f, axarr = mpl.pyplot.subplots(1, n_plots*2, figsize=(2+n_plots*2, 6), gridspec_kw={'width_ratios': [1, 0.1] * n_plots})
    f.subplots_adjust(wspace=0.5, left=0.05, right=0.99)
    for ii,tag in enumerate(plotkeys):
        plotmat = mega_plotdict[tag]
        #plotmat = plotmat0[mysortinds]
        ax, cax = axarr[ii * 2:(ii * 2) + 2]
        if not tag.count('counts') and not tag.count('fano'):
            cmap = 'RdBu_r'
            if tag.count('perc'):
                vminmax = [0, 100]
            elif tag.count('levels'):
                vminmax = [-3, 3]
            else:
                vminmax = [-zlim, zlim]
        else:
            cmap = 'inferno'
            vminmax = [plotmat.min(), plotmat.max()]
        im = ax.imshow(plotmat, cmap=cmap, aspect='auto', origin='lower', vmax=vminmax[0], vmin=vminmax[1])
        if ii == 0:
            ax.set_ylabel('roi')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('cluster')
        ax.set_title(tag)
        ax.set_xticks(np.arange(ncl))
        ax.set_xticklabels(np.arange(ncl) + 1)
        if extend and tag=='z from shuff': f.colorbar(im, cax=cax,extend='both')
        else: f.colorbar(im, cax=cax)
        apos = ax.get_position()
        cpos = cax.get_position()
        cax.set_position(
            [apos.x1 + 0.5 * cpos.width, cpos.y0, cpos.width, cpos.height])  # [left, bottom, width, height]
        if tag.count('levels'):
            cax.remove()
    return f,axarr


def make_roi_ylabels(axarr,rois,roi_colors,fs=20):
    ny = len(rois)
    for ax in axarr:
        ax.set_yticks(np.arange(ny))
        ax.set_yticklabels(['$\\bullet$']*ny, va='center')
        for ticklab,roi in zip(ax.get_yticklabels(),rois):
            #ticklab.set_text('o')
            ticklab.set_color(roi_colors[roi])
            ticklab.set_fontsize(fs)

def calc_plevel_mat(percmat,signif_levels = np.array([0.05,0.01,0.001])):
    #significance version of the precmat

    perc_levels = np.array(signif_levels)*100
    pmat = np.zeros_like(percmat)
    for rr in np.arange(percmat.shape[0]):
        for cc in np.arange(percmat.shape[1]):
            for pp,plev in enumerate(perc_levels):
                if percmat[rr,cc]>100-plev:
                    pmat[rr,cc] = pp+1
                elif percmat[rr,cc]<plev:
                    pmat[rr,cc] = -(pp+1)
    return pmat


color_map = {1: 'pink', 2: 'indianred',3: 'firebrick',\
             -1: 'lightblue', -2: 'dodgerblue',-3: 'mediumblue',0:'w'} # blue

def colorlize_pmat(pmat,color_map):
    color_map_rgb = {key:np.array(mpl.colors.to_rgb(val))*255 for key,val in color_map.items()}
    cmat = np.ndarray(shape=(pmat.shape[0], pmat.shape[1], 3), dtype=int)
    for rr in np.arange(pmat.shape[0]):
        for cc in np.arange(pmat.shape[1]):
            cmat[rr,cc] = color_map_rgb[pmat[rr,cc]]
    return cmat


def get_all_percstats(myUnits,attr1,attr2,signif_levels = np.array([0.05,0.01,0.001]),nreps=1000):
    uvals1, uvals2, vals1, vals2, n1, n2 = extract_vals(myUnits, attr1, attr2)#here the possible string vals are already converted to numbers
    matchmat = get_matchmat_efficient(uvals1, uvals2, vals1, vals2, n1, n2)
    shuffdict = get_shuffledict_const(myUnits, attr1, attr2, nreps=nreps)
    percmat = calc_percofscoremat(matchmat, shuffdict)
    pmat = calc_plevel_mat(percmat, signif_levels=signif_levels)
    cmat = colorlize_pmat(pmat, color_map)
    sep_idx = get_sep_idx(percmat)
    uvals1, uvals2, _, _ = extract_vals_naive(myUnits, attr1, attr2) #here the string vals are not converted
    return matchmat,percmat,cmat,sep_idx,uvals1,uvals2