import matplotlib as mpl
import numpy as np
from cdlib import algorithms
import networkx as nx
import leidenalg as la
import igraph as ig
from scipy.spatial.distance import pdist, squareform
from pfcmap.python.utils import category_matching as matchfns

import igraph
from netgraph import Graph
import string

def get_similarity_mat(Xmat):
    dists = pdist(Xmat)
    Dmat =  squareform(dists)
    return 1-Dmat/Dmat.max()

def get_adjacency_mat(sim_mat,thr,fill_val=np.nan,replace_diag=True):
    Amat = np.zeros_like(sim_mat) * fill_val
    Amat[sim_mat >= thr] = sim_mat[sim_mat >= thr]
    if replace_diag:
        Amat[np.arange(len(Amat)), np.arange(len(Amat))] = fill_val
    return Amat

def shuffle_simmat(sim_mat,keep_diagonal=True):
    sivals = sim_mat[np.triu_indices_from(sim_mat, k=1)]
    shuff_sivals = np.random.permutation(sivals)
    shuff_mat = np.zeros_like(sim_mat)
    shuff_mat[np.triu_indices_from(sim_mat, k=1)] = shuff_sivals #fill upper triangle
    shuff_mat = shuff_mat + shuff_mat.T - np.diag(np.diag(shuff_mat))#make it symmetric
    if keep_diagonal:
        #only for display
        N = len(sim_mat)
        shuff_mat[np.arange(N),np.arange(N)] = 1
    return shuff_mat

def get_igraph_from_adj(A,thr):
    iG = igraph.Graph.Adjacency((A > thr).tolist())
    iG.to_undirected()
    iG.es['weight'] = A[A > thr]
    return iG

def calc_q_vs_res(adjmat,thr,community_fn,weight_fn,resolution_vec,with_n=False,with_comm=False,**kwargs):
    qvec = np.zeros_like(resolution_vec)

    if 'comparelist' in kwargs:
        labels2 = kwargs['comparelist']
        qvec2 = np.zeros_like(resolution_vec)

    if with_n:
        nvec = np.zeros(len(resolution_vec),dtype=int)

    if with_comm:
        comm_mat = np.zeros((len(resolution_vec),len(adjmat)),dtype=int)

    for rr,res in enumerate(resolution_vec):
        iG = get_igraph_from_adj(adjmat, thr)
        com = community_fn(iG)(weights=weight_fn(iG),resolution=res)
        if 'comparelist' in kwargs:
            qvec2[rr] = iG.modularity(labels2, resolution=res)
        if with_comm:
            comm_mat[rr] = com.membership
        qvec[rr] = com.modularity #===.q
        if with_n:
            nvec[rr] = len(np.unique(com.membership))
    outlist = [qvec]
    if 'comparelist' in kwargs:
        outlist += [qvec2]
    if with_n:
        outlist += [nvec]
    if with_comm:
        outlist += [comm_mat]
    return outlist



def get_q_vs_res_shuffled(sim_mat,thr,community_fn,weight_fn,resolution_vec,nreps=10):
    shuff_mat = np.zeros((len(resolution_vec),nreps))
    for ii in np.arange(nreps):
        shuff_sim = shuffle_simmat(sim_mat, keep_diagonal=True)
        shuff_A = get_adjacency_mat(shuff_sim,thr,fill_val=np.nan)#this is actually not necessary if you dont scale the weights in the adj mat calculation!
        shuff_mat[:,ii] =  calc_q_vs_res(shuff_A, thr, community_fn, weight_fn, resolution_vec)[0]
    return shuff_mat


def plot_q_vs_res(qvec,shuff_mat,resolution_vec,idx,**kwargs):
    res = resolution_vec[idx]
    shuff_mean,shuff_std = shuff_mat.mean(axis=1),shuff_mat.std(axis=1)

    ncol = 'grey'
    shuffcol = 'b'
    f, ax = mpl.pyplot.subplots(figsize=(4.5, 3))
    f.subplots_adjust(left=0.15, bottom=0.2, right=0.85)
    ax.plot(resolution_vec, qvec, '.k-', zorder=10, lw=2)
    ax.plot(resolution_vec, shuff_mean, color=shuffcol)
    ax.fill_between(resolution_vec, shuff_mean - shuff_std, shuff_mean + shuff_std, color=shuffcol, alpha=0.5, zorder=-10)
    ax.axvline(resolution_vec[idx], color='r')
    if 'qvec2' in kwargs:
        ax.plot(resolution_vec, kwargs['qvec2'], 'orange', zorder=-2, lw=2)
    ax.set_xlabel('resolution')
    ax.set_ylabel('modularity (Q)')
    ax.set_xlim([resolution_vec[0], resolution_vec[-1]])
    if 'nvec' in kwargs:
        ax2 = ax.twinx()
        nvec = kwargs['nvec']

        ax2.plot(resolution_vec, nvec, color=ncol)
        ax2.set_ylabel('n modules', color=ncol, rotation=-90, labelpad=20)
        ax2.tick_params(axis='y', colors=ncol)
        ax.text(resolution_vec[idx] + 0.025, qvec[idx] + qvec.max() / 20., 'res:%1.2f\nN:%i' % (res, nvec[idx]), ha='left',
                va='bottom', color='r')


    return f,ax


def get_styledict_rois_and_comms(roivals,community_dict,roi_color_dict,cstr_communities='jet'):
    cdict_com = get_community_cmap(community_dict, cstr=cstr_communities)

    dot_colors = np.array([roi_color_dict[roi] for roi in roivals])
    com_labelvec = np.array(
        [[comname for comname, comvals in community_dict.items() if roi in comvals][0] for roi in roivals])
    return {'cdictcom':cdict_com,'dotcolvec':dot_colors,'labelvec':com_labelvec}

def plot_all_stats_with_labels(mystats,selsortinds,styledict,plotsel='full',aw_frac=1/10.,labels_on=True,**kwargs):
    com_labelvec, cdict_com, dot_colors = [styledict[key] for key in ['labelvec','cdictcom','dotcolvec']]
    f, axarr = plot_all_stats(mystats, sortinds=selsortinds,plotsel=plotsel,**kwargs)
    labaxarr = make_clusterlabels_as_bgrnd(axarr[::2], com_labelvec, cdict_com,\
                                           aw_frac=aw_frac, output=True,labels_on=labels_on)
    make_roilabeldots_on_bgrnd(labaxarr, dot_colors, ms=6, mec='w',mew=0.5)
    return f,axarr


def plot_all_stats(statsdict, plotsel='full', **kwargs):
    mean_shuff, std_shuff = [statsdict[key] for key in ['meanshuff', 'stdshuff']]
    zmat = (statsdict['matches'] - mean_shuff) / std_shuff
    if 'zmat_levelbounded' in kwargs:
        zmat[statsdict['levels']==0] = np.nan
        zcmap = mpl.cm.get_cmap('RdBu_r')
        zcmap.set_bad('w')


    elif 'zmat_zerolims' in kwargs:
        zlower, zupper = kwargs['zmat_zerolims']
        zmat[(zmat <= zupper) & (zmat >= zlower)] = np.nan
        zcmap = mpl.cm.get_cmap('RdBu_r')
        zcmap.set_bad('w')
    if 'zlim' in kwargs:
        zlim = kwargs['zlim']
    else:
        zlim = 'maxabs'

    ny, ncl = zmat.shape
    if 'sortinds' in kwargs:
        # N.B.: sortinds can actually be selection inds!
        mysortinds = kwargs['sortinds']
    else:
        mysortinds = np.arange(ny)

    if plotsel == 'full':
        plotkeys = ['mean shuff counts', 'std shuff counts', 'fano shuff', 'orig data counts', 'z from shuff',
                    'perc of score', 'p-levels']
    else:
        plotkeys = plotsel[:]

    mega_plotdict = {}
    for tag, myplotmat in zip(
            ['mean shuff counts', 'std shuff counts', 'fano shuff', 'orig data counts', 'z from shuff',
             'perc of score', 'p-levels'], \
            [mean_shuff, std_shuff, std_shuff ** 2 / mean_shuff, statsdict['matches'], zmat, statsdict['pofs'],
             statsdict['levels']]):
        mega_plotdict[tag] = myplotmat[mysortinds]

    if zlim == 'maxabs':
        zlim, extend = np.max(np.abs([mega_plotdict['z from shuff'].min(), mega_plotdict['z from shuff'].max()])), False
    else:
        extend = True

    n_plots = len(plotkeys)
    f, axarr = mpl.pyplot.subplots(1, n_plots * 2, figsize=(2 + n_plots * 2, 6),
                                   gridspec_kw={'width_ratios': [1, 0.1] * n_plots})
    f.subplots_adjust(wspace=0.5, left=0.05, right=0.99)
    for ii, tag in enumerate(plotkeys):
        plotmat = mega_plotdict[tag]
        # plotmat = plotmat0[mysortinds]
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
        if tag == 'z from shuff':
            cmap = zcmap
            #print (vminmax)
        im = ax.imshow(plotmat, cmap=cmap, aspect='auto', origin='lower', vmax=vminmax[0], vmin=vminmax[1])
        if ii == 0:
            ax.set_ylabel('roi')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('cluster')
        ax.set_title(tag)
        ax.set_xticks(np.arange(ncl))
        ax.set_xticklabels(np.arange(ncl) + 1)
        if extend and tag == 'z from shuff':
            f.colorbar(im, cax=cax, extend='both')
        else:
            f.colorbar(im, cax=cax)
        apos = ax.get_position()
        cpos = cax.get_position()
        cax.set_position(
            [apos.x1 + 0.5 * cpos.width, cpos.y0, cpos.width, cpos.height])  # [left, bottom, width, height]
        if tag.count('levels'):
            cax.remove()
    return f, axarr


def plot_all_stats_with_labels_regs(mystats,selsortinds,styledict,reglabs,plotsel='full',\
                                    check_bold=False,aw_frac=1/10.,labels_on=True,fs=9,**kwargs):
    f, axarr = plot_all_stats(mystats, sortinds=selsortinds, plotsel=plotsel,**kwargs)
    f.subplots_adjust(left=0.1, wspace=0.5, right=0.99)

    if len(styledict)>0:
        com_labelvec, cdict_com = [styledict[key] for key in ['labelvec','cdictcom']]
        labaxarr = make_clusterlabels_as_bgrnd([axarr[0]], com_labelvec, cdict_com, aw_frac=aw_frac, output=True,labels_on=labels_on)


    for myax in axarr[::2]: myax.set_yticks(np.arange(len(reglabs)))
    #axarr[0].set_yticklabels(reglabs)
    #for tt, tick in enumerate(axarr[0].yaxis.get_ticklabels()):
    #    if check_bold(tick.get_text()):  # S.check_pfc(tick.get_text()):
    #        tick.set_fontweight('bold')

    for cax in axarr[1::2]:
        cpos = cax.get_position()
        cax.set_position(
            [cpos.x0 - 0.035, cpos.y0, cpos.width, cpos.height])
    axarr[0].set_ylabel('')
    make_writelabels_on_bgrnd(labaxarr,reglabs,which='y',fs=fs,check_bold=check_bold)
    return f,axarr

def offset_axes_position(axarr,pos_offset):
    for ax in axarr:
        apos = ax.get_position()
        ax.set_position(
            [apos.x0 +pos_offset[0], apos.y0+pos_offset[1], apos.width, apos.height])

def plot_similarity_mat(smat, styledict,cmap='viridis',clab='SI',labels_on=True,tickflav='roi',extend='neither',**kwargs):
    #works well with adjaceny matrix too, but then provide badcol in kwargs


    if not 'axarr' in kwargs:
        f, axarr = mpl.pyplot.subplots(1, 2, figsize=(5.25,4.5),gridspec_kw={'width_ratios': [1, 0.05]})
        f.subplots_adjust(wspace=0.05,right=0.85,left=0.15,bottom=0.15)
    else:
        axarr = kwargs['axarr']
        f = axarr[0].get_figure()

    mycmap = mpl.cm.get_cmap(cmap)
    if np.isnan(smat).sum()>0 and 'badcol' in kwargs:
        mycmap.set_bad(kwargs['badcol'])


    ax, cax = axarr
    if 'vminmax' in kwargs:
        vminmax = kwargs['vminmax']
    else:
        vminmax = [np.nanmin(smat),np.nanmax(smat)]
    im = ax.imshow(smat, origin='lower', aspect='auto', cmap=mycmap,vmin=vminmax[0],vmax=vminmax[1])
    f.colorbar(im, cax=cax,extend=extend)
    cax.set_title(clab)

    if len(styledict)>0:
        fs = kwargs['fs'] if 'fs' in kwargs else 9
        if 'aw_frac' in kwargs:
            aw_frac = kwargs['aw_frac']
        else:
            aw_frac = 1/20 if tickflav == 'roi' else 1/10
        check_bold = kwargs['check_bold'] if 'check_bold' in kwargs else lambda x:False
        com_labelvec, cdict_com, dot_colors = [styledict[key] for key in ['labelvec','cdictcom','dotcolvec']]

        labaxarr = make_clusterlabels_as_bgrnd([ax], com_labelvec, cdict_com, aw_frac=aw_frac, which='y', output=True,labels_on=labels_on)
        if tickflav == 'roi':make_roilabeldots_on_bgrnd(labaxarr, dot_colors, which='y', ms=6, mec='w',mew=0.5)
        elif tickflav == 'reg':
            make_writelabels_on_bgrnd(labaxarr,kwargs['regvals_sorted'],which='y',fs=fs,check_bold=check_bold)
        labaxarr = make_clusterlabels_as_bgrnd([ax], com_labelvec, cdict_com, aw_frac=aw_frac, which='x', output=True,labels_on=labels_on)
        if tickflav == 'roi': make_roilabeldots_on_bgrnd(labaxarr, dot_colors, which='x', ms=6, mec='w',mew=0.5)
        elif tickflav == 'reg':
            make_writelabels_on_bgrnd(labaxarr,kwargs['regvals_sorted'],which='x',fs=fs,check_bold=check_bold)

        #else:

    return f,axarr




def plot_community_sqaures_on_mat(ax,cdict_comm,labelvec,lw=3,bg_on=False,bg_col='w',bg_fac=1.5):


    for lab,col in cdict_comm.items():
        minpos,maxpos = np.where(labelvec==lab)[0][np.array([0,-1])]+np.array([-0.5,0.5])
        if bg_on:
            ax.plot([maxpos,maxpos,minpos,minpos,maxpos],[minpos,maxpos,maxpos,minpos,minpos],color=bg_col,lw=lw*bg_fac,zorder=20)
        ax.plot([maxpos,maxpos,minpos,minpos,maxpos],[minpos,maxpos,maxpos,minpos,minpos],color=col,lw=lw,zorder=22)

def plot_simval_hist(sivals,si_thr,xtag='corr'):
    f, ax = mpl.pyplot.subplots(figsize=(3.6, 2.36))
    f.subplots_adjust(left=0.2, bottom=0.23)
    ax.hist(sivals, np.max([int(len(sivals) / 10),5]), color='k')
    ax.axvline(np.mean(sivals), color='grey')
    ax.axvline(si_thr, color='r')
    ax.set_xlabel('sim. values (%s)' % (xtag))
    ax.set_ylabel('counts')
    return f,ax

manual_colors = ['red','deepskyblue','violet','sienna','orange','limegreen','darkviolet','peru','deeppink','darkturquoise','gold','green']

def get_community_cmap(com_dict,cstr='jet'):
    if cstr == 'manual':
        nrep_clist = int(np.ceil(len(com_dict)/len(manual_colors)))
        mancollist = manual_colors*nrep_clist#in case we have more communities than manual colors available, repeat
        #print(nrep_clist)
        cdict_com = {cname:mancollist[rr] for rr,cname in enumerate(com_dict.keys())}
    else:
        comcmap = mpl.cm.get_cmap(cstr)#
        maxn = len(com_dict)
        comnorm = mpl.colors.Normalize(vmin=0, vmax=maxn-1)
        cdict_com = {cname:comcmap(comnorm(rr)) for rr,cname in enumerate(com_dict.keys())}
    return cdict_com

def make_roilabeldots_on_bgrnd(bgrnd_ax,cvec_dots,which='y',ms=6,mec='none',mew=0.5):
    for ax2 in bgrnd_ax:
        lims = [ax2.get_ylim(),ax2.get_xlim()]
        if which == 'y':
            for tt,dotcol in enumerate(cvec_dots):
                ax2.plot(0.5,tt,'o',mfc=dotcol,mec=mec,ms=ms,mew=mew)
        elif which == 'x':
            for tt,dotcol in enumerate(cvec_dots):
                ax2.plot(tt,0.5,'o',mfc=dotcol,mec=mec,ms=ms,mew=mew)
        ax2.set_xlim(lims[1])
        ax2.set_ylim(lims[0])

def make_writelabels_on_bgrnd(bgrnd_ax,regvals_sorted,which='y',fs=9,check_bold=lambda x:False):
    for ax2 in bgrnd_ax:
        if which == 'y':
            for jj, regval in enumerate(regvals_sorted):
                fw = 'bold' if check_bold(regval) else 'normal'
                ax2.text(1, jj, regval, ha='right', va='center', fontweight=fw, fontsize=fs)
        elif which == 'x':
            for jj, regval in enumerate(regvals_sorted):
                fw = 'bold' if check_bold(regval) else 'normal'
                ax2.text(jj,1, regval, ha='center', va='top', fontweight=fw, fontsize=fs,rotation=-90)


def make_clusterlabels_as_bgrnd(axarr,com_labelvec,cdict_com,aw_frac=1/10.,which='y',output=False,alpha=0.8,labels_on=True):
    #if labels off it is just preparing the background for the dots

    if labels_on:
        cvec = np.array([cdict_com[lab] for lab in com_labelvec])

    else:
        cvec = np.array(['w'])

    #if colors are str and not rgb, convert the to rgb
    if np.ndim(cvec[0])==0:
       cvec = np.array([mpl.colors.to_rgb(cval) for cval in cvec])

    ny = len(cvec)
    axlist = []
    for ax in axarr:
        pos = ax.get_position()
        if which == 'y':
            aw = pos.width *aw_frac
            ax2 = ax.get_figure().add_axes([pos.x0 - aw - 0.3 * aw, pos.y0, aw, pos.height])  # [left, bottom, width, height]

            ax2.imshow(np.dstack(cvec[:,:3].T).transpose(1,0,2),origin='lower',aspect='auto',extent=[0,1,*ax.get_ylim()],alpha=alpha)
            ax2.set_ylim(ax.get_ylim())
            ax2.set_xlim([0,1])
            ax.set_ylabel('')
            ax.set_yticks(np.arange(ny))
            ax.set_yticklabels([])
            ax.get_shared_y_axes().join(ax, ax2)
        elif which == 'x':
            ah = pos.height *aw_frac
            ax2 = ax.get_figure().add_axes([pos.x0 , pos.y0- ah - 0.3 * ah, pos.width,ah])  # [left, bottom, width, height]
            ax2.imshow(np.dstack(cvec[:,:3].T),origin='lower',aspect='auto',extent=[*ax.get_xlim(),0,1],alpha=alpha)
            ax2.set_xlim(ax.get_xlim())
            ax2.set_ylim([0,1])
            ax.set_xlabel('')
            ax.set_xticks(np.arange(ny))
            ax.set_xticklabels([])
            ax.get_shared_x_axes().join(ax, ax2)

        ax2.set_axis_off()

        axlist += [ax2]
    if output:
        return axlist




def calc_graph_from_smat(Simmat,avals,s_thr=0.1):
    N = len(Simmat)
    Smat = np.vstack([val for val in Simmat])#this is deep copying
    np.fill_diagonal(Smat,-1)
    G = nx.DiGraph()
    for aval in avals:
        if not aval == 'na':
            G.add_node(aval)
    for aa in np.arange(N):
        for bb in np.arange(aa+1,N):
            #print(avals[aa],avals[bb],Smat[aa,bb])
            if Smat[aa,bb]>=s_thr:
                G.add_edge(avals[aa], avals[bb], weight=Smat[
                            aa, bb])
    return G




def get_community_dict(community,roivals):
    labels = np.array(community.membership)
    ulabels = np.unique(labels)
    assert (ulabels == np.arange(len(ulabels))).all(), 'community labels must be 0 to len(communties)-1, got %s'%(str(ulabels))
    sortinds_com = np.argsort(labels)

    rois_sorted = roivals[sortinds_com]
    com_dict = {ll: roivals[labels == ll] for ll in ulabels}
    return com_dict,sortinds_com


def graphprops_to_dict(G,communities,roivals,roicolor_dict):
    nodes_to_community = {aa: communities.membership[aa] for aa, memb in enumerate(communities.membership)}
    node_color_dict = {aa: roicolor_dict[aval] for aa, aval in enumerate(roivals)}
    edge_width_dict = {(e.source, e.target): e['weight'] for e in G.es()}
    edge_color_dict = {(e.source, e.target): roicolor_dict[roivals[e.source]] for e in G.es()}
    return {'nodesToComm':nodes_to_community,'nc':node_color_dict,'ew':edge_width_dict,'ec':edge_color_dict}


def graphprops_to_dict2(G,labelvec,roivals,roicolor_dict):
    nodes_to_community = {aa: memb for aa, memb in enumerate(labelvec)}
    node_color_dict = {aa: roicolor_dict[aval] for aa, aval in enumerate(roivals)}
    edge_width_dict = {(e.source, e.target): e['weight'] for e in G.es()}
    edge_color_dict = {(e.source, e.target): roicolor_dict[roivals[e.source]] for e in G.es()}
    return {'nodesToComm':nodes_to_community,'nc':node_color_dict,'ew':edge_width_dict,'ec':edge_color_dict}


def plot_graph(G,graphpropdict,ealpha=0.5,node_labels=False,node_size=3,node_fs=9):
    f, ax = mpl.pyplot.subplots(figsize=(5,5))
    mygraph = Graph(G,
                    node_color=graphpropdict['nc'], edge_alpha=ealpha,
                    node_layout='community', node_layout_kwargs=dict(node_to_community=graphpropdict['nodesToComm']),
                    node_edge_width=0, edge_width=graphpropdict['ew'], \
                    edge_layout='bundled', ax=ax, edge_color=graphpropdict['ec'], node_labels=node_labels,
                    node_label_fontdict=dict(size=node_fs), node_size=node_size)  # edge_layout_kwargs=dict(k=2000),node_labels=True
    return f,ax,mygraph





def add_hulls_to_graph(ax,graphhand,communities,**kwargs):

    ulabels = np.unique(communities.membership)
    if 'cdict' in kwargs:
        cdict = kwargs['cdict']
    else:
        cdict = {ii:'k' for ii in np.arange(ulabels)}
    for ii in ulabels:
        mynodenames = np.where(np.array(communities.membership) == ii)[0]
        coord_mat = np.zeros((len(mynodenames), 2))
        for nn, nodename in enumerate(mynodenames):
            my_artist = graphhand.node_artists[nodename]
            coord_mat[nn] = my_artist.xy
        # coord_dict[ii] = coord_mat #coordinates of points for each community
        try:
            hullfns.draw_rounded_hull(coord_mat, padding=0.05, ax=ax, line_kwargs={'color': cdict[ii]})
        except:
            print('cannot plot hulls for community %i'%ii)


def relabel_clusters_by_enrichment(community_mat,enr_mat,clustinds_of_interest,avgfn=lambda x:np.median(x)):
    '''
    :param community_mat: ndarray, resolution x nsamples (e.g. resolution times x areas)
    :param enr_mat: ndarray, nsamples x nclust
    :param clustinds_of_interest: 1-d array with indices of enr_mat to be averaged
    :param avgfn: fn according to which enr_mat gets averaged at clustinds_of_interest
    :return: relabeled community matrix
    '''

    commat_relabeled = np.zeros_like(community_mat)

    for cc, commlabs in enumerate(community_mat):
        ucomm = np.unique(commlabs)
        meds = np.array([avgfn(enr_mat[commlabs == lab][:, clustinds_of_interest]) for lab in ucomm])
        newlabs = np.zeros_like(commlabs)
        for oldval, newval in enumerate(np.argsort(meds)[::-1]):
            newlabs[commlabs == oldval] = newval
        commat_relabeled[cc] = newlabs
    return commat_relabeled


def harmonize_labels(Aseq,Bseq,mode='desc',verbose=False):
    '''
    Aseq and Bseq are sequences of labels
    output: adjusted labels in Bseq, such that labels between Aseq and Bseq match better
    if verbose: the translation dict from Bseq-->newBseq is also returned
    '''
    assert len(Aseq) == len(Bseq), 'both sequences need to be equal in length'
    newBseq = np.zeros_like(Bseq) - 1  # to be filled
    writeboolvec = np.ones_like(Bseq).astype(bool)
    uA = np.unique(Aseq)
    uB = np.unique(Bseq)
    uAsorted = uA[np.argsort([np.sum(Aseq == lab) for lab in uA])[::-1]]
    uBsorted = uB[np.argsort([np.sum(Bseq == lab) for lab in uB])[::-1]]
    trans_dict = {}
    if mode == 'desc':
        for a in uAsorted:
            matchers = Bseq[(Aseq == a) & writeboolvec]
            if len(matchers) > 0:
                umatchers = np.unique(matchers)
                b = umatchers[np.argmax([np.sum(matchers == matchval) for matchval in umatchers])]
                newBseq[Bseq == b] = a
                writeboolvec[Bseq == b] = False
                trans_dict[b] = a

    elif mode == 'asc':
        for b in uBsorted:
            matchers = Aseq[(Bseq == b) & writeboolvec]
            if len(matchers) > 0:
                umatchers = np.unique(matchers)
                a = umatchers[np.argmax([np.sum(matchers == matchval) for matchval in umatchers])]
                newBseq[Bseq == b] = a
                writeboolvec[Bseq == b] = False
                trans_dict[b] = a

    used_labels = np.unique(newBseq[~writeboolvec])
    unmatched_labels = np.unique(Bseq[writeboolvec])
    labels_avail = np.array([val for val in np.arange(len(uB)) if not val in used_labels])
    for newlab, oldlab in zip(labels_avail, unmatched_labels):
        newBseq[Bseq == oldlab] = newlab
        writeboolvec[Bseq == oldlab] = False
        trans_dict[oldlab] = newlab

    # now just quality control
    for b in uB:
        c1 = len(np.unique(newBseq[Bseq == b])) == 1
        c2 = len(newBseq[Bseq == b]) == len(Bseq[Bseq == b])
        assert c1 and c2, 'quality control failed [uniqueness,matchlen] --> %s' % str([c1, c2])

    if verbose:
        return [newBseq,trans_dict]
    else:
        return newBseq


def harmonize_whole_communitymat(comm_mat,refsrcidx):
    '''
    :param comm_mat: ndarray, resolution x nsamples (e.g. resolution times x areas)
    :param refsrcidx: int, indexing the resoluting that provides the seed community according to which the other communities will be harmonized
    '''
    comm_mat2 = np.zeros_like(comm_mat)
    comm_mat2[refsrcidx] = comm_mat[refsrcidx]
    for refidx in np.arange(refsrcidx, len(comm_mat) - 1):
        #print(refidx)
        comm_mat2[refidx+1]= harmonize_labels(comm_mat2[refidx],comm_mat[refidx+1],mode='desc')
    for refidx in np.arange(1,refsrcidx+1)[::-1]:
        comm_mat2[refidx-1] = harmonize_labels(comm_mat2[refidx],comm_mat[refidx-1],mode='asc')
    return comm_mat2


def get_sortinds_community_mat(comm_mat):
    map = np.array(list(string.ascii_uppercase))
    # instead of the following, make it a letter combination up to 26 per region and then stringsort!!!!
    n_letters = len(map)
    n_levels = len(comm_mat)
    sub_comms = comm_mat[:np.min([n_letters-1,n_levels])]
    toy_labels0 = np.vstack([map[sub_labels] for sub_labels in sub_comms])  # .sum(axis=0)#labels*100+spec_labels
    strlist = [''.join(toys) for toys in toy_labels0.T]
    return np.argsort(strlist)  # for sorting vertically!


def plot_document_community_sorting(cmat_orig,cmat_relabeled,cmat_harmonized,sortinds,refsrcidx,maxshowind=10):
    f,axarr = mpl.pyplot.subplots(1,4,figsize=(9,5))
    for ax,pltmat,headstr in zip(axarr,[cmat_orig,cmat_relabeled,cmat_harmonized,cmat_harmonized[:,sortinds]],['orig','relabeled','harmonized','y-sorted']):
        ax.imshow(pltmat[:maxshowind].T,origin='lower',cmap='jet',aspect='auto')
        nr_temp,nc_temp = pltmat[:maxshowind].T.shape
        for xx in np.arange(nr_temp):
            for yy in np.arange(nc_temp):
                ax.text(yy,xx,'%i'%(pltmat.T[xx,yy]),ha='center',va='center')
        ax.axvline(refsrcidx,color='w',alpha=0.5)
        ax.set_xlabel('res. idx')
        ax.set_title(headstr)
    axarr[0].set_ylabel('sample')
    for myax in axarr[1:]:
        myax.set_yticklabels([])
    return f

def plot_clustspect(X,communities,sortinds_com,refsrcidx,res_vec,qvec,shuff_mean,maxind,cmapstr,ylabs,cdict_clust,\
                    boldlabfn=lambda reglab:False,cstr_comm='jet', zlim=[-5,5],extend='both',clab='enr'):
    #ylabs = sdict['alabels']
    #boldlabfn = S.check_pfc
    ncl = X.shape[1]
    f, axarr = mpl.pyplot.subplots(2, 3, figsize=(4.5, 7),
                            gridspec_kw={'height_ratios': [0.1, 1], 'width_ratios': [0.5, 1, 0.1]})
    f.subplots_adjust(hspace=0.01, wspace=0.1, bottom=0.08, left=0.15, top=0.92)
    for myax in axarr[0, 1:]:
        myax.set_axis_off()
    lax, ax, cax = axarr[1]
    qax = axarr[0, 0]
    qax.plot(res_vec[:maxind], qvec[:maxind], 'k')
    qax.plot(res_vec[:maxind], shuff_mean[:maxind], 'grey')
    qax.set_xlabel('res.')
    qax.set_ylabel('Q')
    qax.xaxis.tick_top()
    qax.xaxis.set_label_position('top')
    qax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    # qax.set_xticklabels([])
    im = ax.imshow(X[sortinds_com], origin='lower', aspect='auto', cmap=cmapstr, vmin=zlim[0], vmax=zlim[1])
    lax.imshow(communities[:maxind, sortinds_com].T, cmap=cstr_comm, origin='lower', aspect='auto',
               extent=[res_vec[0], res_vec[maxind - 1], -0.5, len(X) - 0.5], alpha=1, \
               interpolation='nearest')  # ,vmin=-0.5,vmax=n_max+0.5,alpha=0.8
    cb = f.colorbar(im, cax=cax, extend=extend)
    qax.plot([res_vec[refsrcidx]] * 2, [0, qvec[refsrcidx]], 'k:')
    lax.axvline(res_vec[refsrcidx], color='k', linestyle=':')
    pos = lax.get_position()
    posx = ax.get_position()
    ax.set_position([pos.x1, posx.y0, posx.width, posx.height])
    ax.set_yticks([])
    lax.set_xticklabels([])
    qax.set_xlim([res_vec[0], res_vec[maxind - 1]])
    lax.set_axis_off()
    # ax.xaxis.tick_top()
    cax.set_title(clab)
    ax.set_xticks(np.arange(ncl))
    ax.set_xticklabels(np.arange(ncl) + 1)
    for tt, tick in enumerate(ax.xaxis.get_ticklabels()):
        tick.set_color(cdict_clust[tt])
        tick.set_fontweight('bold')
    ax.set_yticks(np.arange(len(X)))
    ax.set_yticklabels(ylabs[sortinds_com], fontsize=9)
    for tt, tick in enumerate(ax.yaxis.get_ticklabels()):
        if boldlabfn(tick.get_text()):
            tick.set_fontweight('bold')
    return f,axarr