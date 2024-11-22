from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
import numpy as np
import scipy.cluster.hierarchy as hac
from sklearn.metrics import davies_bouldin_score,silhouette_score
from pfcmap.python.utils import cluster_validation as valid
from scipy.stats import scoreatpercentile
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def get_clusterfn_dict(roivals):
    clusterfn_dict = {'ward':lambda nclusters:hac.fcluster(hac.linkage(roivals,method='ward'),nclusters,'maxclust')-1,\
                    'centroid':lambda nclusters:hac.fcluster(hac.linkage(roivals,method='centroid'),nclusters,'maxclust')-1,\
                    'kmeans': lambda nclusters: KMeans(n_clusters=nclusters).fit_predict(roivals),\
                    'gmm':lambda nclusters: KMeans(n_clusters=nclusters).fit_predict(roivals)}
    return clusterfn_dict

def get_cfn_dict():
    clusterfn_dict = {'ward':lambda nclusters,vals:hac.fcluster(hac.linkage(vals,method='ward'),nclusters,'maxclust')-1,\
                    'centroid':lambda nclusters,vals:hac.fcluster(hac.linkage(vals,method='centroid'),nclusters,'maxclust')-1,\
                    'kmeans': lambda nclusters,vals: KMeans(n_clusters=nclusters).fit_predict(vals),\
                    'gmm':lambda nclusters,vals: KMeans(n_clusters=nclusters).fit_predict(vals)}
    return clusterfn_dict


def get_evalfn_dict():
    return {'DB': davies_bouldin_score, 'Dunn': valid.dunn_fast, 'sil': silhouette_score}


def make_rand_corr(Xorig,weightvals,covariancemat):
    randv = norm.rvs(size=np.array(Xorig.shape).T)
    decomp = cholesky(covariancemat)
    randv_corr = np.dot(decomp,randv.T)
    randX = randv_corr.T*weightvals
    return randX

def get_randdict(X,nclustvec,clustfn,evalfns,nrep = 100):
    weightvec = np.ones(X.shape[1])
    covmat = np.cov(X.T)

    randict = {tag: np.zeros((nrep, len(nclustvec))) for tag in evalfns.keys()}

    for rr in np.arange(nrep):
        randX = make_rand_corr(X,weightvec , covmat)
        for nn, nclust in enumerate(nclustvec):
            labels = clustfn(nclust,randX)
            for tag,fn in evalfns.items():
                randict[tag][rr][nn] = fn(randX,labels)
    return randict


def evaluate_clustering(featmat,labels,evalfns):
    eval_dict = {}
    for tag,fn in evalfns.items():
        eval_dict[tag] = fn(featmat,labels)
    return eval_dict

def eval_asfn_nclust(featmat,nclustvec,clustfn,evalfns):
    Evaldict = {tag: np.zeros((len(nclustvec))) for tag in evalfns.keys()}

    for nn, nclust in enumerate(nclustvec):
        labels = clustfn(nclust,featmat)
        eval_dict = evaluate_clustering(featmat,labels,evalfns)
        for key,evalval in eval_dict.items():
            Evaldict[key][nn] = evalval
    return Evaldict

def rand_to_scoredict(randict,percentiles = [20,50,80]):
    nclusts_tested = (list(randict.values())[0]).shape[1]
    scoredict = {tag: np.zeros((len(percentiles),nclusts_tested)) for tag in randict.keys()}
    for tag in randict.keys():
        randvals = randict[tag]
        for pp,perc in enumerate(percentiles):
            score = scoreatpercentile(randvals,perc,axis=0)
            scoredict[tag][pp] = score
    return scoredict


def sort_labels(labels0,featmat,featlist,sortfeat,avgfn=lambda x:np.median(x)):
    '''featlist is list of strings with feature names
    featmat: nsamples x nfeatures'''
    assert sortfeat in featlist, '%s not in featlist'%sortfeat
    luniques = np.unique(labels0)
    ff = featlist.index(sortfeat)
    labelavgs = np.array(
        [avgfn([featmat[ii,ff] for ii, lab in enumerate(labels0) if lab == mylab]) for mylab in luniques])
    lranks = labelavgs.argsort().argsort()
    labels = np.zeros_like(labels0)
    for ll, lab0 in enumerate(luniques):
        ranked_val = lranks[ll]
        labels[labels0 == lab0] = ranked_val
    return labels



def sort_labels_featsimilarity(labels0,featmat,featlist,sortfeat,avgfn=lambda x:np.median(x,axis=0),mode='low_to_high'):
    '''featlist is list of strings with feature names
    featmat: nsamples x nfeatures'''
    assert sortfeat in featlist, '%s not in featlist'%sortfeat
    luniques = np.unique(labels0)
    avgs =  np.array([avgfn([featmat[ii] for ii, lab in enumerate(labels0) if lab == mylab]) for mylab in luniques])
    ff = featlist.index(sortfeat)
    maxlab = avgs[:,ff].argmax()
    refvec = avgs[maxlab]
    if mode == 'low_to_high':
        lranks = (np.linalg.norm(avgs - refvec, axis=1).argsort().argsort() - luniques.max())*-1
    else:
        lranks = np.linalg.norm(avgs - refvec, axis=1).argsort().argsort()
    labels = np.zeros_like(labels0)
    for ll, lab0 in enumerate(luniques):
        ranked_val = lranks[ll]
        labels[labels0 == lab0] = ranked_val
    return labels


def sort_labels_by_idx(ff,labels0,featmat,avgfn=lambda x:np.median(x,axis=0),mode='low_to_high'):
    '''featlist is list of strings with feature names
    featmat: nsamples x nfeatures'''
    luniques = np.unique(labels0)
    avgs = np.array([avgfn([featmat[ii] for ii, lab in enumerate(labels0) if lab == mylab]) for mylab in luniques])
    maxlab = avgs[:,ff].argmax()
    refvec = avgs[maxlab]

    if mode == 'low_to_high':
        lranks = (np.linalg.norm(avgs - refvec, axis=1).argsort().argsort() - luniques.max())*-1
    else:
        lranks = np.linalg.norm(avgs - refvec, axis=1).argsort().argsort()
    labels = np.zeros_like(labels0)
    for ll, lab0 in enumerate(luniques):
        ranked_val = lranks[ll]
        labels[labels0 == lab0] = ranked_val
    return labels






def plot_clustquality(eval_dict,nclustvec,show_rand=True,**kwargs):
    flist = []
    for tag in eval_dict.keys():
        #f,ax = plt.subplots()
        f, ax = plt.subplots(figsize=(4,3))
        f.subplots_adjust(left=0.2,bottom=0.2)
        ax.text(0.5,0.99,tag,fontsize=10,ha='center',va='top',transform=ax.transAxes)
        ax.plot(nclustvec,eval_dict[tag],'o-k',mfc='w',mec='k',lw=2,ms=5,mew=1,zorder=20)
        #ax.axvline(clustsom,color='r',linestyle='--',zorder=30)
        if show_rand and 'scoredict' in kwargs:
            scoredict = kwargs['scoredict']
            ax.fill_between(nclustvec, scoredict[tag][0], scoredict[tag][2], color='grey', alpha=0.5, zorder=0, linewidth=0)
            ax.plot(nclustvec, scoredict[tag][1], '-', color='grey', lw=1)
        ax.set_xlim([nclustvec.min(),nclustvec.max()])
        ax.set_xticks([5, 10, 15])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,-1))
        #ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel('N clusters')
        ax.set_ylabel('score')
        flist += [f]
    return flist



def get_sortidx_featdist(featmat,linkmethod='ward'):
    dists = pdist(featmat)
    Z = hac.linkage(dists, method='ward')
    dn = hac.dendrogram(Z, no_plot=True)
    return np.array(dn['leaves'])


def plot_distmat(featmat,link_method='ward',axlab='samples', cmap='viridis_r',showticks=True,**kwargs):
    '''featmat: nsamps x nfeats'''

    N_samps = len(featmat)

    dists = pdist(featmat)
    Z = hac.linkage(dists, method=link_method)
    dn = hac.dendrogram(Z, no_plot=True)
    idx = dn['leaves']

    f, axarr = plt.subplots(1, 2, figsize=(4.82, 4.2), gridspec_kw={'width_ratios': [1, 0.05]})
    f.subplots_adjust(bottom=0.13)
    ax, cax = axarr
    D = squareform(dists)  # in order not to overwrite
    D = D[idx, :]
    D = D[:, idx]
    im = ax.imshow(D, aspect='auto', origin='lower', cmap=cmap)
    if showticks:
        ax.set_xticks(np.arange(N_samps))
        ax.set_yticks(np.arange(N_samps))
    if 'samplabs' in kwargs:
        samplabs = kwargs['samplabs']
        ax.set_xticklabels(samplabs[idx], rotation=90)
        ax.set_yticklabels(samplabs[idx])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylabel(axlab)
    ax.set_xlabel(axlab)
    cb = f.colorbar(im, cax=cax)
    cax.set_title('dist.')
    return f,axarr





def plot_thorndike(nclustvec,featmat,linkage_method = 'centroid'):
    Z = hac.linkage(featmat, method=linkage_method)
    nback = nclustvec[-1]
    mergedists = Z[-nback:, 2][::-1]
    f, ax = plt.subplots(figsize=(4, 3))
    f.subplots_adjust(left=0.2, bottom=0.2)
    ax.plot(np.arange(nback) + 1, mergedists, 'o-k', ms=5, mec='k', mfc='none')
    # ax.axvline(nclust,color='grey',linestyle='--')
    ax.text(0.5, 0.99, 'thorn. %s'%linkage_method, fontsize=10, ha='center', va='top', transform=ax.transAxes)
    ax.set_xlabel('N clusters')
    ax.set_ylabel('D(clust)')
    return f