import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hac
from scipy.stats import scoreatpercentile
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE,MDS,Isomap
from copy import deepcopy
from sklearn.decomposition import PCA
from .. import plotting_basics as pb
from ..ica_tools import icafns as icaf
from scipy.stats import kurtosis
import matplotlib as mpl
import matplotlib.ticker as ticker
import yaml
import h5py

make_z = lambda vals: (vals-vals.mean(axis=1)[:,None])/vals.std(axis=1)[:,None]


def storedprop(fn):
	attr_name = '_' + fn.__name__

	@property
	def _lazyprop(self):
		if not hasattr(self, attr_name):
			setattr(self, attr_name, fn(self))
		return getattr(self, attr_name)
	return _lazyprop





class ICASSO(object):
    def __init__(self,ica_class,ica_params,n_iter,bootstrap=True,id=''):
        self.ica_class = ica_class
        self.ica_params = ica_params
        self.n_comps = self.ica_params['n_components']
        self.n_iter = n_iter
        self.bootstrap = bootstrap
        self.id = id

    def set_data(self,X):
        '''must be featuers x data'''
        self.n_features,self.n_samples = X.shape
        self.X = X
        self.W_dim1 = self.n_iter*self.n_comps #K in the Himberg 2004 paper


    def set_seeds(self,seeds_ica,seeds_boot=None):
        #if you want to replicate exact experiment
        self._seeds_boot = seeds_boot
        self._seeds_ica = seeds_ica

    @property
    def seeds_boot(self):
        if not hasattr(self,'_seeds_boot'):
           self._seeds_boot = np.random.randint(0, 2**20,self.n_iter)
        return self._seeds_boot

    @property
    def seeds_ica(self):
        if not hasattr(self, '_seeds_ica'):
            self._seeds_ica = np.random.randint(0, 2 ** 20, self.n_iter)
        return self._seeds_ica

    def get_What(self,dim2='n_features',**kwargs):

        self.bootcut = False

        if 'maxsamples' in kwargs:
            if self.n_samples<kwargs['maxsamples']: self.n_boot = self.n_samples
            else:
                self.n_boot = kwargs['maxsamples']
                self.bootcut = True
        else:
            self.n_boot = self.n_samples

        self.K_listed = []

        if dim2.count('feat'): d2 = self.n_features
        elif dim2.count('comp'): d2 = self.n_comps
        else: assert 0, 'What 2nd dim must be feat or comps'
        self.W_hat = np.zeros((self.W_dim1,d2))

        for ii in np.arange(self.n_iter):

            ica = self.ica_class(random_state=self.seeds_ica[ii], **self.ica_params)

            if self.bootstrap:
                np.random.seed(self.seeds_boot[ii])
                inds = np.random.choice(np.arange(self.n_samples), size=self.n_boot,replace=True)
            else: inds = np.arange(self.n_samples)

            Xboot = self.X[:, inds]
            #print (ii,Xboot[20,2000])# to test whether retrieve_bootstrap really works!
            ica.fit(Xboot.T)

            if d2 == self.n_features: self.W_hat[ii * self.n_comps:self.n_comps * (ii + 1), :] = ica.components_
            elif d2 == self.n_comps: self.W_hat[ii * self.n_comps:self.n_comps * (ii + 1), :] = ica._unmixing
            self.K_listed += [ica.whitening_]

    def retrieve_bootstrap(self,boot_idx):
        np.random.seed(self.seeds_boot[boot_idx])
        inds = np.random.choice(np.arange(self.n_samples), size=self.n_boot,replace=True)
        return self.X[:,inds]





    def calc_dissimilarity(self,src='W',method='corr',dm_fn=lambda sm: np.sqrt(1-sm),norm_on=False):
        '''calculate dissimilarity matrix
        dm_fn: function applied to dissmilarity matrix to obtain similarity matrix'''
        self._DM_basis = src
        if src=='W': WC = self.W_hat
        elif src=='C': WC = self.C
        else: assert 0, 'src must be = W or C'

        if norm_on:
            WC_ = (WC-WC.mean(axis=0))/WC.std(axis=0)
        else:
            WC_ = WC

        self.distmethod = method
        if method == 'corr':
            self.dm_fn = dm_fn#to get it printed when saving model: inspect.getsource(dm_fn)
            DM = dm_fn(np.abs(np.corrcoef(WC_)))#similarity matrix
            #there can be super small diffs such that the mat is not symm, i.e. np.triu(DM)!=np.tril(DM).T --> make it super-symmetric for MDS

        elif method == 'euclid':
            DM = euclidean_distances(WC_)
            print ('not fully developed --> should give lowest distances between each pair as components may be flipped')
        else: assert 0, 'unknown method for distance calculation'
        self.DM = make_symmetric(DM)#dissimilarity matrix
        self.DM[np.arange(WC.shape[0]), np.arange(WC.shape[0])] = 0 #important to make exactly zero diag for squareform to work
        #self.disvec = squareform(self.DM)# squarefrom just gets the relevant entries from the matrix into array format, via pdist from hac you do the same (distance+flattening the pairwise) in one go for euclidean
        #disvec is usable as input to linkage
        self.SM = np.abs(np.corrcoef(WC_))

    @storedprop
    def iter_labels(self):
        return np.hstack([[ii]*self.n_comps for ii in np.arange(self.n_iter)])


    def set_cluster_labels(self,labellist):
        self.labellist = labellist[:]
        self.n_clust = len(labellist)

    def calc_components(self,W_src='W_hat'):
        W_mat = getattr(self,W_src)
        self.C = np.zeros((self.W_dim1,self.n_features))
        for ii,[W, iter] in enumerate(zip(W_mat,self.iter_labels)):
            self.C[ii] = np.dot(W,self.K_listed[iter])


    def calc_clusterWC(self,attr='C',mode='avg',perc=70):

        #per default for each cluster!
        #AVG-COMPONENTS
        if not hasattr(self,attr+'_aligned'):self.align(attr)
        WC_aligned = getattr(self,attr+'_aligned')
        N = len(self.labellist)


        WC_clust = np.zeros((N,WC_aligned.shape[1]))

        if mode == 'avg':
            for cc,myinds in enumerate(self.labellist):
                WC_clust[cc] = WC_aligned[np.array(myinds)].mean(axis=0)
        elif mode == 'repr':
            clustWC_repr = np.zeros(N)
            for cc,myinds in enumerate(self.labellist):
                if len(myinds)==1:
                    clustWC_repr[cc] = myinds
                    WC_clust[cc] = WC_aligned[myinds]
                else:
                    #make distance-matrix the one with the 3 highest correlation scores
                    temp = WC_aligned[np.array(myinds)]

                    smat = np.corrcoef(temp)

                    rep_idx = np.argmax(scoreatpercentile(smat,perc,axis=1))
                    clustWC_repr[cc] = myinds[rep_idx]
                    WC_clust[cc] = WC_aligned[myinds[rep_idx]]
            setattr(self,'clust%s_repr'%attr,clustWC_repr)
        setattr(self,attr+'_clust',WC_clust)


    def align(self, which='both'):
        if which == 'both' or which == 'W':
            self.W_aligned = clustalign(self.W_hat, self.labellist, self.DM, reps=2)
        if which == 'both' or which == 'C':
            if not hasattr(self,'C'): self.calc_components(W_src='W_hat')
            self.C_aligned = clustalign(self.C, self.labellist, self.DM, reps=2)

    def calc_S(self):
        self.S = np.dot(self.C_clust, self.X)

    def calc_A(self):
        self.A = np.linalg.pinv(self.C_clust)

    def flip_C_and_A(self):
        if not hasattr(self,'A'):self.calc_A()
        flipfacs = np.sign(self.A.sum(axis=0)).astype(int)
        self.A = self.A*flipfacs[None,:]
        self.C_clust = self.C_clust*flipfacs[:,None]
        C_alflipped = np.zeros_like(self.C_aligned)
        for ii, myinds in enumerate(self.labellist):
            flipfac = flipfacs[ii]
            C_alflipped[np.array(myinds)] = flipfac * self.C_aligned[np.array(myinds)]
        self.C_aligned = C_alflipped
        self.flipped = True

    def calc_Xrecon(self,clinds=None,adjustToX=True):
        if not hasattr(self,'A'):self.calc_A()
        if not hasattr(self,'S'):self.calc_S()
        if isinstance(clinds,type(None)):
            clinds = np.arange(len(self.labellist))
        if isinstance(clinds,int):
            self.Xrecon = np.dot(self.A[:,clinds][:,None],self.S[clinds][None,:])
        else: self.Xrecon = np.dot(self.A[:,clinds],self.S[clinds])
        if adjustToX:
            reconz = make_z(self.Xrecon)
            self.Xrecon = reconz * self.X.std(axis=1)[:, None] + self.X.mean(axis=1)[:, None]



    def project_distances(self, flav='tsne', fn_params={},rs=42):
        # W_ = (I.W_hat-I.W_hat.mean(axis=0))/I.W_hat.std(axis=0)
        self.proj_flav = flav
        self.proj_params = fn_params
        if flav == 'tsne':
            if not 'perplexity' in fn_params:
                print('WARNING: no perplexity specified')
            self.proj = TSNE(n_components=2, metric='precomputed', random_state=rs, **fn_params).fit_transform(
                self.DM)

        elif flav == 'mds':
            self.proj = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=rs, dissimilarity="precomputed",
                      **fn_params).fit_transform(self.DM)

        elif flav == 'isomap':
            if not 'n_neighbors' in fn_params: print('WARNING: no n_neighbors specified')
            self.proj = Isomap(metric='precomputed', **fn_params).fit_transform(self.DM)



class ICASSO_fromData(ICASSO):
    def __init__(self,h5path):

        self.h5path = h5path

    def retrieve_fromh5(self,pathfile,to_unpack=['results','info','methods']):
        with open(pathfile,'r') as f: h5paths = yaml.safe_load(f)
        with h5py.File(self.h5path) as hfile:
            for superdir in to_unpack:
                DD = h5paths[superdir]
                for Dkey in DD.keys():
                    if Dkey == 'dsets': myfn = lambda mypath,mykey,srcfile: srcfile[mypath+'/'+mykey][()]
                    elif Dkey == 'attrs': myfn = lambda mypath,mykey,srcfile: srcfile[mypath].attrs[mykey]
                    for key,path in DD[Dkey].items():
                        #print (key,Dkey)
                        if key == 'labellist':
                            data =  [tuple([clust.attrs[attr] for attr in clust.attrs if attr.count('i')]) for clust in hfile[path+'/'+key].values()]
                        elif key.count('seeds'):
                            data = hfile[path+'/'+key.replace('seeds_','')][()]
                            key = '_'+key
                            #print (data.shape)
                            #print (key)
                        else: data = myfn(path,key,hfile)
                        if type(data) == np.bytes_: data = data.decode('UTF-8')
                        #print (key)
                        setattr(self,key,data)

    def getX_fromfile(self,extractfn):
        self.datalist = [extractfn(tstart, tstop) for tstart, tstop in self.epocs]
        self.X = np.squeeze(np.concatenate(self.datalist, axis=1))





def sort_kurtosis_kept(I,CC,clinds_old):
    I.calc_S()
    I.calc_Xrecon()
    srcmat = I.S[clinds_old]
    kurt = kurtosis(srcmat, axis=1)
    sortinds = np.argsort(kurt)[::-1]
    setattr(CC,'kept',[CC.kept[idx] for idx in sortinds])
    CC.get_reforder()
    clinds = CC.kept_reforder[:]
    return clinds

def make_symmetric(mat):
    return np.triu(mat.T, k=1) + np.tril(mat)

calc_div = lambda vals: len(np.unique(vals)) / len(vals)

class ConditionalClustering(object):
    def __init__(self,distance_mat,iter_labels):
        self.DM = distance_mat
        self.nsamp = self.DM.shape[0]
        self.labels_orig = np.arange(self.nsamp)
        self.iterlabs = [[val] for val in iter_labels]
        self.iters_orig = iter_labels[:]
        self.n_iter = len(np.unique(self.iters_orig))


    def weed_out(self,min_scope=0.8):

        n_min = np.floor(self.n_iter * min_scope)#minimum number of size of a cluster

        self.discarded = []
        self.kept = []
        self.N_clust = len(self.labellist)

        for cc in np.arange(self.N_clust):
            if len(np.unique(self.iterlabs[cc])) >= n_min:
                self.kept += [self.labellist[cc]]
            else:
                self.discarded += [self.labellist[cc]]

        self.n_kept = len(self.kept)
        self.n_discarded = len(self.discarded)

    def sort_clusters(self,quality_fn,reverse=True):
        # M is similarity matrix for I_q and dissimilarity matrix for

        for attr in ['kept','discarded']:
            lablist = deepcopy(getattr(self,attr))
            if len(lablist)>0:
                delattr(self, attr)
                iq_vec = quality_fn(lablist)
                sortinds = np.argsort(iq_vec)
                if reverse: sortinds = sortinds[::-1]
                ordered_list = [lablist[cc] for cc in sortinds]
                #refinds = np.array([[jj for jj in np.arange(self.N_clust) if lab==self.labellist[jj]][0] for lab in lablist])
                #setattr(self,attr+'_reforder',refinds)
                setattr(self,attr,ordered_list)

            self.get_reforder()

    def get_reforder(self):
        for attr in ['kept','discarded']:
            refinds = np.array([[jj for jj in np.arange(self.N_clust) if lab == self.labellist[jj]][0] for lab in getattr(self,attr)])
            setattr(self,attr+'_reforder',refinds)


class ClustEspositoCond(ConditionalClustering):


    def run(self,purge_doublettes=True,diversity_thresh=0.8,minscope=0.5):
        distvec = squareform(self.DM)
        self.minscop = minscope
        # threshvec = np.unique(distvec)
        self.threshvec = np.arange(0., 1., distvec.min())
        linkage = hac.linkage(distvec, method='complete')
        self.minsize = int(self.n_iter * minscope)
        self.diversity_thresh = diversity_thresh
        self.purge = purge_doublettes
        self.exiles = []

        self.candidate_dict = {}

        for ii, thr in enumerate(self.threshvec):
            # thr = threshvec[50]
            parts = hac.fcluster(linkage, thr, 'distance')
            clust_ids = np.unique(parts)
            candidates = []
            for cid in clust_ids:
                cand_inds = np.where(parts == cid)[0]
                cand_inds = np.array([cand for cand in cand_inds if not cand in self.exiles])
                iterlabs = self.iters_orig[cand_inds]
                diversity = calc_div(iterlabs)
                if len(cand_inds) >= self.minsize and diversity >= diversity_thresh:

                    if diversity < 1.:
                        # low_div += [tuple(cand_inds)]
                        if purge_doublettes:
                            double_iters = [iter for iter in np.unique(iterlabs) if np.sum(iterlabs == iter) > 1]
                            for diter in double_iters:
                                submat = self.DM[np.ix_(cand_inds, cand_inds)]
                                # print(len(iterlabs),len(cand_inds))
                                cond = iterlabs == diter
                                doublet_ids = cand_inds[cond]
                                win_idx = np.argmin(
                                    submat[cond].mean(axis=1))  # the one with the lowest mean dissimilarity wins
                                winner = doublet_ids[win_idx]
                                losers = list(doublet_ids[np.arange(len(doublet_ids)) != win_idx])
                                # take out losers
                                for loser in losers:
                                    iterlabs = np.delete(iterlabs, np.where(cand_inds == loser))
                                    cand_inds = np.delete(cand_inds, np.where(cand_inds == loser))

                                self.exiles += losers
                    candidates += [tuple(cand_inds)]
            self.candidate_dict[ii] = [candidates, thr]

    def weed_out(self,max_dissim_soft=0.4,max_dissim_hard=0.6,thresh_int=0.02,minperc=0.3):
        #instead of the weed out
        self.n_comps = int(self.nsamp / len(np.unique(self.iters_orig)))
        self.minperc = minperc
        self.hardthresh = max_dissim_hard
        self.softthresh = max_dissim_soft
        self.thresh_int = thresh_int

        get_clustsizes = lambda cllist: np.array([len(sublist) for sublist in cllist]) if len(cllist)>0 else np.array([0])
        self.nclusts = np.array([len(mylist[0]) for mylist in self.candidate_dict.values()])
        #threshvec = np.array([mylist[1] for mylist in self.candidate_dict.values()])
        self.meansizes = np.array([get_clustsizes(mylist[0]).mean() for mylist in self.candidate_dict.values()])

        self.idx,self.relaxed = self.run_threshfinder(self.softthresh)

        #self.max_dissim = max_dissim

        #idx = np.where(self.threshvec <= max_dissim)[0][-1]
        self.labellist = self.candidate_dict[self.idx][0]
        self.kept = self.candidate_dict[self.idx][0]
        self.discarded = []#here there are no discarded existing clusters
        self.N_clust = len(self.labellist)
        self.n_kept = len(self.kept)
        self.n_discarded = len(self.discarded)

    def run_threshfinder(self,softthresh):
        relaxed = False
        soft_thresh = float(softthresh)
        cond = (self.threshvec <= soft_thresh)
        idx = np.argmax(self.nclusts[cond])
        Nclusts = self.nclusts[idx]
        # print(softthresh,idx)
        if Nclusts > self.n_comps:
            bestmatch_n = np.abs(self.nclusts[cond] - self.n_comps).min()
            idx = np.where(np.abs(self.nclusts[cond] - self.n_comps) == bestmatch_n)[0][0]
        if Nclusts <= (self.minperc * self.n_comps) and soft_thresh <= (self.hardthresh + self.thresh_int):
            idx, rel = self.run_threshfinder(soft_thresh + self.thresh_int)
            relaxed = True
        return idx, relaxed

    def plot_threshfinding(self):
        dissim_thresh = self.threshvec[self.idx]
        f, ax = plt.subplots()
        ax.plot(self.threshvec, self.nclusts, 'ok-', ms=2)
        ax2 = ax.twinx()
        ax2.plot(self.threshvec, self.meansizes, 'grey')
        ax.axvline(self.softthresh, color='b', alpha=0.5, zorder=-10)
        ax.axvspan(self.softthresh, 1, color='b', alpha=0.1, zorder=-10)
        if self.relaxed:
            ax.axvspan(self.hardthresh, 1, color='b', alpha=0.1, zorder=-10)
        ax.plot(dissim_thresh, self.nclusts[self.idx], 'o', mfc='none', mec='r', ms=4, mew=2)
        ax.plot([dissim_thresh, dissim_thresh], [0, self.nclusts[self.idx]], color='r')
        for tickl in ax2.yaxis.get_ticklabels(): tickl.set_color('grey')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_title('max dissim: %1.2f, dissim: %1.2f, Nclust: %i' % (self.softthresh, dissim_thresh, self.nclusts[self.idx]))
        ax.set_xlabel('dissim. threshold')
        ax.set_ylabel('N clusters')
        ax2.set_ylabel('mean clustsize', color='grey', rotation=-90,labelpad=20)
        return f




class ClustAggloCond(ConditionalClustering):


    def run(self,blank_nb=9,max_distperc=30,min_diversity=0.9):
        self.max_distperc = max_distperc
        self.max_dissim = scoreatpercentile(self.DM, max_distperc)
        self.min_diversity = min_diversity

        #blank_nb is number to insert into the distance matrix, where pairwise distances should not be considered (effectively masking)
        self.dm_mat = np.triu(self.DM, k=1) + np.tril(np.ones_like(self.DM) * blank_nb)
        self.labellist = [(val,) for val in self.labels_orig]
        self.exiles = []
        self.dmat_min = self.dm_mat.min()
        self.counter = 0
        maxcounter = 2000 #to prevent endless while-loops

        while self.dmat_min <= self.max_dissim and self.counter <= maxcounter:
            a, b = np.where(self.dm_mat == self.dmat_min)  # assumption here: its just a single index
            assert len(a) == 1 and len(b) == 1, 'non-unique minimum --> please loop!'
            a, b = a[0], b[0]
            setA, setB = self.labellist[a], self.labellist[b]
            # check whether it is allowed to merge set A and B
            newset = setA + setB
            diversity_ratio = len(np.unique(self.iterlabs[a] + self.iterlabs[b])) / len(newset)#how different is setA from setB
            if diversity_ratio >= self.min_diversity:

                # REMOVE
                # remove a and b from the matrix and the labels
                self.dm_mat = np.delete(self.dm_mat, np.array([a, b]), 0)
                self.dm_mat = np.delete(self.dm_mat, np.array([a, b]), 1)
                # remove the indices
                self.iterlabs.pop(a)
                self.labellist.pop(a)
                self.iterlabs.pop(b - 1)
                self.labellist.pop(b - 1)

                # FILL
                # fill with new column
                new_col = np.array([np.mean([self.DM[lab1, lab2] for lab1 in labs1 for lab2 in newset]) for labs1 in self.labellist])
                self.dm_mat = np.c_[self.dm_mat, new_col]
                self.dm_mat = np.vstack([self.dm_mat, np.ones(len(new_col) + 1) * blank_nb])
                # append the indices
                self.labellist += [newset]
                self.iterlabs += [[self.iters_orig[idx] for idx in newset]]

            else:
                if len(setA) > 1:
                    self.dm_mat[a, b], self.dm_mat[b, a] = [blank_nb, blank_nb]
                else:
                    self.dm_mat = np.delete(self.dm_mat, a, 0)
                    self.dm_mat = np.delete(self.dm_mat, a, 1)
                    self.iterlabs.pop(a)
                    self.labellist.pop(a)
                    self.exiles += [a]
            self.dmat_min = self.dm_mat.min()
            self.counter += 1

        # SANITY CHECK
        x = np.array([item for sublist in self.labellist for item in sublist])
        assert len(x) == len(np.unique(x)), 'doublettes found, no clear cluster identity'






def calc_diversity(iter_labels,lablist):
    iterlists = [iter_labels[np.array(labs)] for labs in lablist]
    return np.array([len(np.unique(iters))/len(iters) for iters in iterlists])


def clustalign(data,labellist,distmat,reps=2):
    """data is (n_comps*n_iter)x(n_features)"""
    data_aligned = deepcopy(data)

    #self.representatives = np.zeros(len(self.labellist))

    for cc,myinds in enumerate(labellist):
        # get the similarity matrix cutout

        #set the most-cluster-similar as a first reference
        dm = distmat[np.ix_(np.array(myinds), np.array(myinds))]  # I.DM[myinds[2],myinds[3]] == dm[2,3]

        min_idx = np.argmin(np.median(dm, axis=1))
        seedidx = myinds[min_idx]
        #self.representatives[cc] = seedidx
        for rr in np.arange(reps):
            if rr ==0:
                reftrace = deepcopy(data_aligned[seedidx])
                tracestack = reftrace[:]

            facarray = np.zeros(len(myinds))
            counter = 1
            for ii, idx in enumerate(myinds):
                trace = data_aligned[idx][:]
                fac = 1 if np.linalg.norm(reftrace - trace) < np.linalg.norm(reftrace - trace * -1) else -1
                # fac = np.sign(np.corrcoef(reftrace,trace)[0,1])
                # print(fac,np.corrcoef(reftrace,trace)[0,1])
                # W_aligned[idx] = trace*fac
                if rr == 0:
                    tracestack += trace * fac
                    reftrace = tracestack / (counter + 1)
                counter += 1
                facarray[ii] = fac

                # counter +=1
            data_aligned[np.array(myinds)] = data_aligned[np.array(myinds)] * facarray[:, np.newaxis]
    return data_aligned
    # cc = 1
    # myinds = I.labellist[cc]
    # f, ax = plt.subplots()
    # for ii, idx in enumerate(myinds):
    #     col = 'r' if idx == I.representatives[cc] else 'k'
    #     # ax.plot(I.W_hat[idx],col)
    #     ax.plot(I.W_aligned[idx], col)

    # PROMBLEM: kmeans can get odd with few repetitions
    # kmeans = KMeans(init="random", n_clusters=2)
    # self.W_aligned = deepcopy(self.W_hat)
    # # make the max positive
    # for myinds in self.labellist:
    #     kmeans.fit(self.W_hat[np.array(myinds)])
    #     switchinds = np.array(myinds)[kmeans.labels_ == 1]
    #     self.W_aligned[switchinds] = self.W_hat[switchinds] * -1



def get_plot_quality(llist,I,plot=True,verbose=False):
    """llist: list of tuples, each tuple containing indices of a specific cluster;
    I is an Icasso object"""
    iq_vec = calc_Iq(llist,I.SM,summary_fn=np.median)
    mmm_dists = calc_within_distances(llist,I.DM,summary_fn=np.median)
    sizes = np.array([len(labs) for labs in llist])
    diversity = calc_diversity(I.iter_labels, llist)
    qualdict = {'I_q':iq_vec,'dists':mmm_dists,'sizes':sizes,'diversity':diversity}
    outlist = []

    if plot:
        f,axarr = plt.subplots(2,2,figsize=(12,10))

        plot_clustscore(iq_vec,'I_q',ax=axarr[0,0])

        plot_clustscore(mmm_dists,'within clust dissim [min,med.,max]',ax=axarr[1,0])


        plot_clustscore(sizes,'cluster size','k',ax=axarr[0,1])
        plot_clustscore(diversity,'diversity ratio','k',ax=axarr[1,1])
        outlist += [f]

    if verbose: outlist += [qualdict]
    if np.sum([plot,verbose])!=2: return outlist[0]
    else: return outlist

def plot_clustscore(scorevec,score_label,color='k',**kwargs):

    if not 'ax' in kwargs:
        f, ax = plt.subplots()
    else:
        ax = kwargs['ax']

    N = scorevec.shape[-1]
    xvec = np.arange(N)+1
    if scorevec.ndim ==1:
        ax.plot(xvec, scorevec, 'o-',color=color)
    else:
        for cc in (xvec):
            ax.plot([cc]*scorevec.shape[0], scorevec[:, cc - 1], 'o-',color=color)

    ax.set_xlim([0.5, N + 0.5])
    ax.set_xlabel('Cluster')
    ax.set_ylabel(score_label)
    if not 'ax' in kwargs: return f

def calc_Iq(labellist,SM,summary_fn=np.mean):
    #SM is similarity matrix
    #labellist can also contain only a sublist
    all_ids = np.arange(SM.shape[0])
    N_clust = len(labellist)
    iq_vec = np.zeros((N_clust))
    for cc in np.arange(N_clust):
        myinds = labellist[cc]
        outside_inds = tuple([id for id in all_ids if not id in myinds])

        sim_inside = summary_fn([SM[ii, jj] for ii, jj in itertools.combinations(myinds,
                                                                              2)])  # /div# sum and then /div = len(myinds)*(len(myinds)-1)/2.
        sim_outside = summary_fn([SM[ii, jj] for ii, jj in itertools.product(myinds, outside_inds)])
        iq_vec[cc] = sim_inside - sim_outside
    return iq_vec


def calc_within_distances(labellist,DM,summary_fn=np.mean):
    N_clust = len(labellist)
    fn_list = [np.min, summary_fn, np.max]
    mmm_dists = np.zeros((3, N_clust))
    for cc in np.arange(N_clust):
        myinds = labellist[cc]
        within_dists = [DM[ii, jj] for ii, jj in itertools.combinations(myinds, 2)]
        mmm_dists[:, cc] = np.array([fn(within_dists) for fn in fn_list])
    return mmm_dists



def clust_list_to_inds(labellist,blank_nb=-2,n_samples=None):
    n_samp = len([item for sublist in labellist for item in sublist]) if n_samples == None else n_samples
    clindarray = np.zeros((n_samp)).astype(int) + blank_nb
    for cc, inds in enumerate(labellist): clindarray[np.array(inds)] = cc
    return clindarray

def clust_inds_to_list(clindarray,blank_nb=-2):
    # switch_back
    labellist = []
    for cc in np.unique(clindarray):
        if not cc == blank_nb: labellist += [tuple(np.where(clindarray == cc)[0])]  # ordered ascendingly within tuple but otherwise == labellist



def plot_WhatsComps(I,M,labellist,centers=None,cmap='jet',centmode='front',sharey=True):
    # M is what or components
    #n.b. plots maximally 9 clusters
    mycmap = plt.cm.get_cmap(cmap)
    colors = mycmap(np.linspace(0, 1, I.n_iter))
    xvec = np.arange(M.shape[1])
    figlist = []
    N = len(labellist)
    n_figs = int(np.ceil(N/9))
    for nn in np.arange(n_figs):
        f, axarr = plt.subplots(3, 3, figsize=(13, 10),sharey=sharey)
        f.subplots_adjust(hspace=0.4,left=0.07,bottom=0.08, right=0.93)
        f.text(0.95,0.98,'iter.',ha='left',va='top',fontsize=14)
        for jj,iter in enumerate(np.unique(I.iter_labels)):
            col = colors[iter]
            f.text(0.95,0.95-(0.015*jj),'%i'%iter,color=col,fontweight='bold',fontsize=12,ha='left',va='top')
        thisdata = labellist[nn*9:(nn+1)*9]
        n_thisdata = len(thisdata)
        #overhead = 9-n_thisdata
        for ii, myinds in enumerate(thisdata):
            ax = axarr.flatten()[ii]
            thisind = nn*9+ii
            ax.set_title('clust %i' % (thisind + 1),fontweight='bold')
            for idx in myinds:
                col = colors[I.iter_labels[idx]]
                ax.plot(xvec, M[idx], color=col)

            if type(centers) == np.ndarray:
                if centmode == 'front':
                    ax.plot(xvec, centers[thisind], 'w', lw=4.5,zorder=22)
                    ax.plot(xvec, centers[thisind], 'k', lw=2,zorder=23)
                elif centmode == 'back':

                    ax.plot(xvec, centers[thisind], 'k', lw=10, alpha=0.4, zorder=-10)

            if np.mod(ii,3)==0: ax.set_ylabel('loading [A.U.]')
            #if ii>=6:ax.set_xlabel('channel')
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-1,2))

            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.set_xlim([xvec.min(), xvec.max()])

            if ii>=(n_thisdata-3):ax.set_xlabel('channel')
            else: ax.set_xticklabels([])



        for oo in np.arange(n_thisdata,9):
            axarr.flatten()[oo].set_axis_off()

        figlist += [f]
    return figlist



def project_clusters(I,inds,cl_labels,cmapstr='tab10',marker='s',ms=20,alpha=0.7,show_iters=True,axlabs=True,suptitle=True,**kwargs):
    y_off = kwargs['y_off'] if 'y_off' in kwargs else 0.99
    x_off = kwargs['x_off'] if 'x_off' in kwargs else 1
    ecol = 'k'
    ew = kwargs['ew'] if 'ew' in kwargs else 0.5

    if not 'raw_inds' in kwargs:
        llist = list(map(I.labellist.__getitem__, inds))
    else: llist = [kwargs['raw_inds']]
    #parts = clust_list_to_inds(CC.kept, n_samples=CC.nsamp)
    titlestr = '%s - '% (I.proj_flav.upper()) +'; '.join(['%s: %s'% (key, str(val)) for key, val in I.proj_params.items()])
    if not 'ax' in kwargs:
        f, ax = plt.subplots()

    else: ax = kwargs['ax']
    if suptitle: ax.get_figure().suptitle(titlestr)
    #my_cmap = plt.cm.get_cmap(cmap).colors
    ica_cols = icaf.get_colors(cmapstr, len(llist), **kwargs)

    #plotfn = lambda ax,x,y: ax.plot(x, y,'s', mec=ecol, mfc=col, mew=ew, zorder=-1, ms=ms,alpha=alpha)

    for cc,myinds in enumerate(llist):
        xx,yy = I.proj[np.array(myinds)].T
        iters = I.iter_labels[np.array(myinds)]
        col = ica_cols[cc]
        ax.plot(xx, yy,marker, mec=ecol, mfc=col, mew=ew, zorder=-1, ms=ms,alpha=alpha)
        if show_iters:
            for x,y,iter in zip(xx,yy,iters):
                ax.text(x, y, iter, color='k', fontsize=12, ha='center', va='center', zorder=1)
        if len(cl_labels)>0: ax.text(x_off, y_off - 0.05 * cc, cl_labels[cc], ha='left', va='top', fontweight='bold', color=col,transform=ax.transAxes,alpha=alpha)

    if axlabs:
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    if not 'ax' in kwargs: return f


def plot_multi_cluster_levels(I,CC,dispdict,show_exiles=True):
    f, ax = plt.subplots()
    f.subplots_adjust(right=0.8)



    # prev_inds = np.arrapy([])
    for ii in np.arange(len(dispdict)):
        sdict = dispdict[ii]
        inds = sdict['inds']
        labs = ['C_%s%i' % (sdict['tag'], ii) for ii in np.arange(len(inds)) + 1]
        x_off = 1 + ii * 0.1
        if ii == 0:
            iterlabs = True
        else:
            iterlabs = False
        project_clusters(I, inds, labs, cmapstr=sdict['cmap'], marker=sdict['marker'], ms=sdict['ms'], alpha=sdict['alpha'], \
                             show_iters=iterlabs, axlabs=iterlabs, suptitle=iterlabs, ax=ax, x_off=x_off)
    #and the exiles
    if show_exiles and len(CC.exiles)>0:
        project_clusters(I, [], [], marker='x', ms=7, alpha=1, \
                             show_iters=False, axlabs=False, suptitle=False, ax=ax, raw_inds=CC.exiles, ew=2)

    return f


def do_pca(X):
    X_z = make_z(X)
    pca = PCA(svd_solver='full', whiten=False)
    pcs = pca.fit_transform(X_z.T)
    return pcs,pca

def do_pca2(X):
    X_z = X-X.mean(axis=1)[:,None]
    pca = PCA(svd_solver='full', whiten=False)
    pcs = pca.fit_transform(X_z.T)
    return pcs,pca

def get_ncomp_suggestion(pca,var_thresh=98.):
    xvec = np.arange(pca.n_features_) + 1
    cum_perc = np.cumsum(pca.explained_variance_ratio_) * 100.

    return xvec[cum_perc >= var_thresh][0]


def plot_pca(pca,pcs,X,aRec,els,var_thresh=98.,maxcomp=None):
    xvec = np.arange(pca.n_features_) + 1
    cum_perc = np.cumsum(pca.explained_variance_ratio_) * 100.
    comp_suggested = xvec[cum_perc >= var_thresh][0]
    f = plt.figure(constrained_layout=False,figsize=(12,5))

    gs1 = f.add_gridspec(nrows=1, ncols=1, left=0.07, right=0.4,
                            wspace=0.05)
    ax = f.add_subplot(gs1[:,0])
    ax.set_title('thresh: %1.2f %% explained var.' % var_thresh)
    ax.bar(xvec, pca.explained_variance_ratio_)
    ax2 = ax.twinx()
    ax2.plot(xvec, cum_perc, 'ok-')
    ax2.axhline(var_thresh, color='grey', alpha=0.5, zorder=-3)
    ax2.axvline(comp_suggested, color='r', alpha=0.5)
    ax2.plot(comp_suggested, cum_perc[xvec == comp_suggested], 'o', mec='r', mew=2, mfc='none')
    ax2.set_ylim([cum_perc.min(), 100])
    ax.set_xlim([0, comp_suggested + 5])
    ax.set_ylabel('explained variance [0,1]')
    ax2.set_ylabel('cumulative expl. var.', rotation=-90)
    ax.set_xlabel('# component')
    ax2.text(comp_suggested + 1, var_thresh - 0.2 * (var_thresh - cum_perc.min()), '# %i' % comp_suggested, color='r',
             ha='left', va='top')

    if type(maxcomp) != type(None):
        ax2.axvline(maxcomp, color='r', alpha=0.5)
        ax2.text(maxcomp + 1, var_thresh - 0.4 * (var_thresh - cum_perc.min()), 'forced max # %i' % maxcomp, color='r',
                     ha='left', va='top')

    corr_1 = np.array([np.corrcoef(pcs[:, 0], x)[0, 1] for x in X])
    corr_2 = np.array([np.corrcoef(pcs[:, 1], x)[0, 1] for x in X])
    angles = np.linspace(0, 2 * np.pi + 0.001, 500)
    circle_facs = [0.5, 1.]
    el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])

    gs2 = f.add_gridspec(nrows=1, ncols=3, left=0.52, right=0.93, hspace=0.05,width_ratios=[15,1,1])


    #f, axarr = plt.subplots(1, 3, gridspec_kw={'width_ratios': [15, 1, 1]})
    ax = f.add_subplot(gs2[0,0])
    colax = f.add_subplot(gs2[0, 1])
    locax = f.add_subplot(gs2[0, 2])

    sc = ax.scatter(corr_1, corr_2, c=np.arange(pcs.shape[1]), cmap='jet', vmin=0, vmax=pcs.shape[1] - 1)  # ,c=colors
    for circle_fac in circle_facs:
        ax.plot(np.sin(angles) * circle_fac, np.cos(angles) * circle_fac, color='grey', alpha=0.5)
    ax.axhline(0, color='grey', alpha=0.25)
    ax.axvline(0, color='grey', alpha=0.25)
    ax.set_xlabel('corrcoef comp1')
    ax.set_ylabel('corrcoef comp2')
    ax.set_aspect('equal')

    f.colorbar(sc, cax=colax, orientation='vertical')
    # mpl.colorbar.ColorbarBase(axarr[1], cmap=cmap, orientation = 'vertical')
    colax.set_axis_off()
    pb.make_locax(locax, aRec.el_locs[el_inds], linecol='k', tha='left', textoff=1.1, cols=['k', 'lightgray'],
                  boundary_axes=[colax])
    return f



def plot_kurtosis(I,srcmat,cmapstr='tab10',nbins=200,labels=True,resid_col='grey',orig_col='k',**kwargs):

    ncomps = srcmat.shape[0]
    ica_cols = icaf.get_colors(cmapstr,ncomps,**kwargs)

    f,kax = plt.subplots(figsize=(5,4))
    f.subplots_adjust(left=0.1,bottom=0.15)
    colors = ica_cols[:, :3]
    kurt = kurtosis(srcmat,axis=1)
    xvec = np.arange(srcmat.shape[0])+1
    kax.scatter(xvec, kurt, c=colors)
    kax.set_xticks(xvec)
    kax.set_xticklabels(list(xvec.astype(str)))
    kax.axhline(0.,color='grey',alpha=0.3)
    kax.set_xlabel('source')
    kax.set_ylabel('kurtosis')
    for xtick, color in zip(kax.get_xticklabels(), colors):
        xtick.set_color(color)
        xtick.set_fontweight('bold')
    return f

def plot_kurt_simple(kurt,ica_cols):
    f, kax = plt.subplots(figsize=(5, 4))
    f.subplots_adjust(left=0.1, bottom=0.15)
    #colors = np.vstack([ica_cols[:, :3], np.array([mpl.colors.to_rgb(col) for col in [resid_col,orig_col]])])
    colors = ica_cols[:, :3]
    #kurt_resid = kurtosis(resid.flatten())
    #kurt_orig = kurtosis(X_z.flatten())
    #xvec = np.arange(srcmat.shape[0]+2)+1
    xvec = np.arange(len(kurt)) + 1
    #kax.scatter(xvec,np.r_[kurt,kurt_resid,kurt_orig],c=colors)
    kax.scatter(xvec, kurt, c=colors)
    kax.set_xticks(xvec)
    #kax.set_xticklabels(list(xvec[:-2].astype(str))+['resid','orig'])
    kax.set_xticklabels(list(xvec.astype(str)))
    kax.axhline(0., color='grey', alpha=0.3)
    kax.set_xlabel('source')
    kax.set_ylabel('kurtosis')
    for xtick, color in zip(kax.get_xticklabels(), colors):
        xtick.set_color(color)
        xtick.set_fontweight('bold')
    return f


def plot_kurtosis_channels(I,aRec,els,resid_col='grey',orig_col='k'):
    X_z = make_z(I.X)
    resid = X_z - make_z(I.Xrecon)
    el_inds = np.array([np.where(aRec.eois == el)[0][0] for el in els])

    chanvec = np.arange(1, resid.shape[0] + 1)
    f, axarr = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.1, 2.]})
    locax, ax = axarr
    for col,vals in zip([resid_col,orig_col],[resid,X_z]):
        ax.plot(kurtosis(vals, axis=1), chanvec, col,lw=3)
    for tt,[tag,col] in enumerate(zip(['resid','orig'],[resid_col,orig_col])):
        ax.text(1.01,0.99-(0.04*tt),tag,color=col,fontweight='bold',va='top',ha='left',transform=ax.transAxes)

    pb.make_locax(locax, aRec.el_locs[el_inds], cols=['gray','lightgrey'], boundary_axes=[ax],
                  lim=[chanvec[0], chanvec[-1]])
    ax.axvline(0, color='grey', linewidth=2,alpha=0.3)
    ax.set_ylim([chanvec[0], chanvec[-1]])
    ax.set_xlabel('kurtosis')
    return f
