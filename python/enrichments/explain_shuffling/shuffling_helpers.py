import numpy as np
import task_setups as setup
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import multivariate_normal
from scipy.stats import scoreatpercentile as sap
from scipy.stats import percentileofscore as pofs

def complete_input(cond_dict,reg_prob_dict,cat_labels):

    cats_in_cond_dict = np.unique([key[0] for key in cond_dict.keys()])
    regs_avail = np.unique([key[1] for key in cond_dict.keys()])
    cat_missing = [cat for cat in cat_labels if not cat in cats_in_cond_dict][0]

    cond_dict2 = {key:vals for key,vals in cond_dict.items()}#copying
    for reg in regs_avail:
        already_there = np.sum([cond_dict[key] for key in cond_dict.keys() if reg in key])
        assert already_there<=1.,'cond dict extension probabilities are >1 %1.2f'%(already_there)
        cond_dict2[(cat_missing,reg)] = 1-already_there

    reg_prob_dict2 = {key:val for key,val in reg_prob_dict.items()}#copying

    reg_missing = [reg for reg in regs_avail if not reg in reg_prob_dict.keys()][0]
    already_there = np.sum([reg_prob_dict[reg] for reg in reg_prob_dict.keys()])
    assert already_there <= 1., 'probabilities are >1 %1.2f' % (already_there)
    reg_prob_dict2[reg_missing] = 1-already_there

    return cond_dict2,reg_prob_dict2


def make_a_probe(cond_dict,reg_prob_dict,Nus=12):

    reg_labels = list(reg_prob_dict.keys())
    cat_labels = np.unique([key[0] for key in cond_dict.keys()])

    prob_per_reg = np.array([reg_prob_dict[key] for key in reg_labels])

    probe_mat = np.zeros((Nus, 2), dtype='U10')
    regvec = np.random.choice(reg_labels, size=Nus, replace=True, p=prob_per_reg)
    reg_counts = np.array([(regvec == reg).sum() for reg in reg_labels])
    cum_vec = np.r_[0, np.cumsum(reg_counts)]
    for rr, reg in enumerate(reg_labels):
        probe_mat[cum_vec[rr]:cum_vec[rr + 1], 0] = reg
        cat_probs = np.array([cond_dict[(cat, reg)] for cat in cat_labels])
        probe_mat[cum_vec[rr]:cum_vec[rr + 1], 1] = np.random.choice(cat_labels, size=reg_counts[rr], replace=True,
                                                                     p=cat_probs)
    return probe_mat



def plot_dotstack(catvec,rad=2,s=50,lw=1,ec='w'):

    cols = np.random.permutation(np.array([setup.cdict[cat] for cat in catvec]))

    t = np.random.random(len(catvec))
    u = np.random.random(len(catvec))
    x = rad * np.sqrt(t) * np.cos(2 * np.pi * u)
    y = rad * np.sqrt(t) * np.sin(2 * np.pi * u)

    f,ax = plt.subplots(figsize=(1.5,1.5))
    ax.scatter(x,y,c=cols,s=s,edgecolor=ec,lw=lw)
    ax.set_aspect('equal')
    ax.set_axis_off()
    f.tight_layout()

    return f,ax


def plot_dotpile(catvec,s=30,lw=0.5,ec='w'):
    cols = np.random.permutation(np.array([setup.cdict[cat] for cat in catvec]))
    cov = np.array([[1, 0], [0, 1]])
    distr = multivariate_normal(cov = cov, mean = np.array([0,0]))
    xy = distr.rvs(size = len(catvec))
    f,ax = plt.subplots(figsize=(1.5,1.5))
    ax.scatter(xy[:,0],xy[:,1],c=cols,s=s,edgecolor=ec,lw=lw)
    ax.set_aspect('equal')
    ax.set_axis_off()
    f.tight_layout()
    return f,ax

def plot_dotcrate(catvec,height_to_width=0.5,s=30,lw=0.5,ec='w',show_crate=True):
    cols = np.random.permutation(np.array([setup.cdict[cat] for cat in catvec]))
    x = np.random.uniform(0,1,len(cols))
    y = np.random.uniform(0,1*height_to_width,len(cols))
    f, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.scatter(x, y, c=cols, s=s, edgecolor=ec, lw=lw)
    ax.set_aspect('equal')
    if show_crate:
        ax.spines[['top']].set_visible(False)
        ax.spines[['bottom','left','right']].set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([0,ax.get_ylim()[1]*1.1])
        ax.set_xlim([0,1])
    else:
        ax.set_axis_off()
    f.tight_layout()
    return f,ax


def plot_a_probe(ax,probe,pos_offset = [0, 0],width=1,rad = 0.4,lw = 0.5,ec = 'w',noactivity=False,nocolor='w'):
    Nus = len(probe)
    centers = np.vstack([np.zeros(Nus), np.arange(Nus)]).T + np.array(pos_offset)

    regvals = np.unique(probe[:, 0])
    startinds = np.array([np.where(probe[:, 0] == reg)[0][0] for reg in regvals])
    rangevec = np.r_[startinds, np.array(Nus)]


    for rr, reg in enumerate(regvals):
        range_start = rangevec[rr]
        range_stop = rangevec[rr + 1]
        x_anch = centers[rr][0] - 0.5
        height = range_stop - range_start
        myrect = plt.Rectangle((x_anch, centers[range_start][1] - 0.5), width, range_stop - range_start,
                               color=setup.cdict[reg], zorder=-10, alpha=1, lw=0)
        ax.add_patch(myrect)

    if noactivity:
        for cent, cat in zip(centers, probe[:, 1]):
            circle = plt.Circle(cent, rad, fc=nocolor, lw=lw, ec=ec)
            ax.add_patch(circle)
    else:
        for cent, cat in zip(centers, probe[:, 1]):
            circle = plt.Circle(cent, rad, fc=setup.cdict[cat], lw=lw, ec=ec)
            ax.add_patch(circle)



def plot_multi_probe_example(task_dict,Nprobes_show,mytasks=['task1','task2'], width=1, rad=0.4, lw=0.5, ec='w',noactivity=False,nocolor='w'):
    Nus = len(task_dict[mytasks[0]][0])
    f, ax = plt.subplots(figsize=(4, 5))
    for tt, ttag in enumerate(mytasks):
        for pp in np.arange(Nprobes_show):
            my_offset = [pp * 1.5, tt * (Nus + 2)]
            plot_a_probe(ax, task_dict[ttag][pp], pos_offset=my_offset, width=width, rad=rad, lw=lw, ec=ec,noactivity=noactivity,nocolor=nocolor)

    ax.set_xlim([-0.6, my_offset[0] + 0.6])
    ax.set_ylim([-0.5, my_offset[1] + Nus - 0.5])
    ax.set_aspect('equal')
    ax.set_axis_off()
    f.tight_layout()
    return f,ax

def get_countmat(superstack,uregs,ucats):
    count_mat = np.zeros((len(uregs),len(ucats)),dtype=int)
    for rr,reg in enumerate(uregs):
        for cc,cat in enumerate(ucats):
            count_mat[rr,cc] = np.sum((superstack[:,0] == reg)&(superstack[:,1]==cat))
    return count_mat


def plot_coincidence_mat(mymat,regs,cats,cmap='Greys',write_text=True,txt_style='%i',fs=12,**kwargs):
    vminmax = kwargs['vminmax'] if 'vminmax' in kwargs else [mymat.min(),mymat.max()]
    f,ax = plt.subplots(figsize=(1.5,1.2))
    f.subplots_adjust(left=0.2,top=0.8)
    ax.imshow(mymat,cmap=cmap,vmin=vminmax[0],vmax=vminmax[1])
    ax.set_yticks(np.arange(len(regs)))
    ax.set_xticks(np.arange(len(cats)))
    ax.xaxis.set_ticks_position('top')
    tlabsy = ax.set_yticklabels(regs,fontweight='bold')
    tlabsx = ax.set_xticklabels(cats,fontweight='bold')
    for cc,cat in enumerate(cats):
        col = setup.cdict[cat]
        tlabsx[cc].set_color(col)
    for cc,reg in enumerate(regs):
        col = setup.cdict[reg]
        tlabsy[cc].set_color(col)
    ax.tick_params(axis='both', which='both', length=0,pad=0)
    if write_text:
        for rr,reg in enumerate(regs):
            for cc,cat in enumerate(cats):
                ax.text(cc,rr,txt_style%(mymat[rr,cc]),color='k',ha='center',va='center',fontsize=fs,bbox={'facecolor':'w','edgecolor':'k','alpha':0.8, 'boxstyle':'round,pad=0.2'})
        ax.set_aspect('equal')
    f.tight_layout()
    return f,ax



def shuffle_per_tasks(task_dict):
    task_dict_tshuff = {}
    for ttag in task_dict.keys():
        superstack = np.array([task_dict[ttag][jj] for jj in task_dict[ttag].keys()])
        allcats = superstack[:,:,1].flatten()
        permuted_cats = np.random.permutation(allcats)
        newprobemat = np.array([superstack[:,:,0],permuted_cats.reshape(superstack.shape[:2])]).transpose(1,2,0)
        newprobedict = {jj:newprobemat[jj] for jj in np.arange(len(task_dict[ttag]))}
        task_dict_tshuff[ttag] = newprobedict
    return task_dict_tshuff

def shuffle_across_tasks(task_dict):
    task_tags  = list(task_dict.keys())
    N_probes = len(task_dict[task_tags[0]].keys())
    superstack = np.array([task_dict[ttag][jj] for ttag in task_tags for jj in task_dict[ttag].keys()])
    allcats = superstack[:, :, 1].flatten()
    permuted_cats = np.random.permutation(allcats)
    newprobemat = np.array([superstack[:, :, 0], permuted_cats.reshape(superstack.shape[:2])]).transpose(1, 2, 0)
    task_dict_ashuff = {ttag: {jj: newprobemat[jj + tt * N_probes] for jj in np.arange(len(task_dict[ttag]))} for
                        tt, ttag in enumerate(task_tags)}
    return task_dict_ashuff



def plot_shuffling_distribution(reg,cat,uregs,ucats,count_mat_orig,shuff_dict,meancol='grey',distro_col='silver',signif_pldict={}):


    rr = np.where(uregs==reg)[0][0]
    cc = np.where(ucats==cat)[0][0]

    orig_val = count_mat_orig[rr,cc]
    distro = shuff_dict[rr,cc]
    myhist,mybins = np.histogram(distro,30)
    plbins = mybins[:-1]+np.diff(mybins)[0]/2
    dstd = distro.std()
    dmean = distro.mean()
    enr = (orig_val-dmean)/dstd

    pofval = pofs(distro,orig_val)

    if len(signif_pldict)>0:
        perc_levels = signif_pldict['perc_levels']
        saps_vec = np.array([sap(distro,perc) for perc in perc_levels])

        n_halfplevels = int(len(perc_levels)/2)


        if pofval<50:
            mylevs0 = np.where(perc_levels[:n_halfplevels]>=pofval)[0]
            pofidx = mylevs0[0] if np.size(mylevs0) !=0 else np.nan
        else:
            mylevs0 = np.where(perc_levels[n_halfplevels:]<=pofval)[0]
            pofidx = mylevs0[-1]+n_halfplevels if (np.size(mylevs0) !=0) else np.nan



    f,ax = plt.subplots(figsize=(3,2))
    #ax.plot(plbins,myhist,color=setup.cdict[cat])
    ax.fill_between(plbins,np.zeros_like(myhist),myhist,color=distro_col,lw=0,alpha=0.5)
    #ax.axvline(orig_val,color='k',lw=6)
    ax.axvline(orig_val,color=setup.cdict[cat],lw=4)
    ax.axvline(dmean,color=meancol,lw=2)
    for val in [dmean+dstd,dmean-dstd]:
        ax.axvline(val,color=meancol,lw=2,linestyle=':')
    if len(signif_pldict)>0:
        signif_colors = signif_pldict['scolors']
        for mysap,mycol in zip(saps_vec,signif_colors):
            ax.axvline(mysap,color=mycol,lw=2,linestyle='--')

    ha_txt,x_txt = ['right',0.8] if pofval>50 else ['left',0.2]
    arrow = mpl.patches.FancyArrowPatch((dmean, myhist.mean()), (orig_val, myhist.mean()),
                                   arrowstyle='->,head_width=.15', mutation_scale=20,color=meancol,lw=2)
    ax.annotate("%1.2f"%enr, (x_txt, 0.6), xycoords=arrow, ha=ha_txt, va='bottom',color=meancol,fontweight='bold')#, bbox=dict(boxstyle="round", fc="w",alpha=0.5,lw=0)
    ax.add_patch(arrow)
    ax.set_ylim([0,ax.get_ylim()[1]])

    if ~np.isnan(pofidx) and len(signif_pldict)>0:
        arrow = mpl.patches.FancyArrowPatch((saps_vec[pofidx], myhist.mean()*1.5), (orig_val, myhist.mean()*1.5),
                                       arrowstyle='->,head_width=.15', mutation_scale=20,color=signif_colors[pofidx],lw=2)
        ax.annotate("%s"%signif_pldict['stars'][pofidx], (.5, 0.6), xycoords=arrow, ha='center', va='bottom',color=signif_colors[pofidx])
        ax.add_patch(arrow)
    ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_xlabel('# coincidences (%s,%s)'%(reg,cat))
    ax.set_ylabel('count')
    f.tight_layout()
    return f,ax
