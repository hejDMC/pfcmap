import sys
import numpy as np
import yaml
import os
import matplotlib as mpl
from matplotlib import pyplot as plt



pathpath = 'PATHS/filepaths_carlen.yml'
fformat = 'svg'
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(os.path.join(pathdict['code']['workspace'],'pfcmap','python','enrichments','explain_shuffling'))

import shuffling_helpers as sh
import task_setups as setup

genfigdir = pathdict['figdir_root'] + '/shuffling_explained'


#################
# edit the follow three lines if you want to change the setup
genfigtag = 'imbalanced_cats23'
task1_setup = {key:val for key,val in setup.Adom_c23Imbalance.items()}
task2_setup = {key:val for key,val in setup.Bdom_c23Imbalance.items()}
task_tags = ['task1','task2']
N_probes = 100
Nus = 12
nreps = 1000#number of shuffles

cmap_counts = 'Greys'
Nprobes_show = 10#in the examples


def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(genfigdir,genfigtag, nametag + '__%s.%s'%(genfigtag,fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



#making the tasks originally
task_dict = {ttag:{} for ttag in task_tags}
setup_dict = {}
for ttag,tsetup in zip(task_tags,[task1_setup,task2_setup]):
    cond_dict,reg_prob_dict = sh.complete_input(tsetup['cond_dict'],tsetup['reg_prob_dict'],tsetup['cat_labels'])
    setup_dict[ttag] = {'conditionals':cond_dict,'reg_probs':reg_prob_dict}
    task_dict[ttag] = {ii:sh.make_a_probe(cond_dict,reg_prob_dict,Nus=Nus) for ii in np.arange(N_probes)}

uregs = np.array(list(reg_prob_dict.keys()))
ucats = np.unique(list([key[0] for key in cond_dict.keys()]))




# plot setup for each trial
Nus_total = N_probes*Nus
count_mat_dict = {}
for ttag in task_tags:

    cond_dict = setup_dict[ttag]['conditionals']
    rprob_dict= setup_dict[ttag]['reg_probs']

    cond_mat = np.vstack([np.array([cond_dict[(cat,reg)] for cat in ucats]) for reg in uregs])
    regprob_array = np.array([rprob_dict[reg] for reg in uregs])
    coincidence_prob_mat =cond_mat*regprob_array[:,None]

    expected_count_mat = coincidence_prob_mat*Nus_total
    count_mat_dict['%s_expected'%ttag] = expected_count_mat#will be plotted later

    f,ax = sh.plot_coincidence_mat(cond_mat,uregs,ucats,cmap=cmap_counts,write_text=True,txt_style='%1.2f',fs=8)
    ax.axhline(0.5,color='m',lw=3)
    figsaver(f,'setup/%s_condMat__SETUP'%(ttag))

    f,ax = sh.plot_coincidence_mat(coincidence_prob_mat,uregs,ucats,cmap=cmap_counts,write_text=True,txt_style='%1.2f',fs=8)
    figsaver(f,'setup/%s_coincMat__SETUP'%(ttag))

    f,ax = sh.plot_coincidence_mat(regprob_array[:,None],uregs,[ucats[0]],cmap=cmap_counts,write_text=True,txt_style='%1.2f',fs=8)
    ax.set_xticks([])
    figsaver(f,'setup/%s_regprobs__SETUP'%(ttag))

count_mat_dict['overall_expected'] = np.array([count_mat_dict['%s_expected'%(ttag)] for ttag in task_tags]).sum(axis=0)#will be plotted later

overall_coincidence_prob = count_mat_dict['overall_expected']/(Nus_total*len(task_tags))
f,ax = sh.plot_coincidence_mat(overall_coincidence_prob,uregs,ucats,cmap=cmap_counts,write_text=True,txt_style='%1.2f',fs=8)
figsaver(f,'setup/ovarall_coincMat__SETUP')




#explaining shuffling: dotheaps and empty probes

# plot empty probes for both tasks
for ttag in task_tags:
    f,ax = sh.plot_multi_probe_example(task_dict,Nprobes_show,mytasks=[ttag], width=1, rad=0.4, lw=0.5, ec='w',noactivity=True)
    ax.set_title('%s noactivity'%ttag)
    figsaver(f, 'probeexamples/%s_noActivity'%ttag)

# piles of dots
for ttag in task_tags:
    cats = np.array([task_dict[ttag][jj][:,1] for jj in task_dict[ttag].keys()]).flatten()

    f,ax = sh.plot_dotstack(cats,rad=2,s=50,lw=0.6,ec='w')
    figsaver(f, 'dotpool/%s_dotstack'%ttag)

    f,ax = sh.plot_dotpile(cats,s=30,lw=0.5,ec='w')
    figsaver(f, 'dotpool/%s_dotpile'%ttag)

    f,ax = sh.plot_dotcrate(cats,height_to_width=0.5,s=60,lw=0.5,ec='w')
    figsaver(f, 'dotpool/%s_dotcrate'%ttag)

#t one large heap(twice figsize) for both combined (this is for the idiotic task-trancending shuffling)
cats = np.array([task_dict[ttag][jj][:,1] for ttag in task_tags for jj in task_dict[ttag].keys()]).flatten()
f, ax = sh.plot_dotstack(cats, rad=2, s=50, lw=0.6, ec='w')
figsaver(f, 'dotpool/taskcombined_dotstack')
f, ax = sh.plot_dotpile(cats, s=30, lw=0.5, ec='w')
figsaver(f, 'dotpool/taskcombined_dotpile')
f, ax = sh.plot_dotcrate(cats, height_to_width=0.5, s=50, lw=0.5, ec='w')
figsaver(f, 'dotpool/taskcombined_dotcrate')

# shuffle taskwise
task_dict_tshuff = sh.shuffle_per_tasks(task_dict)

# shuffle overall
task_dict_ashuff = sh.shuffle_across_tasks(task_dict)


# collect all countmats and then plot with one gradient together at the end
# on the way, plot the example probes for each setup

for my_task_dict,shufftag in zip([task_dict_tshuff,task_dict_ashuff,task_dict],['taskShuff','allShuff','orig']):
    for ttag in task_tags:
        f, ax = sh.plot_multi_probe_example(my_task_dict, Nprobes_show, mytasks=[ttag], width=1, rad=0.4, lw=0.5,
                                            ec='w')
        ax.set_title('%s %s' %(ttag,shufftag))
        figsaver(f, '%s_probes/probeexamples/%s_%s_probexamples' %(shufftag,ttag,shufftag))

        superstack = np.array([my_task_dict[ttag][jj] for jj in my_task_dict[ttag].keys()]).reshape(N_probes*Nus,2)
        count_mat = sh.get_countmat(superstack,uregs,ucats)
        count_mat_dict['%s_%s'%(ttag,shufftag)] = count_mat
    overall_countmat = np.array([count_mat_dict['%s_%s'%(ttag,shufftag)] for ttag in task_tags]).sum(axis=0)
    count_mat_dict['overall_%s'%shufftag] = overall_countmat


allcountvals = np.array(list(count_mat_dict.values()))
vminmax = [allcountvals.min(),allcountvals.max()]
for key,count_mat in count_mat_dict.items():

    f,ax = sh.plot_coincidence_mat(count_mat,uregs,ucats,cmap=cmap_counts,write_text=True,txt_style='%i',vminmax=vminmax,fs=8)
    f.text(0.01,0.98,key,ha='left',va='top',fontsize=8)
    figsaver(f, 'count_mats/%s__countmat' %(key))

    fracmat = count_mat / count_mat.sum(axis=1)[:, None]
    f,ax = sh.plot_coincidence_mat(fracmat,uregs,ucats,cmap='Purples',write_text=True,txt_style='%1.1f',fs=8)
    f.text(0.01,0.98,key,ha='left',va='top',fontsize=8)
    ax.axhline(0.5,color='m',lw=3)

    figsaver(f, 'frac_mats/%s__fracmat' %(key))


# get the actuall shuffled count matrices!

shuff_rep_dict = {}
for shufflefn,shuffletag in zip([sh.shuffle_per_tasks,sh.shuffle_across_tasks],['taskShuff','allShuff']):
    print('shuffling %i times %s'%(nreps,shuffletag))
    shuff_mat = np.zeros((len(uregs),len(ucats),nreps),dtype=int)
    for ii in np.arange(nreps):
        shuff_dict = shufflefn(task_dict)
        count_mat = np.zeros((len(uregs),len(ucats)))
        for ttag in task_tags:
            superstack = np.array([shuff_dict[ttag][jj] for jj in shuff_dict[ttag].keys()]).reshape(N_probes*Nus,2)
            count_mat += sh.get_countmat(superstack,uregs,ucats)
        shuff_mat[:,:,ii] = count_mat
    shuff_rep_dict[shuffletag] = shuff_mat


meanstd_dict = {}
for shuffletag in ['taskShuff','allShuff']:
    meanstd_dict[shuffletag] = {'mean':shuff_rep_dict[shuffletag].mean(axis=2),'std':shuff_rep_dict[shuffletag].std(axis=2)}

allstdvals = np.array([meanstd_dict[shuffletag]['std'] for shuffletag in ['taskShuff','allShuff']])
stdvminmax = [allstdvals.min(),allstdvals.max()]
cmap_std = 'Purples'

#plot mean and std of the shuffled
for shuffletag in ['taskShuff','allShuff']:
    f,ax = sh.plot_coincidence_mat(meanstd_dict[shuffletag]['mean'],uregs,ucats,cmap=cmap_counts,write_text=True,txt_style='%i',vminmax=vminmax,fs=8)
    f.text(0.01,0.98,shuffletag,ha='left',va='top',fontsize=8)
    figsaver(f, '%s_probes/mats/meanmat_%s' % (shuffletag, shuffletag))

    f,ax = sh.plot_coincidence_mat(meanstd_dict[shuffletag]['std'],uregs,ucats,cmap=cmap_std,write_text=True,txt_style='%i',vminmax=stdvminmax,fs=8)
    f.text(0.01,0.98,shuffletag,ha='left',va='top',fontsize=8)
    figsaver(f, '%s_probes/mats/stdmat_%s' % (shuffletag, shuffletag))

#plot enrichment matrices
count_mat_orig = count_mat_dict['overall_orig']
enr_dict = {}
for shuffletag in ['taskShuff','allShuff']:
    enr_dict[shuffletag] = (count_mat_orig - meanstd_dict[shuffletag]['mean'])/meanstd_dict[shuffletag]['std']


cmap_enr = 'RdBu_r'
allenrvals = np.array([enr_dict[shuffletag] for shuffletag in ['taskShuff','allShuff']])
eminmax = [allenrvals.min(),allenrvals.max()]
emax = np.abs(eminmax).max()
evminmax = [-emax,emax]

for shuffletag in ['taskShuff','allShuff']:
    f,ax = sh.plot_coincidence_mat(enr_dict[shuffletag],uregs,ucats,cmap=cmap_enr,write_text=True,txt_style='%1.1f',fs=8,vminmax=evminmax)
    f.text(0.01,0.98,shuffletag,ha='left',va='top',fontsize=8)
    figsaver(f, '%s_probes/mats/enrichment_%s_genvminmax' % (shuffletag, shuffletag))

    eminmax2 = [enr_dict[shuffletag].min(),enr_dict[shuffletag].max()]
    emax2 = np.abs(eminmax2).max()
    evminmax2 = [-emax2,emax2]
    f,ax = sh.plot_coincidence_mat(enr_dict[shuffletag],uregs,ucats,cmap=cmap_enr,write_text=True,txt_style='%1.1f',fs=8,vminmax=evminmax2)
    f.text(0.01,0.98,shuffletag,ha='left',va='top',fontsize=8)
    figsaver(f, '%s_probes/mats/enrichment_%s' % (shuffletag, shuffletag))


###
#plot the underlying distributions with significance levels
percs = np.array([0.05,0.01,0.001])*100
cmap_levels = mpl.cm.get_cmap('RdBu_r')
cnorm = mpl.colors.Normalize(vmin=-4, vmax=4)
signif_colors = np.r_[np.array([cmap_levels(cnorm(-ii-1)) for ii in np.arange(len(percs))])[::-1],np.array([cmap_levels(cnorm(ii+1)) for ii in np.arange(len(percs))])]
signif_pldict = {'stars':np.array(['*','**','***'][::-1]+ ['*','**','***']),\
                 'scolors':signif_colors,\
                 'perc_levels':np.r_[percs[::-1],100-percs]}

for shuffletag in shuff_rep_dict.keys():
    #shuffletag = 'taskShuff'
    for reg in uregs:
        for cat in ucats:
            f,ax = sh.plot_shuffling_distribution(reg,cat,uregs,ucats,count_mat_orig,shuff_rep_dict[shuffletag],meancol='grey',distro_col=setup.cdict[cat],signif_pldict=signif_pldict)
            figsaver(f, '%s_probes/shuffdistros/%sand%s_shuffdistro_%s' % (shuffletag,reg,cat,shuffletag))







#row-wise cond_mat
#one-column
#expected counts
#expected summed counts

