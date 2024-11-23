import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import scoreatpercentile as sap

pathpath,myrun = sys.argv[1:]

plot_example_recs = False


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python.utils import som_helpers as somh
from pfcmap.python import settings as S
from pfcmap.python.utils import som_plotting as somp

rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)


kshape = somdict['kshape']
figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__kShape%i_%i'%(myrun,kshape[0],kshape[1]))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])


Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc = (not myrun.count('_brain')),rois_from_path=False)
assert (somfeats == somdict['features']).all(),'mismatching features'
assert (weightvec == somdict['featureweights']).all(),'mismatching weights'


get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])


#project on map
refmean,refstd = somdict['refmeanstd']
featureweights = somdict['featureweights']
dmat = np.vstack([get_features(elobj) for elobj in Units])
wmat = uloader.reference_data(dmat,refmean,refstd,featureweights)

#set BMU for each
weights = somdict['weights']
allbmus = somh.get_bmus(wmat,weights)
for bmu,U in zip(allbmus,Units):
    U.set_feature('bmu',bmu)


uloader.assign_parentregs(Units,pfeatname='preg',cond=lambda uobj:True,na_str='na')


parent_regs = np.array([reg for reg in S.parents_ordered if reg in np.unique([U.preg for U in Units if not U.preg=='na'])])


#plot quantization error per region
utype = 'all'
utypechecker = lambda element: True if utype=='all' else element.utype==utype
qe_dict = {}
for reg in parent_regs:
    reginds = np.array([ee for ee,el in enumerate(Units) if el.preg == reg and utypechecker(el)])
    bmus = somh.get_bmus(wmat[reginds],weights)
    qe_dict[reg] = np.array([np.linalg.norm(roi-weights[:,bmu]) for roi,bmu in zip(wmat[reginds],bmus)])


sortinds_pregs = np.argsort([np.median(qe_dict[reg]) for reg in parent_regs])
f,ax = plt.subplots(figsize=(4,3.5))
f.subplots_adjust(left=0.2,bottom=0.2,right=0.95)
ax.set_xticks(np.arange(len(parent_regs)))
ax.set_xticklabels(parent_regs[sortinds_pregs],rotation=90)
for rr,reg in enumerate(parent_regs[sortinds_pregs]):
    col = S.parent_colors[reg]
    vals = qe_dict[reg]
    ax.plot(rr,np.median(vals),'o',mfc=col,mec='none')
    ax.plot([rr,rr],[sap(vals,25),sap(vals,75)],color=col)
    ax.get_xticklabels()[rr].set_color(col)
ax.set_ylabel('QError')
figsaver(f, 'QError_parentregs')




cvec = [mpl.colors.to_rgb(S.parent_colors[reg]) for reg in parent_regs]


n_regs = len(parent_regs)
n_nodes = np.prod(kshape)

sizefac = 0.7
hw_hex = 0.35
radius = hw_hex * sizefac * 0.75

hex_dict = somp.get_hexgrid(kshape, hw_hex=hw_hex * sizefac)
fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                    sizefac=sizefac)

show_hits = False
show_hexid = False
utype = 'all'
utypechecker = lambda element: True if utype=='all' else element.utype==utype

region_mat = np.zeros((n_regs, n_nodes))

for rr,reg in enumerate(parent_regs):
    #reg = region_keys[rr]
    reginds = np.array([ee for ee,el in enumerate(Units) if el.preg.count(reg) and utypechecker(el)])
    regdata = wmat[reginds]
    bmus = somh.get_bmus(regdata,weights)
    for bmu in bmus:
        region_mat[rr,bmu] += 1

frac_mat = region_mat/region_mat.sum(axis=0)[None,:]*100

elinds =  np.array([ee for ee,el in enumerate(Units) if utypechecker(el)])
mybmus = somh.get_bmus(wmat[elinds],weights)
allcounts = np.zeros(n_nodes)
for bmu in mybmus:
    allcounts[bmu] += 1

rankvec = allcounts.argsort().argsort()

plotvals = allcounts[:] #rankvec#np.ma.masked_invalid(np.sqrt(allcounts))


f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
f.subplots_adjust(right=0.89)


if show_hits:
    somp.plot_hexPanel(kshape,plotvals,ax,hw_hex=hw_hex*sizefac,ec='k',showHexId=show_hexid\
                                     ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',hexcol=S.cmap_count)#rankvec

    norm = mpl.colors.Normalize(vmin=allcounts.min(), vmax=allcounts.max())
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    cax.set_title('# hits')
else:
    for hexid, hexparams in list(hex_dict.items()):
        center, verts = hexparams
        ax.add_patch(plt.Polygon(verts, fc='w', ec='k', alpha=1))
    cax.set_axis_off()

xxlim,yylim = ax.get_xlim(),ax.get_ylim()
for hh,hexvals in enumerate(hex_dict.values()):
    c = hexvals[0]
    #print(np.sum(frac_mat[:,hh]))
    if not np.isnan(frac_mat[0,hh]):
        ax.pie(frac_mat[:,hh],radius=radius,center=c,colors=cvec)
ax.set_xlim(xxlim)
ax.set_ylim(yylim)
ax.set_title('%s'%(utype.upper()))
toppos = cax.get_position().y1
for rr,reg in enumerate(parent_regs):
    f.text(0.99,toppos-rr*0.04,reg,color=S.parent_colors[reg],ha='right',va='top',fontweight='bold',fontsize=18)
figsaver(f, 'pies/regionpie_parentregs_%s'%(utype))

#region cartoon


#for each hex find what region is contributing most
frac_mat1 = region_mat/region_mat.sum(axis=1)[:,None]
frac_mat2 = region_mat/region_mat.sum(axis=0)[None,:]
dcolors = np.array([S.parent_colors[reg] for reg in parent_regs])# np.array([cdict_clust[lab] for lab in labels])

for fracmat,fractag in zip([frac_mat1,frac_mat2],['relwinners','winners']):
    wincats_idx = np.argmax(fracmat,axis=0).astype(float)
    wincats_idx[np.isnan(frac_mat.sum(axis=0))] = np.nan
    f, ax = plt.subplots(figsize=(fwidth, fheight))
    f.subplots_adjust(left=l_w / fwidth, right=1. - r_w / fwidth-0.1, bottom=b_h / fheight,
                      top=1. - t_h / fheight-0.05, wspace=0.05)  # -tspace/float(fheight)
    somp.plot_hexPanel(kshape,wincats_idx,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                     ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',quality_map=dcolors)#rankvec

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_axis_off()
    toppos = ax.get_position().y1
    for rr,reg in enumerate(parent_regs):
        f.text(0.99,toppos-rr*0.05,reg,color=S.parent_colors[reg],ha='right',va='top',fontweight='bold',fontsize=18)
    ax.set_title(fractag)
    figsaver(f, 'pies/regionCartoon_parentregs_%s'%(fractag))


for rr,reg in enumerate(parent_regs):
    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])

    somp.plot_hexPanel(kshape, region_mat[rr], ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, hexcol=S.cmap_count, alphahex=1., idcolor='k')

    norm = mpl.colors.Normalize(vmin=region_mat[rr].min(), vmax=region_mat[rr].max())
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    ax.set_title('hits %s'%reg)
    cax.tick_params(axis='y', which='major', labelsize=15)
    f.tight_layout()
    figsaver(f, 'hitmaps/hits_parentreg_%s'%(reg))


filter_dict = {'PFC_wo_AI_ORB_FRP':lambda U: U.preg=='PFC' and not U.area.count('ORB') and not U.area.count('AI') and not U.area.count('FRP'),\
               'onlyORB':lambda U: U.area.count('ORB'),\
               'onlyAI':lambda U: U.area in ['AIv','AId','AI'] }


for regspecname,filterfn in filter_dict.items():

    regvec = np.zeros(n_nodes)
    reginds = np.array([ee for ee,U in enumerate(Units) if filterfn(U)])
    regdata = wmat[reginds]
    bmus = somh.get_bmus(regdata,weights)
    for bmu in bmus:
        regvec[bmu] += 1

    f, ax, cax = somp.make_fig_with_cbar(fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax, add_width=1.,
                                         width_ratios=[1., 0.035])

    somp.plot_hexPanel(kshape, regvec, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                       , scalefree=True, return_scale=False, hexcol=S.cmap_count, alphahex=1., idcolor='k')

    norm = mpl.colors.Normalize(vmin=regvec.min(), vmax=regvec.max())
    cb = somp.plot_cmap(cax, S.cmap_count, norm)
    ax.set_title('hits %s'%regspecname)
    cax.tick_params(axis='y', which='major', labelsize=15)
    f.tight_layout()
    figsaver(f, 'hitmaps/hits_parentreg_SPEC_%s'%(regspecname))



