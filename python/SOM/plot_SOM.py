import sys
import yaml
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import scoreatpercentile as sap

pathpath,myrun = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
'''
#myrun = runIBLPasdMP3_brain_pj

#pathpath = 'PATHS/filepaths_IBL.yml'
#myrun = 'runIBLPasdMP3_brain_pj'

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




####################
#plot features
for feat in somfeats:
    featvals = np.array([getattr(U,feat) for U in Units])
    f,ax = plt.subplots(figsize=(3,2.5))
    f.subplots_adjust(left=0.18,bottom=0.22)
    ax.set_title('N=%i'%(len(Units)))
    ax.hist(featvals,100,histtype='step',color='k', weights=np.ones_like(featvals)/len(featvals))
    if feat in S.featfndict:
        featname = S.featfndict[feat]['repl']
    else: featname = str(feat)
    ax.set_xlabel(featname)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel('prob.')
    figsaver(f, 'features/%s'%(feat))

#save settings
with open(os.path.join(figdir_mother,'params__%s.yml'%myrun), 'w') as outfile:
    yaml.dump(rundict, outfile, default_flow_style=False, sort_keys=False)

# plot compontent planes


sizefac =0.4
hw_hex = 0.35
fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax = somp.get_figureparams(kshape,hw_hex=hw_hex,sizefac=sizefac)

add_width = 1

diverge_dict = {'B':0,'LvR':1,'Lv':1,'Rho':0,'M':0}
diverge_dict2 = {key+'_mean':val for key,val in diverge_dict.items()}
diverge_dict.update(diverge_dict2)

for somfeat in somfeats:
    if somfeat.count('PC'):
        diverge_dict.update({somfeat:0})

def get_cmap_and_vminmax(feattag,values):
    if feattag in diverge_dict.keys():
        cmap = ['RdBu_r']
        dcenter = diverge_dict[feat]
        vminmax = np.max(np.abs([values.min()-dcenter,values.max()-dcenter]))*np.array([-1,1])+dcenter
    else:
        cmap = ['magma','binary']
        vminmax = [values.min(),values.max()]
    return cmap,vminmax

for ff,feat in enumerate(somfeats):
    #ff = 1
    #feat = somfeats[ff]


    featvals = (weights[ff]/weightvec[ff]*refstd[ff])+refmean[ff]
    cmaps,vminmax = get_cmap_and_vminmax(feat,featvals)
    for cmap in cmaps:

        f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
        if feat in S.featfndict:
            featname = S.featfndict[feat]['repl']
        else:
            featname = str(feat)

        ax.set_title(featname)

        #print(vminmax)
        somp.plot_hexPanel(kshape,featvals,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                         ,scalefree=True,return_scale=False,hexcol=cmap,alphahex=1.,idcolor='k',vminmax=vminmax)

        norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
        norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
        cb = somp.plot_cmap(cax, cmap, norm)
        f.tight_layout()
        figsaver(f, 'components/%s_CMAP%s'%(feat,cmap))


Nfeats = len(somfeats)
k_ratio = np.divide(*somdict['kshape'])

for ratecmap in ['magma','binary']:
    f,axarr = plt.subplots(1,Nfeats,figsize=(Nfeats*k_ratio*2.3,1.5/k_ratio))#,gridspec_kw={'width_ratios':[1,0.1]*Nfeats}
    #f.text(0.005,0.99,'%s'%(myrun),ha='left',va='top')
    f.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.85)
    for ff,feat in enumerate(somfeats):
        #feat = somfeats[2]
        featvals = (weights[ff]/weightvec[ff]*refstd[ff])+refmean[ff]
        cmaps,vminmax = get_cmap_and_vminmax(feat,featvals)
        cmap = str(ratecmap) if feat.count('rate') else cmaps[0]

        ax = axarr[ff]
        featlab = S.featfndict[feat]['repl'] if feat in S.featfndict else feat
        ax.set_title(featlab)
        #norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
        #print(ncl,np.unique(labels),dcolors.shape)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        somp.plot_hexPanel(kshape,featvals,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                             ,scalefree=True,return_scale=False,hexcol=cmap,alphahex=1.,idcolor='k',vminmax=vminmax)
        ax.set_axis_off()
    figsaver(f,'components/all_mini__CMAP%s'%ratecmap)

#HITMAP
n_nodes = np.prod(kshape)
allcounts = np.zeros(n_nodes)
for bmu in allbmus:
    allcounts[bmu] += 1
f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])

somp.plot_hexPanel(kshape,allcounts,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                 ,scalefree=True,return_scale=False,hexcol=S.cmap_count,alphahex=1.,idcolor='k')

norm = mpl.colors.Normalize(vmin=allcounts.min(), vmax=allcounts.max())
cb = somp.plot_cmap(cax, S.cmap_count, norm)
ax.set_title('hits')
cax.tick_params(axis='y', which='major', labelsize=15)
f.tight_layout()
figsaver(f, 'hits')


unique_regs = np.unique([U.area for U in Units])

#plot quantization error per region
utype = 'all'
utypechecker = lambda element: True if utype=='all' else element.utype==utype
myregs = np.array([reg for reg in S.PFC_sorted if reg in unique_regs])
qe_dict = {}
for reg in myregs:
    reginds = np.array([ee for ee,el in enumerate(Units) if el.region.count(reg) and utypechecker(el)])
    bmus = somh.get_bmus(wmat[reginds],weights)
    qe_dict[reg] = np.array([np.linalg.norm(roi-weights[:,bmu]) for roi,bmu in zip(wmat[reginds],bmus)])

f,ax = plt.subplots(figsize=(4,3.5))
f.subplots_adjust(left=0.2,bottom=0.2,right=0.95)
ax.set_xticks(np.arange(len(myregs)))
ax.set_xticklabels(myregs,rotation=90)
for rr,reg in enumerate(myregs):
    col = S.cdict_pfc[reg]
    vals = qe_dict[reg]
    ax.plot(rr,np.median(vals),'o',mfc=col,mec='none')
    ax.plot([rr,rr],[sap(vals,25),sap(vals,75)],color=col)
    ax.get_xticklabels()[rr].set_color(col)
ax.set_ylabel('QError')
figsaver(f, 'QError')


dont_show_regs = []#['AId','AIv','FRP']
region_keys = [key for key in S.cdict_pfc.keys() if key in unique_regs and not key in dont_show_regs]
n_regs = len(region_keys)

sizefac = 0.7
hw_hex = 0.35
radius = hw_hex * sizefac * 0.75

hex_dict = somp.get_hexgrid(kshape, hw_hex=hw_hex * sizefac)
fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                    sizefac=sizefac)

show_hits = False
show_hexid = False
#for utype in np.append(utypes,'all'):
utype = 'all'
region_mat = np.zeros((n_regs, n_nodes))
utypechecker = lambda element: True if utype=='all' else element.utype==utype

for rr,reg in enumerate(region_keys):
    #reg = region_keys[rr]
    reginds = np.array([ee for ee,el in enumerate(Units) if el.region.count(reg) and utypechecker(el)])
    regdata = wmat[reginds]
    bmus = somh.get_bmus(regdata,weights)
    for bmu in bmus:
        region_mat[rr,bmu] += 1

frac_mat = region_mat/region_mat.sum(axis=0)[None,:]*100
cvec = np.array([S.cdict_pfc[reg] for reg in region_keys])

elinds =  np.array([ee for ee,el in enumerate(Units) if utypechecker(el)])
mybmus = somh.get_bmus(wmat[elinds],weights)
allcounts = np.zeros(n_nodes)
for bmu in mybmus:
    allcounts[bmu] += 1

rankvec = allcounts.argsort().argsort()

plotvals = allcounts[:] #rankvec#np.ma.masked_invalid(np.sqrt(allcounts))


f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
f.subplots_adjust(right=0.85)
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
for rr,reg in enumerate(region_keys):
    f.text(0.99,toppos-rr*0.02,reg,color=S.cdict_pfc[reg],ha='right',va='top',fontweight='bold',fontsize=18)
figsaver(f, 'pies/regionpie_%s'%(utype))




#region cartoon

#for each hex find what region is contributing most
frac_mat1 = region_mat/region_mat.sum(axis=1)[:,None]
frac_mat2 = region_mat/region_mat.sum(axis=0)[None,:]
dcolors = np.array([S.cdict_pfc[reg] for reg in region_keys])  # np.array([cdict_clust[lab] for lab in labels])

for fracmat,fractag in zip([frac_mat1,frac_mat2],['relwinners','winners']):
    wincats_idx = np.argmax(fracmat,axis=0)

    f, ax = plt.subplots(figsize=(fwidth, fheight))
    f.subplots_adjust(left=l_w / fwidth, right=1. - r_w / fwidth-0.1, bottom=b_h / fheight,
                      top=1. - t_h / fheight-0.05, wspace=0.05)  # -tspace/float(fheight)
    somp.plot_hexPanel(kshape,wincats_idx,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                     ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',quality_map=dcolors)#rankvec

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_axis_off()
    toppos = ax.get_position().y1
    for rr,reg in enumerate(region_keys):
        f.text(0.99,toppos-rr*0.05,reg,color=S.cdict_pfc[reg],ha='right',va='top',fontweight='bold',fontsize=18)
    ax.set_title(fractag)
    figsaver(f, 'pies/regionCartoon_%s'%(fractag))




if myrun.count('_brain'):
    utype = 'all'
    utypechecker = lambda element: True if utype=='all' else element.utype==utype
    region_mat = np.zeros((n_regs+1, n_nodes))

    for rr,reg in enumerate(region_keys):
        #reg = region_keys[rr]
        reginds = np.array([ee for ee,el in enumerate(Units) if el.region.count(reg) and utypechecker(el)])
        regdata = wmat[reginds]
        bmus = somh.get_bmus(regdata,weights)
        for bmu in bmus:
            region_mat[rr,bmu] += 1
    #now the non-pfc
    reginds = np.array([ee for ee,el in enumerate(Units) if not S.check_pfc(U.area) and utypechecker(el)])
    #np.unique(np.array([Units[idx].region for idx in reginds_nonPFC]))
    regdata = wmat[reginds]
    bmus = somh.get_bmus(regdata,weights)
    for bmu in bmus:
        region_mat[-1,bmu] += 1

    frac_mat = region_mat/region_mat.sum(axis=0)[None,:]*100
    cvec = np.r_[np.array([S.cdict_pfc[reg] for reg in region_keys]),np.array(['w'])]

    f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
    f.subplots_adjust(right=0.85)
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
    for rr,reg in enumerate(region_keys):
        f.text(0.99,toppos-rr*0.02,reg,color=S.cdict_pfc[reg],ha='right',va='top',fontweight='bold',fontsize=18)
    figsaver(f, 'pies/regionpiePFCnoPFC_%s'%(utype))


###################
#tasks, datasets, layers, unittypes
allcounts = np.zeros(n_nodes)
for bmu in allbmus:
    allcounts[bmu] += 1
#rankvec = allcounts.argsort().argsort()
plotvals = allcounts[:] #rankvec#np.ma.masked_invalid(np.sqrt(allcounts))

sizefac = 0.7
hw_hex = 0.35
radius = hw_hex * sizefac * 0.75

hex_dict = somp.get_hexgrid(kshape, hw_hex=hw_hex * sizefac)
fwidth, fheight, l_w, r_w, b_h, t_h, xmin, xmax, ymin, ymax = somp.get_figureparams(kshape, hw_hex=hw_hex,
                                                                                    sizefac=sizefac)

for U in Units:
    lnum = 'NA' if U.layer == 'NA' else U.layer[0]
    U.set_feature('layernum',lnum)


for attr,cdict in zip(['task','dataset','layernum','utype'],[S.cdict_task,S.cdict_ds,S.cdict_layer,S.cdict_utype]):
    attrtypes = np.unique([getattr(U,attr) for U in Units])
    n_attrs = len(attrtypes)
    if n_attrs>1:
        attr_mat = np.zeros((n_attrs,n_nodes))
        #for reg in cdict_pfc:
        for aa,attrval in enumerate(attrtypes):
            #reg = region_keys[rr]
            attrinds = np.array([ee for ee,el in enumerate(Units) if getattr(el,attr)==attrval])
            attrdata = wmat[attrinds]
            bmus = somh.get_bmus(attrdata,weights)
            for bmu in bmus:
                attr_mat[aa,bmu] += 1

        frac_mat = attr_mat/attr_mat.sum(axis=0)[None,:]*100
        cvec = np.array([cdict[key] for key in attrtypes])

        f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
        f.subplots_adjust(right=0.85)
        if show_hits:
            somp.plot_hexPanel(kshape,allcounts,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=show_hexid\
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
        toppos = cax.get_position().y1
        for aa,attrval in enumerate(attrtypes):
            f.text(0.99,toppos-aa*0.02,attrval[:5],color=cdict[attrval],ha='right',va='top',fontweight='bold',fontsize=18)
        figsaver(f, 'pies/%s_pie'%attr)

        frac_mat1 = attr_mat/attr_mat.sum(axis=1)[:,None]
        frac_mat2 = attr_mat/attr_mat.sum(axis=0)[None,:]
        dcolors = np.array([cdict[attrval] for attrval in attrtypes])  # np.array([cdict_clust[lab] for lab in labels])

        for fracmat,fractag in zip([frac_mat1,frac_mat2],['relwinners','winners']):
            wincats_idx = np.argmax(fracmat,axis=0)

            f, ax = plt.subplots(figsize=(fwidth, fheight))
            f.subplots_adjust(left=l_w / fwidth, right=1. - r_w / fwidth-0.1, bottom=b_h / fheight,
                              top=1. - t_h / fheight-0.05, wspace=0.05)  # -tspace/float(fheight)
            somp.plot_hexPanel(kshape,wincats_idx,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                             ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',quality_map=dcolors)#rankvec

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_axis_off()
            toppos = ax.get_position().y1
            for aa, attrval in enumerate(attrtypes):
                f.text(0.99, toppos - aa * 0.02, attrval[:5], color=cdict[attrval], ha='right', va='top', fontweight='bold',
                       fontsize=18)
            ax.set_title(fractag)
            figsaver(f, 'pies/%sCartoon_%s'%(attr,fractag))


##### HOW MANY RECS PER HEX? AS PIECHART
cmapstr = 'tab10'
N_colors = 10

recids = np.unique([el.recid for el in Units])
n_recs = len(recids)

N_crepeats = int(np.ceil(n_recs/N_colors))
cmap_pie = plt.cm.get_cmap(cmapstr)
colors_ = cmap_pie(np.arange(N_colors))
cchain = np.repeat(colors_[None],N_crepeats,axis=0).reshape(-1,4)

r_mat = np.zeros((n_recs,n_nodes))
for rr,recid in enumerate(recids):
    #reg = region_keys[rr]
    rinds = np.array([ee for ee,el in enumerate(Units) if el.recid==recid])
    rdata = wmat[rinds]
    bmus = somh.get_bmus(rdata,weights)
    for bmu in bmus:
        r_mat[rr,bmu] += 1


f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
f.subplots_adjust(right=0.85)
if show_hits:
    somp.plot_hexPanel(kshape,allcounts,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
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
    vals = r_mat[:,hh]
    nonz = vals[vals != 0]
    if np.size(nonz)>0:
        fracs = np.sort(nonz)[::-1] / np.sum(nonz) * 100
        ax.pie(fracs,radius=radius,center=c,colors=cchain[:len(fracs)])
ax.set_xlim(xxlim)
ax.set_ylim(yylim)
ax.set_title('N recs: %i'%(len(recids)))
figsaver(f, 'pies/Nrecs_pie')


recs_participating = np.array([np.sum(submat!=0) for submat in r_mat.T])

f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
f.subplots_adjust(right=0.85)
somp.plot_hexPanel(kshape,recs_participating,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                 ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',hexcol=S.cmap_count)#rankvec

norm = mpl.colors.Normalize(vmin=0, vmax=n_recs)
cb = somp.plot_cmap(cax, S.cmap_count, norm)
cax.set_title('# recs')
ax.set_xlim(xxlim)
ax.set_ylim(yylim)
ax.set_title('N recs: %i'%(len(recids)))
figsaver(f, 'nrecs/Nrecs_absolute')


units_per_rec = allcounts/recs_participating
f,ax,cax = somp.make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035])
f.subplots_adjust(right=0.85)
somp.plot_hexPanel(kshape,units_per_rec,ax,hw_hex=hw_hex*sizefac,showConn=False,showHexId=False\
                                 ,scalefree=True,return_scale=False,alphahex=1,idcolor='k',hexcol=S.cmap_count)#rankvec

norm = mpl.colors.Normalize(vmin=units_per_rec.min(), vmax=units_per_rec.max())
cb = somp.plot_cmap(cax, S.cmap_count, norm)
cax.set_title('#units/#recs')
ax.set_xlim(xxlim)
ax.set_ylim(yylim)
ax.set_title('N recs: %i'%(len(recids)))
figsaver(f, 'nrecs/nunits_per_nrecs')









#plot Umat

sizefac = 0.4
hw_hex = 0.35*sizefac

widthU = 2*kshape[0]-1
heightU = 2*kshape[1]-1
#kshape1test = int(kshape[1]/4) #just for testing --> works faster
#heightU = 2*kshape1test-1
kshapeU = (widthU,heightU)

hex_dict =  somp.get_hexgrid(kshapeU,hw_hex=hw_hex)
hexd2 = {key:val for key,val in hex_dict.items()}
allinds = np.arange(np.prod(kshapeU))
totind_mat = allinds.reshape(heightU,widthU)

#shift every 3, 7, ... row
rshiftinds = np.arange(2,heightU,4)
shiftids = totind_mat[rshiftinds].flatten()
for idx in shiftids:
    centers,verts = hex_dict[idx]
    hexd2[idx] = [[centers[0]+hw_hex*2,centers[1]],np.vstack([verts[:,0]+2*hw_hex,verts[:,1]]).T ]


#label the retainers
labeldict = {ii:-1 for ii in allinds}
origrows = np.arange(0,heightU,2)
originds = totind_mat[origrows,::2].flatten()
labtemp = {oind:ii for ii,oind in enumerate(originds)}
labeldict.update(labtemp)

#label the betweeners
#first the ones in the row inbetweens
inbetws = totind_mat[origrows,1::2].flatten()
tempd = {}
for idx in inbetws:
    idx1 = labeldict[idx-1]
    idx2 = labeldict[idx+1]
    tempd[idx] = (idx1,idx2)
labeldict.update(tempd)
#then the ones connecting rows
inrows = np.arange(1,heightU,2)
inrowers = totind_mat[inrows]
tempd2 = {}
for rr,myrow in enumerate(inrowers):
    for idx in myrow[::2]:
        idx1 = labeldict[idx-widthU]
        idx2 = labeldict[idx+widthU]
        tempd2[idx] = (idx1,idx2)
    for idx in myrow[1::2]:
        if rr%2==0:
            idx1 = labeldict[idx-widthU+1]
            idx2 = labeldict[idx+widthU-1]
        else:
            idx1 = labeldict[idx-widthU-1]
            idx2 = labeldict[idx+widthU+1]
        tempd2[idx] = (idx1,idx2)
labeldict.update(tempd2)


getdist = lambda p1, p2: np.linalg.norm(
    weights[:, p1] - weights[:, p2])  # euclidean distance, short for np.sqrt(np.sum((w1-w2)**2))

plotvec = np.zeros(len(allinds))

# first the connectors
tupinds = np.delete(allinds, originds)  # all pairing hexagons
for idx in tupinds:
    t1, t2 = labeldict[idx]
    plotvec[idx] = getdist(t1, t2)
# now mean for the centerguys
alltup = plotvec[tupinds]
tuplabs = np.array([labeldict[idx] for idx in tupinds])
for idx in originds:
    # idx = originds[18]
    lab = labeldict[idx]
    neighvals = np.array([tupval for tupval, tuplab in zip(alltup, tuplabs) if lab in tuplab])
    plotvec[idx] = np.mean(neighvals)

if len(hexd2[0]) == 2:
    for ii in sorted(hexd2.keys()): hexd2[ii] += [plotvec[ii]]
elif len(hexd2[0]) == 3:  # overwrite
    for ii in sorted(hexd2.keys()): hexd2[ii][2] = plotvec[ii]
else:
    assert 1, 'something rotten'


xmax, ymax = np.max(hex_dict[np.prod(kshapeU) - 1][1], axis=0)
xmin, ymin = np.min(hex_dict[0][1], axis=0)
xmax = xmax + hw_hex
if np.mod(kshapeU[1], 2.) == 1: xmax = xmax + hw_hex * sizefac
pan_h, pan_w = ymax - ymin, xmax - xmin
t_h, b_h, l_w, r_w = [0.4 * sizefac] + [0.2 * sizefac] * 3
fheight = t_h + pan_h + b_h
fwidth = l_w + pan_w + r_w

cmap = 'gist_earth'
add_width = 0.5
f, axarr = plt.subplots(1, 2, figsize=(fwidth + add_width, fheight), gridspec_kw={'width_ratios': [1,0.05]})

ax,cax = axarr

somp.plot_hexPanel(kshapeU, plotvec, ax, hw_hex=hw_hex * sizefac, showConn=False, showHexId=False \
                 , scalefree=True, return_scale=False, hexcol=cmap, alphahex=1., hexdict=hexd2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_axis_off()

myinds = originds[:] #if disp_mode == 'centerlabs' else allinds[:]
for ii in myinds:
    cent1, cent2 = hexd2[ii][0]
    ax.text(cent1, cent2, str(labeldict[ii]), fontsize=9, ha='center', va='center', color='k')
norm = mpl.colors.Normalize(vmin=plotvec.min(), vmax=plotvec.max())
cb = somp.plot_cmap(cax, cmap, norm)
cax.set_title('dist.')
figsaver(f, 'Umat')

