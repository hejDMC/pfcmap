import sys
import yaml
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py


pathpath,myrun,ncluststr,cmethod = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runCrespZP_brain'
cmethod = 'ward'
ncluststr = '8'
'''
with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import category_matching as matchfns




#kickout_lay3pfc = True
signif_levels=np.array([0.05,0.01,0.001])
nshuff = 1000


outfile = os.path.join(pathdict['statsdict_dir'],'statsdictZETA__%s__ncl%s_%s.h5'%(myrun,ncluststr,cmethod))

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s_zeta/zeta_enrichments'%(myrun))

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)



with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']

zeta_flav = 'onset_time_(half-width)'
zeta_pthr = 0.01
missing_zeta = 999
zeta_pattern = 'RECID__%s__%s%s__TSELpsth2to7__STATEactive__all__zeta.h5'%(myrun,cmethod,ncluststr)
zeta_dir =  os.path.join(pathdict['src_dirs']['zeta'],'%s__%s%s'%(myrun,cmethod,ncluststr))


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])
Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')))

#now set the zeta:
recids = np.unique([U.recid for U in Units])

uloader.get_set_zeta_delay(recids,zeta_dir,zeta_pattern,Units,zeta_feat=zeta_flav,zeta_pthr=zeta_pthr,missingval=missing_zeta)
Units = [U for U in Units if not U.zeta_respdelay == missing_zeta]




for U in Units:
    U.set_feature('clust',int(U.zetap<=zeta_pthr))

Units_pfc = np.array([U for U in Units if not U.roi==0 and S.check_pfc(U.area)])

for U in Units:
    U.set_feature('roistr',str(U.roi))


uloader.assign_parentregs(Units,pfeatname='area',cond=lambda uobj: S.check_pfc(uobj.area)==False,na_str='na')
Units_ctx = np.array([U for U in Units if U in Units_pfc or U.area in S.ctx_regions])

print('pfc: areas',np.unique([U.area for U in Units_pfc]))
print('no pfc areas',np.unique([U.area for U in Units if not S.check_pfc(U.area)]))
print('ctx areas',np.unique([U.area for U in Units_ctx]))



#ux = [U for U in Units if U.area in ctx_regions]

for U in Units:
    if U in Units_pfc:
        U.set_feature('roistrreg',str(U.roi))
    else:
        U.set_feature('roistrreg',str(U.area))



for U in Units:
    if U in Units_ctx:
        laysimple = re.sub("[^0-9]", "", U.layer)
        U.set_feature('arealay','%s|%s'%(U.area,laysimple))
        U.set_feature('layctx',laysimple)
        if U in Units_pfc: U.set_feature('roilay','%s|%s'%(str(U.roi),laysimple))
        else: U.set_feature('roilay',str(U.area))
    else:
        U.set_feature('arealay',str(U.area)) #there can be a few ones in PFC that have no layer assigned
        U.set_feature('roilay',str(U.area))
        U.set_feature('layctx','na')


#allroilays = [U.arealay for U in Units]
#myUs = [U for U in Units if U.arealay=='ACAd|5']

for U in Units:
    if U in Units_ctx:
        is_deep = S.check_layers(U.layer, ['5', '6'])
        laydepth = 'deep' if is_deep else 'sup'
        U.set_feature('arealaydepth', '%s|%s' % (U.area, laydepth))
        if U in Units_pfc:U.set_feature('roilaydepth', '%s|%s' % (str(U.roi), laydepth))
        else: U.set_feature('roilaydepth', str(U.area))
    else:
        U.set_feature('arealaydepth', str(U.area))
        U.set_feature('roilaydepth', str(U.area))

for U in Units: #just to make the shuffling function work when nothing should stay constant for shuffling
    U.set_feature('fake_const','a')


#const_attr='task'
#attr1 = 'roistr'
#attr2 = 'clust'
#Units_pfc56 = [U for Units in Units_pfc if  S.check_layers(U.layer, ['5','6'])]



#laydepth: split into superficical (123) and deep (56)
supertags = ['rois','regs','tasks','utype','recid','layers']

reftags = ['refall','refPFC']
statsdict = {supertag:{subtag:{} for subtag in reftags} for supertag in supertags}
attr2 = 'clust'
const_attr = 'task' #only not used in attr1=task!

for reftag, Usel in zip(['refall','refPFC'],[Units,Units_pfc]):
    statsdict['regs'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'area',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['regs'][reftag]['lays'] = matchfns.get_all_shuffs(Usel,'arealay',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    statsdict['regs'][reftag]['laydepth'] = matchfns.get_all_shuffs(Usel,'arealaydepth',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    #statsdict['tasks'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'task',attr2,const_attr='fake_const',nshuff=nshuff,signif_levels=signif_levels)
    #statsdict['utype'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'utype',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    #statsdict['recid'][reftag]['nolays'] = matchfns.get_all_shuffs(Usel,'recid',attr2, const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)
    #statsdict['layers'][reftag]['justlays'] = matchfns.get_all_shuffs(Usel,'layctx',attr2,const_attr=const_attr,nshuff=nshuff,signif_levels=signif_levels)

uloader.save_dict_to_hdf5(statsdict, outfile)

##now the plotting
cdict_clust = {0:'grey',1:'k'}

statstag = 'enr'
statsfn = S.enr_fn

omnidict = {reftag:{statstag:{} for statstag in [statstag]} for reftag in reftags}
replace_dict = {'|deep':'|d','|sup':'|s'}
def replace_fn(mystr):
    for oldstr,newstr in replace_dict.items():
        mystr = mystr.replace(oldstr, newstr)
    return mystr


#reftag = 'refall'
with h5py.File(outfile,'r') as hand:

    for reftag in reftags:


        ####
        thistag = 'deepCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval.count('|sup') and not aval=='na' \
                                and not (S.check_pfc_full(aval) and not aval.count('|deep'))])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ###
        thistag = 'layeredregsCtx'
        statshand = hand['regs'][reftag]['lays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}


        ####
        thistag = 'deepSupCtx_nolayOtherwise'
        statshand = hand['regs'][reftag]['laydepth']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and not aval=='na' \
                                and not (S.check_pfc_full(aval) and not aval.count('|'))])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}

        ####
        thistag = 'deepSupCtx'
        statshand = hand['regs'][reftag]['laydepth']#from this select the deep ones
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)
        countvec = mystats['matches'][()].sum(axis=1)
        presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if countvec[aa]  > S.Nmin_maps and aval.count('|')])
        sdict = {key: mystats[key][presel_inds] for key in ['avals1','levels','matches','meanshuff','stdshuff','pofs']}
        sdict['alabels'] = np.array([replace_fn(aval) for aval in sdict['avals1']])
        sdict['src'] = statshand.name
        omnidict[reftag][statstag][thistag] = {'src_stats':sdict}


seltags = [key for key in omnidict[reftag][statstag].keys() ]

#make simple correlation plots of the different categories
for reftag in reftags:
    for thistag in seltags:

        #reftag = 'refall'
        #thistag = seltags[0]
        sdict = omnidict[reftag][statstag][thistag]['src_stats']
        def make_savename(plotname):
            return '%s/%s__%s/%s__%s_%s_%s' % (reftag, thistag, statstag, plotname, reftag, thistag, statstag)

        X = statsfn(sdict)

        nareas = len(X)
        xvec = np.arange(nareas)
        sortinds = np.argsort(X[:,1])

        xlab_simp = [replace_fn(lab) for lab in sdict['alabels'][sortinds]]
        # plot sorted
        def make_nice_axes(ax):
            ax.set_xticks(xvec)
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2))
            ax.set_xticklabels(xlab_simp, rotation=-90)
            ax.set_ylabel('enrichment')
            myxlim = np.array(ax.get_xlim())
            ax.fill_between(myxlim,np.zeros_like(myxlim)-2,np.zeros_like(myxlim)+2,color='grey',alpha=0.5)
            ax.set_ylim([-10,10])
            ax.set_xlim(myxlim)

            ax.get_figure().tight_layout()




        f, ax = plt.subplots(figsize=(2+0.1*len(X), 3))
        f.subplots_adjust(bottom=0.18)
        for cc in np.arange(2):
            col = cdict_clust[cc]
            ax.plot(xvec, X[sortinds, cc], '.-', color=col, ms=12, mec='none', mfc=col, lw=0)
        for xx in np.arange(nareas):
            vals = X[sortinds[xx], :]
            if not len(np.unique(vals)) == 1:
                minval = np.nanmin(vals)
                maxval = np.nanmax(vals)
                ax.plot([xx, xx], [minval, maxval], color='silver', zorder=-5)
        make_nice_axes(ax)
        figsaver(f,make_savename('vert_conn'))

        f, ax = plt.subplots(figsize=(2+0.1*len(X), 3))
        f.subplots_adjust(bottom=0.18)
        for cc in np.arange(2):
            col = cdict_clust[cc]
            ax.plot(xvec, X[sortinds, cc], '.-', color=col, ms=12, mec='none', mfc=col, lw=1)
        make_nice_axes(ax)
        figsaver(f,make_savename('default_conn'))

uloader.save_dict_to_hdf5(omnidict, outfile.replace('statsdict','enrichmentdict'))
