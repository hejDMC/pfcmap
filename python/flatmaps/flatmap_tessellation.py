import numpy as np
import h5py
import yaml
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import shapely
from shapely.ops import polylabel

pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
layers = ['5','6']


with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_gen = pathdict['savepath_gen']

tesselation_dir = pathdict['tesselation_dir']

figdir_gen = pathdict['figdir_root'] + '/methods/flatmap_tesselation'

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_gen, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


sys.path.append(pathdict['code']['workspace'])
from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import tessellation_tools as ttools


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)
#reftag,layerlist,datasets,tasks,tsel,statetag,utypes,nnodes = [rundict[key] for key in ['reftag','layerlist','datasets','tasks','tsel','state','utypes','nnodes']]
#wmetrics,imetrics,wmetr_weights,imetr_weights = [rundict[metrtype][mytag] for mytag in ['features','weights'] for metrtype in ['wmetrics','imetrics']]#not strictly necessary: gets called in uloader



metricsfiles = S.get_metricsfiles_auto(rundict,srcdir,tablepath=pathdict['tablepath'])

Units,somfeats,weightvec =  uloader.load_all_units_for_som(metricsfiles,rundict,check_pfc= (not myrun.count('_brain')),rois_from_path=False)




Units_pfc = [U for U in Units if S.check_pfc(U.area)]
roiless_pfc = [U for U in Units_pfc if U.roi==0]
print('N roiless pfc units %i'%(len(roiless_pfc)))
Units_pfc_sel = [U for U in Units_pfc if S.check_layers(U.layer,layers) and not U.roi==0]
#layu = np.unique([U.layer for U in Units_pfc])



####just for plotting
v_fac = -1

roimage_file = os.path.join(tesselation_dir,'flatmap_PFCregions.h5')
with h5py.File(roimage_file,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

def plot_flatmap(axobj,**kwargs):
    for key in polygon_dict.keys():
        points = polygon_dict[key]
        axobj.plot(points[:, 0], points[:, 1] * v_fac, **kwargs)


def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])


gensizes =  [100,200]
roiattr = 'roi2'
sortax = 0
marg = 0.1
frac_range=[1-marg,1+marg]
write_thr = 1500
outline_file = os.path.join(tesselation_dir,'flatmap_PFC_outline.txt')
with open(outline_file) as myfile:
    mytxt = myfile.readline()
outcoords = np.array([[subvals.strip().split(' ')] for subvals in mytxt.split(',')]).astype(float)[:, 0, :]


region_file = os.path.join(tesselation_dir,'flatmap_PFCregions.h5')
with h5py.File(region_file, 'r') as hand:
    regpolygon_dict = {key: hand[key][()] for key in hand.keys()}
Units_pfc_all = [U for U in Units if S.check_pfc(U.area) and not U.roi==0]

for gensize in gensizes:
    print('gensize %i'%gensize)
    #do it for the whole outline
    '''
    mode = 'free'
    print('mode %s'%mode)

    X = np.array([[U.u,U.v] for U in Units_pfc])

    midpoints,clustlabels = ttools.get_clusters(X,NperPatch=gensize,frac_range=frac_range) #cluster_centers_*np.array([1,v_fac])
    rois,vor_polygons = ttools.get_voronoi_polygons_outbounded(midpoints,outcoords)
    myroidict = ttools.make_roidict(rois,sortax=sortax,reverse=True)


    #save it

    roidict_dst = os.path.join(tesselation_dir,'flatmap_PFC_ntesselated_%s_res%i.h5'%(mode,gensize))
    ttools.save_roidict(roidict_dst,myroidict)



    #example how to load and assign to units
    with h5py.File(roidict_dst,'r') as hand:
        myroidict = {key: hand[key][()] for key in hand.keys()}



    uloader.assign_roi_to_units(Units_pfc_all,myroidict,roiattr =roiattr)
    '''

    ### now per sub-area!
    mode = 'obeyRegions'
    print('mode %s'%mode)



    X = np.array([[U.u,U.v] for U in Units_pfc_sel])

    roi_superdict = {}
    for regkey,regpolycoords in regpolygon_dict.items():
        regpoly = shapely.Polygon(regpolycoords)
        reg_inds = np.array([uu for uu,coords in enumerate(X) if regpoly.contains(shapely.Point(coords))])
        X_reg = X[reg_inds]
        midpoints,clustlabels = ttools.get_clusters(X_reg,NperPatch=gensize,frac_range=frac_range) #cluster_centers_*np.array([1,v_fac])
        rois,vor_polygons = ttools.get_voronoi_polygons_outbounded(midpoints,regpolycoords)
        roi_superdict[regkey] = ttools.make_roidict(rois,sortax=sortax,reverse=True)


    allroislist = [shapely.Polygon(roi_superdict[regkey][subkey])  for regkey in roi_superdict.keys()for subkey in roi_superdict[regkey].keys()]
    myroidict = ttools.make_roidict(allroislist,sortax=sortax,reverse=True)


    roidict_dst =  os.path.join(tesselation_dir,'flatmap_PFC_ntesselated_%s_res%i.h5'%(mode,gensize))
    ttools.save_roidict(roidict_dst,myroidict)



###now the plotting!
for gensize in gensizes:
    for mode in ['obeyRegions']:#,'free'
        roidict_dst = os.path.join(tesselation_dir,'flatmap_PFC_ntesselated_%s_res%i.h5'%(mode,gensize))

        with h5py.File(roidict_dst,'r') as hand:
            myroidict = {key: hand[key][()] for key in hand.keys()}

        uloader.assign_roi_to_units(Units_pfc_all,myroidict,roiattr =roiattr)


        ####check-plotting
        #just plotting here
        areas = np.array([ttools.PolyArea(coords[:,0],coords[:,1]) for coords in myroidict.values() ])


        f,ax = plt.subplots(figsize=(12,10))
        for roikey,polyg in myroidict.items():

            ax.plot(polyg[:,0],polyg[:,1]*v_fac,'k')
            area = ttools.PolyArea(polyg[:,0],polyg[:,1])
            pttemp = polylabel(shapely.Polygon(polyg))
            m0,m1 = pttemp.x,pttemp.y
            #m0,m1 = polyg.mean(axis=0)
            ax.text(m0,m1*v_fac,roikey,ha='center',va='center',color='r')
        plot_flatmap(ax,color='skyblue',alpha=0.5,lw=6)
        set_mylim(ax)
        ax.set_axis_off()
        figsaver(f, '%s/Ngen%i/%s_N%i__rois_labeled_FULLYNUMBERED'%(mode,gensize,mode,gensize), closeit=True)




        f,ax = plt.subplots()
        for roikey,polyg in myroidict.items():

            ax.plot(polyg[:,0],polyg[:,1]*v_fac,'k')
            area = ttools.PolyArea(polyg[:,0],polyg[:,1])
            pttemp = polylabel(shapely.Polygon(polyg))
            m0,m1 = pttemp.x,pttemp.y
            #m0,m1 = polyg.mean(axis=0)
            if area>=write_thr:

                ax.text(m0,m1*v_fac,roikey,ha='center',va='center',color='r')
            else:
                ax.plot(m0,m1*v_fac,'r.')
        plot_flatmap(ax,color='skyblue',alpha=0.5,lw=6)
        set_mylim(ax)
        ax.set_axis_off()
        figsaver(f, '%s/Ngen%i/%s_N%i__rois_labeled'%(mode,gensize,mode,gensize), closeit=True)

        # assign U to roi


        #checking units assigned
        labelvec = np.array([str(getattr(U,roiattr)) for U in Units_pfc_sel])

        allu = np.array([U.u for U in Units_pfc_sel])
        allv = np.array([U.v for U in Units_pfc_sel])
        myX = np.vstack([allu,allv]).T


        f,ax = plt.subplots()

        for roikey,polyg in myroidict.items():

            phand = ax.plot(polyg[:,0],polyg[:,1]*v_fac)
            col = phand[0].get_color()
            selxy = myX[labelvec==roikey]
            ax.plot(selxy[:,0],selxy[:,1]*v_fac,'.',color=col,alpha=0.3,ms=2)

        plot_flatmap(ax,color='silver')
        set_mylim(ax)
        ax.set_axis_off()
        figsaver(f, '%s/Ngen%i/%s_N%i__units_assigend'%(mode,gensize,mode,gensize), closeit=True)



        #final check count per reg also separate for L5 and L6 and toghether
        # different selections as input here
        for tag,layers_allowed in zip(['L5/6','L5','L6','L2/3'],[['5','6'],['5'],['6'],['2','3']]):
            Usel = [U for U in Units_pfc_all if S.check_layers(U.layer, layers_allowed)]
            roikeys = np.array(list(myroidict.keys()))
            count_dict = {roikey: len([U for U in Usel if getattr(U,roiattr) == int(roikey)]) for roikey in roikeys}
            count_dict = {roikey:count_dict[roikey] if not np.isinf(count_dict[roikey]) else np.nan for roikey in count_dict.keys()}
            countvals = np.array(list(count_dict.values()))
            countvals = countvals[~np.isnan(countvals)]
            vmin,vmax = [np.min(countvals),np.max(countvals)]
            cmap = plt.cm.get_cmap(S.cmap_count)
            cNorm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)

            f, ax = plt.subplots(figsize=(5, 4))
            f.subplots_adjust(left=0, right=1, top=1, bottom=0)
            f.text(0.01, 0.98, '%s counts' % (myrun), ha='left', va='top')
            plot_flatmap(ax, color='k', zorder=-10)
            for roikey,polyg in myroidict.items():
                ax.plot(polyg[:,0],polyg[:,1]*v_fac,color='k')
                col = scalarMap.to_rgba(count_dict[roikey])
                ax.add_patch(plt.Polygon(polyg * np.array([1, v_fac])[None, :], fc=col, ec='k', alpha=1.))
            cax = f.add_axes([0.2,0.1,0.05,0.45])#[left, bottom, width, height]
            set_mylim(ax)
            ax.set_axis_off()
            cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.cm.get_cmap(S.cmap_count), norm=cNorm, orientation='vertical')
            cax.set_ylabel('%s count'%tag,rotation=-90,labelpad=15)
            figsaver(f, '%s/Ngen%i/%s_N%i__counts_%s'%(mode,gensize,mode,gensize,tag.replace('/','_')), closeit=True)


'''
centlist = [roi.centroid for roi in rois]
centvec = np.array([[cent.x, cent.y] for cent in centlist])

sortinds = np.argsort(centvec[:, sortax])
sortinds = sortinds[::-1]
for ii, roi in enumerate(np.array(rois)[sortinds]):
    xx, yy = roi.exterior.coords.xy
pgs = list(roi.geoms)
f,ax = plt.subplots()
for pg in pgs:
    xx, yy = pg.exterior.coords.xy
    ax.plot(np.array(list(xx)),np.array(list(yy))*-1)
plot_flatmap(ax,color='skyblue',alpha=0.5,lw=2)
set_mylim(ax)
'''