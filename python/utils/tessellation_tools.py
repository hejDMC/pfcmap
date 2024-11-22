from k_means_constrained import KMeansConstrained
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

import shapely

def get_clusters(X,NperPatch=100,frac_range=[0.8,1.2]):
    n_clusters = int(len(X) / NperPatch)
    minsize = NperPatch*frac_range[0]
    maxsize = NperPatch*frac_range[1]
    if len(X)<=maxsize:
        return X.mean(axis=0)[None,:],np.array([0])
    if n_clusters * maxsize < len(X):
        maxsize = int(np.ceil(len(X)/n_clusters))
    clf = KMeansConstrained(n_clusters, size_min=minsize,
                            size_max=maxsize)
    clf.fit(X)
    #cluster_centers_ = clf.cluster_centers_
    #labels_ = clf.labels_
    return clf.cluster_centers_, clf.labels_

def get_voronoi_polygons_outbounded(midpoints,outcoords,print_on=False):
    #outcoords = outcoords * np.array([1, v_fac])
    outline_pg = shapely.Polygon(outcoords)
    centerpts = shapely.MultiPoint(midpoints)
    vor_polygons = shapely.voronoi_polygons(centerpts, extend_to=outline_pg)  #
    clipped = []
    counter = 0
    for pg in vor_polygons.geoms:
        xx, yy = pg.exterior.coords.xy
        mycoords = np.vstack([xx.tolist(), yy.tolist()]).T
        mypg = shapely.Polygon(mycoords)
        # clipped_pg = shapely.intersection(mypg, outline_pg)
        clipped_pg = shapely.intersection(outline_pg, mypg)
        if clipped_pg.is_empty:
            clipped += [mypg]
            if print_on: print('empty-clip')
        else:
            clipped += [clipped_pg]
            if print_on: print('clipped!')
            counter += 1
    return clipped,vor_polygons


def make_roidict(rois,sortax=1,reverse=False):
    centlist = [roi.centroid for roi in rois]
    centvec = np.array([[cent.x, cent.y] for cent in centlist])
    sortinds = np.argsort(centvec[:, sortax])
    if reverse:
        sortinds = sortinds[::-1]

    roi_dict = {}
    #cent_dict = {ii:cent for ii,cent in enumerate(centvec[sortinds])}
    for ii, roi in enumerate(np.array(rois)[sortinds]):
        xx, yy = roi.exterior.coords.xy
        roi_dict[str(ii + 1)] = np.array([xx.tolist(), yy.tolist()]).T
    return roi_dict#,cent_dict


def save_roidict(dstpath,roidict):
    with h5py.File(dstpath, 'w') as fdest:
        for key,poly in roidict.items():
            fdest.create_dataset(str(key),data=poly,dtype='f')

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))




def get_regnamedict_of_rois(region_file,polygon_dict):
    with h5py.File(region_file, 'r') as hand:
        rdict = {key: hand[key][()] for key in hand.keys()}

    regnames = np.array(list(rdict.keys()))
    roidict_regs = {}
    for roiname, roiverts in polygon_dict.items():
        roipolyg = shapely.geometry.Polygon(roiverts)
        myreg = regnames[np.argmax(np.array([shapely.geometry.Polygon(rdict[regname]).intersection(roipolyg).area for regname in regnames]))]
        roidict_regs[roiname] = myreg
    return roidict_regs


def colorfill_polygons(ax,polygon_dict,plotdict,subkey=0,cmap='',clab='',na_col='grey',ec='k',nancol='gainsboro',show_cmap=True,**kwargs):
    mode = 'depth' if  type(list(plotdict.items())[0][-1])==type(np.array([1.2])) else 'direct'
    #print(mode)

    f = ax.get_figure()
    for key, verts in polygon_dict.items():
        #print (key)
        if key in plotdict:
            plotval = plotdict[key][subkey] if mode == 'depth' else plotdict[key]#subkey is e.g. often clidx
            #print(type(plotval))
            if type(plotval)==type('bla'):
                col = str(plotval)
            else:
                #print(plotval)
                if np.isnan(plotval):
                    col =nancol
                else: col = cmap.to_rgba(plotval)
            #print(col)
            # elif mode == 'perc': col = perc_dict[]
        else:
            col = na_col
        ax.add_patch(plt.Polygon(verts * np.array([1, -1])[None, :], fc=col, ec=ec, alpha=1.))

    if show_cmap:
        pos = ax.get_position()
        cax = f.add_axes(
            [pos.x0 + pos.width * 0.2, pos.y0 + 0.05, pos.width / 12, pos.height / 4])  # [left, bottom, width, height]
        cax.set_title(clab)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap.cmap, norm=cmap.norm, orientation='vertical')
    if 'mylimfn' in kwargs:
        kwargs['mylimfn'](ax)
    ax.set_axis_off()

def get_scalar_map(cmapstr,vminmax):
    cmap_z = plt.cm.get_cmap(cmapstr)
    cNorm_z = plt.cm.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    return mpl.cm.ScalarMappable(norm=cNorm_z, cmap=cmap_z)


