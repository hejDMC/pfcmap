import os
import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
import h5py
import yaml
import sys
from glob import glob


pathpath = 'PATHS/filepaths_carlen.yml'


figdir_gen =  pathdict['figdir_root']+'flatmap_rasterization'
fformat = 'png'#png to avoid too many dots
dpi = 300

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_gen, nametag + '.png')
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname,dpi=dpi)
    if closeit: plt.close(fig)

roimapfiles = [roimapfile for roimapfile in glob(os.path.join(pathdict['tesselation_dir'], '*.h5')) if not roimapfile.count('PFCregions')]
rasterdir = pathdict['tesselation_dir'].replace('flatmaps','flatmaps_rasterized')#output where stuff gets saved


outline_file = os.path.join(pathdict['tesselation_dir'], 'flatmap_PFC_outline.txt')
with open(outline_file) as myfile:
    mytxt = myfile.readline()
outcoords = np.array([[subvals.strip().split(' ')] for subvals in mytxt.split(',')]).astype(float)[:, 0, :]


res = 1
x_min, x_max = np.min(outcoords[:, 0]), np.max(outcoords[:, 0])
y_min, y_max = np.min(outcoords[:, 1]), np.max(outcoords[:, 1])
fine_x = np.arange(x_min, x_max, res)
fine_y = np.arange(y_min, y_max, res)
fine_grid = np.array(np.meshgrid(fine_x, fine_y)).reshape(2, -1).T

p = path.Path(outcoords)
nx,ny = len(fine_x),len(fine_y)

isonmap_bool = p.contains_points(fine_grid).reshape(ny,nx)



for roimapfile in roimapfiles:

    value_mat = np.zeros((ny,nx))

    value_mat[isonmap_bool==False] = np.nan

    with h5py.File(roimapfile,'r') as hand:
        polygon_dict= {key: hand[key][()] for key in hand.keys()}

    for myroikey,mycoords in polygon_dict.items():
        p = path.Path(mycoords)
        roi_bools = p.contains_points(fine_grid).reshape(value_mat.shape)
        value_mat[roi_bools] = int(myroikey)

    #remove zero-coords (usally happending at roi-borders)
    vshape = value_mat.shape
    coord_mask = np.array(np.meshgrid(np.arange(-2,3),np.arange(-2,3))).reshape(2,-1).T

    zero_coords = np.vstack(np.where(value_mat==0)).T

    for coord in zero_coords:
        #coord = zero_coords[10]
        masked_coord = coord+coord_mask
        admissible_mask = masked_coord[(masked_coord[:,0]>=0) &(masked_coord[:,1]>=0) & (masked_coord[:,0]<vshape[0]) & (masked_coord[:,1]<vshape[1])]
        myvals = value_mat[admissible_mask[:,0],admissible_mask[:,1]]
        value_mat[coord[0],coord[1]] = int(np.nanmedian(myvals[myvals>0]))

    fname = os.path.basename(roimapfile).replace('.h5','_rasterized')
    f,ax = plt.subplots()
    im = ax.imshow(value_mat[::-1],cmap='jet',origin='lower')
    cb = f.colorbar(im)
    cb.set_label('ROI id',rotation=-90,labelpad=15)
    ax.set_axis_off()
    figsaver(f,fname)

    outfile = os.path.join(rasterdir,os.path.basename(roimapfile).replace('.h5','__rasterized.h5'))
    with h5py.File(outfile,'w') as hand:
        hand.create_dataset('rastermat',data=value_mat,dtype='f')#f to keep the nans
        hand.create_dataset('valuevec',data=value_mat[isonmap_bool],dtype='i')#nice for direct comparisons/correlations
        hand.attrs['srcfile'] = roimapfile
        hand.attrs['resolution'] = res