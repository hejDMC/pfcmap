
import h5py
from glob import glob
import sys
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pathpath,myrun,roi_tag = sys.argv[1:]

'''
pathpath = 'PATHS/filepaths_carlen.yml'
myrun = 'runC00dMP3_brain'
#pathpath = 'PATHS/filepaths_IBL.yml'#
#myrun = 'runIBLPasdMP3_brain_pj'
roi_tag = 'gaoRois'
'''

with open(pathpath, 'r') as myfile: pathdict = yaml.safe_load(myfile)
sys.path.append(pathdict['code']['workspace'])

from pfcmap.python.utils import unitloader as uloader
from pfcmap.python import settings as S
from pfcmap.python.utils import tessellation_tools as ttools


with open(pathpath) as yfile: pathdict = yaml.safe_load(yfile)

rundict_path = pathdict['configs']['rundicts']
srcdir = pathdict['src_dirs']['metrics']
savepath_SOM = pathdict['savepath_SOM']


region_file = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCregions.h5')

if roi_tag == 'dataRois':
    flatmapfile = str(S.roimap_path)
elif roi_tag == 'gaoRois':
    flatmapfile = os.path.join(pathdict['tessellation_dir'],'flatmap_PFCrois.h5')
else:
    assert 0, 'unknown roi_tag: %s'%roi_tag

statsfile = glob(os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl*_*.h5'%(roi_tag,myrun)))[0]
#statsfile = os.path.join(pathdict['statsdict_dir'],'statsdict_rois_%s__%s__ncl%s_%s.h5'%(roi_tag,myrun,ncluststr,cmethod))


rundict_folder = os.path.dirname(rundict_path)
rundict = uloader.get_myrun(rundict_folder,myrun)

somfile = uloader.get_somfile(rundict,myrun,savepath_SOM)
somdict = uloader.load_som(somfile)

kshape = somdict['kshape']

stylepath = os.path.join(pathdict['plotting']['style'])
plt.style.use(stylepath)

figdir_mother = os.path.join(pathdict['figdir_root'],'SOMruns',myrun,'%s__flatmapCounts'%myrun,roi_tag)

#figsaver
def figsaver(fig, nametag, closeit=True):
    figname = os.path.join(figdir_mother, nametag + '__%s.%s'%(myrun,S.fformat))
    figdir = os.path.dirname(figname)
    if not os.path.isdir(figdir): os.makedirs(figdir)
    fig.savefig(figname)
    if closeit: plt.close(fig)


#prepare flatmap plotting
with h5py.File(flatmapfile,'r') as hand:
    polygon_dict= {key: hand[key][()] for key in hand.keys()}

def set_mylim(myax):
    myax.set_xlim([338,1250])
    myax.set_ylim([-809,-0])


reftag = 'refall'#N.B: only for one REFTAG--refall and refPFC should be identical
with h5py.File(statsfile,'r') as hand:

        ####
        statshand = hand[reftag]['lays']
        mystats = uloader.unpack_statshand(statshand,remove_srcpath=True)

presel_inds = np.array([aa for aa, aval in enumerate(mystats['avals1']) if aval.count('|')])
avals = mystats['avals1'][presel_inds]
countvec = mystats['matches'][presel_inds].sum(axis=1)

roivec_overall = np.unique([aval.split('|')[0] for aval in avals])


###how to select layers
#layers_allowed = ['5','6']
ulays = np.unique([aval.split('|')[1] for aval in avals])
layer_dict = {'deep':['5','6'],'sup':['23','1']}
layer_dict.update({'L%s'%mylay:[mylay] for mylay in ulays })


#color setup for plotting
na_col = 'skyblue'
ec = 'firebrick'
nancol = 'r'
mapstrs = ['binary_r','binary','Greys','Greys_r','bone','bone_r']
fndict = {'N':lambda x:x,'logN':lambda x:np.log10(x)}

collection_dict = {}
for laylab,layers_allowed in layer_dict.items():
    #laylab = 'sup'
    #layers_allowed = layer_dict[laylab]

    plotdict = {}
    for rr,roi in enumerate(roivec_overall):
        roivals_allowed = [roi+'|'+lay for lay in layers_allowed]
        roi_inds = np.array([int(np.where(avals==rval)[0]) for rval in roivals_allowed if rval in avals])
        if len(roi_inds)>0:
            #mycounts[rr] = countvec[roi_inds].sum()
            plotdict[roi] = int(countvec[roi_inds].sum())

    collection_dict[laylab] = plotdict
    #print(mycounts)


    # loop over log and no-log
    for myfnlab,myfn in fndict.items():
        #myfnlab = 'logN'
        #myfn = fndict[myfnlab]
        showdict = {key:myfn(val) for key,val in plotdict.items()}
        myvals = np.array([list(showdict.values())])

        for mymapstr in mapstrs:
            #mymapstr = mapstrs[0]
            savename = 'dispmode_%s/CMAP_%s/countstatsFlatmap_%s_%s__CMAP%s'%(myfnlab,mymapstr,laylab,myfnlab,mymapstr)

            mycmap = ttools.get_scalar_map(mymapstr, [myvals.min(), myvals.max()])

            f,ax = plt.subplots(figsize=(4,4))
            ttools.colorfill_polygons(ax, polygon_dict, showdict, cmap=mycmap, clab=myfnlab, na_col=na_col, nancol=nancol,
                                      ec=ec,
                                      show_cmap=True, mylimfn=set_mylim)
            ax.set_title('%s %s'%(myrun,laylab))
            figsaver(f,savename)


outfile = os.path.join(os.path.join(figdir_mother,'flatmapCounts__%s_%s.xlsx'%(myrun,roi_tag)))
with pd.ExcelWriter(outfile, engine='openpyxl') as writer:
    for laylab,mydict in collection_dict.items():
        #df = pd.DataFrame.from_dict(mydict, orient='index',\
        #                            columns=['roi_id','count'])#

        df = pd.DataFrame(list(mydict.items()),
                     columns=['roi_id', 'count'])
        df.to_excel(writer, sheet_name=laylab)

if roi_tag == 'gaoRois':

    import csv
    gao_hier_path = pathdict['gao_hierarchy_file']



    with open(gao_hier_path, newline='') as csvfile:
        hdata = list(csv.reader(csvfile, delimiter=','))[1:]
    hdict = {str(int(float(el[0]))): float(el[1]) for el in hdata}
    condcol_dict = {'neither':'w','both':'green','Gao':'dodgerblue','data':'gold'}

    for laylab,ddict in collection_dict.items():
        roicol_dict = {}
        #ddict = collection_dict[laylab]
        for roi in hdict.keys():
            d_cond = False
            if roi in ddict:
                if ddict[roi]>=S.Nmin_maps:
                    d_cond = True
            if np.isnan(hdict[roi]):
                g_cond = False
            else:
                g_cond = True
            if g_cond and d_cond:
                roicol_dict[roi] = condcol_dict['both']
            elif g_cond:
                roicol_dict[roi] = condcol_dict['Gao']
            elif d_cond:
                roicol_dict[roi] = condcol_dict['data']
            else:
                roicol_dict[roi] = condcol_dict['neither']


        f, ax = plt.subplots(figsize=(4, 4))
        ttools.colorfill_polygons(ax, polygon_dict, roicol_dict, na_col='k', ec='k',
                                   show_cmap=False, mylimfn=set_mylim)
        for tt,tag in enumerate(['both','GaoOnly','dataOnly']):
            f.text(0.25,0.45-tt*0.07,tag,color=condcol_dict[tag.replace('Only','')],transform=ax.transAxes,fontsize=10,fontweight='bold')
        savename = 'data_availablity/dataAvailability_%s'%(laylab)
        ax.set_title('%s %s'%(myrun,laylab))
        figsaver(f,savename)







