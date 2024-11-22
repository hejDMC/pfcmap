import numpy as np
import re


class Chan(object):
    def __init__(self,recid,channum,task):
        self.recid = recid
        self.chan = channum
        self.task = task
        self.id = '%s__%i'%(recid,channum)

    def set_feature(self,featurename,featureval):
        setattr(self,featurename,featureval)

    def set_coords(self,coords):
        self.AP,self.DV,self.ML = coords

    def set_region(self,region):
        self.region = region
        dpos = re.search(r"\d",region)
        if not type(dpos) == type(None):
            didx = dpos.start()
            self.area, self.layer = region[:didx], region[didx:]
        else:
            self.area,self.layer = region,'NA'


calc_succ_dist = lambda data: np.linalg.norm((data[1:]-data[:-1]).T,axis=0)#data dim: samples x features

#used to medfilt only within permitted range

def apply_to_closeby_coords(featmat,coordmat,uvmat, valfn=lambda x:calc_succ_dist, maxd=0.06,rankon=False):
    distance = calc_succ_dist(coordmat)
    jumptemp = np.where(distance > maxd)[0]
    #print(jumptemp)
    if len(jumptemp) > 0:
        jumpinds = jumptemp + 1
        uvcoords = np.split(uvmat, jumpinds)
        valmats = np.split(featmat, jumpinds)
    else:
        valmats = [featmat]
        uvcoords = [uvmat]
    #print(valmats)
    plotvals = np.hstack([valfn(valmat) for valmat in valmats])
    plotcoords = np.vstack([uvcoordvec for uvcoordvec in uvcoords])
    #plotvals = medfilt(valdists, kwidth)
    if rankon:
        plotvals = plotvals.argsort().argsort()

    return plotcoords, plotvals



