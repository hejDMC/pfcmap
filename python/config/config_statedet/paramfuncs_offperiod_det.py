import yaml
import numpy as np

from pfcmap.python.utils import accesstools as act
from pfcmap.python.utils import statedet_helpers as shf
from pfcmap.python.utils import filtering as filt
from scipy.ndimage import gaussian_filter

pfc_regs = ['ACA', 'ILA', 'MOs', 'ORB', 'PL']
#check_region = lambda locname: len([1 for reg in pfc_regs if locname.count(reg)]) > 0
class DataFuncs(object):
    def __init__(self,recpath,my_regs=pfc_regs):
        self.recretrievalpath = recpath
        self.regs_of_interest = my_regs

    def check_region(self,locname):
        return len([1 for reg in self.regs_of_interest if locname.count(reg)]) > 0

    def get_aRec_exptype(self,filename_g,get_eoi_bool=False,get_freetimes_bool=False):
        aSetting = act.Setting(pathpath=self.recretrievalpath)

        aRec = aSetting.loadRecObj(filename_g, get_eois_col=get_eoi_bool)
        if get_freetimes_bool: aRec.get_freetimes()  # aRec.freetimes now has the free times
        recid = aRec.id.replace('-', '_')
        exptype = (aRec.h_info['exptype'][()]).decode()
        self.aRec = aRec
        self.exptype = exptype
        self.recid = recid

    def get_allspikes(self):
        if not hasattr(self,'allspikes'):
            self.allspikes = self.aRec.h_units['units/spike_times'][()]
    def get_spikes_unit(self,unit_id):
        r1, r0 = self.aRec.get_ridx(unit_id)
        return self.allspikes[r0:r1]

    def get_select_spiketimes(self):
        self.spikes = np.hstack([self.get_spikes_unit(unit_id) for unit_id, unit_elid in zip(self.aRec.unit_ids, self.aRec.unit_electrodes) \
                            if self.check_region(self.aRec.ellocs_all[unit_elid])])
        self.n_units_used = len([1 for unit_elid in self.aRec.unit_electrodes if self.check_region(self.aRec.ellocs_all[unit_elid])])


    def get_ratevec(self,bw):
        return shf.make_spikehist(self.spikes,bw, [0, self.aRec.dur], n_units=self.n_units_used)

    def smooth_ratevec(self,ratevec,Pobj,style='savgol'):
        self.smoothingstyle = style
        if style == 'savgol':
            temp = filt.savitzky_golay(ratevec, Pobj.savgol_params[0], Pobj.savgol_params[1])  # problem savgol goes too low
            return temp.clip(min=0) #because rate cant be negative and savgol sometimes overshoots
        elif style == 'gauss':
            gaussian_filter(ratevec,int(Pobj.gauss_bw/Pobj.bw))


    def get_detectiondict(self,ratevec_smoothed,Pobj):
        return shf.get_detdict(ratevec_smoothed, self.aRec.dur, ratethr_frac=Pobj.ratethr_frac, mindur_trough_pts=Pobj.mindur_trough_pts, \
                        maxint_between_off=Pobj.maxint_between_off, bw=Pobj.bw, \
                        nmin_super=Pobj.nmin_super, marg_super=Pobj.marg_super, marg_nonsuper=Pobj.marg_nonsuper,
                        minpts_free=Pobj.minpts_free, \
                        maxpts_off=Pobj.maxpts_off, print_on=False)


class Paramgetter(object):
    def __init__(self,defaultdictpath,**kwargs):
        with open(defaultdictpath, 'r') as myfile:
            default_dict = yaml.safe_load(myfile)

        for key in default_dict.keys():
            if key in kwargs:
                setattr(self,key,kwargs[key])
            else:
                setattr(self,key,default_dict[key])


        self.mindur_trough_pts = int(self.mindur_trough / self.bw)
        self.maxint_between_off = int(self.maxint_between_off_sec/ self.bw)

        self.marg_super = (np.array(self.marg_super_sec) / self.bw).astype(int)
        self.marg_nonsuper = (np.array(self.marg_nonsuper_sec) / self.bw).astype(int)
        self.minpts_free = int(self.mindur_free / self.bw)  # after blocking and everything

        self.maxpts_off = int(self.maxdur_off / self.bw)  # maximal duration of off episode

