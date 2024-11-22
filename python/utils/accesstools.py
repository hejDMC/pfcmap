import yaml
import sys


class Setting(object):
    def __init__(self,pathpath = 'PATHS/general_paths.yml',exptype='Passive'):
        with open(pathpath, 'r') as f: self.pdict = yaml.safe_load(f)

        workdir = self.pdict['workspace_dir']
        if not workdir in sys.path:
            print('Appending sys %s'%workdir)
            sys.path.append(workdir)

        from . import data_classes as dc
        self.dc = dc
        self.exptype = exptype

    def get_cfg(self):
        self.cfgpath, self.dspath = self.dc.get_paths(self.exptype, self.pdict)

    def loadRecObj(self,filename,get_eois_col=True):
        self.recfile = filename
        if not hasattr(self,'cfgpath'): self.get_cfg()
        aRec = self.dc.RecPassive(filename, self.cfgpath, self.dspath)
        if get_eois_col:
            aRec.set_eois(aRec.get_column_chans(1))
            aRec.get_elecs(eoi_only=True)
        return aRec




