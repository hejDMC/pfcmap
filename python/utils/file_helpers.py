import yaml
#def check_filexists(ymlfilepath):
import logging
import logging.config
import os
import importlib.util
import sys
import numpy as np
import pandas
import collections

def iter_paths(d):
    def iter1(d, path):
        paths = []
        for k, v in d.items():
            if isinstance(v, dict):
                paths += iter1(v, path + [k])
            paths.append((path + [k], v))
        return paths
    return iter1(d, [])

def check_presence(pathdict):
    '''pathdict is a ymlfile that may contains paths as the lowest-level entry
    in the hierarchy _file marks files to be checked and _dir marks directories'''

    output = iter_paths(pathdict)
    endpaths1 = [out for out in output if type(out[1])==str]
    endpaths = ['/'.join(myp[0])+'XFNAMEX'+myp[1] for myp in endpaths1]
    filetests = [path.split('XFNAMEX')[1] for path in endpaths if path.count('_file')]
    dirtests = [path.split('XFNAMEX')[1] for path in endpaths if path.count('_dir')]
    for file in filetests:
        assert os.path.isfile(file), 'LACKING file %s'%file
    for mydir in dirtests:
        assert os.path.isdir(mydir), 'LACKING dir %s'%mydir
    return 1

def make_my_logger(loggerpath,log_outpath,default_level=logging.INFO):
    with open(loggerpath, 'rt') as lfile: ldict = yaml.safe_load(lfile.read())
    ldict['handlers']['file']['filename'] = log_outpath
    try:
        logging.config.dictConfig(ldict)
    except:
        print('Error in loading logging configuration. Using default configs')
        logging.basicConfig(level=default_level)
    return logging.getLogger()


def retrieve_module_by_path(pathtofile):
    spec = importlib.util.spec_from_file_location("module.name", pathtofile)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    return foo

def get_githash():
    import git
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

def set_ellocs_all_to_acronym(aRec,regionfile):
    regdict0 = pandas.read_csv(regionfile).to_dict()
    regdict = {str(regdict0['id'][ii]): regdict0['acronym'][ii] for ii in np.arange(len(regdict0['id']))}
    regdict['0'] = 'XXXX'
    ellocs_all = np.array([regdict[idloc] for idloc in aRec.ellocs_all])
    aRec.set_ellocs_all(ellocs_all)


def read_ok_recs_from_excel(tablepath,sheetname='sheet0',not_allowed=['-', '?'],output='verbose'):
    df = pandas.read_excel(tablepath, sheet_name=sheetname)
    # find the ones that need to be ignored
    allrecs = list(df.loc[df['recid'].str.contains('probe')]['recid'].values)
    recids_bad = list(df.loc[df['usable_gen'].isin(not_allowed)]['recid'].values)
    recids_ok = [rec for rec in allrecs if not rec in recids_bad]
    if output == 'verbose':
        return recids_ok,allrecs,df
    else:
        return recids_ok

def write_ordered_yml(filename,datadict):

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_mapping(_mapping_tag, data.iteritems())

    def dict_constructor(loader, node):
        return collections.OrderedDict(loader.construct_pairs(node))

    yaml.add_representer(collections.OrderedDict, dict_representer)
    yaml.add_constructor(_mapping_tag, dict_constructor)


    with open(filename, 'w') as outfile:
        yaml.dump(datadict, outfile, default_flow_style=False)