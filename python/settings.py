import numpy as np
from glob import glob
import os
import pandas as pd
import re

ctx_regions = ['AUD','MO','SS','VIS']#append if you use more!!!

runsets = {'wwspont':'runC00dMP3_brain','nwspont':'runC00dMI3_brain',\
           'wwresp':'runCrespZP_brain','nwresp':'runCrespZI_brain',\
           'wwIBL':'runIBLPasdMP3_brain_pj','nwIBL':'runIBLPasdMI3_brain_pj'}

cmethod = 'ward'
nclust_dict = {'wwspont':8,'nwspont':5,'wwresp':8,'nwresp':5,'wwIBL':8,'nwIBL':5}

task_replacer = {'Opto':'Passive'}

enr_fn = lambda sdict: ((sdict['matches']-sdict['meanshuff'])/sdict['stdshuff'])
frac_fn = lambda sdict: sdict['matches']/sdict['matches'].sum(axis=1)[:,None]


src_acquistion_types = {'Carlen':'table',\
                        'IBL_Passive':'glob'}

fformat = 'svg'

cmap_count = 'Greys'
cmap_div = 'RdBu_r'
cmap_rate = 'magma'
cmap_hier = 'viridis'
cmap_clust = 'rainbow_r'
Nmin_maps = 20


niter = 200

ntints_min = 12
timescalepath =  'ZENODOPATH/preprocessing/metrics_extraction'

response_path = 'ZENODOPATH/preprocessing/metrics_extraction/psth_scores'
responsefile_pattern = 'RECID__TSELpsth2to7__STATEmystate__all_psth_ks10__PCAscoresMODE.h5'#--> replace recid, mystate,MODE
responsetint_pattern =  'RECID__TSELpsth2to7__STATEmystateREFTAG.h5'

roimap_path = 'ZENODOPATH/flatmaps/flatmap_PFC_ntesselated_obeyRegions_res200.h5'\



featfndict = {}
#featfndict = {'rate':{'fn':lambda x:np.log10(x) if x>0 else np.nan,'repl':'logRate'},\
#                'rate_mean':{'fn': lambda x: np.log10(np.e**x),'repl':'lrate_mean'},\
#              'rate_std':{'fn': lambda x: np.log10(np.e**x),'repl':'lrate_std'}}

#pfc_rois = ['ACA', 'ILA', 'MOs', 'ORB', 'PL','AI','FRP']

#check_pfc = lambda locname: len([1 for reg in pfc_rois if locname.count(reg) and not locname.count('VPL')]) > 0
pfc_set = {'ACA', 'ILA', 'MOs', 'ORB', 'PL','AI','FRP','ACAd', 'ACAv', 'ORBm', 'ORBvl','ORBl','AId','AIv','AI'}
check_pfc = lambda locname: locname in pfc_set

numbered_regions = {'CA1','CA2','CA3'}#for uloader so it doesnt count the numbers here as layers

check_layers = lambda mylayer,layers_allowed: len([1 for lay in layers_allowed if mylayer.count(lay)]) > 0


def decompose_area_layer(regname):
    dpos = re.search(r"\d", regname)  # find the digit position

    if regname in numbered_regions or type(dpos) == type(None):
        area, layer = str(regname), 'NA'

    else:
        didx = dpos.start()
        area, layer = regname[:didx], regname[didx:]
    return area,layer

def strip_to_area(regname,depth_separator='|'):
    '''returns region name with layers and anything after depth_separator removed'''
    rname = regname.split(depth_separator)[0]
    dpos = re.search(r"\d",rname)#find the digit position
    if rname in numbered_regions or type(dpos) == type(None):
        return rname
    else:
        return rname[:dpos.start()]


def get_layer_from_region(regname):
    return decompose_area_layer(regname)[1]


check_pfc_full = lambda regname: check_pfc(strip_to_area(regname))



physbound_dict = {'abRatio':[-1.1,1],
             'p2tRatio':[-0.8,0.3],\
             'LvR':[0,2.5],\
                'LvR_mean':[0,2.5],\
                  'LvR_std':[0,1.5],\
            'Lv_mean':[0.,2.5],\
             'IR_mean':[0,3.],\
                  'IR_std':[0,1.5],\
            'B_std':[0,0.5]}

check_bound = lambda uobj,feat: True if not hasattr(uobj,feat) else physbound_dict[feat][0]<=getattr(uobj,feat)<=physbound_dict[feat][1]
check_physbounds = lambda uobj: np.sum([check_bound(uobj,feat) for feat in physbound_dict])==len(physbound_dict)

anat_fdict = {'U':'u','V':'v','ap':'AP','dv':'DV','ml':'ML','location':'region','main channel':'chan','roi':'roi'}
unittype_options = ['nw','ww']

imetrics_fdict = {'B':'B','LvR':'LvR','rate':'rate','CV2':'CV2', 'IR':'IR', 'LKappa':'LKappa', 'Lv':'Lv', 'M':'M', 'Rho':'Rho'}
waveform_fdict = {'peak2Trough':'peak2Trough','abRatio':'abRatio','p2tRatio':'p2tRatio'}

#cdict_pfc = {'MOs':'forestgreen','ACAd':'goldenrod', 'ACAv':'darkorange', 'PL':'darkorchid','ILA':'hotpink',\
#             'ORBl':'skyblue', 'ORBm':'dodgerblue', 'ORBvl':'mediumblue'}

cdict_pfc = {'MOs':'#4B6A2E','ACAd':'#E8CD00', 'ACAv':'#E5A106', 'PL':'#CE6161','ILA':'#9F453B',\
             'ORBl':'#505770', 'ORBm':'#5A8DAF', 'ORBvl':'#3D6884',\
             'ACA':'#E7B703','ORB':'#4C7B9A','FRP':'burlywood','AId':'purple','AIv':'orchid'}

PFC_sorted = ['MOs','ACAd', 'ACAv', 'PL','ILA', 'ORBm', 'ORBvl','ORBl','FRP','AId','AIv']

cdict_task = {'Passive':"#004194", 'Opto':"#56dada",'Aversion':"hotpink", 'Detection':"forestgreen",'Context':'yellowgreen','IBL':'khaki',\
              'Attention':'gold'}#"#98634c""#ff5db8"
cdict_ds = {'Carlen':'mediumorchid','IBL':'darkkhaki'}
cdict_layer = {'1':'orange','2':'forestgreen','3':'forestgreen','4':'saddlebrown','5':'skyblue','6':'orchid','NA':'grey'}
cdict_utype = {'ww':'firebrick','nw':'cornflowerblue','na':'grey'}

pc_std_allowed = 5 #units with pc scores smaller or bigger than this get disregarded





def get_metricsfiles_auto(rundict,srcdir,**kwargs):
    acq_types = [src_acquistion_types[ds] for ds in rundict['datasets']]
    #acq_typesu = np.unique(acq_types)
    subkwargs = {key:val for key,val in kwargs.items() if not key=='tablepath'}
    if type(srcdir)==str:
        srcdirs = [srcdir]
    else:
        srcdirs = srcdir
    fileslist = []
    for acqtype,mysrcdir in zip(acq_types,srcdirs):
        if acqtype == 'glob':
            newfiles =  glob(os.path.join(mysrcdir,'*'))
        elif acqtype == 'table':
            newfiles = get_metricsfiles(mysrcdir,from_table=True,\
                                        tablepath=kwargs['tablepath'],**subkwargs)
            print('getting files from table %s'%kwargs['tablepath'])
        fileslist += newfiles
    return fileslist


def get_allowed_recids_from_table(tablepath='config/datatables/allrecs_allprobes.xlsx',sheet='sheet0',not_allowed=['-', '?']):
    df = pd.read_excel(tablepath, sheet_name=sheet)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    allrecrows = df['recid']
    isrec_bool = allrecrows.str.contains('probe',na=False)
    allowed_bool = ~df['usable_gen'].isin(not_allowed)
    #exptype_bool = df['exptype'].isin(exptypes_allowed)
    bools_list = [isrec_bool,allowed_bool]
    cond_bool = np.array([vals.astype(int) for vals in bools_list]).sum(axis=0) == len(bools_list)
    return list(allrecrows[cond_bool].values)

def get_metricsfiles(srcdir,from_table=True,tablepath='config/datatables/allrecs_allprobes.xlsx',sheet='sheet0',not_allowed=['-', '?'],**kwargs):
    metrics_pool = glob(os.path.join(srcdir,'*'))
    #tablepath = 'config/datatables/allrecs_allprobes.xlsx'
    if not from_table:
        return metrics_pool
    else:
        recids = get_allowed_recids_from_table(tablepath=tablepath,sheet=sheet,not_allowed=not_allowed)
        metricsfiles = []
        for recid in recids:
            if 'mustcontain' in kwargs:
                matchfile = [f for f in metrics_pool if f.count(recid) and f.count(kwargs['mustcontain'])]
            else:
                matchfile = [f for f in metrics_pool if f.count(recid)]
            assert len(matchfile) == 1, 'not exactly one metrics file for %s N=%i'%(recid,len(matchfile))
            metricsfiles += matchfile
        return metricsfiles






parent_dict = {
    'PFC':['MOs','ACAd','ACAv','PL','ILA','FRP','ACA']+['ORBl','ORBvl','ORBm','ORB']+['AId','AIv'],\
     #'ORB':['ORBl','ORBvl','ORBm','ORB'],\
     #'AI':['AId','AIv'],\
    'DP':['DP'], \
    'TT':['TTd','TTv'], \
    'LS':['LSr','SH','LSv','LSc'],\
    'CP':['CP'],\
    'ACB':['ACB'],\
    'HPF':['CA1','CA2','CA3','DG-mo','DG-po','DG-sg','HPF'],\
    'MO':['MOp'], \
    'AON':['AON'], \
    'PIR':['PIR'],\
    'SS':['SSp-bfd','SSp-m','SSs','SSp-tr','SSp-n','SSp-ul','SSp-ll','SSp-un'],
    'VIS':['VISa', 'VISal',  'VISli','VISp','VISpm','VISam','VISpor','VISpl','VISl','VISrl'],
    'AUD':['AUDpo','AUDd','AUDp'],\
    'EPd':['EPd'],\
    'TH':['RT','VPM','LP','LGv','LD','VPL','Eth','PO','LGd-co']}

all_ctx_regions =  set(np.hstack([parent_dict[key] for key in ctx_regions]))


parent_colors = {
    'PFC':'silver',\
    'DP':'purple', \
    'TT':'deeppink', \
    'LS':'skyblue',\
    'CP':'dodgerblue',\
    'ACB':'steelblue',\
    'HPF':'gold',\
    'MO':'darkgreen', \
    'AON':'violet', \
    'PIR':'lightpink',\
    'SS':'limegreen',
    'AUD':'forestgreen',\
    'VIS': 'yellowgreen',\
    'EPd':'orange',\
    'TH':'saddlebrown'}

parents_ordered = ['PFC', 'MO','AUD', 'SS', 'VIS','DP', 'TT','AON', 'PIR','HPF', 'ACB', 'CP','LS','EPd','TH']#,'ORB', 'AI'
#olf: AON, TT, DP, PIR -purples: violet, purple, deeppink, orchid
# cerebral nuclei - LS, ACB, CP - blues: skyblue, steelblue, dodgerblue
# ctx: AUD, SS, MO, AI - greens: forestgreen, limegreen, seagreen teal
# th: TH - brown peru
# HC: hpf - gold gold
# EPd, EPd - orange orange
# PFC: ORB, PFC -greys silver grey

def get_parentreg(regionval,na_str='na'):
    temp =  [key for key,vals in parent_dict.items() if strip_to_area(regionval) in vals]
    return na_str if len(temp) == 0 else temp[0]
