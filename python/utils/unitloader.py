import h5py
import numpy as np
import re
from glob import glob
import os
import yaml
from pfcmap.python.utils import file_helpers as fhs
from pfcmap.python import settings as S
import git


class SortedUnit(object):
    def __init__(self):
        pass

    @property
    def id(self):
        return '%s__%i'%(self.recid,self.uid)

    def set_feature(self,featurename,featureval):
        setattr(self,featurename,featureval)


    def get_area_and_layer(self):
        self.area,self.layer = S.decompose_area_layer(self.region)


def set_various_features(U,uu,featlists,labeldicts,collection_dict):
    for featlist,labeldict in zip(featlists,labeldicts):
        for feat in featlist:

            featlabel = labeldict[feat] if feat in labeldict else str(feat)
            U.set_feature(featlabel,collection_dict[feat][uu])


def set_various_features_tagged(U, uu, featlists, labeldicts,collection_dict,tags=['prestim','poststim'],featsep='__'):
    for featlist, labeldict in zip(featlists, labeldicts):
        for feat in featlist:
            featlabel = labeldict[feat] if feat in labeldict else str(feat)
            for ttag in tags:
                U.set_feature('%s%s%s'%(featlabel,featsep,ttag), collection_dict['%s%s%s'%(feat,featsep,ttag)][uu])


def set_boolstr(U,uu,keys,collection_dict,featname='utype',na_tag='na'):
    for key in keys:
        if collection_dict[key][uu]:
            U.set_feature(featname, key)
        if not hasattr(U, featname):
            U.set_feature(featname, na_tag)


def extract_feats_metrics(hand, feat,tsel,statetag,reftag,spec):
    featname = feat.split('_')[0]
    if feat.count('_'):
        flav = feat.split('_')[1]
        dspath = 'interval_metrics/{featname}/{tsel}/{stag}/{rtag}/{spec}/{flav}'.format(featname=featname,
                                                                                      tsel=tsel, stag=statetag,
                                                                                      rtag=reftag, spec=spec,flav=flav)
    else:
        dspath = 'interval_metrics/{featname}/{tsel}/{stag}/{rtag}/all/all_isis'.format(featname=featname,
                                                                                        tsel=tsel,
                                                                                        stag=statetag,
                                                                                        rtag=reftag)
    #print(dspath,hand.filename)
    return hand[dspath][()]


def translate_tsel(tsel,statetag):
    #this is for the IBL
    if tsel == 'passive':
        mytsel = 'prestim'
        mystatetag = 'active'
    else:
        mytsel,mystatetag = str(tsel),str(statetag)
    return mytsel,mystatetag



def get_units_from_metricsfile(filename,anat_feats='all',imetrics='all',wavemetrics='all',rmetrics=[],tsel='prestim',reftag='all',statetag='active',\
                               spec='na',get_unittype=True,filterfn=lambda uobj:True,responsefile=None):

    if tsel == 'prepost':
        suffix = '%s/%s'%(statetag,reftag)
    else:
        #suffix = '%s/%s/%s'%(tsel,statetag,reftag)
        mytsel,mystatetag = translate_tsel(tsel,statetag)
        featextrfn_metrics = lambda hand, feat:extract_feats_metrics(hand, feat,mytsel,mystatetag,reftag,spec)

    if anat_feats == 'all': anat_feats = list(S.anat_fdict.keys())
    if imetrics == 'all': imetrics= list(S.imetrics_fdict.keys())
    if wavemetrics == 'all': wavemetrics = list(S.waveform_fdict.keys())

    collection_dict = {}

    with h5py.File(filename,'r') as hand:
        dataset = hand['Dataset'][()].decode()
        task = hand['Task'][()].decode()

        uids = (hand['uids'][()]).astype(int)

        if tsel == 'prepost':
            for feat in imetrics:
                for ttag in 'prestim','poststim':
                    collection_dict['%s__%s'%(feat,ttag)] = hand['interval_metrics/%s/%s/%s'%(feat,ttag,suffix)][()]
        else:
            for feat in imetrics:
                #print(feat)
                collection_dict[feat] = featextrfn_metrics(hand, feat)
                #print('extr_feat',feat)

        for feat in wavemetrics:
            collection_dict[feat] = hand['waveform_metrics/%s'%(feat)][()]



        #if len(wavemetrics)>0:
        collection_dict['wavequality'] =hand['waveform_metrics/waveform_quality'][()]

        collection_dict['quality'] =  hand['quality/quality'][()]
        #print('WAVEFORMS!!!',collection_dict['wavequality'].min())

        for feat in anat_feats:
            if feat == 'roi':
                collection_dict[feat] = hand['anatomy/%s'%feat][()].astype(int)
            else: collection_dict[feat] = hand['anatomy/%s'%feat][()]
        if get_unittype:
            #print (hand.filename)
            for feat in S.unittype_options:
                collection_dict[feat] =  hand['unit_type/%s'%(feat)][()]


    #decoding strings
    for key,vals in collection_dict.items():
        if type(vals[0]) == bytes:
            collection_dict[key] = np.array([val.decode() for val in vals])


    if tsel == 'prepost':
        set_imetrics = lambda uobj,uid: set_various_features_tagged(uobj,uid,[imetrics],[{}],collection_dict)
    else:
        set_imetrics = lambda uobj,uid: set_various_features(uobj,uid,[imetrics],[{}],collection_dict)

    n_units = len(uids)
    #print(n_units)
    for feat in imetrics:
        assert n_units==len(collection_dict[feat]), 'uids and feature extracted mismatch!'

    if not len(rmetrics) == 0:
        n_comps = len(rmetrics)
        with h5py.File(responsefile) as rhand:
            rscores = rhand['scores'][:,:n_comps]
            ruids = rhand['uids'][()]

    if task in S.task_replacer:
        task_new = S.task_replacer[task]
        print('replacing task %s with %s for %s'%(task,task_new,get_recid_from_mtrfile(filename)))
    else: task_new = str(task)

    Units = []
    for uu in np.arange(n_units):
        U = SortedUnit()
        uid = uids[uu]
        U.set_feature('uid',uid)
        if len(rmetrics)>0:
            uidx = int(np.where(ruids==uid)[0])
            uscores = rscores[uidx]
            for rfeat,uscore in zip(rmetrics,uscores):
                U.set_feature(rfeat,uscore)

        #set_various_features(U,uu,[anat_feats,imetrics,wavemetrics],[anat_fdict,imetrics_fdict,waveform_fdict])
        set_various_features(U,uu,[anat_feats,wavemetrics],[S.anat_fdict, {}],collection_dict)
        #if len(wavemetrics)>0:
        #print ('QUALITY',collection_dict['wavequality'][uu])
        U.set_feature('wavequality', collection_dict['wavequality'][uu])
        U.set_feature('quality', collection_dict['quality'][uu])

        if len(imetrics)>0: set_imetrics(U,uu)
        U.get_area_and_layer()
        if get_unittype:
            set_boolstr(U,uu,S.unittype_options,collection_dict,featname='utype',na_tag='na')
        #if filterfn(U) and len(rfeats)==0:
        #    Units += [U]
        #elif filterfn(U) and ~np.isnan(getattr(U,rfeats[0])):
        #    Units += [U]
        if filterfn(U):
            Units += [U]
        #else:
        #    print('%s discarding %i quality%i'%(os.path.basename(filename),uid,U.wavequality))
        for featname,val in zip(['dataset','task'],[dataset,task_new]):U.set_feature(featname,val)

    return Units

    #except:
    #    print('Error at file %s'%filename)
    #    return []


get_recid_from_mtrfile = lambda filename: (os.path.basename(filename).split('metrics_')[1]).split('.')[0]

def replace_by_dict(mytxt,repldict):
    newtxt = str(mytxt)
    for rkey,rval in repldict.items():
        newtxt = newtxt.replace(rkey,rval)
    return newtxt


def get_psth_file(recid,statetag,mode):
    repl_dict = {'RECID':recid,'mystate':statetag,'MODE':mode}
    response_tag = replace_by_dict(S.responsefile_pattern,repl_dict)
    response_file = os.path.join(S.response_path,response_tag)
    assert os.path.isfile(response_file),'responsefile does not exist %s'%response_file
    return response_file


def load_all_units_for_som(filenames,rundict,check_pfc=True,rois_from_path=True,feat_adj_dict=S.featfndict,physbound_checker=S.check_physbounds):
    #print ('hehey')
    #reftag, layerlist, datasets, tasks, tsel, statetag = [rundict[key] for key in
    #                                                         ['reftag', 'layerlist', 'datasets', 'tasks', 'tsel',
    #                                                          'state']]

    reftag, layerlist, datasets, tasks, tsel, statetag, utypes, nnodes,spec = [rundict[key] for key in
                                                                          ['reftag', 'layerlist', 'datasets', 'tasks',
                                                                           'tsel', 'state', 'utypes', 'nnodes','spec']]

    wmetrics, imetrics, wmetr_weights, imetr_weights = [rundict[metrtype][mytag] for mytag in ['features', 'weights']
                                                        for metrtype in ['wmetrics', 'imetrics']]
    if 'rmetrics' in rundict.keys():
        recid_temp = get_recid_from_mtrfile(filenames[0])
        response_file = get_psth_file(recid_temp,statetag,spec)
        assert os.path.isfile(response_file), 'responsefile not  found %s'%response_file
        with h5py.File(response_file) as rhand:
            rscores = rhand['scores'][()]
        if rundict['rmetrics']['features'] == 'all':
            pass
        else:
            assert type(rundict['rmetrics']['features']) == type(1), 'unfitting feature type for response pc %s'%(str(rundict['rmetrics']['features']))
            uptocomp = rundict['rmetrics']['features']
            rscores = rscores[:,:uptocomp]

        rfeats = ['PC%i'%(pcnum) for pcnum in np.arange(rscores.shape[1])+1]
        if rundict['rmetrics']['weights'] == []:
            rweights = [1]*len(rfeats)
        else:
            rweights =  rundict['rmetrics']['features']['weights']

    else:
        rfeats = []
        rweights = []

    somfeats = wmetrics + imetrics + rfeats

    weightvec = np.array(wmetr_weights + imetr_weights+rweights)  # np.array([1]*len(bandfeatures) + [2]*len(ampfeatures))

    if tsel == 'prepost':
        somfeats = ['%s_prestim' % feat for feat in somfeats] + ['%s_poststim' % feat for feat in somfeats]
        weightvec = np.r_[weightvec, weightvec]

    check_layer = lambda locname: True if layerlist == 'all' else len(
        [1 for mylayer in layerlist if locname.count(mylayer)]) > 0
    check_utype = lambda utype: True if utypes == 'all' else utype in utypes
    check_task = lambda task: len([1 for mytask in tasks if task == mytask]) > 0
    check_dataset = lambda dataset: len([1 for myds in datasets if dataset == myds]) > 0

    check_quality = lambda uobj: uobj.quality==1 and uobj.wavequality==1
    get_features = lambda uobj: np.array([getattr(uobj, sfeat) for sfeat in somfeats])
    check_fullfeat = lambda uobj: ~np.isnan(
        get_features(uobj).mean())  # select only those that have not nan at all features

    def filterfn_unit(uobj):
        mybool = check_layer(uobj.region) & check_utype(uobj.utype)\
                & check_fullfeat(uobj) & check_quality(uobj)
        if check_pfc:
            mybool = (S.check_pfc(uobj.area)) & mybool

        return mybool


    def filterfn_rec(fhand):
        dataset = fhand['Dataset'][()].decode()
        task = fhand['Task'][()].decode()
        recid = get_recid_from_mtrfile(filename)
        #print(recid)
        tintcheck = True
        #print('here',len(rundict['rmetrics']['features']),recid)
        if len(rundict['imetrics']['features'])>0 or len(rundict['rmetrics']['features'])>0:
            tintfile = get_tintfile_rec(recid, dataset, rundict, timescalepath=S.timescalepath)
            #print('tintfile',tintfile)
            #print('tintfile %s'%tintfile)
            with h5py.File(tintfile, 'r') as hand: ntints = hand['tints'][()].shape[0]
            if ntints<=S.ntints_min:
                tintcheck = False
                print('not enough tints %s: %i<%i'%(recid,ntints,S.ntints_min))

        return check_task(task) & check_dataset(dataset) & tintcheck



    Units = []
    #(S.)responsefile_pattern = 'RECID__TSELpsth2to7__STATEmystate__all_psth_ks10__PCAscoresMODE.h5'#--> replace recid, mystate,MODE

    for filename in filenames:
        #print (filename)
        # filename = filenames[10]
        try:
            with h5py.File(filename,'r') as fhand:
                #dataset = fhand['Dataset'][()].decode()
                #print(filename)
                include_rec = filterfn_rec(fhand)
                #print(filename,include_rec)

        except:
            include_rec = False
            print('WARNING - not including %s'%filename)
        if include_rec:

            recid = get_recid_from_mtrfile(filename)#(os.path.basename(filename).split('metrics_')[1]).split('.')[0]
            responsefile = get_psth_file(recid,statetag,spec) if len(rfeats) > 0 else None


            Us = get_units_from_metricsfile(filename, anat_feats='all', imetrics=imetrics, wavemetrics=wmetrics,rmetrics=rfeats,\
                                                    tsel=tsel, reftag=reftag, statetag=statetag, spec=spec,\
                                                    get_unittype=True, filterfn=filterfn_unit,responsefile=responsefile)
            #print (os.path.basename(filename),'N',len(Us))

            for U in Us:
                U.set_feature('recid',recid)
            Units += Us
        else:
            print('not including %s'%filename)

    for rfeat in rfeats:
        vals = np.array([getattr(U, rfeat) for U in Units])
        normed_vals = (vals - np.mean(vals)) / vals.std()
        Units = [U for uu, U in enumerate(Units) if (normed_vals[uu] <= S.pc_std_allowed) & (
                    normed_vals[uu] >= -S.pc_std_allowed)]  # Units[(normed_vals<=std_allowed)&(normed_vals>=-std_allowed)]



    if hasattr(S,'roimap_path') and rois_from_path:
        if len(S.roimap_path)>0:
            with h5py.File(S.roimap_path, 'r') as hand:
                myroidict = {key: hand[key][()] for key in hand.keys()}

            print('assigning pfc rois from file %s'%S.roimap_path)
            Units_pfc_all = [U for U in Units if S.check_pfc(U.area) and not U.roi == 0]
            assign_roi_to_units(Units_pfc_all, myroidict, roiattr='roi')
            for U in Units:
                if not U in Units_pfc_all:
                    U.set_feature('roi',0)

    for feat, subdict in feat_adj_dict.items():
        for U in Units:
            if hasattr(U, feat):
                setattr(U, feat, subdict['fn'](getattr(U, feat)))

    Units = [U for U in Units if physbound_checker(U)]

    return Units,somfeats,weightvec





def get_tintfile_rec(recid,dataset,rundict,timescalepath =S.timescalepath):
    reftag = '__all' if rundict['reftag'] == 'all' else ''
    if dataset == 'Carlen':
        if rundict['tsel'] == 'response':
            tintfilepath = os.path.join(timescalepath,'timeselections_psth')
            #print('tfilepath',tintfilepath)
            #responsetint_pattern =  'RECID__TSELpsth2to7__STATEmystateREFTAG.h5'
            repl_dict = {'RECID': recid, 'mystate': rundict['state'], 'REFTAG': reftag}
            responsetint_tag = replace_by_dict(S.responsetint_pattern, repl_dict)
            tintfile =  os.path.join(tintfilepath, responsetint_tag)
            #print('mytintfile', tintfile)

        else:
            tintfilepath = os.path.join(timescalepath, 'timeselections_quietactive')
            tintfile =  os.path.join(tintfilepath, '%s__TSEL%s%s__STATE%s%s.h5' % (recid, rundict['tsel'], \
                                                                                   rundict['spec'].replace('dur',''),rundict['state'],reftag))

    else:
        tintfilepath = os.path.join(timescalepath, 'timeselections_external', dataset)
        tintfile = glob(os.path.join(tintfilepath, '%s*__TSEL%s%s.h5' % (recid, rundict['tsel'],rundict['spec'].replace('dur',''))))
        assert len(tintfile) == 1, 'not exactly one tintfile %s'%(str(tintfile))
        tintfile = tintfile[0]
    assert os.path.isfile(tintfile),'does not exist tintfile %s'%(tintfile)

    return tintfile


def get_tintfile(U,rundict,timescalepath = S.timescalepath):
    return get_tintfile_rec(U.recid,U.dataset,rundict,timescalepath = timescalepath)

def get_ntints(U,rundict,timescalepath = S.timescalepath):
    tintfile = get_tintfile(U, rundict,timescalepath)
    with h5py.File(tintfile,'r') as hand: ntints = hand['tints'][()].shape[0]
    return ntints





def get_anatref(nwbpool,recids,dataset_dict):

    Carlen_bregma = [530, 65, 570]
    IBL_bregma = [540, 33, 566]

    anat_refdict = {}
    for recid in recids:
        try:
            nwbfile = [nwbfile for nwbfile in nwbpool if nwbfile.count(recid)][
                0]
            with h5py.File(nwbfile, 'r') as hand:
                dataset = dataset_dict[hand['identifier'][()].decode()]
                if dataset == 'Carlen':
                    coord_labs = ['AP', 'DV', 'ML']
                    transfn = lambda coords: np.vstack(
                        [coords[0] * -1000, coords[1] * -1000, coords[2] * 1000. +Carlen_bregma[2] * 10])
                elif dataset == 'IBL':
                    coord_labs = ['AP', 'DV', 'ML']
                    transfn = lambda coords: np.vstack(
                        [IBL_bregma[0] * 10 * (-coords[0]), coords[1] * (-IBL_bregma[1] + 10) * -1, coords[2] + IBL_bregma[2] * 10])

                all_coords = np.vstack(
                    [hand['general/extracellular_ephys/electrodes/%s' % coord_lab][()] for coord_lab in coord_labs])
                locations = np.array([loc.decode() for loc in hand['general/extracellular_ephys/electrodes/location'][()]])

            all_coords = np.round(transfn(all_coords))
            refinds_pfc = np.array([S.check_pfc(loc) for loc in locations])
            anat_refdict[recid] = [all_coords.T[refinds_pfc], locations[refinds_pfc]]
        except:
            pass
    return anat_refdict

stringsave_h5 = lambda mygroup, dsname, strlist: mygroup.create_dataset(dsname, data= [mystr.encode("ascii", "ignore") for mystr
                                                                             in strlist],dtype='S30')
def save_som(savepath,myrun,rundict,kshape,weights,weightvec,refmean,refstd,somfeats,qe,te,recids,niter):
    with h5py.File(savepath,'w') as dhand:

        dhand.attrs['git_hash'] = fhs.get_githash()

        mgroup = dhand.create_group('selection')

        mgroup.attrs['run'] = myrun
        for attrkey in ['state','tsel','reftag']:
            mgroup.attrs[attrkey] = rundict[attrkey]

        layerinput = ['all'] if rundict['layerlist'] == 'all' else rundict['layerlist']
        stringsave_h5(mgroup, 'layers', layerinput)
        utypeinput = ['all'] if rundict['utypes'] == 'all' else rundict['utypes']
        stringsave_h5(mgroup, 'utypes', utypeinput)


        stringsave_h5(mgroup, 'datasets', rundict['datasets'])
        stringsave_h5(mgroup, 'tasks', rundict['tasks'])
        stringsave_h5(mgroup, 'recids', recids)

        results = dhand.create_group('results')
        results.create_dataset('kshape', data=np.array(kshape), dtype='i')
        results.create_dataset('weights', data=weights, dtype='f')
        results.create_dataset('refmeanstd', data=np.vstack([refmean, refstd]), dtype='f')
        stringsave_h5(results, 'features', somfeats)
        results.create_dataset('featureweights', data=weightvec, dtype='f')

        results.attrs['qe'] = qe
        results.attrs['te'] = te


def load_som(srcfile):
    outdict = {}

    with h5py.File(srcfile,'r') as fhand:

        for key in ['layers','utypes','datasets','tasks','recids']:
            vals = fhand['selection/%s'%key][()]
            if type(vals[0]) == np.bytes_:
                outdict[key] = np.array([val.decode() for val in vals])
            else: outdict[key] = vals

        for key in ['kshape','weights','refmeanstd','features','featureweights']:

            vals = fhand['results/%s'%key][()]
            if type(vals[0]) == np.bytes_:
                outdict[key] = np.array([val.decode() for val in vals])
            else: outdict[key] = vals

        for attr in ['qe','te']:
            outdict[attr] = fhand['results'].attrs[attr]

    return outdict


def reference_data(dmat,refmean,refstd,weightvec):
    zmat = (dmat-refmean[None,:])/refstd[None,:]# standardize
    return zmat*weightvec[None,:]#wheight





def get_myrun(rundict_folder,myrun,print_mother=True):
    for rpath in glob(os.path.join(rundict_folder,'*.yml')):
        with open(rpath, 'r') as myfile: rsuper = yaml.safe_load(myfile)
        if myrun in rsuper.keys():
            if print_mother: print('getting rundict from %s'%os.path.basename(rpath))
            return rsuper[myrun]
    return None

def get_somfile(rundict,myrun,savepath_gen):
    somfile = glob(os.path.join(savepath_gen, '%s_SOM_kShape*.h5' % myrun))

    if is_projection(rundict):
        somfile = glob(os.path.join(savepath_gen, '%s_SOM_kShape*.h5' % rundict['srcrun']))
        print('getting somfile from external run %s' % rundict['srcrun'])
    assert len(somfile) == 1, 'not exactly one somfile matching found %s' % (str(somfile))

    return somfile[0]

def is_projection(rundict):
    myflag = False
    if 'srcrun' in rundict:
        if len(rundict['srcrun']) > 0:
            myflag = True
    return myflag

def assign_roi_to_units(Units,roidict,roiattr = 'roi'):
    import shapely

    roikeys = list(roidict.keys())
    polylist = [shapely.Polygon(roidict[key]) for  key in roikeys]
    for U in Units:
        mypoint = shapely.Point(U.u,U.v)
        temp = np.array([pp for pp,polyg in enumerate(polylist) if polyg.contains(mypoint)])
        if np.size(temp) == 0:
            #take the closet border instead
            myidx = np.argmin([polyg.exterior.distance(mypoint) for polyg in polylist])
        else:
            myidx = int(temp[0])
        myroilab = roikeys[myidx]
        U.set_feature(roiattr,int(myroilab))


def extract_clusterlabeldict(somfile,cmethod,get_resorted=True):
    with h5py.File(somfile, 'r') as hand:
        #nclustkeys = list(hand['clustering/%s'%cmethod].keys())
        ddict = {int(key): vals[()] for key, vals in hand['clustering/%s' % cmethod].items()}
        has_resorted = 'clustering_resorted' in hand
        if not has_resorted or not get_resorted:
            return ddict
        else:
            resorted_ds =  hand['clustering_resorted/%s' % cmethod]
            for ncl in ddict.keys():
                if str(ncl) in resorted_ds:
                    print('loading resorted ncl:%i from %s'%(ncl, somfile))
                    ddict[ncl] = resorted_ds[str(ncl)][()]
            return ddict


def save_dict_to_hdf5(mydict, filename,strtype='S10',include_githash=True):
    """
    ....
    """
    if include_githash:
        repo = git.Repo(search_parent_directories=True)
        mydict.update({'git_hash':repo.head.object.hexsha})
    with h5py.File(filename, 'w') as hand:
        recursively_save_dict_contents_to_group(hand, '/', mydict,strtype=strtype)

def recursively_save_dict_contents_to_group(hand, path, mydict,strtype='S10'):
    """
    ....
    """

    for key, item in mydict.items():
        if isinstance(item,tuple):
            item = np.array(item)
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,int,list)):
            #print(path,key)
            ascii_written = 0
            if isinstance(item,list):
                print('Waring transforming list to array: %s'%str(item))
                item = np.array(item)
            if type(item) == np.ndarray:
                if item.dtype.kind in {'U', 'S'}:
                    asciiList = [val.encode("ascii", "ignore") for val in item]
                    hand.create_dataset(path + key, shape=(len(asciiList), 1),
                                 data=asciiList, dtype=strtype)
                    ascii_written= 1
            if ascii_written == 0:
                if hasattr(item,'dtype'):
                    if item.dtype.kind in {'U', 'S'}:
                        hand[path + str(key)] = item.encode("ascii", "ignore")
                    else:
                        hand[path + str(key)] = item
                else:
                    hand[path + str(key)] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(hand, path + key + '/', item,strtype=strtype)
        else:
            print(key,item)
            raise ValueError('Cannot save %s type'%type(item))

def unpack_statshand(statshand,remove_srcpath=True):
    ds_dict = {}
    def get_ds_dictionaries(dummy, node):
        fullname = node.name
        if isinstance(node, h5py.Dataset):
            # node is a dataset
            #print(f'Dataset: {fullname}; adding to dictionary')
            if node.dtype.kind == 'S':
                nodevals = np.array([aval.decode() for aval in node[:,0]])
            else:
                nodevals = node[()]
                if type(nodevals) == bytes:
                    nodevals = nodevals.decode()
            ds_dict[fullname] = nodevals
            #print('ds_dict size', len(ds_dict))
        else:
            #node is a group
            print(f'Group: {fullname}; skipping')

    statshand.visititems(get_ds_dictionaries)
    if remove_srcpath:
        ds_dict = {key.replace(statshand.name+'/',''):vals for key,vals in ds_dict.items()}

    return ds_dict


def assign_parentregs(Us,pfeatname='preg',cond=lambda uboj:True,na_str='na'):
    for U in Us:
        if cond(U):
            temp =  [key for key,vals in S.parent_dict.items() if U.area in vals]
            parentreg = na_str if len(temp) == 0 else temp[0]
            U.set_feature(pfeatname,parentreg)


def get_set_zeta_delay(recids,zeta_dir,zeta_pattern,Units,zeta_feat='onset_time_(half-width)',zeta_pthr=0.01,missingval=999):
    for recid in recids:
        # print(recid)
        zeta_file = os.path.join(zeta_dir, zeta_pattern.replace('RECID', recid))
        # zeta_file = zeta_file.replace(myrun,'runC00dMI3_brain')#
        recUs = [U for U in Units if U.recid == recid]

        if os.path.isfile(zeta_file):
            with h5py.File(zeta_file, 'r') as hand:
                myuids = hand['uids'][()].astype(int)
                zeta = hand['ZETA_test/%s'%zeta_feat][()]
                zetap = hand['ZETA_test/pvalue'][()]

            for U in recUs:
                matches = np.where(myuids == U.uid)[0]
                if len(matches) == 1:
                    uidx = int(matches[0])
                    thiszeta, thisp = zeta[uidx], zetap[uidx]
                    U.set_feature('zetaval', thiszeta)
                    U.set_feature('zetap', thisp)

                    if thisp <= zeta_pthr:
                        U.set_feature('zeta_respdelay', thiszeta)
                    else:
                        U.set_feature('zeta_respdelay', np.nan)
                else:
                    print(
                        'WARNING - no single matching uid found for %s-uid:%i found:%s' % (recid, U.uid, str(matches)))
                    U.set_feature('zeta_respdelay', missingval)
        else:
            for U in recUs:
                print('WARNING - zetafile found for %s-uid:%i' % (recid, U.uid))
                U.set_feature('zeta_respdelay', missingval)