import numpy as np
from scipy.stats import gamma,spearmanr

def get_interval_stats(tints,stimes,minisis=5):
    isis, isis1, isis2 = collect_intervals(tints,stimes)
    if np.size(isis1) >= minisis:
        #print (np.size(isis1))
        M = np.corrcoef(isis1,isis2)[0,1]
    else:
        M = None
        
    if np.size(isis) >= minisis:
        B = (np.std(isis)-np.mean(isis))/(np.std(isis)+np.mean(isis))
    else:
        B = None
    
    return {'B':B,'M':M,'N_isi':len(isis),'N_isiM':len(isis1)}

getspikes = lambda stimes,tinterval: stimes[(stimes>=tinterval[0]) & (stimes<=tinterval[1])]

def collect_intervals(tints,stimes):
    isi_list = collect_intervals_per_tint(tints,stimes)
    return np.hstack(isi_list),np.hstack([[isis[:-1],isis[1:]] for isis in isi_list if len(isis)>1])

def collect_intervals_per_tint(tints,stimes):
    isi_list = []
    for tint in tints:
        spikes = getspikes(stimes,tint)
        isis_tint = np.diff(spikes)
        isi_list += [isis_tint]
    return isi_list

getM = lambda v1,v2: np.corrcoef(v1,v2)[0,1]
getB = lambda vals: (np.std(vals)-np.mean(vals))/(np.std(vals)+np.mean(vals))
getLv = lambda v1,v2: np.sum(((v1-v2)/(v1+v2))**2)*3/(len(v1))
getLvR = lambda v1,v2,R: np.sum((1-4*v1*v2/((v1+v2)**2))*(1+4*R/(v1+v2)))*3/(len(v1))
getLvR5 = lambda v1,v2: getLvR(v1,v2,5/1000)
getCV2 = lambda v1,v2: 2*np.mean(np.abs((v2-v1))/np.abs((v2+v1)))
getIR = lambda v1,v2: np.mean(np.abs(np.log(v2/v1)))
getRho = lambda v1,v2: spearmanr(v1,v2).correlation
getLKappa = lambda vals: np.log(gamma.fit(vals,loc=0)[0])
getRate = lambda vals: 1/np.median(vals)

intfn_dict = {'M':[getM,2],'B':[getB,1],'Lv':[getLv,2],'LvR':[getLvR5,2],'CV2':[getCV2,2],'IR':[getIR,2],\
           'Rho':[getRho,2],'LKappa':[getLKappa,1],'isiRate':[getRate,1]}


def calc_quantities(isis,isipairs,isi_list,pairs_list,fn_dict,varname,mininputsize=5,segwise=True):
    Ntints = len(isi_list)
    myfn, n_inputs = fn_dict[varname]
    myinputs = [isis] if n_inputs == 1 else [isipairs[0], isipairs[1]]
    mainout  = myfn(*myinputs) if myinputs[0].size > mininputsize else np.nan
    if segwise:
        segvec = np.zeros(Ntints)
        for tt in np.arange(Ntints):
            myinputs = [isi_list[tt]] if n_inputs == 1 else [pairs_list[tt][0], pairs_list[tt][1]]
            segvec[tt] = myfn(*myinputs) if myinputs[0].size > mininputsize else np.nan
        return mainout,segvec
    else:
        return mainout,None

def calc_quant_segwise(isi_list,pairs_list,varname,fn_dict,mininputsize=3):
    Ntints = len(isi_list)
    myfn, n_inputs = fn_dict[varname]
    segvec = np.zeros(Ntints)
    for tt in np.arange(Ntints):
        myinputs = [isi_list[tt]] if n_inputs == 1 else [pairs_list[tt][0], pairs_list[tt][1]]
        segvec[tt] = myfn(*myinputs) if myinputs[0].size >= mininputsize else np.nan
    return segvec