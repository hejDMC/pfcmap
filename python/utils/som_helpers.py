import numpy as np
import h5py





def get_bmu(sample, som):
    loc = np.argmin(((som.K - sample) ** 2).sum(axis=2))
    bmu = np.divide(loc, som.kshape[1]), loc % som.kshape[1]
    return bmu

def distance_map(K):
    """ Returns the average distance map of the weights.
    (Each mean is normalized in order to sum up to 1) """
    um = np.zeros((K.shape[0], K.shape[1]))
    it = np.nditer(um, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
            for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                if ii >= 0 and ii < K.shape[0] and jj >= 0 and jj < K.shape[1]:
                    um[it.multi_index] += np.linalg.norm(K[ii, jj, :] - K[it.multi_index])
        it.iternext()
    um = um / um.max()
    return um


def reference_datamat(datamat,refmean,refstd,**kwargs):
    '''datamat dimension is samplex x featurs'''
    zmat = (datamat-refmean[None,:])/refstd[None,:]# standardize
    if 'weightvec' in kwargs:
        return zmat*kwargs['weightvec'][None,:]#wheight
    else:
        return zmat


stringsave_h5 = lambda mygroup, dsname, strlist: mygroup.create_dataset(dsname, (len(strlist), 1), 'S30',
                                                                            [mystr.encode("ascii", "ignore") for mystr
                                                                             in strlist])





def get_simple_roimat(roi_dict, my_vars, norm='zscore',**kwargs):

    roi_mat = np.empty((0, len(my_vars)))
    for roi_id in sorted(roi_dict.keys()):
        # fill roi_mat with parameters
    
        mr = roi_dict[roi_id]
        roilist = [mr[my_var] for my_var in my_vars if my_var in list(mr.keys())]
        if 'r_fano' in my_vars:roilist.append((mr['r_std'] ** 2) / mr['r_mean'])
        
        roi_mat = np.vstack([roi_mat, roilist])
    
    if norm == 'L2': roi_mat = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, roi_mat)
    elif norm == 'zscore':
        if 'z_ref' in kwargs:meanmat,stdmat = kwargs['z_ref']
        else: meanmat,stdmat = np.mean(roi_mat, axis=0), np.std(roi_mat, axis=0)
        roi_mat = (roi_mat - meanmat) / stdmat
    # elif norm=='01'
    return roi_mat

def get_mean_std(roi_dict,my_vars):
    roi_mat = np.empty((0, len(my_vars)))
    for roi_id in sorted(roi_dict.keys()):
        # fill roi_mat with parameters
    
        mr = roi_dict[roi_id]
        roilist = [mr[my_var] for my_var in my_vars if my_var in list(mr.keys())]
        if 'r_fano' in my_vars:roilist.append((mr['r_std'] ** 2) / mr['r_mean'])
        
        roi_mat = np.vstack([roi_mat, roilist])
    
    
    return np.mean(roi_mat, axis=0), np.std(roi_mat, axis=0)

    

def get_kshape(roimat, nneurons='default'):
    
    covmat = np.cov(roimat.T)
    evals, evecs = np.linalg.eig(covmat)
    eratio = np.sort(evals)[-2] / np.sort(evals)[-1]
    if nneurons == 'default':nneurons = np.sqrt(roimat.shape[0]) * 5
    x = np.sqrt(nneurons / eratio)
    kshape = (int(np.ceil(x * eratio)), int(np.ceil(x)))  # swap for longitudinal map
    return kshape

def get_distance_to_ref(sommap, ref, norm=True):
    edist = np.linalg.norm(sommap - ref, axis=2)
    if norm == True:
     return (edist - np.min(edist)) / (np.max(edist) - np.min(edist))
    else: return edist


  

def plot_somInput(roimat, catlist, varnames):
    from matplotlib.pylab import figure
    xvec = np.arange(len(catlist))
    # cols = ["#7BC05E","#961859","#6DBCFF","#B48812","#1A6C33"]
    mymin, mymax = np.min(roimat[:, 0]) - 1., np.max(roimat[:-1]) + (roimat.shape[1] - 1) * 7 + 1
    f = figure(facecolor='w', figsize=(14, 7))
    f.subplots_adjust(left=0.05, right=0.98)
    ax = f.add_subplot(111)
    ax.set_title('Input Variables', fontsize=16)
    ax.vlines(xvec[np.array(catlist) == 1], mymin, mymax, 'r', linewidth=2, alpha=0.6)

    for ii in range(roimat.shape[1]):
        ax.plot(xvec, roimat[:, ii] + 7 * ii, 'o-k', lw=2)
        ax.plot(xvec, [7 * ii] * len(xvec), 'grey', lw=2)
        ax.plot(xvec, [7 * ii - 1] * len(xvec), 'grey', ls='--', lw=2)
        ax.plot(xvec, [7 * ii + 1] * len(xvec), 'grey', ls='--', lw=2)
        xcoor, ycoor = 5., 7 * ii + 3
        ax.text(xcoor, ycoor, varnames[ii], color='k', fontsize=15, ha='left', va='bottom', bbox=dict(ec='1', fc='1'))
    
    means = np.arange(0., roimat.shape[1] * 7, 7.)
    ax.set_yticks(np.unique(np.r_[means - 1, means, means + 1]))
    ax.set_yticklabels([-1, 0, 1] * roimat.shape[1])
    ax.set_xlim([xvec.min(), xvec.max()])
    ax.set_ylim([mymin, mymax])
    ax.set_xlabel('ROI ID')
    ax.set_ylabel('Normalised Magnitude')
    
    return f


def plot_distancePDFCDF(roimat, catlist):
    from matplotlib.pyplot import figure, ticklabel_format
    from matplotlib.ticker import ScalarFormatter
    mean_vec = np.mean(np.vstack([mydat for mydat, cat in zip(roimat, catlist) if cat == 1]), axis=0)
    edist = np.linalg.norm(roimat - mean_vec, axis=1)
    edist = (edist - np.min(edist)) / (np.max(edist) - np.min(edist))
    
    mybins = np.linspace(0., 1., 100)
    hist, bins = np.histogram(edist, mybins)
    yhist = hist / float(len(edist))
    xbins = bins[:-1]
    f = figure(facecolor='w', figsize=(7, 4))
    f.subplots_adjust(bottom=0.2, right=0.87, top=0.85)
    
    ax = f.add_subplot(111)
    ax.plot(xbins, yhist, 'k', lw=4)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ticklabel_format(axis='y', style='sci', scilimits=(1, 2))
    
      
    ax2 = ax.twinx()
    ax2.plot(xbins, np.cumsum(yhist), 'grey', lw=4)
    ax2.vlines(edist[np.array(catlist) == 1], -0.001, 1.1, 'r', alpha=0.6, lw=2)
    
    ax.set_ylabel('Probability Density')  
    ax.set_xlabel('distance to mean seizure')
    ax2.set_ylim([-0.001, 1.1])
    ax2.set_ylabel('Cumulative Distribution', rotation=-90, color='grey', labelpad=20)
    
    return f

def get_winnersAndcounts(roimat, catlist, som, catval=1):
    cat_winners = [get_bmu(roimat[ii], som) for ii in range(roimat.shape[0]) if catlist[ii] == catval]
    rect_dict = {}
    for win_rect in cat_winners:
        if not win_rect in list(rect_dict.keys()): 
            rect_dict[win_rect] = cat_winners.count(tuple(win_rect))
    return rect_dict



def plot_som(roimat, catlist, som, ref='none'):
    from matplotlib.pyplot import figure
    
    
    if ref == 'none':
       dm = distance_map(som.K) 
    else:
        dm = get_distance_to_ref(som.K, ref)
        center_bmu = get_bmu(ref, som)


     
    f = figure(facecolor='w')
    ax = f.add_subplot(111)
    # ax.set_title('SOM  (r_fano,r_max,r_mean)')
    
    im = ax.imshow(dm.T, cmap='jet', origin='lower', aspect='auto', interpolation='none')  # distance map as background
    im.set_extent([0, som.kshape[0], 0, som.kshape[1]])
    cbar = f.colorbar(im)
    
    cbar.ax.set_ylabel('Distance', rotation=-90, labelpad=20)
    
    
    seizureWinners = get_winnersAndcounts(roimat, catlist, som, catval=1)
    nosWinners = get_winnersAndcounts(roimat, catlist, som, catval=0)

    newInt = [4, 20]  # mincirc und maxcirc marker
    oldInt = [1, np.max(np.r_[list(seizureWinners.values()), list(nosWinners.values())])]  # 1 to maximum nb of counts
    scaleToInt = lambda x: (newInt[1] - newInt[0]) * (x - oldInt[0]) / (oldInt[1] - oldInt[0]) + newInt[0]

    seizcol, noscol = ['r', 'ForestGreen']
   
    for xx in range(som.kshape[0]):
        for yy in range(som.kshape[1]):
            if (xx, yy) in seizureWinners and (xx, yy) in nosWinners:
                seizsize = scaleToInt(seizureWinners[xx, yy])
                nossize = scaleToInt(nosWinners[xx, yy])
                
                if seizsize > nossize:
                    for catsize, catcol in zip([seizsize, nossize], [seizcol, noscol]):
                        # print 'Catsize,catcol',catsize,catcol
                        ax.plot(xx + 0.5, yy + 0.5, marker='o', markerfacecolor=catcol, \
                     markeredgecolor='None', markersize=catsize)

                elif seizsize < nossize:
                    for catsize, catcol in zip([nossize, seizsize], [noscol, seizcol]):
                        # print 'Catsize,catcol',catsize,catcol
                        ax.plot(xx + 0.5, yy + 0.5, marker='o', markerfacecolor=catcol, \
                     markeredgecolor='None', markersize=catsize)
                elif seizsize == nossize:
                    ax.plot(xx + 0.5, yy + 0.5, marker='o', markerfacecolor='DarkKhaki', \
                     markeredgecolor='None', markersize=seizsize)
                    
            elif (xx, yy) in seizureWinners and (xx, yy) not in nosWinners:
                seizsize = scaleToInt(seizureWinners[xx, yy])
                ax.plot(xx + 0.5, yy + 0.5, marker='o', markerfacecolor=seizcol, \
                     markeredgecolor='None', markersize=seizsize)
            elif (xx, yy) not in seizureWinners and (xx, yy) in nosWinners:
                nossize = scaleToInt(nosWinners[xx, yy])
                ax.plot(xx + 0.5, yy + 0.5, marker='o', markerfacecolor=noscol, \
                     markeredgecolor='None', markersize=nossize)
                # plot only the marker
    biggestSeiz = max(iter(seizureWinners.keys()), key=lambda k: seizureWinners[k])
    ax.text(biggestSeiz[0] + 0.5, biggestSeiz[1] + 0.5, str(seizureWinners[biggestSeiz])\
            , fontsize=12, color='w', fontweight='bold', va='center', ha='center')
    
    if 'center_bmu' in locals():
        ax.plot(center_bmu[0] + .5, center_bmu[1] + 0.5, marker='o', markerfacecolor='None', \
                     markeredgecolor='DarkOrange', markersize=28, mew=5)
        # cx,cy = center_bmu
        # ax.plot([cx,cx+1,cx+1,cx,cx],[cy,cy,cy+1,cy+1,cy],color='DarkOrange',linewidth=4)
    '''
    # use different colors and markers for each label
    for mydat,cat in zip(roimat,catlist):
        w = get_bmu(mydat,som)#the winner
        if cat==1:
            
            
            ax.plot(w[0]+.5,w[1]+.5,marker='o',markerfacecolor='None',\
                 markeredgecolor='DarkOrange',markersize=18,markeredgewidth=2)
        elif cat==0:
            ax.plot(w[0]+.5,w[1]+.5,'*g',markersize=4)
    
    
    ax.plot(center_bmu[0]+.5,center_bmu[1]+.5,'or',markeredgecolor='none',markersize=16) 
    ax.plot(center_bmu[0]+.5,center_bmu[1]+.5,'ow',markeredgecolor='none',markersize=10) 
    ax.plot(center_bmu[0]+.5,center_bmu[1]+.5,'ok',markeredgecolor='none',markersize=6)  
    '''
    ax.set_xticks([int(mytick) for mytick in ax.get_xticks() if np.mod(mytick, 1) == 0])
    ax.set_yticks([int(mytick) for mytick in ax.get_yticks() if np.mod(mytick, 1) == 0])    
    
    myxticks = ax.get_xticks()
    myyticks = ax.get_yticks()
    ax.set_xticks(myxticks + 0.5)
    ax.set_yticks(myyticks + 0.5)
    ax.set_xticklabels(myxticks[:-1])
    ax.set_yticklabels(myyticks[:-1])
    
    ax.axis([0, som.K.shape[0], 0, som.K.shape[1]])
    
    
    return f





#------------------------------------------------------------------------------ 
# DRAW HEXAGONAL MAP
def getHexagon(center=[0., 0.], hw=0.5, fc='k'):
    import matplotlib.pyplot as plt
    
    p2c = lambda rho, phi: [rho * np.cos(np.radians(phi)), rho * np.sin(np.radians(phi))]
    
    phis = np.arange(30., 360., 60.)
    r = hw / np.cos(np.radians(30))
    # print r
    verts = np.vstack([p2c(r, phis[ii]) for ii in range(6)]) + center
    return verts
    # myHexagon = plt.Polygon(verts,fc=fc,ec=fc)
    # return myHexagon

def getHexConn(hw=0.04, center=[0., 0.], hw_hex=0.5, align='m', fc='k'):
    import matplotlib.pyplot as plt
    
    p2c = lambda rho, phi: [rho * np.cos(np.radians(phi)), rho * np.sin(np.radians(phi))]
    
    l1 = 0.5 * hw_hex / np.cos(np.radians(30))
    lx = hw * np.sin(np.radians(30))
    l2 = np.sqrt(hw ** 2 + (l1 - lx) ** 2)
    alpha = np.rad2deg(np.arccos(hw / l2))
    beta = 90. - alpha
    # print alpha,beta
    polars = [[l2, alpha], [l1, 90.], [l2, 90. + beta], [l2, 180. + alpha], [l1, 270.], [l2, 270. + beta]]
    # print polars
    verts = np.vstack([p2c(mypol[0], mypol[1]) for mypol in polars]) + center
    if align == 'm':  # middle
        pass
    elif align == 'r':  # right top
        rotcent = [center[0] + hw_hex, center[1]]
        verts = rotate2D(verts, rotcent, ang=np.radians(-60.))
    elif align == 'l':  # left top
        rotcent = [center[0] - hw_hex, center[1]]
        verts = rotate2D(verts, rotcent, ang=np.radians(60.))
    # print verts
    return verts
    # myConn = plt.Polygon(verts,fc=fc,ec=fc)
    # return myConn

def rotate2D(pts, cnt, ang=np.pi / 4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt


def count_winnersPerHex(roimat, weights):
    
    bmu_vec = np.zeros((weights.shape[1]))
    
    for roi in roimat:
        my_bmu = np.argmin(np.linalg.norm(weights.T - roi, axis=1))
        bmu_vec[my_bmu] += 1
    return bmu_vec.astype('int')

def get_bmus(roimat,weights):
    
    bmus = np.zeros((roimat.shape[0]))
    
    for ii,roi in enumerate(roimat):
        my_bmu = np.argmin(np.linalg.norm(weights.T - roi, axis=1))
        bmus[ii] = my_bmu
    return bmus.astype('int')
    

def get_hexgrid(kshape,hw_hex=0.5):
    
    hw_hex1 = float(hw_hex)
    hw_hex = 0.5
    hexratio = hw_hex1/hw_hex
    
    dx, dy = kshape  # x and y dimension of map
    xcents1 = np.arange(0., dx * (hw_hex) * 2, hw_hex * 2)
    xcents2 = np.r_[xcents1, xcents1 + hw_hex]
    xcents = np.array(list(xcents2) * int((np.floor(dy / 2.))) + int(np.floor(np.mod(dy, 2))) * list(xcents1))
    ydiff = np.sqrt(3) * hw_hex
    ycents = np.array([[ii * ydiff] * dx for ii in range(dy)]).flatten()
    
    # get properties of the hexagons
    hex_dict = {}
    for ii in range(dx * dy):
        center = [xcents[ii], ycents[ii]]
        hex_dict[ii] = [list(np.array(center)*hexratio), getHexagon(center=center, hw=hw_hex)*hexratio]
    
    return hex_dict
    
def get_conngrid(kshape,hw_hex=0.5,hw_conn=0.04):
    
    hw_hex1 = float(hw_hex)
    hw_hex = 0.5
    hexratio = hw_hex1/hw_hex
    dx, dy = kshape
    xcents1 = np.arange(0., dx * (hw_hex) * 2, hw_hex * 2)
    xcents2 = np.r_[xcents1, xcents1 + hw_hex]
    xcents = np.array(list(xcents2) * int(np.floor(dy / 2.)) + int(np.floor(np.mod(dy, 2))) * list(xcents1))
    ydiff = np.sqrt(3) * hw_hex
    ycents = np.array([[ii * ydiff] * dx for ii in range(dy)]).flatten()
    
    #------------------------------------------------------------------------------ 
    # describe connectors in terms of their centers and whether they are at the middle, left or right
    nconns = (dx - 1) * dy + (dy - 1) * ((dx - 1) * 2 + 1)
    modes_pool = ['m'] * (dx - 1) + ['l', 'r'] * (dx - 1) + ['l'] + ['m'] * (dx - 1) + ['r', 'l'] * (dx - 1) + ['r']
    conn_modes = (modes_pool * int(np.ceil(nconns / float(len(modes_pool)))))[:nconns]
    conn_ycents = np.array([[ii * ydiff] * (3 * (dx - 1) + 1) for ii in range(dy)]).flatten()[:nconns] 
    
    row1 = np.arange(hw_hex, hw_hex * 2 * (dx - 1), 2 * hw_hex)
    row2 = np.r_[np.array([[prev] * 2 for prev in row1]).flatten(), row1[-1] + 2 * hw_hex]
    row3 = row1 + hw_hex
    row4 = np.r_[row3[0] - 2 * hw_hex, np.array([[prev] * 2 for prev in row3]).flatten()]
    rows_pool = list(np.r_[row1, row2, row3, row4])
    conn_xcents = np.array((rows_pool * int(np.ceil(nconns / float(len(rows_pool)))))[:nconns])
    
    conn_dict = {}
    for ii in range(nconns):
        cent = [conn_xcents[ii], conn_ycents[ii]]
        mode = conn_modes[ii]
        verts = getHexConn(hw=hw_conn, center=cent, hw_hex=hw_hex, align=mode)
        
        # fill in which pair of hexagons connected by the connector
        if mode == 'm':
            centy1, centy2 = [cent[1]] * 2
            centx1, centx2 = cent[0] - hw_hex, cent[0] + hw_hex
        elif mode == 'l':
            centy1, centy2 = cent[1], cent[1] + hw_hex * np.sqrt(3)
            centx1, centx2 = cent[0] - hw_hex, cent[0]
        elif mode == 'r':
            centy1, centy2 = cent[1], cent[1] + hw_hex * np.sqrt(3)
            centx1, centx2 = cent[0] + hw_hex, cent[0]
            
        p1 = np.where(np.isclose(xcents, centx1) & np.isclose(ycents, centy1))[0][0]
        p2 = np.where(np.isclose(xcents, centx2) & np.isclose(ycents, centy2))[0][0]
        
        conn_dict[ii] = [list(np.array(cent)*hexratio), mode, verts*hexratio, (p1, p2)]
    return conn_dict


def plot_hexsom(kshape, weights, roimat, type='classic', **kwargs):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.pyplot import figure, setp, get_cmap, Polygon, Circle
    from matplotlib import colorbar,ticker
    
    default_dict = {'hexcol':'Greys', 'conncol':'afmhot_r', 'hextick_col':'k', 'conntick_col':'k', \
                    'circcol':'crimson','time_map':'Reds','traj_col':'firebrick','timetick_col':'r'}
    print('type:  ', type)
    for varname in list(default_dict.keys()):
        # print varname
        if varname in kwargs:
            
            exec('%s = "%s"' % (varname, kwargs[varname]))
        else:
            # print varname,default_dict[varname]
            exec('%s = "%s"' % (varname, default_dict[varname]))
    
   
    # how to color the hexagons and define the circle size
    if type == 'classic':
        nos_winners = count_winnersPerHex(roimat[np.array(kwargs['catlist']) == 0], weights)
        seiz_winners = count_winnersPerHex(roimat[np.array(kwargs['catlist']) == 1], weights)
    
    elif type == 'hist':
        nos_winners = count_winnersPerHex(roimat, weights)
    
    elif type == 'trajectory':
        nos_winners = np.zeros((weights.shape[1]))
    
    #------------------------------------------------------------------------------ 
    #hexagonal grid
    hw_hex = 0.5
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    # append data to the hexgrid
    for ii in sorted(hex_dict.keys()):
        hex_dict[ii] += [weights[:, ii],nos_winners[ii]]
    
    #------------------------------------------------------------------------------ 
    #connector grid
    hw_conn = 0.04
    conn_dict =  get_conngrid(kshape,hw_hex=hw_hex,hw_conn=hw_conn)
    # append data to the conngrid
    for jj in sorted(conn_dict.keys()):
        p1,p2 = conn_dict[jj][-1]
        comb_dist = np.linalg.norm(weights[:, p1] - weights[:, p2])  # euclidean distance between the two hexagons the connector connects
        conn_dict[jj]+=[comb_dist]
    
    # connector values (distances) ordered according to connector id, used for scaling the colormap of the connectors
    comb_dists = [conn_dict[key][-1] for key in sorted(conn_dict.keys())]  # collects all euclidean distance between neighbours, used to scale the cmap! 
    
    #extract xcenters and ycenters from hexgrid
    xcents,ycents = np.array([hex_dict[key][0] for key in sorted(hex_dict.keys())]).T

    
    #------------------------------------------------------------------------------ 
    # PLOTTING
    cmap_hex = get_cmap(hexcol) 
    cNorm_hex = colors.Normalize(vmin=np.min(nos_winners), vmax=np.max(nos_winners))
    scalarMap_hex = cmx.ScalarMappable(norm=cNorm_hex, cmap=cmap_hex)
    # print scalarMap.get_clim()
    cmap_conn = get_cmap(conncol) 
    cNorm_conn = colors.Normalize(vmin=np.min(comb_dists), vmax=np.max(comb_dists))
    scalarMap_conn = cmx.ScalarMappable(norm=cNorm_conn, cmap=cmap_conn)
    
    if type == 'trajectory': mycol_h, mycol_c = 'darkgrey', 'grey'
    
    fheight = 8
    xstart, ystart, xlen, ylen, pad, barxlen = 0.11, 0.1, 0.64, 0.8, 0.05, 0.04
    f = figure(figsize=(fheight / float(kshape[1]) * kshape[0] + 0.25 * fheight, fheight), facecolor='w')
    ax = f.add_axes([xstart, ystart, xlen, ylen])
    for hexid, hexvals in list(hex_dict.items()):
        center, verts, weight, nwins = hexvals  
        if type != 'trajectory': mycol_h = scalarMap_hex.to_rgba(nwins)
        ax.add_patch(Polygon(verts, fc=mycol_h, ec=mycol_h, alpha=0.7))
        # ax.text(hexvals[0][0],hexvals[0][1],hexid,color='k',va='center',ha='center')
        
        if 'seiz_winners' in locals():
            if seiz_winners[hexid] > .0:
                circle = Circle(center, float(seiz_winners[hexid]) / np.max(seiz_winners) * 0.7 * hw_hex, \
                                facecolor=circcol, edgecolor=circcol, alpha=1.)    
                ax.add_patch(circle)
                if seiz_winners[hexid] == np.max(seiz_winners):
                    ax.text(center[0], center[1], int(seiz_winners[hexid]), fontsize=14, color='w', fontweight='bold', va='center', ha='center')
                
    for connid, connvals in list(conn_dict.items()):
        center, mode, verts, pair, comb_dist = connvals
        if type != 'trajectory': mycol_c = scalarMap_conn.to_rgba(comb_dist)
        ax.add_patch(Polygon(verts, fc=mycol_c, ec=mycol_c, alpha=1.))
    
    

    # ax.set_xlim([-1.,3.5])
    # ax.set_ylim([-1.,4.5])
    #ax.axis('equal')
    ax.set_yticks(np.unique(ycents)[::2])
    ax.set_yticklabels(np.arange(len(np.unique(ycents)))[::2])
    ax.set_xticks(np.unique(xcents)[::2])
    ax.set_xticklabels(np.arange(len(np.unique(xcents)[::2])))
    
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].axis.axes.tick_params(direction='outward') 
    for sppos in ['top', 'right']:
        ax.spines[sppos].set_color('none')
        ax.spines[sppos].set_visible(False)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.set_xlim([-2*hw_hex,kshape[0]-1+2*hw_hex])
    ax.set_ylim([-3*hw_hex,kshape[1]-1+3*hw_hex])
    ax.axis('equal')
    
    if type != 'trajectory':
        hex_ax = f.add_axes([xstart + xlen + pad, ystart, barxlen, ylen]) 
        hex_cb = colorbar.ColorbarBase(hex_ax, cmap=cmap_hex,
                                           norm=cNorm_hex,
                                           orientation='vertical')
        hex_cb.set_ticks([int(nos_winners.min()), int(nos_winners.max())])
        setp(hex_ax.get_yticklabels(), rotation=-90, color=hextick_col)
        
        
        conn_ax = f.add_axes([xstart + xlen + 2 * pad + barxlen, ystart, barxlen, ylen]) 
        conn_cb = colorbar.ColorbarBase(conn_ax, cmap=cmap_conn,
                                           norm=cNorm_conn,
                                           orientation='vertical')
        setp(conn_ax.get_yticklabels(), rotation=-90, color=conntick_col)
        
        if type == 'classic': hexbar_label = '#(non-seizure hits)'
        elif type == 'hist': hexbar_label = '#(hits)'
        
        f.text(xstart + xlen + pad + 0.5 * barxlen, 0.5, hexbar_label, ha='center', va='center', color='k', rotation=-90., fontweight='bold')
        f.text(xstart + xlen + 2 * pad + barxlen + 0.5 * barxlen, 0.5, 'euclidean distance', ha='center', va='center', color='k', rotation=-90., fontweight='bold')

    elif type=='trajectory':
        jit = 0.3
        
        bmu_seq = get_bmus(roimat,weights)
        bmu_x = [xcents[bmu_id] for bmu_id in bmu_seq]
        bmu_y = [ycents[bmu_id] for bmu_id in bmu_seq]
        
        x_jit = np.random.uniform(-jit, +jit, len(bmu_x))
        y_jit = np.random.uniform(-jit, +jit, len(bmu_y))
        bmu_x,bmu_y = bmu_x+x_jit,bmu_y+y_jit
        ax.plot(bmu_x,bmu_y,color=traj_col,lw=1,alpha=0.7)
        
        cmap_time = get_cmap(time_map)
        cNorm_time = colors.Normalize(vmin=0, vmax=len(bmu_x))
        scalarMap_time = cmx.ScalarMappable(norm=cNorm_time, cmap=cmap_time)
        for tt,[bx,by] in enumerate(zip(bmu_x,bmu_y)):
            mycol = scalarMap_time.to_rgba(tt)
            ax.plot(bx,by,'o',mfc=mycol,mec=mycol,ms=10)
            
        time_ax = f.add_axes([xstart + xlen + pad, ystart, barxlen*2, ylen])  
        time_cb = colorbar.ColorbarBase(time_ax, cmap=cmap_time,
                                           norm=cNorm_time,
                                           orientation='vertical')
        #return time_cb
        time_cb.locator =  ticker.MaxNLocator(integer=True)
        time_cb.update_ticks()
        setp(time_ax.get_yticklabels(), rotation=-90, color=timetick_col)
        f.text(xstart + xlen + pad + barxlen, 0.5,'roi id', ha='center', va='center', color='k', rotation=-90., fontweight='bold',fontsize=17)
        
    
    
    return f


def plot_hexPanel(kshape,hex_values,ax,hw_hex=0.5,showConn=True,showHexId=True,\
                  labelsOn=False,logScale=False,scalefree=False,return_scale=False,**kwargs):
    
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.pyplot import get_cmap, Polygon
    from matplotlib import colorbar,ticker
     

    hexcol = kwargs['hexcol'] if 'hexcol' in kwargs else 'Blues'
    conncol = kwargs['conncol'] if 'conncol' in kwargs else 'afmhot_r'
    hw_conn = kwargs['hw_conn'] if 'hw_conn' in kwargs else 0.04
    idcolor = kwargs['idcolor'] if 'idcolor' in kwargs else 'k'
    alphahex = kwargs['alphahex'] if 'alphahex' in kwargs else 0.7
    conn_ecol = kwargs['conn_ecol'] if 'conn_ecol' in kwargs else 'None'
    
    if logScale:
        hex_values = np.ma.array(np.log10(hex_values),mask=(~np.isfinite(np.log10(hex_values))))

    
    #get grids and fill with values
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    #print len(hex_values)
    #print hex_dict.keys()
    for ii in sorted(hex_dict.keys()): hex_dict[ii] += [hex_values[ii]]
    

    if 'conn_ref' in kwargs:conn_ref = kwargs['conn_ref']
    else: conn_ref = hex_values[None, :]#just take the distance between the hexvalues then!
    
    #print conn_ref.min(),conn_ref.max()
    
    conn_dict =  get_conngrid(kshape,hw_hex=hw_hex,hw_conn=hw_conn)
    # append data to the conngrid
    for jj in sorted(conn_dict.keys()):
        p1,p2 = conn_dict[jj][-1]
        pair_dists = np.linalg.norm(conn_ref[:, p1] - conn_ref[:, p2])  # euclidean distance between the two hexagons the connector connects
        conn_dict[jj]+=[pair_dists]
        
        #pair_dists = np.abs(hex_values[p2]-hex_values[p1])  # euclidean distance between the two hexagons the connector connects
        #conn_dict[jj]+=[pair_dists]

    allpairs = np.array([conn_dict[key][-1] for key in sorted(conn_dict.keys())])
        
    xcents,ycents = np.array([hex_dict[key][0] for key in sorted(hex_dict.keys())]).T
    

    #set up the colormaps and draw 
    if 'vminmax' in kwargs: vmin,vmax = kwargs['vminmax']
    else: vmin,vmax = np.min(hex_values),np.max(hex_values)
    cmap_hex = get_cmap(hexcol) 
    cNorm_hex = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap_hex = cmx.ScalarMappable(norm=cNorm_hex, cmap=cmap_hex)
    
    if type(hex_values) == np.ma.MaskedArray:
        for hexid in np.where(hex_values.mask)[0]:
            center,temp1,temp2 = hex_dict[hexid]
            ax.text(center[0],center[1],'--',fontsize=15,color='k',fontweight='bold',ha='center',\
                    va='bottom')
            
    
    for hexid, hexparams in list(hex_dict.items()):
        center, verts, hexval = hexparams 
        if 'quality_map' in kwargs: mycol_h = kwargs['quality_map'][hexval.astype('int')] 
        else: mycol_h = scalarMap_hex.to_rgba(hexval)
        if 'ec' in kwargs: ec_col = kwargs['ec']
        else: ec_col = mycol_h
        ax.add_patch(Polygon(verts, fc=mycol_h, ec=ec_col, alpha=alphahex))
        if showHexId==True:
            ax.text(center[0]-0.3*hw_hex,center[1]-0.3*hw_hex,str(hexid),fontsize=12,ha='center',va='center',color=idcolor)

    
    if showConn==True:
        cmap_conn = get_cmap(conncol) 
        cNorm_conn = colors.Normalize(vmin=np.min(allpairs), vmax=np.max(allpairs))
        scalarMap_conn = cmx.ScalarMappable(norm=cNorm_conn, cmap=cmap_conn)
        for connid, connparams in list(conn_dict.items()):
            center, mode, verts, pair, pair_dist = connparams
            mycol_c = scalarMap_conn.to_rgba(pair_dist)
            ax.add_patch(Polygon(verts, fc=mycol_c, ec=conn_ecol, alpha=1.))
    
    if not scalefree:
        if labelsOn==True:    
            ax.set_yticks(np.unique(ycents)[::2])
            ax.set_yticklabels(np.arange(len(np.unique(ycents)))[::2])
            ax.set_xticks(np.unique(xcents)[::2])
            ax.set_xticklabels(np.arange(len(np.unique(xcents)[::2])))
       
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].axis.axes.tick_params(direction='outward') 
            for sppos in ['top', 'right']:
                ax.spines[sppos].set_color('none')
                ax.spines[sppos].set_visible(False)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        
        else: ax.set_axis_off()
    
        ax.set_xlim([-1.5*hw_hex,kshape[0]-1+1.5*hw_hex])
        ax.set_ylim([-1.5*hw_hex,kshape[1]-1+1.5*hw_hex])
        
        ax.set_aspect('equal',adjustable='datalim',anchor='SW')
        
        
        #ax.set_xlim([-1.5*hw_hex,kshape[0]-1+1.5*hw_hex])
        #ax.set_ylim([-1.5*hw_hex,kshape[1]-1+1.5*hw_hex])
        
        #ax.set_xlim([-2*hw_hex,kshape[0]-1+2*hw_hex])
        #ax.set_ylim([-3*hw_hex,kshape[1]-1+3*hw_hex])
        
    
    
    if 'hex_ax' in kwargs:
        hex_ax = kwargs['hex_ax']
        hex_cb = colorbar.ColorbarBase(hex_ax, cmap=cmap_hex,
                                           norm=cNorm_hex,
                                           orientation='vertical')
        #hex_cb.set_ticks([])
        hightext,lowtext = str(np.around(hex_values.max(),1)),str(np.around(hex_values.min(),1))
        if logScale: hightext,lowtext='10^%s'%(hightext),'10^%s'%(lowtext)
        hex_ax.text(0.5,0.9,hightext,rotation=-90,fontsize=15,color='w',ha='center',va='center',transform = hex_ax.transAxes)
        hex_ax.text(0.5,0.1,lowtext,rotation=-90,fontsize=15,color='k',ha='center',va='center',transform = hex_ax.transAxes)
        hex_ax.set_axis_off()
        #hex_cb.set_ticks([np.around(hex_values.min(),2), np.around(hex_values.max(),2)])
        #setp(hex_ax.get_yticklabels(), rotation=-90, color='k')
    
    if 'conn_ax' in kwargs:
        conn_ax = kwargs['conn_ax']
        
        conn_cb = colorbar.ColorbarBase(conn_ax, cmap=cmap_conn,
                                           norm=cNorm_conn,
                                           orientation='vertical')
        #conn_cb.set_ticks([])
        conn_ax.text(0.5,0.9,str(np.around(allpairs.max(),1)),rotation=-90,fontsize=15,color='w',ha='center',va='center',transform = conn_ax.transAxes)
        conn_ax.text(0.5,0.1,str(np.around(allpairs.min(),1)),rotation=-90,fontsize=15,color='k',ha='center',va='center',transform = conn_ax.transAxes)
        conn_ax.set_axis_off()
        #setp(conn_ax.get_yticklabels(), rotation=-90, color='k')
    if return_scale:
        return {'hexint':[hex_values.min(),hex_values.max()],'connint':[allpairs.min(),allpairs.max()]}


def plot_addCircles(ax,kshape,circlevalues,hw_hex = 0.5,scalefac=0.7,color='firebrick',textOn=True):
    from matplotlib.pyplot import Circle
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    for hexid, hexparams in list(hex_dict.items()):
        center, verts = hexparams  
        if circlevalues[hexid] > .0:
            circle = Circle(center, float(circlevalues[hexid]) / np.max(circlevalues) * scalefac * hw_hex, \
                            facecolor=color, edgecolor=color, alpha=1.)    
            ax.add_patch(circle)
            if circlevalues[hexid] == np.max(circlevalues):
                if textOn:ax.text(center[0], center[1], int(circlevalues[hexid]), fontsize=14, color='w', fontweight='bold', va='center', ha='center')

def plot_addText(ax,kshape,textvalues,hw_hex = 0.5,**kwargs):
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    textstrs = [str(val) for val in textvalues]
    for hexid, hexparams in list(hex_dict.items()):
        center, verts = hexparams  
        if textstrs[hexid] !='':
            #print textstrs[hexid]
            #print center
            ax.text(center[0], center[1], textstrs[hexid],va='center', ha='center',\
                    **kwargs)
                        

def plot_addTrajectory(ax,kshape,bmu_seq,hw_hex=0.5,traj_col='k',time_map='Blues'):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.pyplot import get_cmap, Polygon
    from matplotlib import colorbar,ticker
    
    jit = 0.3
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex) 
    xcents,ycents = np.array([hex_dict[key][0] for key in sorted(hex_dict.keys())]).T  
    
    bmu_x = [xcents[bmu_id] for bmu_id in bmu_seq]
    bmu_y = [ycents[bmu_id] for bmu_id in bmu_seq]
    
    x_jit = np.random.uniform(-jit, +jit, len(bmu_x))
    y_jit = np.random.uniform(-jit, +jit, len(bmu_y))
    bmu_x,bmu_y = bmu_x+x_jit,bmu_y+y_jit
    ax.plot(bmu_x,bmu_y,color=traj_col,lw=1,alpha=0.7)
    
    cmap_time = get_cmap(time_map)
    cNorm_time = colors.Normalize(vmin=0, vmax=len(bmu_x))
    scalarMap_time = cmx.ScalarMappable(norm=cNorm_time, cmap=cmap_time)
    for tt,[bx,by] in enumerate(zip(bmu_x,bmu_y)):
        mycol = scalarMap_time.to_rgba(tt)
        ax.plot(bx,by,'o',mfc=mycol,mec=mycol,ms=10)


def plot_addScatter(ax,kshape,bmu_seq,hw_hex=0.5,marker='o',color='k',alpha=1.,ms=10):
    
    jit = 0.3
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex) 
    xcents,ycents = np.array([hex_dict[key][0] for key in sorted(hex_dict.keys())]).T  
    
    bmu_x = [xcents[bmu_id] for bmu_id in bmu_seq]
    bmu_y = [ycents[bmu_id] for bmu_id in bmu_seq]
    
    x_jit = np.random.uniform(-jit, +jit, len(bmu_x))
    y_jit = np.random.uniform(-jit, +jit, len(bmu_y))
    bmu_x,bmu_y = bmu_x+x_jit,bmu_y+y_jit

    for bx,by in zip(bmu_x,bmu_y):ax.plot(bx,by,ls='',marker=marker,mfc=color,mec=color,ms=ms,alpha=alpha)


def plot_componentPlanes(kshape,weights,**kwargs): 
    from matplotlib.pyplot import figure
    import matplotlib.gridspec as gridspec
    
    nvars = weights.shape[0]    
    
    means = kwargs['means'] if 'means' in kwargs else 0
    stds = kwargs['stds'] if 'stds' in kwargs else 1
    nrows,ncols = kwargs['rc_num'] if 'rc_num' in kwargs else [weights.shape[0],1]
    varweights = kwargs['varweights'] if 'varweights' in kwargs else np.ones((nvars))
    
    #fill values into hex_dict!
    
    #print nvars
    #print means
    #print stds
    weight_mat = np.array([weights[ii]/varweights[ii]*stds[ii]+means[ii] for ii in range(nvars)])
    
    
    #set proper figure dimensions establish a nice grid
    fh_unit = 6
    fpadh_unit = 0.1*fh_unit
    fw_unit = fh_unit/float(kshape[1])*kshape[0]
    bar_unit = fw_unit/5.
    barpad_unit = fw_unit/10.
    fpadw_unit = fw_unit/4.
    
    fh = nrows*fh_unit + (nrows*-1)*fpadh_unit
    fw = ncols*(fw_unit+barpad_unit+bar_unit)+(ncols-1)*fpadw_unit
    f = figure(facecolor='w',figsize = (fw,fh))
    f.subplots_adjust(left = 0.04,right=0.96,wspace=0.2)
    
    #make a grid with subgrids
    gs0 = gridspec.GridSpec(nrows, ncols)#the super-grid
    for vv in range(nvars):
        
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[vv],width_ratios=[10,1,1])
        ax = f.add_subplot(gs00[0])
        if 'varnames' in kwargs: ax.set_title(kwargs['varnames'][vv],fontsize=16)
        hex_ax = f.add_subplot(gs00[1])
        conn_ax = f.add_subplot(gs00[2])
        hex_values = weight_mat[vv]
        if vv==0: hexstr = False
        else: hexstr = False
        plot_hexPanel(kshape,hex_values,ax,conn_ax=conn_ax,hex_ax=hex_ax,showHexId=hexstr)
    return f


def getInitialMap_pca(data,msize):
    from sklearn.decomposition import PCA
    '''
    msize is grid dimensions, [rows,columns]
    ''' 
    #data = roimat[:]
    #msize = kshape[::-1]
    
    rows = msize[0]
    cols = msize[1]
    nnodes = np.prod(msize)
    coord = np.zeros((nnodes, 2))
    for i in range(0,nnodes):
        coord[i,0] = int(i/cols) #x
        coord[i,1] = int(i%cols) #y
    mx = np.max(coord, axis = 0)
    mn = np.min(coord, axis = 0)
    coord = (coord - mn)/(mx-mn)#normalise each direction between 0 and 1
    coord = (coord - .5)*2 # now from -1 to 1
    
    me = np.mean(data, 0)#calculate the mean of the data
    data = (data - me)
    codebook = np.tile(me, (nnodes,1))
    pca = PCA(n_components=2, svd_solver='randomized', whiten=True) #Randomized PCA is scalable
    #pca = PCA(n_components=2)
    pca.fit(data)
    eigvec = pca.components_
    eigval = pca.explained_variance_
    norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
    eigvec = ((eigvec.T/norms)*eigval).T; eigvec.shape
    for j in range(nnodes):
        for i in range(eigvec.shape[0]):
            codebook[j,:] = codebook[j, :] + coord[j,i]*eigvec[i,:]
            
    initcode = np.around(codebook, decimals = 6)
    return initcode.T

def get_quantisationError(roimat,weights):
    error = 0
    bmus = get_bmus(roimat,weights)
    for bmu,roi in zip(bmus,roimat): error += np.linalg.norm(roi-weights[:,bmu])
    return error/roimat.shape[0]

def get_topologicalError(roimat,weights,kshape):
    
    #make a nearest neighbour dictionary
    dx, dy = kshape  # x and y dimension of map
    neighdeg = 2#implicitly using a hw == 1, giving a distance of 2 for nearest neighbour
    xcents1 = np.arange(0., dx * 2,  2)
    xcents2 = np.r_[xcents1, xcents1 + 1.]
    xcents = np.array(list(xcents2) * int((np.floor(dy / 2.))) + int(np.floor(np.mod(dy, 2))) * list(xcents1))
    ydiff = np.sqrt(3) 
    ycents = np.array([[ii * ydiff] * dx for ii in range(dy)]).flatten()
    coords = np.transpose(np.vstack([xcents,ycents]))
    neighbour_dict = {}
    for ii,coord in enumerate(coords):
        distances = np.linalg.norm(coords-coord,axis=1)
        neighbour_dict[ii] =  np.where(np.isclose(distances,neighdeg))[0]

    error = 0.
    for roi in roimat:
        bmu1,bmu2 = np.argsort(np.linalg.norm(weights.T - roi, axis=1))[:2]
        if not bmu2 in neighbour_dict[bmu1]: error+=1.
    return error/roimat.shape[0]

def plot_mapQuality(iter_dict):
    from matplotlib.pyplot import figure
    f = figure(facecolor='w',figsize=(15,8))
    f.subplots_adjust(left = 0.06,right=0.92)
    nrows = 2
    ncols = 3
    qe_col = 'navy'
    te_col = 'firebrick'
    
    for ii,iter in enumerate(sorted(iter_dict.keys())):
        x_vals = np.array(sorted(iter_dict[iter].keys()))
        qe_mean,te_mean = np.array([ np.mean(iter_dict[iter][x_val],axis=0) for x_val in x_vals]).T
        qe_std,te_std = np.array([ np.std(iter_dict[iter][x_val],axis=0) for x_val in x_vals]).T
        
        ax = f.add_subplot(nrows,ncols,ii+1)
        ax.text(0.5,0.9,'niter '+str(iter),ha='center',fontsize=16,fontweight='bold',color='grey',transform = ax.transAxes)
        for jj,[err_mean,err_std,err_col] in enumerate(zip([qe_mean,te_mean],[qe_std,te_std],[qe_col,te_col])):
            if jj==1: ax = ax.twinx()
            ax.plot(x_vals,err_mean,'o-',color=err_col,mec=err_col,lw=2)
            ax.fill_between(x_vals,err_mean,err_mean+err_std,color=err_col,alpha=0.4)
            ax.fill_between(x_vals,err_mean,err_mean-err_std,color=err_col,alpha=0.4)
            if jj==0: 
                ax.set_ylim([0.2,0.7])
                if ii==0 or ii==3: 
                    ax.set_ylabel ('Quantisation Error',color=qe_col)
                    ax.tick_params(axis='y', colors=qe_col)
                else:ax.set_yticks([])
            else: 
                ax.set_ylim([0.,0.1])
                if ii==2 or ii==5: 
                    ax.set_ylabel ('Topological Error',color=te_col,rotation=-90,labelpad=20)
                    ax.tick_params(axis='y', colors=te_col)
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(1,2))
                else:ax.set_yticks([])
            if ii>2:ax.set_xlabel('#(neurons)')
            else: ax.set_xticks([])
        ax.set_xlim([x_vals.min(),x_vals.max()])
    #then all panels in a figure
    return f




def get_clusterColors(nclust):    
    all_colors = np.array(['firebrick','darkorange','olivedrab','steelblue','dimgrey','darkgrey'])
    
    if nclust ==2: colors = all_colors[np.array([0,-1])]
    elif nclust==3: colors = all_colors[np.array([0,1,-1])]
    elif nclust==4: colors = all_colors[np.array([0,1,2,-1])]
    elif nclust==5: colors = all_colors[np.array([0,1,2,-2,-1])]
    elif nclust==6: colors = all_colors[np.array([0,1,2,-3,-2,-1])]
    return colors

def cluster_hierarchy(weights,nclust=2,cmethod='complete',pos=None,**kwargs):
    import scipy.cluster.hierarchy as hac
    
    colors = kwargs['colors'] if 'colors' in kwargs else get_clusterColors(nclust)
    
    dists = hac.distance.pdist(weights.T)
    if cmethod == 'ward':z = hac.linkage(weights.T,method='ward')
    else:z = hac.linkage(dists, method=cmethod)
    parts = hac.fcluster(z, nclust, 'maxclust')
    
    if 'ref_hex' in kwargs:
        clustermeans = np.array([np.mean(weights[:,parts==cln],axis=1) for cln in np.unique(parts)]).T
        dist = clustermeans-weights[:,kwargs['ref_hex']][:,None]
        edist = np.sqrt(np.sum(dist**2,axis=0))
        clust_sorted = np.hstack([np.arange(nclust)[np.argsort(edist)+1==cc] for cc in parts])+1
        
    if 'ax' in kwargs :
        colors_swap = colors[edist.argsort().argsort()]
        ax = kwargs['ax']
        d = hac.dendrogram(z,no_plot=True)
        dcoord,icoord = np.array(d['dcoord']),np.array(d['icoord'])
        color_list = []
        label_colors = []
        for part,mycol in zip(np.unique(parts),colors_swap):
            color_list += [mycol]*(np.bincount(parts)[part]-1)#-1
            label_colors += [mycol]*np.bincount(parts)[part]#for coloring xticklabels
        color_list += ['k']*(len(dcoord)-len(color_list))
        
        xmin, xmax = icoord.min(), icoord.max()
        ymin, ymax = dcoord.min(), dcoord.max()
        
        if pos:
            icoord = icoord[pos]
            dcoord = dcoord[pos]
            color_list = color_list[pos]
        
        
        for xs, ys, color in zip(icoord, dcoord, color_list):
            #print xs,ys,color
            ax.plot(xs, ys,  color,lw=3)
        ax.set_xlim( xmin-10, xmax + 10 )
        ax.set_ylim( ymin, ymax + 0.1*np.abs(ymax) )
        ax.set_xticks( np.arange(5,len(d['leaves'])*10,10))
        ax.set_xticklabels(d['leaves'],fontsize=15,rotation=-30,ha='left')
        for xtick, color in zip(ax.get_xticklabels(),label_colors):xtick.set_color(color)        #ax.tick_params(axis='x',direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].axis.axes.tick_params(direction='outward') 
        for sppos in ['top', 'right']:
            ax.spines[sppos].set_color('none')
            ax.spines[sppos].set_visible(False)
        ax.set_ylabel('Distance')
        ax.set_xlabel('Hexagon ID')
        
    if 'ref_hex' in kwargs:return clust_sorted,colors#0 will then be the highest cluster
    else:return parts,colors

def sort_clustersToRef(weights,parts,ref_idx):
    nclust = len(np.unique(parts))
    cmeans = np.array([np.mean(weights[:,parts==cln],axis=1) for cln in range(nclust)]).T
    dist = cmeans-weights[:,ref_idx][:,None]
    edist = np.sqrt(np.sum(dist**2,axis=0))
    rankclusts = np.hstack([np.arange(nclust)[np.argsort(edist)==cc] for cc in parts])
    return rankclusts
    
def plot_fancyDendrogram(weights,nclust=4,return_verbose = False,**kwargs):
    '''Truncates dendrogram cluster-wise: useful for a representation in limited space
    ref_hex: cluster ranks will be interpreted as mean distance of cluster to ref_hex'''
    
    import scipy.cluster.hierarchy as hac
    from matplotlib.ticker import MaxNLocator
    from matplotlib.pyplot import figure,axhline,show
    
    triangConn = lambda x,y: [np.r_[x[0],(x[3]-x[0])/2.+x[0],x[3]],y[[0,1,3]]]
    
    ref_hex = kwargs['ref_hex'] if 'ref_hex' in kwargs else 0
    rank_colors = kwargs['rank_colors'] if 'rank_colors' in kwargs else {cc:col for cc,col in enumerate(get_clusterColors(nclust))}    
    rank_names = kwargs['rank_names'] if 'rank_names' in kwargs else {cc:chr(numb).upper() for cc,numb in enumerate(np.arange(97,123)[:nclust])}
    cmethod = kwargs['cmethod'] if 'cmethod' in kwargs else 'ward'
    seiz_winners = kwargs['seiz_winners'] if 'seiz_winners' in kwargs else np.zeros(weights.shape[1])
    rank_colors['conn'] = kwargs['conncol'] if 'conncol' in kwargs else 'k'
    nbins = kwargs['nbins'] if 'nbins' in kwargs else 4 #for distance display on y-axis
    specialleaves = kwargs['specialleaves'] if 'specialleaves' in kwargs else []
    specialcol = kwargs['specialcol'] if 'specialcol' in kwargs else 'm'
    #make a simple dendrogram
    
    z = hac.linkage(weights.T,method=cmethod)

    parts = hac.fcluster(z, nclust, 'maxclust')-1
    ct=z[-(nclust-1),2]#calculate cutoff-threshold from number of clusters  

    d = hac.dendrogram(z,color_threshold=ct,no_plot=True)#,p=20,truncate_mode='lastp',no_plot=False
    #d = hac.dendrogram(z,no_plot=False)

    print(ct)
    #unpack dendrogram dictionary
    leaves = np.array(d['leaves'])
    icoords = np.array( d['icoord'] )
    dcoords = np.array( d['dcoord'] )
    colarray = np.array(d['color_list'])
    
    #check which cluster from parts has which rank
    rankclusts = sort_clustersToRef(weights,parts,ref_hex)
    
    return_dict = {}
    if return_verbose:        
        return_dict['rank_colors'] = rank_colors
        return_dict['clusterids'] = rankclusts
     
    #identify which original leave-color would correspond to which rank
    lclusts = rankclusts[leaves]
    u,ind = np.unique(lclusts,return_index=True)
    uclusts = u[np.argsort(ind)]#ranking order clusters appear on dendrogram
    u,ind = np.unique([col for col in colarray if not col=='b'],return_index=True)
    ucols = u[np.argsort(ind)]

   
    coldict = {uclusts[ii]:ucols[ii] for ii in range(nclust)}#rank_colors only for original dendrogram colors
    coldict ['conn'] = 'b'#non-leafy connector is blue by default

    
    
    if 'axlist' not in kwargs:
        f = figure(facecolor='w')
        axlist = [f.add_subplot(nclust+1,1,aa+1)for aa in range(nclust+1)]
        return_dict['fh'] = f
    else: 
        axlist = kwargs['axlist']


    for ii,rank in enumerate(['conn']+list(np.arange(nclust))):
        oldcol = coldict[rank]
        newcol = rank_colors[rank]
        myIs,myDs = icoords[colarray == oldcol],dcoords[colarray == oldcol]
        ax = axlist[ii]
        ax.patch.set_alpha(0.)
        for x,y in zip(myIs,myDs):
            
            x,y = triangConn(x,y)
            ax.plot(x,y,newcol,linewidth=2)
        if rank == 'conn':    
            loose_endsX = np.array([myIs[ll][idx] for ll in range(myDs.shape[0]) for idx in [0,-1]  if len(myDs.flatten()[myDs.flatten()==myDs[ll][idx]])==1])
            linds = np.argsort(loose_endsX)
            loose_endsX = loose_endsX[linds]
            loose_endsY = np.array([myDs[ll][idx] for ll in range(myDs.shape[0]) for idx in [0,-1]  if len(myDs.flatten()[myDs.flatten()==myDs[ll][idx]])==1])[linds]
            
            #ax.scatter(loose_endsX,loose_endsY,c=[rank_colors[cc] for cc in uclusts],s=40.,marker='o',edgecolors='face',zorder=10,alpha=1.)
            #ax.set_ylabel('Distance',fontsize=13,fontweight='normal')  
            for cc,[lx,ly] in enumerate(zip(loose_endsX,loose_endsY)):
                mycol = rank_colors[uclusts[cc]]
                ax.text(lx,ly,rank_names[uclusts[cc]],color='w',fontsize=11,fontweight='bold',va='top',ha='center',\
                         bbox=dict(boxstyle="round", ec =mycol,fc=mycol,alpha = 0.8),zorder=0)
            ax.set_xlim([loose_endsX.min()-100,loose_endsX.max()+100])
            
        else:
            leavesX = np.sort(myIs.flatten()[myDs.flatten()==0])
            leavesNames = leaves[rankclusts[leaves]==rank]
            for leafpos,leafname in zip(leavesX,leavesNames):
                namestr,bbox = str(leafname),None
                if leafname in np.where(seiz_winners>0)[0]: 
                    namestr = namestr+'_'+str(seiz_winners[leafname])
                if leafname == ref_hex: bbox = dict(boxstyle="round", ec ='k',fc='w')
                if leafname in specialleaves: leafcol = specialcol
                else: leafcol = newcol
                ax.text(leafpos,0.,namestr,va='top',ha='center',color=leafcol,fontsize=10,rotation=-70,bbox=bbox)
            ax.set_xlim([leavesX.min()-10,leavesX.max()+10])
    
        
        for pos in ['top','right','bottom']:ax.spines[pos].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(tickdir='out', labelsize=12,width=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins,integer=True))
        ax.set_xticks([]) 
       
    if return_verbose:
        return return_dict


def copy_roisOfHex(hexids,roiids,roimat,weights,fpath_dest,figformat='.png'):
    import shutil
    import os
    import load_data,mod_config 
    
    load_data.createPath(os.path.join(fpath_dest,'hexid_rois'))
    
    bmus = get_bmus(roimat,weights)
    for hex_id in hexids:
        load_data.createPath(os.path.join(fpath_dest,'hexid_rois',str(hex_id)))
        
        hexrois = [roiid for ii,roiid in enumerate(roiids) if bmus[ii]==hex_id]
        
        for roi_id in hexrois:
            print('Copying ',roi_id, 'to ', hex_id)
            my_id = ('_').join(roi_id.split('_')[:-1])
            if my_id.count('_') == 2:
                animal,loc,day = my_id.split('_')
                genpath = os.path.join(animal,animal+'_'+loc,my_id)
            elif my_id.count('_') == 3:
                animal,loc,day,sess = my_id.split('_')
                genpath = os.path.join(animal,animal+'_'+loc,animal+'_'+loc+'_'+day,my_id)
                
            figpath_source = os.path.join(mod_config.fig_directory,genpath,my_id+'__bliprois'\
                                          , roi_id+figformat)
            figpath_dest = os.path.join(fpath_dest,'hexid_rois',str(hex_id),roi_id+figformat)
            shutil.copyfile(figpath_source,figpath_dest)
    return 0


def plot_roisOfHex(hex_id,roidict,roiids,roimat,weights,fpath_gen,**kwargs):
    import load_data,mod_config
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    figformat = kwargs['figformat'] if 'figformat' in kwargs else '.png'
    margL = kwargs['margL'] if 'margL' in kwargs else 5.
    margR = kwargs['margR'] if 'margR' in kwargs else 10.#minimal duration of datadisplay after burst
    sr = kwargs['sr'] if 'sr' in kwargs else 500.
    max_snip = kwargs['max_snip'] if 'max_snip' in kwargs else 40.
    longdur = kwargs['longdur'] if 'longdur' in kwargs else 400.
    suptag = kwargs['suptag'] if 'suptag' in kwargs else ''
    
    color_list = ['k','grey','steelblue','darkgreen','darkviolet']
    max_yticks = 3
    
    
    hexpath = os.path.join(fpath_gen,'hexid_rois',str(hex_id))
    load_data.createPath(os.path.join(fpath_gen,'hexid_rois'))
    load_data.createPath(hexpath)
    bmus = get_bmus(roimat,weights)
    #all rois that are located in hex_id
    hexrois = [roiid for ii,roiid in enumerate(roiids) if bmus[ii]==hex_id]
    
    #find ids that have hexrois somewhere in them
    my_IDS = sorted(list(set(['_'.join(hexroi.split('_')[:-1]) for hexroi in hexrois])))
    
    for my_id in my_IDS:    
        print(my_id)
        animal,my_loc = my_id.split('_')[:2]
        animal_path = os.path.join(mod_config.home_directory,animal)
        
        chanlist = [subdir.split('_')[-1] for subdir in os.listdir(animal_path) \
                                 if os.path.isdir(os.path.join(animal_path,subdir))]
        chanlist.remove(my_loc)
        
        
        data_list,bt_list,tag_list = [],[],[]
        for cc,loc in enumerate([my_loc]+chanlist):    
        
            temp_id = my_id.replace(my_loc,loc)
            data,tvec = load_data.get_IDdata(temp_id,mod_config.home_directory,t_offset=0.,sr=sr)
            btimes = load_data.get_bdict(temp_id,mod_config.home_directory,appendix = '__blipdict_clean')['btimes']
            if cc==0:tref = tvec[:]
            if len(tvec)==len(tref):#sometimes tvecs are different for different positions        
                data_list.append(data)
                tag_list.append(loc)
                bt_list.append(btimes)
        
        
        #find rois that are present in exactly this recording
        my_rois = [roi for roi in hexrois if roi.count(my_id)]
        
        n_chans = len(data_list)
        
        #now plot each roi: 1) in an overview, then a 
        for my_roi in my_rois:
            #determine how many subplots are needed --> figuredimensions
            start,stop = roidict[my_roi]['roi_int']
            snipdur = (stop-start)+margL+margR
            nsnips = int(np.ceil(snipdur/max_snip))
            n_panels = 1+n_chans*nsnips
            
            long_pre,long_post = (longdur-snipdur)*0.3,(longdur-snipdur)*0.7
            lstart,lstop = (start-long_pre)*sr,(stop+long_post)*sr
            sstart,sstop = (start-margL)*sr,(stop+margR)*sr
            
            #break up the datatrace into snippets
            
            snipstarts = np.array([sstart+max_snip*sr*ii for ii in range(nsnips)])
            snipstops = np.r_[snipstarts[1:],snipstarts[-1]+max_snip*sr]
            #durratio = ((snipstops-snipstarts)/(max_snip*sr))[-1]
            
            
            f = plt.figure(figsize=(16,4+0.6*n_panels),facecolor='w')
            f.suptitle(my_roi,fontsize=15,y=0.98)
            f.subplots_adjust(left=0.05,right=0.98,top=0.92,bottom=0.12,hspace=0.1)
            gs0 = gridspec.GridSpec(nsnips+1, 1, hspace=0.3,height_ratios=[0.7]+[1]*nsnips)
            
            #the overview
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
            longax = f.add_subplot(gs00[0])
            longax.plot(tref[lstart:lstop],data_list[0][lstart:lstop],color='k')
            longax.set_xlim([lstart/sr,lstop/sr])
            longax.axvspan(stop, stop+lstop/sr, facecolor='khaki', alpha=0.5)
            longax.axvspan(start-lstart/sr, start, facecolor='khaki', alpha=0.5)
            longax.set_ylabel('mV',fontsize=15,labelpad=-2)
            
            #the detailed view
            for ss in range(nsnips):
                gs01 = gridspec.GridSpecFromSubplotSpec(n_chans, 1, subplot_spec=gs0[ss+1],hspace=0.)
                
                for ch in range(n_chans):
                    bscale = np.std(data_list[ch][sstart:sstop])
                    btimes = bt_list[ch]
                    ax = f.add_subplot(gs01[ch])
            
                      
                    ax.plot(tref[snipstarts[ss]:snipstops[ss]],data_list[ch][snipstarts[ss]:snipstops[ss]],\
                            color=color_list[ch],label=tag_list[ch])
                    ax.vlines(btimes,[5*bscale]*len(btimes),[8*bscale]*len(btimes),color='firebrick',linewidth=2)
                    if ch==0:
                       ax.axvspan(start-margL, start, facecolor='khaki', alpha=0.5)
                       ax.axvspan(stop, snipstops[-1]*sr, facecolor='khaki', alpha=0.5) 
                    if ss==0:
                        ax.legend()   
                    ax.set_xlim([snipstarts[ss]/sr,snipstops[ss]/sr]) 
                    #ax.set_ylim()
                    #ax.set_ylim([np.min(data_list[ch][sstart:sstop]),np.max(data_list[ch][sstart:sstop])])
                    if ch==n_chans-1 and ss==nsnips-1: ax.set_xlabel('Time [s]',fontsize=15)
            for ax in f.get_axes():ax.yaxis.set_major_locator(plt.MaxNLocator(max_yticks))       
                #ax.plot(tref[start*sr:stop*sr],data_list[0][start*sr:stop*sr])
            f.text(0.02,0.95,suptag,fontsize=14)
            f.savefig(os.path.join(hexpath,my_roi+figformat))
            plt.close(f)
    

def plot_SIdistribution(seizidx,bmus,**kwargs):
    '''len(seizidx) == np.prod(kshape)
    len(bmus) == number of rois
    '''
    from matplotlib.pyplot import figure
    
    clusterids = kwargs['clusterids'] if 'clusterids' in kwargs else np.zeros((len(seizidx)))
    dcolors = kwargs['dcolors'] if 'dcolors' in kwargs else ['k']*len(np.unique(clusterids))
    
    x = seizidx[:]
    y = np.array([np.sum(seizidx[bmus]==si) for si in x])
    f = figure(figsize=(5,4),facecolor='w')
    f.subplots_adjust(left=0.17,bottom=0.15)
    ax = f.add_subplot(111)
    ax.vlines(x,[0]*len(x),[y.max()+500]*len(x),color=np.array(dcolors)[clusterids.astype('int')])
    ax.vlines(x[y==0],[0]*len(x[y==0]),[y.max()+500]*len(x[y==0]),color='w',lw=2,alpha=0.3)
    ax.scatter(x[y==0],len(x[y==0])*[0.9],marker='v',s=100,\
               c=np.array(dcolors)[clusterids[y==0].astype('int')],zorder=10)
    #ax.vlines(x,[0]*len(x),y,color=np.array(dcolors)[hexvalues.astype('int')],lw=2)
    ax.scatter(x,y,s=100,c=np.array(dcolors)[clusterids.astype('int')],zorder=5,marker='h')
    ax.set_yscale('log',nonposy='clip')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([0.8,y.max()+500])
    ax.set_xlabel('Seizure Index',fontweight='normal')
    ax.set_ylabel('Counts',fontweight='normal')
    return f


def plotExample_intSI(si_int,class_dict,**kwargs):

    from load_data import get_IDdata,get_bdict
    from matplotlib.pyplot import figure
    import matplotlib.gridspec as gridspec
    from mod_config import home_directory
    
    margL = kwargs['margL'] if 'margL' in kwargs else 2.
    margR = kwargs['margR'] if 'margR' in kwargs else 2.
    n_panels = kwargs['n_panels'] if 'n_panels' in kwargs else 5
    snipdur = kwargs['snipdur'] if 'snipdur' in kwargs else 40.
    dcolors = kwargs['dcolors'] if 'dcolors' in kwargs else ['w']*10
    


    params = class_dict['params']
    
    #RANDOMLY SELECT ROIS within this interval 
    roiids = np.array([key for key in sorted(class_dict.keys()) if not key=='params'])
    
    sidx = np.array([class_dict[roi][params.index('seizidx')] for roi in roiids])
    my_rois = roiids[(sidx<=si_int[1]) & (sidx>=si_int[0])]#rois that have an SI in the desired range
    if 'nMax_ids' in kwargs:
        nids = int(kwargs['nMax_ids'])
        allids = np.array(['_'.join(roi.split('_')[:-1]) for roi in my_rois])
        idcounts = np.array([np.sum(allids==this_id) for this_id in np.unique(allids)])
        inds = np.argsort(idcounts)
        mysel_ids = np.unique(allids)[inds][-nids:]
        my_rois = np.array([roi for roi in my_rois if '_'.join(roi.split('_')[:-1]) in mysel_ids])
        nmaxids = str(nids)
    else: nmaxids = 'open'
   
    
    if (len(my_rois))==0:
        
        f = figure(figsize=(16,10),facecolor='w')
        f.suptitle('Examples from SI-Interval [%1.2f, %1.2f]'%(si_int[0],si_int[1]),fontsize=15,y=0.98)
        f.text(0.5,0.5,'No ROIs available for this interval.',fontsize=22,fontweight='bold',ha='center',\
               va='center')
        return f
    np.random.shuffle(my_rois)
    durs = np.hstack(np.diff([class_dict[roi][params.index('roi_int')] for roi in my_rois]))
    nrois = np.where(np.cumsum([durs[ii]+margL+margR for ii in range(len(my_rois))])\
                     <=n_panels*snipdur)[0][-1]+1#starting at the beginning of the permuted,how many rois can fit on the figure (n_panels*snipdur)
    rois = my_rois[:nrois]

    #COLLECT DATA
    data_dict = {}
    my_IDS = np.unique(['_'.join(roi.split('_')[:-1]) for roi in rois])#only open each trace once, in case it has several
    #rois randomly selected
    for my_id in my_IDS:
        print('opening to extract rois',my_id)
        
        rois_id = sorted([roi for roi in rois if roi.count(my_id)])#rois that are part of the recording 'my_id'
        roitimes = np.vstack([class_dict[roi][params.index('roi_int')] for roi in rois_id]).T#temporal borders of rois
        sniptimes = np.vstack([roitimes[0]-margL,roitimes[1]+margR])#temporal borders with margins
        
        bdict = get_bdict(my_id,home_directory,appendix = '__blipdict_clean')
        btimes,sr = bdict['btimes'],bdict['methods']['sr']
        
        
        data,tvec = get_IDdata(my_id,home_directory,t_offset=0.,sr=sr)
        
        
        for rr,[start,stop] in enumerate(sniptimes.T*sr):
            data_dict[rois_id[rr]]={'data':data[start:stop],'tvec':tvec[start:stop],\
                                    'btimes':btimes[(btimes<=stop/sr) & (btimes>=start/sr)]}

    #PREPARE PLOTTING
    #sort rois from lowest to highest SI
    rois = rois[np.argsort([class_dict[roi][params.index('seizidx')] for roi in rois])]

    datalist = [data_dict[roi]['data'] for roi in rois]
    datalens = [len(data) for data in datalist]#to know where to visualise breaks
    cumlen = np.r_[0,np.cumsum(datalens[:-1])]
    n_panelsSparse = np.ceil(np.cumsum(datalens)[-1]/(snipdur*sr)).astype('int')#in case you do not have enough data to fill all panels
    
    #return data_dict,rois
    if len(rois)*(margR+margL)<n_panels*snipdur: bticks = np.array([data_dict[roi]['btimes']*sr \
                - data_dict[roi]['tvec'][0]*sr+cumlen[ii] for ii,roi in enumerate(rois)])\
    #ticktimes have to be adapted to dot-time
    else: bticks = np.array([data_dict[roi]['btimes']*sr - data_dict[roi]['tvec'][0]*sr for roi in rois])\
                  +np.r_[0,np.cumsum(datalens[:-1])][:,None]#when only singlets are there
                                                     #of the fused trace
    bticks = np.hstack(bticks)
    trace_fused = np.hstack(datalist)#trace will be plotted as continuum
    timevec = np.arange(len(trace_fused))
    print('Fused len (s): ',len(trace_fused)/sr) 
    
    durratio = np.mod(len(trace_fused),snipdur*sr)/(snipdur*sr)#to allow plotting to scale of the last panel
    dscale = np.std(trace_fused)#to define ylim and placement of text and blipticks etc.
    sb_xAnch,sb_yAnch,sb_xLen,sb_yLen = (0.2*margL)*sr,-6*dscale,5*sr,2 #scale bar parameters


    #PLOTTING
    f = figure(figsize=(16,10),facecolor='w')
    f.suptitle('Examples from SI-Interval [%1.2f, %1.2f]    nMaxIds: %s'%(si_int[0],si_int[1],nmaxids),fontsize=15,y=0.98)
    f.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.95,hspace=0.)
    
    if durratio==0:gs0 = gridspec.GridSpec(n_panels, 2, hspace=0)#the other way (else:) will give a matrix error then
    else: gs0 = gridspec.GridSpec(n_panels, 2, hspace=0,width_ratios=[durratio,1-durratio],wspace=0.)

    
    for ii in range(np.min([n_panels,n_panelsSparse])):

        pstart = ii*snipdur*sr
        if not ii==n_panels-1 or durratio==0:
            pstop = (ii+1)*snipdur*sr
            ax = f.add_subplot(gs0[ii,:])
        else:
            pstop = timevec.max()
            ax = f.add_subplot(gs0[ii,0])
        #if ii==0: ax.plot([sb_offset*sr,(sb_offset+sb_len)*sr],[-8,-8],'k',linewidth=10)
        ax.plot(timevec[pstart:pstop],trace_fused[pstart:pstop],'k')
        bt = bticks[(bticks>=pstart) & (bticks<=pstop)]

        ax.vlines(bt,[4*dscale]*len(bt),[6*dscale]*len(bt),'r')

        for ll in np.cumsum(np.r_[0,datalens]):
            ax.plot(timevec[(ll-margL*sr).clip(min=0):ll+margR*sr],trace_fused[(ll-margL*sr).clip(min=0):ll+margR*sr],'darkgrey')
        ax.vlines(np.cumsum(datalens[:-1]),-dscale*8,dscale*8,color='w',lw=10,zorder=10)#whitespace between rois
        for rr,roi in enumerate(rois):#plot a hexagon in color of cluster, with id inside and write roiid next to it
            xpos = np.cumsum(np.r_[margL*sr,datalens[:-1]])[rr]
            ax.plot(xpos-0.5*margL*sr,4*dscale,'h',ms=20,mec=None,mfc=dcolors[int(class_dict[roi][params.index('clustid')])],zorder=11)
            ax.text(xpos-0.5*margL*sr,4*dscale,'%i'%class_dict[roi][params.index('bmu')],color='k',va='center',ha='center',fontsize=13,zorder=12)
            ax.text(xpos-0.2*margL*sr,6*dscale,'%s SI:%1.2f'%(roi,class_dict[roi][params.index('seizidx')]),\
                    va='bottom',zorder=13,bbox = dict(fc='w', ec='w',alpha=1.))
         
        if ii==0:#plot the scalebar
           ax.plot([sb_xAnch,sb_xAnch+sb_xLen],[sb_yAnch,sb_yAnch],'k',lw=3,zorder=16)
           ax.plot([sb_xAnch,sb_xAnch],[sb_yAnch,sb_yAnch+sb_yLen],'k',lw=3,zorder=16) 
           ax.plot([sb_xAnch,sb_xAnch+sb_xLen],[sb_yAnch,sb_yAnch],'w',lw=3,zorder=15)
           ax.plot([sb_xAnch,sb_xAnch],[sb_yAnch,sb_yAnch+sb_yLen],'w',lw=3,zorder=15)  
           ax.text(sb_xAnch+0.1*sb_xLen,sb_yAnch+0.4*sb_yLen,'2mV 5s',color='k',\
                   fontsize=14,zorder=17,fontweight='bold',bbox=dict(fc='w',ec='w'))   
        ax.set_xlim([pstart,pstop])
        ax.set_ylim([-dscale*8,dscale*9])
        ax.set_axis_off()
    return f

def plot_clustAndSI_trace(my_id,class_dict,data,btimes,tvec,dcolors,sr=500.,**kwargs):
    from matplotlib.pyplot import figure
    
    bar_offset = kwargs['bar_offset'] if 'bar_offset' in kwargs else 4.5
    
    params = class_dict['params']
    
    class_dict = {key:class_dict[key] for key in list(class_dict.keys()) if key.count(my_id)}
    my_rois = sorted(class_dict.keys())
    tags = [roi.split('_')[-1] for roi in my_rois]
    roi_times = np.vstack([class_dict[roi][params.index('roi_int')] for roi in my_rois]).T
    colors = [dcolors[int(class_dict[roi][params.index('clustid')])] for roi in my_rois]
    sidx = [class_dict[roi][params.index('seizidx')] for roi in my_rois]#same as sis
    hidx = [class_dict[roi][params.index('bmu')] for roi in my_rois]#same as bmus


    
    f = figure(figsize=(16,4),facecolor='w')
    f.suptitle(my_id,fontsize=15,y=0.98)
    f.subplots_adjust(left=0.05,right=0.94,bottom=0.16,hspace=0.)
    ax = f.add_subplot(111)

    ax.plot(tvec,data,'k',lw=2)
    ax.hlines([bar_offset]*len(roi_times[0]),roi_times[0],roi_times[1],linewidth=8,color=colors)
    ax.vlines(btimes,3.5,4.5,color='navy',alpha=0.8)
    for roistart,roistop,si,hi,tag,col in zip(roi_times[0],roi_times[1],sidx,hidx,tags,colors):
        ax.text(roistart+0.5*(roistop-roistart),bar_offset+0.05*bar_offset,'%s h%i SI:%1.2f'%(tag,hi,si),\
                fontsize=12,ha='center',color=col,fontweight='bold')
    
    ax.set_ylabel('mV',fontsize=15,labelpad=-2)
    ax.set_xlabel('Time [s]',fontsize=15)

    return f    


def create_seizdictHex(all_ids,roidict,bmus,bmu_seiz,**kwargs):
    
    deltaT_min = kwargs['deltaT_min'] if 'deltaT_min' in kwargs else 20.
    
    roiids = sorted(roidict.keys())
    
    seizsom_dict = {}
    for my_id in all_ids:
        #my_id = 'NP14_ipsi1_04_000'
        try:
            
            la_inds,la_roi = np.array([[ii,roiid] for ii,roiid in enumerate(sorted(roiids)) if roiid.count(my_id)==1]).T
            
            seizcount = np.array([1  if bmu in bmu_seiz else 0 for bmu in bmus[la_inds.astype('int')]])
            Nseiz = np.sum(seizcount)
            #print 'he'
            seizRois = la_roi[np.array(seizcount)==1]
            
            Tseiz = np.vstack([roidict[seiz_id]['roi_int'] for seiz_id in sorted(seizRois) ])
            if len(np.where(np.diff(Tseiz.flatten())[1::2]<deltaT_min)[0])!=0:
                print(my_id,'overlap alert N:',Nseiz) 
                Nseiz = Nseiz - len(np.where(np.diff(Tseiz.flatten())[1::2]<deltaT_min)[0])
                print(my_id,'new N:',Nseiz) 
            seizsom_dict[my_id] = [Nseiz,seizRois]
            
        except:
            seizsom_dict[my_id] = [0,'-']
    return seizsom_dict

def plot_compareAnimals(f,vardict,title=True,**kwargs):

    from matplotlib.pyplot import figure
    from matplotlib.patches import Polygon
    import matplotlib.gridspec as gridspec
    
    
        
    animals = list(set([key.split('_')[0] for key in sorted(vardict.keys())]))
    el_locations = list(set([key.split('_')[1] for key in sorted(vardict.keys())]))
    nrecs_loc = []
    for loc in el_locations:
        nrecs_loc += [len([key for key in list(vardict.keys()) if key.count(loc)==1])]
        
    color_dict = kwargs['color_dict'] if 'color_dict' in kwargs else {animal:'k' for animal in animals}  
    bw = kwargs['bw'] if 'bw' in kwargs else 1. 
    ymax = kwargs['ymax'] if 'ymax' in kwargs else np.mean(list(vardict.values()))+np.std(list(vardict.values()))
    ylab = kwargs['ylab'] if 'ylab' in kwargs else ''
    nrows,ncols = kwargs['plotdim'] if 'plotdim' in kwargs else [1, len(el_locations)]
    nrows2,ncols2 = kwargs['plotdim2'] if 'plotdim2' in kwargs else [1, len(animals)]
    
    y1 = ymax-0.1*ymax
    y2 = y1+0.05*ymax
    y3 = y1+0.05*ymax
    arrowverts = lambda x:np.array([[x-bw/2.,y1],[x-bw/2.,ymax],[x+bw/2.,ymax],[x+bw/2.,y1],[x,y2]])
    
    
    gs0 = gridspec.GridSpec(nrows, ncols,width_ratios=nrecs_loc)

    for ll,loc in enumerate(el_locations):
        
        nrecs_animal = []#calculating with ratios for all animals within particular recording location
        for animal in animals:
            nrecs_animal += [len([key for key in list(vardict.keys()) if key.count(loc)==1 and key.count(animal)==1])]
        
        gs00 = gridspec.GridSpecFromSubplotSpec(nrows2, ncols2, subplot_spec=gs0[ll]\
                                                ,width_ratios=nrecs_animal)
        
        for aa,animal in enumerate(animals):
            print(animal)
            ax = f.add_subplot(gs00[aa])

            if aa==0 and title==True:ax.set_title(loc,fontsize=16)
            
            animal_loc = animal+'_'+loc
            varkeys = [key for key in sorted(vardict.keys()) if key.count(animal_loc)==1]
            yvec = np.array([vardict[key] for key in varkeys])


            index = np.arange(len(yvec))
            #print yvec
            #print index,yvec
            if len(yvec.shape)<2:
                yvec,mycol = [yvec],[color_dict[animal]]
            else: mycol = color_dict[animal]
            for yyvec,col in zip(yvec.T,mycol):
                print(col)
                ax.bar(index, yyvec,edgecolor='w',facecolor=col\
                       ,width=bw,alpha=0.8)
                
            
                for bigid in np.where(yyvec>ymax)[0]:

                    xpos,yval = index[bigid]+bw/2.,yyvec[bigid]
                    
                    triang = Polygon(arrowverts(xpos), fc='w',ec='w')

                    ax.add_artist(triang)
                    ax.text(xpos,y3,str(yval),fontsize=13,color=col,ha='center',fontweight='bold')

            ax.set_ylim([0.,ymax])
            ax.set_xlim([index.min()-0.1,index.max()+1+0.1])
                        
            strs = np.array([(key.split('_')[2]).lstrip('0') for key in varkeys])
            strnames = []
            for mystr in strs:
                if not mystr in strnames: strnames += [mystr]
                else: strnames += [mystr+'\'']
            ax.set_xticks(np.arange(bw/2.,bw/2.+index.max()+0.1))
            ax.set_xticklabels(strnames)
            
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].axis.axes.tick_params(direction='outward')
            
            if aa==0 and ll==0:ax.set_ylabel(ylab)
            else: ax.set_yticklabels([])
            for sppos in ['top', 'right']:
                ax.spines[sppos].set_color('none')
                ax.spines[sppos].set_visible(False)
            if aa>0:
                ax.spines['left'].set_color('none')
                ax.spines['left'].axis.axes.tick_params(axis='y',color='w')
    f.text(0.5,0.04,'recordings',fontsize=16,fontweight='bold',va='center',ha='center')
    return f


def plot_barPanel(ax,vardict,varkeys,**kwargs):
    from matplotlib.patches import Polygon
    
    bw = kwargs['bw'] if 'bw' in kwargs else 1. 
    ymax = kwargs['ymax'] if 'ymax' in kwargs else np.mean(list(vardict.values()))+np.std(list(vardict.values()))
    mycol = kwargs['mycol'] if 'mycol' in kwargs else 'k'
    rounddec = kwargs['rounddec'] if 'rounddec' in kwargs else 0
    
    y1 = ymax-0.1*ymax
    y2 = y1+0.05*ymax
    y3 = y1+0.05*ymax

    arrowverts = lambda x:np.array([[x-bw/2.,y1],[x-bw/2.,ymax],[x+bw/2.,ymax],[x+bw/2.,y1],[x,y2]])
    
    
    
    yvec = np.array([vardict[key] for key in varkeys])
    
   
    index = np.arange(len(yvec))
    
    if len(yvec.shape)<2:
        yvec,mycol = yvec[:,None],[mycol]
    else: mycol = mycol[:]
    

    for yyvec,col in zip(yvec.T,mycol):

        ax.bar(index, yyvec,edgecolor='w',facecolor=col\
               ,width=bw,alpha=0.8)
        
        
        for bigid in np.where(yyvec>ymax)[0]:

            xpos,yval = index[bigid]+bw/2.,np.around(yyvec[bigid],rounddec)
            if rounddec ==0: yval = int(yval)
            
            triang = Polygon(arrowverts(xpos), fc='w',ec='w')

            ax.add_artist(triang)
            ax.text(xpos,y3,str(yval),fontsize=15,color=col,ha='center',fontweight='bold')
        
        
        ax.set_ylim([0.,ymax])
        ax.set_xlim([index.min()-0.1,index.max()+1+0.1])
                    
        strs = np.array([(key.split('_')[2]).lstrip('0') for key in varkeys])
        strnames = []
        for mystr in strs:
            if not mystr in strnames: strnames += [mystr]
            else: strnames += [mystr+'\'']
        ax.set_xticks(np.arange(bw/2.,bw/2.+index.max()+0.1))
        ax.set_xticklabels(strnames)
        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].axis.axes.tick_params(direction='outward')
        for sppos in ['top', 'right']:
            ax.spines[sppos].set_color('none')
            ax.spines[sppos].set_visible(False)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#ROBUSTNESS COMPARISONS
def plot_compareSidx(ax,X,Y,print_stats=True,mode='maps',**kwargs):
    import scikits.statsmodels.api as sm
    
    x_name = kwargs['x_name'] if 'x_name' in kwargs else 'X'
    y_name = kwargs['y_name'] if 'y_name' in kwargs else 'Y'
    
       
    X_const = np.hstack([X[:,None],np.ones(len(X))[:,None]])#input needs 1. appended for sm.OLS
    results = sm.OLS(Y,X_const).fit()

    eq_txt = r'${}$'.format('y') + \
        r'$=\, {:.2f}{} {} {:.2f}$'.format(results.params[0],'x',str(np.sign(results.params[1]))[:1],np.abs(results.params[1]))
    st_txt = r'$R_{adj}^2'+r'=\, {:.2f},\, p=\,{:.2e}$'.format(results.rsquared_adj,results.pvalues[0])
    
    
    if mode=='maps':mt,mc,ms = 'o','grey',8
    if mode=='rois':mt,mc,ms = '.','k',4
    
    stats_col = 'r'
    ax.plot(X,Y,mt,mfc=mc,mec=mc,ms=ms,alpha=0.6)#the scatterplot
    
    ax.plot(X, X*results.params[0] + results.params[1],stats_col)
    ax.plot([0,1],[0,1],'dimgrey',ls='--',lw=2)
    ax.plot([1,1,0],[0,1,1],'dimgrey',ls='--',lw=1)
    
    ax.set_xlabel('SI on {}'.format(x_name),fontweight='normal')
    ax.set_ylabel('SI on {}'.format(y_name),fontweight='normal')
    
    for sppos in ['top', 'right']:
            ax.spines[sppos].set_color('none')
            ax.spines[sppos].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    if print_stats:
        ax.text(0.02,0.92,eq_txt,transform=ax.transAxes,fontsize=15,color=stats_col,fontweight='bold',bbox = dict(fc='w', ec='w',alpha=0.8))
        ax.text(0.02,0.82,st_txt,transform=ax.transAxes,fontsize=15,color=stats_col,fontweight='bold',bbox = dict(fc='w', ec='w',alpha=0.8))
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_aspect('equal', adjustable='box')


def plot_compareClusterIds(ax,clustsX,clustsY,**kwargs):

    import matplotlib.colors as colors
    from matplotlib import colorbar    
    
    #clustrange = np.unique(np.r_[clustsX,clustsY])
    clustrange = np.arange(0.,np.max([clustsX,clustsY])+1)
    nclust = len(clustrange)
    
    cb_ax = kwargs['cb_ax'] if 'cb_ax' in kwargs else None
    my_cmap = kwargs['cmap'] if 'cmap' in kwargs else 'gray_r'
    clustcols = kwargs['clustcols'] if 'clustcols' in kwargs else ['k']*nclust
    x_name = kwargs['x_name'] if 'x_name' in kwargs else 'X'
    y_name = kwargs['y_name'] if 'y_name' in kwargs else 'Y'
    
    mybins = np.arange(0,nclust+1)-0.5
    #print nclust,mybins,clustrange
    clust_mat = np.vstack([np.histogram(clustsY[clustsX==cln],mybins)[0] for cln in clustrange])[:,::-1]
    norm_facs = np.sum(clust_mat,1)#so sum along vertical is 1
    clustmat_norm = clust_mat/norm_facs[:,None].astype(float)



    im = ax.imshow(clustmat_norm.transpose(),cmap=my_cmap,aspect='auto',\
                   interpolation= 'none',filternorm=None)
    im.set_extent([mybins.min(),mybins.max(),mybins.min(),mybins.max()])
    for ii,[mybin,col] in enumerate(zip(mybins,clustcols)):
        #print col
        if ii>0:
            ax.axhline(mybin, lw=9, color='w', zorder=3)
            ax.axvline(mybin, lw=9, color='w', zorder=3)
        
        
        ax.axvline(ii, lw=2, color=col, ls='--',zorder=12)
        ax.plot([ii-0.5,ii+0.5],[-0.5,-0.5],color=col,lw=10)
        ax.plot([-0.5,-0.5],[ii-0.5,ii+0.5],color=col,lw=10)
        #ax.axhline(ii, lw=1, color=col, ls='--',zorder=6)
    ax.axvline(mybins[-1], lw=9, color='w', zorder=3)
    ax.axhline(mybins[-1], lw=9, color='w', zorder=3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([mybins.min(),mybins.max()])
    ax.set_ylim([mybins.min(),mybins.max()])
    ax.set_xticks(clustrange)
    ax.set_yticks(clustrange)
    for sppos in ['top', 'right','bottom','left']:
                ax.spines[sppos].set_color('none')
                ax.spines[sppos].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].axis.axes.tick_params(direction='outward')
    #ax.set_axis_off()
    ax.set_xlabel('clusterID on {}'.format(x_name),fontweight='normal')
    ax.set_ylabel('clusterID on {}'.format(y_name),fontweight='normal')
    
    
    ax.tick_params(axis='both', direction='outward')
    for ii,[xtl,ytl] in enumerate(zip(ax.xaxis.get_ticklabels(),ax.yaxis.get_ticklabels())):
        #print ii
        xtl.set_color(clustcols[ii])
        ytl.set_color(clustcols[ii])
        xtl.set_fontweight('bold')
        ytl.set_fontweight('bold')


    if not cb_ax==None:
        cb = colorbar.ColorbarBase(cb_ax, cmap=my_cmap,norm=colors.Normalize(vmin=0, vmax=1.),\
                                                   orientation='vertical')
        cb_ax.set_axis_off()
        cb_ax.text(0.5,0.95,'1',fontsize=15,color='w',ha='center',va='center',transform = cb_ax.transAxes)
        cb_ax.text(0.5,0.05,'0',fontsize=15,color='k',ha='center',va='center',transform = cb_ax.transAxes)
        cb_ax.text(0.5,0.5,'relat. freq.',fontsize=15,fontweight='bold',rotation=-90,\
                   color='k',ha='center',va='center',transform = cb_ax.transAxes)



def plot_compareMaps(home_map,target_map):
    from matplotlib.pyplot import figure,Circle
    import matplotlib.gridspec as gridspec
    
    
    weights,kshape = target_map['weights'],target_map['kshape']
    
    bmus = get_bmus(home_map['weights'].T,weights)#BonA
    clusterids = target_map['clusterids'][bmus]#which clusterids would home-hex get on target
    sidx = target_map['seizidx'][bmus]#which SI would home-hex get on target map

    f = figure(figsize=(15,5),facecolor='w')
    f.subplots_adjust(left=0.02,right=0.98,bottom=0.13,top=0.9)
    
    gs_main = gridspec.GridSpec(1, 3,width_ratios=[1.5,2,2])
    
    #hitmap
    hexvalues = count_winnersPerHex(home_map['weights'].T,weights)
    
    hw_hex = 0.5
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    max_center = hex_dict[np.argmax(target_map['seizidx'])][0]
    min_center = hex_dict[np.argmin(target_map['seizidx'])][0]
    
    gs_map = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs_main[0],width_ratios=[10,1],wspace=0.0)
    map_ax = f.add_subplot(gs_map[0])
    hist_ax = f.add_subplot(gs_map[1])
    
    
    plot_hexPanel(kshape,hexvalues,map_ax,hw_hex=hw_hex,hex_ax=hist_ax,showConn=True,showHexId=False\
                     ,labelsOn=False,logScale=False,conn_ref=weights,hexcol='Greys',idcolor='navy')
    
    for ii,[col,center,cs] in enumerate(zip(['firebrick','steelblue'],[max_center,min_center],['max','min'])):
        map_ax.add_patch(Circle(center,hw_hex/2. ,facecolor=col, edgecolor=col))
        map_ax.text(0.1,0.8-ii*0.07,cs+'SI',color=col,transform=map_ax.transAxes,fontsize=13,fontweight='bold')          
    
    hist_ax.text(0.5,0.5,'#Hits',rotation=-90,va='center',ha='center',fontweight='bold',fontsize=16)
    #map_ax.set_xlim([0,4])
    #map_ax.set_ylim([-hw_hex,(kshape[1]-1)*2*hw_hex])
    map_ax.set_title('HomeHex on Target Map')
    
    #sidx
    sidx_ax = f.add_subplot(gs_main[1])
    plot_compareSidx(sidx_ax,home_map['seizidx'],sidx,print_stats=True,x_name='home-map',y_name='target-map')
    
    #clusters
    print('Warning- cluster-swap for clusterid-comparison!')
    maxclust = np.max(home_map['clusterids'])
    clustX = maxclust-home_map['clusterids']#reverse such that low cln now is high and appears at the end
    clustY = maxclust-clusterids
    ccols = home_map['dcolors'][::-1]#need to reverse colors as well
    
    clust_main = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs_main[2],width_ratios=[9,1])
    clust_ax = f.add_subplot(clust_main[0])
    cb_ax = f.add_subplot(clust_main[1])
    plot_compareClusterIds(clust_ax,clustX,clustY,cb_ax=cb_ax,\
                              x_name='home-map',y_name='target-map',clustcols=ccols)
    
    return f

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
# make it runnable
class som_run(object):
    
    def __init__(self,n_run=1):
        import datetime
        now = datetime.date.today()
        datestr = str(now.year-2000).zfill(2)+str(now.month).zfill(2)+str(now.day).zfill(2)
        self.run_name = datestr+'_somrun'+str(n_run).zfill(2)
        self.date = now.isoformat()
       
       

    def load_default(self):
        
        self.plot_on = True
        self.plot_examples = True
        
        unit_dict = {'dur':'s','b_n':'','isi_mean':'s','isi_med':'s','isi_std':'s','isi_cv':'',\
             'isi_fano':'s','isi_peak':'s','l_len':'mV','l_rate':'mV/s','lz_len':'z','lz_rate':'z/s'}
        self.myall_vars = ['dur', 'b_n', 'isi_mean','isi_med','isi_std', 'isi_cv', 'isi_fano',\
           'isi_peak', 'l_len', 'l_rate','lz_len','lz_rate']
        self.myall_logFlags = [True, True,True,True,False,False,False,True,True,True,True,True]
        self.all_vars = {ii:[self.myall_vars[ii],self.myall_logFlags[ii]] \
                         for ii in range(len(self.myall_vars))}  
        self.myall_logvars = [val[0] for val in list(self.all_vars.values()) if val[1]]
        self.myall_varnames = ['lg '+my_var  if my_var in self.myall_logvars else my_var for my_var in self.myall_vars]
        self.myall_unitlist = [unit_dict[var] for var in self.myall_vars]
       
        
        self.myvars = ['dur', 'b_n', 'isi_med','isi_cv','isi_peak','isi_mean']
        self.logFlags = [True, True, True, False,True,True]
        self.myvarsWeights = [1.,1.,1.,1.,1.,2.5]
        self.som_vars = {ii:[self.myvars[ii],self.logFlags[ii],self.myvarsWeights[ii]] \
                         for ii in range(len(self.myvars))}        
        self.nvars = len(self.som_vars)
        self.som_varnames = self.get_varnames()
        
        self.norm_method = 'zscore'
        self.with_singlets = True
        self.kshapeMode = 'auto'
        self.niter = 200
        
        self.nclust = 5
        self.cmethod = 'ward'
        self.ref_vars = ['b_n','dur']
        self.maxdur = 'None'
    
    def set_plotOn(self,flag):
        self.plot_on = flag
    
    def set_maxdur(self,maxsec):
        self.maxdur = maxsec
        
    def set_plotExamples(self,flag):
        self.plot_examples = flag    
    
    def set_runName(self,run_name):
        self.run_name = run_name
    
    def set_filepath(self,filepath):
        self.fpath_gen = filepath
        
    def set_kshape(self,kshape):
        self.kshapeMode = 'man'
        self.kshape = kshape 
        
    def use_singlets(self,sflag):
        self.with_singlets = sflag   
         
    def set_animals(self,animal_list):    
        self.animals = animal_list
        
    def set_electrodes(self,loc_list):    
        self.electrodes = loc_list
        
    def set_somvars(self,som_vardict):
        self.som_vars = som_vardict
        self.nvars = len(self.som_vars)
        self.som_varnames = self.get_varnames()
        self.myvars = [som_vardict[key][0] for key in list(som_vardict.keys())]
        self.myvarsWeights = [som_vardict[key][2] for key in list(som_vardict.keys())]
        self.logFlags = [som_vardict[key][1] for key in list(som_vardict.keys())]
    def set_nclust(self,nclust):
        self.nclust = nclust    
        
    def set_cmethod(self,cmethod):
        self.cmethod = cmethod     
    
    def set_refVars_SI(self,reflist):
        self.ref_vars = reflist
    
    def get_varnames(self):
        unit_dict = {'dur':'s','b_n':'','isi_mean':'s','isi_med':'s','isi_std':'s','isi_cv':'',\
             'isi_fano':'s','isi_peak':'s','l_len':'mV','l_rate':'mV/s','lz_len':'z','lz_rate':'z/s'}
        loglist = ['lg ' if val[1] else ' ' for val in list(self.som_vars.values())]
        return [str(val[2])+' *'+loglist[ii]+val[0]+' ['+unit_dict[val[0]]+']' for ii,val in enumerate(self.som_vars.values())]
     
    def kshape_update(self,kshape):
        self.kshape = kshape
        


    def write_params(self,motherSOMpath=None):
        import os
        f = open(os.path.join(self.fpath_gen,self.run_name+'.txt'),'w') #opens file with name of "test.txt"
        
        f.write('path: '+self.fpath_gen +'\n') 
        f.write('run: '+self.run_name +'\n') 
        f.write('host SOM: '+str(motherSOMpath) +'\n') 
        f.write('animals: '+str(self.animals) +'\n') 
        f.write('electrodes: '+str(self.electrodes) +'\n') 
        f.write('singlets: '+str(self.with_singlets) +'\n') 
        f.write('norm: '+self.norm_method +'\n') 
        f.write('kshapeMode: '+self.kshapeMode +'\n') 
        f.write('kshape: '+str(self.kshape) +'\n') 
        f.write('som_vars: '+str(self.get_varnames()) +'\n') 
        f.write('ref_vars(SI): '+str(self.ref_vars) +'\n') 
        f.write('nclust: '+str(self.nclust) +'\n')
        f.write('cmethod: '+self.cmethod +'\n')
        f.write('maxdur: '+str(self.maxdur)+'\n')
        
        f.close() 
        
#------------------------------------------------------------------------------ 
#prettifications
def get_mapstyles(stylestr):
    if stylestr == 'CopperRGreys':
        hexcol,conncol = ['copper_r','Greys']
        hcol,ccol = ['brown','grey']
    elif stylestr == 'InfernoBlues':
        hexcol,conncol = ['inferno_r','Blues']
        hcol,ccol = ['darkviolet','skyblue']
    elif stylestr == 'GreysReds':
        hexcol,conncol = ['Greys','Reds']
        hcol,ccol = ['grey','red']
    return {'hexcol':hexcol,'conncol':conncol,'hcol':hcol,'ccol':ccol}


def plotsummarize_somrun(datadict,mapstyle='GreysReds',seizmode='dots',**kwargs):
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import figure
    
    clusttextcol = kwargs['clusttextcol'] if 'clusttextcol' in kwargs else 'k'
    
    dictget = lambda mydict, *mykeys: [mydict[kk] for kk in mykeys]
    
    #manually set_parameters in inches
    hw_hex = 0.2/2.#0.2 inches widht of horizontal hexagon
    hw_conn = 0.2*hw_hex
    t_h = 1.6
    b_h = 0.8
    l_w = 0.3
    r_w = 0.3
    
    c_space = 0.5
    dend_space = c_space*1.3
    fminwidth = 12.
    
    seizcol = 'y'
    
    name_dict = {'b_n':'lg N_spikes','dur':'lg duration [s]','isi_cv':'IDI_cv','isi_fano':'IDI_ff [s]',\
             'isi_mean':'lg IDI_mean [s]','isi_med':'lg IDI_med [s]','isi_peak':'lg IDI_peak [s]',\
             'isi_std':'IDI_std [s]','lz_len':'lg LL [z]','lz_rate':'lg LLt [z/s]',\
             'l_len':'lg LL [mV]','l_rate':'lg LLt [mV/s]'}

    
    hexcol,conncol,hcol,ccol = dictget(get_mapstyles(mapstyle),'hexcol','conncol','hcol','ccol')
    
    
    qe,te,maprun,corrs = dictget(datadict,'qe','te','maprun','corrs')
    weights_rescaled,weights,vnames = dictget(datadict,'weights_rescaled','weights','vnames')
    nclust,ref_hex,kshape = dictget(datadict,'nclust','ref_hex','kshape')
    seiz_winners,seiz_winnersP,winratios = dictget(datadict,'seiz_winners','seiz_winnersP','winratios')
    swinners,swinnersP = dictget(datadict,'swinners','swinnersP')
    seizidx,clusterids,kclusts = dictget(datadict,'seizidx','clusterids','kclusts')
    dcolors = datadict['dcolors'] if 'dcolors' in datadict else get_clusterColors(nclust)
    bonusranks = datadict['bonusranks'] if 'bonusranks' in datadict else []
    
    ranknamelist = ['S','M','L','XL','XXL','XXXL']
    rank_names = {rr:rankname for rr,rankname in enumerate(ranknamelist[:nclust][::-1])}
    #ascending rank_names so the highest rank will be 'S'
    
    
    
    rank_colors = {cc:col for cc,col in enumerate(dcolors)}
    #dcolors = [rank_colors[cc] for cc in xrange(nclust)]
    nvars = len(vnames)
    scalefacs = np.array([vnames[vv][2] for vv in range(nvars)])
    varstrings = [name_dict[vnames[vv][0]] for vv in range(nvars)]
    
    
    #get individual panel extents
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    xmax,ymax = np.max(hex_dict[np.prod(kshape)-1][1],axis=0)
    xmin,ymin = np.min(hex_dict[0][1],axis=0)
    if np.mod(kshape[1],2.) == 1:xmax = xmax+hw_hex
    pan_h, pan_w = ymax-ymin,xmax-xmin
    
    #calculate figure dimensions
    fheight = 2*t_h+2*pan_h+b_h
    fwidth = l_w+nvars*pan_w+(nvars-1)*c_space+r_w
    if fwidth < fminwidth:
        #cc_space = (fminwidth - l_w - r_w - nvars*pan_w)/float(nvars-1)
        #fwidth = l_w+nvars*pan_w+(nvars-1)*cc_space+r_w
        nnvars = int(np.ceil((fminwidth-l_w-r_w+c_space)/(pan_w+c_space)))#figure should be wide enough
        fwidth = l_w+nnvars*pan_w+(nnvars-1)*c_space+r_w
    else: nnvars = int(nvars)
    t_diff = 0.3/fheight #for text above panels
    t_start = (t_h-0.35)/fheight #for text above panels
    
    
    axes_list = []
    gs0 = gridspec.GridSpec(2,nnvars, hspace  =2*t_h/fheight,wspace=c_space/fwidth)
    f = figure(figsize=(fwidth,fheight),facecolor='w')
    f.subplots_adjust(left=l_w/fwidth,right=1.-r_w/fwidth,bottom=b_h/fheight,top=1.-t_h/fheight)#-tspace/float(fheight)
    f.text(0.98,b_h/fheight/2.,maprun +' % s QE:%1.2f | TE:%1.3f'%(str(kshape),qe,te),ha='right',va='center',fontsize=14)
    
    #component-planes
    for vv in range(nvars):
        ax = f.add_subplot(gs0[0,vv])
        pos = ax.get_position()
        
        f.text(pos.x0,pos.y1+t_start,varstrings[vv],va='top',ha='left')
        f.text(pos.x0,pos.y1+t_start-t_diff,'bias %i  ccSI:%1.2f'%(scalefacs[vv],corrs[vv]),va='top',ha='left')
        scaledict = plot_hexPanel(kshape,weights_rescaled[vv],ax,hw_hex=hw_hex,showHexId=False,scalefree=True,\
                                     return_scale=True,hexcol=hexcol,conncol=conncol,hw_conn=hw_conn,alphahex=1.)
        hexint, connint = dictget(scaledict, 'hexint', 'connint')
        #if vnames[vv][1]: hexint,connint = 10**np.array(hexint),10**np.array(connint)
        for ii,[myint,intcol] in enumerate(zip([hexint,connint],[hcol,ccol])):
            f.text(pos.x0,pos.y1+t_start-t_diff*(ii+2),'[%1.2f,%1.2f]'%(myint[0],myint[1]),\
                   va='top',ha='left',color=intcol)
        axes_list.append(ax)
    
    #seizure-index
    siax = f.add_subplot(gs0[1,0])
    pos = siax.get_position()
    f.text(pos.x0,pos.y1+t_start-t_diff,'SeverIn + Seiz%',va='top',ha='left')
    
    scaledict = plot_hexPanel(kshape,seizidx,siax,hw_hex=hw_hex,showConn=True,showHexId=False\
                             ,scalefree=True,return_scale=True,hexcol=hexcol,conncol=conncol,hw_conn=hw_conn,alphahex=1.)#,alphahex=1.
    #sd.plot_addCircles(siax,kshape,seiz_winners,hw_hex = hw_hex,scalefac=0.8,color=seizcol,textOn=False)
    plot_addText(siax,kshape,winratios,hw_hex = hw_hex,fontsize=10,color=seizcol,fontweight='bold')
    hexint, connint = dictget(scaledict, 'hexint', 'connint')
    for ii,[myint,intcol] in enumerate(zip([hexint,connint],[hcol,ccol])):
            f.text(pos.x0,pos.y1+t_start-t_diff*(ii+2),'[%1.2f,%1.2f]'%(myint[0],myint[1]),\
                   va='top',ha='left',color=intcol)
    axes_list.append(siax)
    
    #dendrogram

    nnclust = nclust -len(bonusranks)
    rrank_colors = {cc:col for cc,col in enumerate(dcolors[[dd for dd in range(nclust) if not dd in bonusranks]])}
    nnames = [rank_names[ranknumb] for kk,ranknumb in enumerate(sorted(rank_names.keys())) if not kk in bonusranks]
    rrank_names = {nn:nname for nn,nname in enumerate(nnames)}
    specialleaves = np.where(clusterids==bonusranks)[0]
    if len(bonusranks)>0:specialcol = dcolors[bonusranks][0]
    else: specialcol = 'error'
    hierspec = gridspec.GridSpecFromSubplotSpec(nnclust+1, 1, subplot_spec=gs0[1,1:-2],hspace=0.4)
    dendaxes = [f.add_subplot(hierspec[cc]) for cc in range(nnclust+1)]
    
    #dendaxes = [f.add_subplot(nnclust+1,1,cc+1) for cc in xrange(nnclust+1)]
    plot_fancyDendrogram(weights,nclust=nnclust,\
                            rank_colors=rrank_colors,ref_hex=ref_hex,axlist=dendaxes\
                           ,rank_names=rrank_names,seiz_winners=seiz_winners,nbins=3,specialleaves=specialleaves,\
                           specialcol = specialcol)#

    pos = gs0[1,1:-1].get_position(f)
    #f.text(pos.x0+0.7*dend_space/fwidth,pos.y0+0.5*pos.height,'Distance',ha='left',va='center',fontsize=13,\
           #rotation=90)
    for gs in dendaxes:
        pos = gs.get_position()
        gs.set_position([pos.x0+dend_space/fwidth, pos.y0, pos.width-2*dend_space/fwidth, pos.height])
    
    
    #cluster-som-hierarcy
    mapaxHier = f.add_subplot(gs0[1,-2])
    mapaxHier.set_title('Hier+SAll')
    plot_hexPanel(kshape,clusterids,mapaxHier,hw_hex=hw_hex,showConn=False,showHexId=False\
                             ,labelsOn=False,quality_map=dcolors,\
                             hw_conn=hw_conn)
    
    if seizmode == 'text':plot_addText(mapaxHier,kshape,swinners,hw_hex = hw_hex,fontsize=10,color=clusttextcol,fontweight='bold')
    elif seizmode == 'dots':plot_addCircles(mapaxHier,kshape,seiz_winners,hw_hex = hw_hex,scalefac=0.8,color='k',textOn=False)
    
    axes_list.append(mapaxHier)
    
    #cluster-som-kmeans
    mapaxKM = f.add_subplot(gs0[1,-1])
    mapaxKM.set_title('KMeans+SGen')
    plot_hexPanel(kshape,kclusts,mapaxKM,hw_hex=hw_hex,showConn=False,showHexId=False\
                             ,labelsOn=False,quality_map=dcolors,hw_conn=hw_conn)
    if seizmode == 'text':plot_addText(mapaxKM,kshape,swinnersP,hw_hex = hw_hex,fontsize=10,color=clusttextcol,fontweight='bold')
    elif seizmode == 'dots':plot_addCircles(mapaxKM,kshape,seiz_winnersP,hw_hex = hw_hex,scalefac=0.8,color='k',textOn=False)
    
    axes_list.append(mapaxKM)
    
    #scaling som-axes
    for ax in axes_list:
        ax.set_aspect('auto',anchor='C')
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_axis_off()
        
    return f


def plotexamples_hexhits(inputdict,sr=500.,allOnOne=False,verbose=False,forceclip=False,**kwargs):
    ''' if verbose a dictionary with the roi-ids plotted will be returned
    if forceclip events longer than xdur (length of panel in seconds) will be clipped
    '''
    
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import figure
    
    removelastN = lambda bla: bla[:-1] if bla[-1]=='\n' else bla
    
    xdur = kwargs['xdur'] if 'xdur' in kwargs else 60.#[0.,xdur]s will be xlim of examples
    myylim = kwargs['myylim'] if 'myylim' in kwargs else [-7.,6.]#mV
    nrows = kwargs['nrows'] if 'nrows' in kwargs else 10
    nlim = kwargs['nlim'] if 'nlim' in kwargs else 2#maximal number of roiindices allowed to be written in one hex
    d_w = kwargs['d_w'] if 'd_w' in kwargs else 11.#width of example panel in inches
    
    #unpack data in inputdict
    dictget = lambda *mykeys: [inputdict[kk] for kk in mykeys]
    kshape,dcolors,clusterids,cdict,ddict = dictget('kshape','dcolors','clusterids','cdict','ddict')
    bef,aft = inputdict['margint']
    bcolor = 'grey'

    
    #figure parameters
    hw_hex = 0.7/2.#0.2 inches widht of horizontal hexagon
    t_h = 0.5
    b_h = 0.2
    l_w = 0.2
    r_w = 0.1
    w_space = 0.5
    bcolor = 'grey'
    #scalebars
    sbx0,sby0 = bef*0.2*sr,myylim[0]+2.#,
    sbx1,sby1 = sbx0+5*sr,sby0+2

    
    params,cparams = ddict['params'],cdict['params']
    ridx,cidx = params.index('roi'),cparams.index('bmu')
    roidurs_sorted = np.array([len(ddict[roiidx][params.index('data')])/sr for roiidx in list(ddict.keys()) if not roiidx=='params'])
    roidurs_sorted = np.clip(roidurs_sorted,0,xdur-xdur/10000000.)
    startidx = 0
    indsList = []#each item in the list will contain array of roiidx present in a single panel
    #print roidurs_sorted
    while startidx < len(roidurs_sorted):
        csum = np.cumsum(roidurs_sorted[startidx:])
        inds = np.where(csum<xdur)[0]+startidx
        if np.size(inds)>0:
            indsList.append(inds)
            startidx = inds[-1]+1
        else:
            startidx = 999
    #print indsList
    #indsList +=[np.array([idx]) for idx in np.arange(len(roidurs_sorted))[indsList[-1][-1]+1:]]#everybody longer than xdur will 
    #print indsList                                                                                                    #get a single entry
    
    #setup figure dimensions
    npanels = len(indsList)+len(np.where(roidurs_sorted>xdur)[0])
    if allOnOne: nrows = int(npanels)
    nfigs = int(np.ceil(npanels/float(nrows)))
    
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    xmax,ymax = np.max(hex_dict[np.prod(kshape)-1][1],axis=0)
    xmin,ymin = np.min(hex_dict[0][1],axis=0)
    if np.mod(kshape[1],2.) == 1:xmax = xmax+hw_hex
    pan_h, pan_w = ymax-ymin,xmax-xmin
    fheight = t_h+pan_h+b_h
    fwidth = l_w+pan_w+w_space+d_w+r_w

    gs0 = gridspec.GridSpec(1,2,wspace=w_space/fwidth,width_ratios=[pan_w,d_w])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec=gs0[1],hspace=0)
    
    ll=0 # counting dataexample panels
    bmus = np.array([cdict[ddict[idROI][ridx]][cidx] for idROI in range(len(roidurs_sorted))])
    flist = []
    exampleIDdict = {}
    for ff in range(nfigs):
        #ff = 0
        if ff == nfigs-1: myROIids = np.hstack(indsList[ll:])
        else:myROIids =  np.hstack(indsList[ll:ll+nrows])  
        mybmus = bmus[myROIids]
        idlist = [myROIids[mybmus==myhex] for myhex in range(np.prod(kshape)) ]
        stringlist = [','.join(str(l)+'\n'*(n%nlim==nlim-1) for n,l in enumerate(sorted(idsROI))) for idsROI in idlist]
        stringlist = [removelastN(mystr) if len(mystr)>1 else mystr for mystr in stringlist] 
        
        f = figure(figsize=(fwidth,fheight),facecolor='w')
        f.subplots_adjust(left=l_w/fwidth,right=1.-r_w/fwidth,bottom=b_h/fheight,top=1.-t_h/fheight)
        #the map
        
        ax = f.add_subplot(gs0[0])
        plot_hexPanel(kshape,clusterids,ax,hw_hex=hw_hex,showConn=False,showHexId=False\
                                     ,labelsOn=False,quality_map=dcolors,alphahex=0.3)
        ax.set_aspect('auto',anchor='C')
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_axis_off()
        #add ids caught in hexagons to hexagon
        plot_addText(ax,kshape,stringlist,hw_hex = hw_hex,fontsize=12,color='k',fontweight='bold')
        
        
        rr = 0 
        axlist = []
        while rr<nrows and ll<len(indsList):
            #if ll<len(indsList):
            sublist = indsList[ll]
            traces = [ddict[idx][params.index('data')] for idx in sublist]
            csum = np.cumsum(roidurs_sorted[sublist[0]:sublist[-1]+1])*sr
            bpoints = [(ddict[idx][params.index('btimes')]-ddict[idx][params.index('roi_int')][0]+bef)*sr+\
                       np.r_[0,csum][cc]for cc,idx in enumerate(sublist)]
            myclusts = [cdict[ddict[idx][params.index('roi')]][cparams.index('clustid')] for idx in sublist ]
            collist = ['w' if np.isnan(cid) else dcolors[cid] for cid in myclusts ]
            rstartstop = np.array([[bts[0],bts[-1]] for bts in bpoints])
            #print collist
            #bscale = np.std(np.hstack(traces))
            ax = f.add_subplot(gs00[rr])
            ax.plot(np.hstack(traces),color='k',zorder=0)
            ax.vlines(csum,-10.,10.,linewidth=8,color='w',zorder=15)
            ax.vlines(np.array(np.hstack(bpoints)),myylim[1]-4.,myylim[1]-2.,color=bcolor)
            ax.hlines([myylim[1]-4.]*rstartstop.shape[0],rstartstop[:,0],rstartstop[:,1],linewidth=5,colors=collist)#
            ax.set_xlim([0.,xdur*sr])
            axlist.append(ax)
            for xx,xpt in enumerate(np.r_[0.,csum[:-1]]+bef*0.5*sr):
                idx = sublist[xx]
                myroi = ddict[idx][params.index('roi')]
                if np.isnan(cdict[myroi][cparams.index('bmu')]):
                    textstr = '{:d}'.format(idx)
                else:
                    textstr = '{:d} bmu{:d} SI:{:1.2f}'.format(idx,cdict[myroi][cparams.index('bmu')],\
                                                             cdict[myroi][cparams.index('seizidx')])
                
                ax.text(xpt,myylim[1],textstr,fontsize=11,va='top',ha='left',bbox={'fc':'w','ec':'w','alpha':0.5},zorder=20)        
            
            if rr==0:#plot the scalebar
                ax.plot([sbx0,sbx0,sbx1],[sby1,sby0,sby0],'k',lw=3,zorder=17)
                ax.text(sbx0,sby1,'2mV, 5s',ha='left',va='top',fontsize=10)
            
            
            rr+=1
            ll+=1
            if len(np.hstack(traces))/sr > xdur:
                if forceclip: ax.text(0.99,0.99,'clipped',transform=ax.transAxes,va='top',ha='right',fontsize=11)
                else:
                    ax = f.add_subplot(gs00[rr])
                    ax.plot(np.hstack(traces),color='k',zorder=0)
                    ax.vlines(csum,-10.,10.,linewidth=8,color='w',zorder=20)
                    ax.vlines(np.array(np.hstack(bpoints)),myylim[1]-4.,myylim[1]-2.,color=bcolor)
                    ax.hlines([myylim[1]-4.]*rstartstop.shape[0],rstartstop[:,0],rstartstop[:,1],linewidth=5,colors=collist)#
                    ax.set_xlim([xdur*sr,2*xdur*sr])
                    rr+=1
                    axlist.append(ax)

        
        for ax in axlist:     #ax.set_ylim([-bscale*8,bscale*9])
            ax.set_ylim(myylim)
            ax.set_axis_off()
        flist.append(f)
        exampleIDdict[ff] = {roiidx:ddict[roiidx][0] for roiidx in myROIids}
    if verbose:  return flist,exampleIDdict
    else: return flist


def plot_clustermapTagged(inputdict,showtags=True,**kwargs):
    from matplotlib.pyplot import figure
    
    removelastN = lambda bla: bla[:-1] if bla[-1]=='\n' else bla
    nlim = kwargs['nlim'] if 'nlim' in kwargs else 2#maximal number of roiindices allowed to be written in one hex
    hw_hex = kwargs['hw_hex'] if 'hw_hex' in kwargs else 0.7/2.
    fs = kwargs['fs'] if 'fs' in kwargs else 12
    fw = kwargs['fs'] if 'fw' in kwargs else 'bold'
    alphahex = kwargs['alphahex'] if 'alphahex' in kwargs else 0.3
    ec = kwargs['ec'] if 'ec' in kwargs else None
    #unpack data in inputdict
    dictget = lambda *mykeys: [inputdict[kk] for kk in mykeys]
    kshape,dcolors,clusterids,cdict,ddict = dictget('kshape','dcolors','clusterids','cdict','ddict')
    params,cparams = ddict['params'],cdict['params']
    ridx,cidx = params.index('roi'),cparams.index('bmu')
    
    idsInFig = np.array([key for key in list(ddict.keys()) if not key=='params'])
    #print idsInFig
    bmus = np.array([cdict[ddict[idROI][ridx]][cidx] for idROI in idsInFig])
    idlist = [np.arange(len(idsInFig))[bmus==myhex] for myhex in range(np.prod(kshape)) ]
    stringlist = [','.join(str(idsInFig[np.array(l)])+'\n'*(n%nlim==nlim-1) for n,l in enumerate(sorted(idsROI))) for idsROI in idlist]
    #print stringlist
    stringlist = [removelastN(mystr) if len(mystr)>1 else mystr for mystr in stringlist ]
    
    #figuresetup
    t_h = 0.6
    b_h = 0.2
    l_w = 0.2
    r_w = 0.2
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex)
    xmax,ymax = np.max(hex_dict[np.prod(kshape)-1][1],axis=0)
    xmin,ymin = np.min(hex_dict[0][1],axis=0)
    if np.mod(kshape[1],2.) == 1:xmax = xmax+hw_hex
    pan_h, pan_w = ymax-ymin,xmax-xmin
    fheight = t_h+pan_h+b_h
    fwidth = l_w+pan_w+r_w
   
    f = figure(figsize=(fwidth,fheight),facecolor='w')
    f.subplots_adjust(left=l_w/fwidth,right=1.-r_w/fwidth,bottom=b_h/fheight,top=1.-t_h/fheight)#-tspace/float(fheight)
    ax = f.add_subplot(111)
    if ec:plot_hexPanel(kshape,clusterids,ax,hw_hex=hw_hex,showConn=False,showHexId=False\
                                 ,labelsOn=False,quality_map=dcolors,alphahex=alphahex,ec=ec)
    else:plot_hexPanel(kshape,clusterids,ax,hw_hex=hw_hex,showConn=False,showHexId=False\
                                 ,labelsOn=False,quality_map=dcolors,alphahex=alphahex)
    ax.set_aspect('auto',anchor='C')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_axis_off()
    #add ids caught in hexagons to hexagon
    if showtags:plot_addText(ax,kshape,stringlist,hw_hex = hw_hex,fontsize=fs,color='k',fontweight=fw)
    return f


def plotexample_bursts(inputdict,showtext=True,sr=500.,**kwargs):
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import figure
    
    xdur = kwargs['xdur'] if 'xdur' in kwargs else 60.#[0.,xdur]s will be xlim of examples
    myylim = kwargs['myylim'] if 'myylim' in kwargs else [-7.,6.]#mV
    nrows = kwargs['nrows'] if 'nrows' in kwargs else 10
    nancol = kwargs['nancol'] if 'nancol' in kwargs else 'w'
    
    #unpack data in inputdict
    dictget = lambda *mykeys: [inputdict[kk] for kk in mykeys]
    kshape,dcolors,clusterids,cdict,ddict = dictget('kshape','dcolors','clusterids','cdict','ddict')
    bef,aft = inputdict['margint']
    bcolor = 'grey'

    
    #figure parameters
    bcolor = 'grey'
    #scalebars
    sbx0,sby0 = bef*0.2*sr,myylim[0]+2.#,
    sbx1,sby1 = sbx0+5*sr,sby0+2

    
    params,cparams = ddict['params'],cdict['params']
    ridx,cidx = params.index('roi'),cparams.index('clustid')
    ids_sorted = np.array(sorted([key for key in list(ddict.keys()) if not key=='params']))
    roidurs_sorted = np.array([len(ddict[roiidx][params.index('data')])/sr for roiidx in ids_sorted])
    

    startidx = 0
    indsList = []#each item in the list will contain array of roiidx present in a single panel
    while startidx < len(roidurs_sorted):
        csum = np.cumsum(roidurs_sorted[startidx:])
        inds = np.where(csum<xdur)[0]+startidx
        if np.size(inds)>0:
            indsList.append(inds)
            startidx = inds[-1]+1
        else:
            startidx = 999
    indsList +=[np.array([idx]) for idx in np.arange(len(roidurs_sorted))[indsList[-1][-1]+1:]]#everybody longer than xdur will get a single entry
    
    #setup figure dimensions
    npanels = len(indsList)+len(np.where(roidurs_sorted>xdur)[0])
    nfigs = int(np.ceil(npanels/float(nrows)))
    
    #bmus = np.array([cdict[ddict[idROI][ridx]][cidx] for idROI in ids_sorted])
    flist = []
    ll=0
    for ff in range(nfigs):
        f = figure(figsize=(16,10),facecolor='w')
        f.subplots_adjust(left=0.05,right=0.99,top=0.98,bottom=0.02,hspace=0)
        rr = 0 
        while rr<nrows and ll<len(indsList):
            #if ll<len(indsList):
            sublist = indsList[ll]
            traces = [ddict[idx][params.index('data')] for idx in ids_sorted[sublist]]
            csum = np.cumsum(roidurs_sorted[sublist[0]:sublist[-1]+1])*sr
            bpoints = [(ddict[idx][params.index('btimes')]-ddict[idx][params.index('roi_int')][0]+bef)*sr+\
                       np.r_[0,csum][cc]for cc,idx in enumerate(ids_sorted[sublist])]
            myclusts = [cdict[ddict[idx][ridx]][cidx] if ddict[idx][ridx] in cdict\
                         else np.nan for idx in ids_sorted[sublist] ]#nan when there is a singlet
            #print myclusts
            collist = [nancol if np.isnan(cid) else dcolors[cid] for cid in myclusts ]
            rstartstop = np.array([[bts[0],bts[-1]] for bts in bpoints])
            #print collist
            #bscale = np.std(np.hstack(traces))
            ax = f.add_subplot(nrows,1,rr+1)
            ax.plot(np.hstack(traces),color='k',zorder=0)
            ax.vlines(csum,-10.,10.,linewidth=8,color='w',zorder=15)
            ax.vlines(np.array(np.hstack(bpoints)),myylim[1]-4.,myylim[1]-2.,color=bcolor)
            ax.hlines([myylim[1]-4.]*rstartstop.shape[0],rstartstop[:,0],rstartstop[:,1],linewidth=5,colors=collist)#
            ax.set_xlim([0.,xdur*sr])
            
            if showtext:
                for xx,xpt in enumerate(np.r_[0.,csum[:-1]]+bef*0.5*sr):
                    idx = ids_sorted[sublist[xx]]
                    myroi = ddict[idx][params.index('roi')]
                    if myroi not in cdict:
                        textstr = 'id:{:d}'.format(idx)
                    elif np.isnan(cdict[myroi][cidx]):
                        textstr = 'id:{:d}'.format(idx)
                    else:
                        textstr = 'id:{:d} bmu{:d} SI:{:1.2f}'.format(idx,cdict[myroi][cparams.index('bmu')],\
                                                                 cdict[myroi][cparams.index('seizidx')])
                    
                    ax.text(xpt,myylim[1],textstr,fontsize=11,va='top',ha='left',bbox={'fc':'w','ec':'w','alpha':0.5},zorder=20)        
            
            if rr==0:#plot the scalebar
                ax.plot([sbx0,sbx0,sbx1],[sby1,sby0,sby0],'k',lw=3,zorder=17)
                if showtext:ax.text(sbx0,sby1,'2mV, 5s',ha='left',va='top',fontsize=10)
            
            
            rr+=1
            ll+=1
            if len(np.hstack(traces))/sr > xdur:
                ax = f.add_subplot(nrows,1,rr+1)
                ax.plot(np.hstack(traces),color='k',zorder=0)
                ax.vlines(csum,-10.,10.,linewidth=8,color='w',zorder=20)
                ax.vlines(np.array(np.hstack(bpoints)),myylim[1]-4.,myylim[1]-2.,color=bcolor)
                ax.hlines([myylim[1]-4.]*rstartstop.shape[0],rstartstop[:,0],rstartstop[:,1],linewidth=5,colors=collist)#
                ax.set_xlim([xdur*sr,2*xdur*sr])
                rr+=1
               
        
        for ax in f.get_axes():     #ax.set_ylim([-bscale*8,bscale*9])
            ax.set_ylim(myylim)
            ax.set_axis_off()
        flist.append(f)
    return flist

#------------------------------------------------------------------------------ 
#RESULT-PLOTS FOR MAPRUNS
