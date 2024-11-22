import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl



def get_cdict_and_vec(cmap_clust,labels):
    ncl = len(np.unique(labels))
    cmap = mpl.cm.get_cmap(cmap_clust)#
    norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
    cdict_clust = {lab:cmap(norm(lab)) for lab in np.unique(labels)}
    cat_cvec = np.array([cdict_clust[ii] for ii in np.arange(ncl)])
    return cdict_clust,cat_cvec


def getHexagon(center=[0., 0.], hw=0.5, fc='k'):
    import matplotlib.pyplot as plt

    p2c = lambda rho, phi: [rho * np.cos(np.radians(phi)), rho * np.sin(np.radians(phi))]

    phis = np.arange(30., 360., 60.)
    r = hw / np.cos(np.radians(30))
    # print r
    verts = np.vstack([p2c(r, phis[ii]) for ii in np.arange(6)]) + center
    return verts

def get_hexgrid(kshape, hw_hex=0.5):
    hw_hex1 = float(hw_hex)
    hw_hex = 0.5
    hexratio = hw_hex1 / hw_hex

    dx, dy = kshape  # x and y dimension of map
    xcents1 = np.arange(0., dx * (hw_hex) * 2, hw_hex * 2)
    xcents2 = np.r_[xcents1, xcents1 + hw_hex]
    xcents = np.array(list(xcents2) * int((np.floor(dy / 2.))) + int(np.floor(np.mod(dy, 2))) * list(xcents1))
    ydiff = np.sqrt(3) * hw_hex
    ycents = np.array([[ii * ydiff] * dx for ii in np.arange(dy)]).flatten()

    # get properties of the hexagons
    hex_dict = {}
    for ii in np.arange(dx * dy):
        center = [xcents[ii], ycents[ii]]
        hex_dict[ii] = [list(np.array(center) * hexratio), getHexagon(center=center, hw=hw_hex) * hexratio]

    return hex_dict


def rotate2D(pts, cnt, ang=np.pi / 4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts - cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt

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

def get_conngrid(kshape, hw_hex=0.5, hw_conn=0.04):
    hw_hex1 = float(hw_hex)
    hw_hex = 0.5
    hexratio = hw_hex1 / hw_hex
    dx, dy = kshape
    xcents1 = np.arange(0., dx * (hw_hex) * 2, hw_hex * 2)
    xcents2 = np.r_[xcents1, xcents1 + hw_hex]
    xcents = np.array(list(xcents2) * int(np.floor(dy / 2.)) + int(np.floor(np.mod(dy, 2))) * list(xcents1))
    ydiff = np.sqrt(3) * hw_hex
    ycents = np.array([[ii * ydiff] * dx for ii in np.arange(dy)]).flatten()

    # ------------------------------------------------------------------------------
    # describe connectors in terms of their centers and whether they are at the middle, left or right
    nconns = (dx - 1) * dy + (dy - 1) * ((dx - 1) * 2 + 1)
    modes_pool = ['m'] * (dx - 1) + ['l', 'r'] * (dx - 1) + ['l'] + ['m'] * (dx - 1) + ['r', 'l'] * (dx - 1) + ['r']
    conn_modes = (modes_pool * int(np.ceil(nconns / float(len(modes_pool)))))[:nconns]
    conn_ycents = np.array([[ii * ydiff] * (3 * (dx - 1) + 1) for ii in np.arange(dy)]).flatten()[:nconns]

    row1 = np.arange(hw_hex, hw_hex * 2 * (dx - 1), 2 * hw_hex)
    row2 = np.r_[np.array([[prev] * 2 for prev in row1]).flatten(), row1[-1] + 2 * hw_hex]
    row3 = row1 + hw_hex
    row4 = np.r_[row3[0] - 2 * hw_hex, np.array([[prev] * 2 for prev in row3]).flatten()]
    rows_pool = list(np.r_[row1, row2, row3, row4])
    conn_xcents = np.array((rows_pool * int(np.ceil(nconns / float(len(rows_pool)))))[:nconns])

    conn_dict = {}
    for ii in np.arange(nconns):
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

        conn_dict[ii] = [list(np.array(cent) * hexratio), mode, verts * hexratio, (p1, p2)]
    return conn_dict


def plot_hexPanel(kshape, hex_values, ax, hw_hex=0.5, showConn=True, showHexId=True, \
                  labelsOn=False, logScale=False, scalefree=False, return_scale=False, maskcol='khaki',\
                  show_mask_str=False,**kwargs):
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.pyplot import get_cmap, Polygon
    from matplotlib import colorbar, ticker

    hexcol = kwargs['hexcol'] if 'hexcol' in kwargs else 'Blues'
    conncol = kwargs['conncol'] if 'conncol' in kwargs else 'afmhot_r'
    hw_conn = kwargs['hw_conn'] if 'hw_conn' in kwargs else 0.04
    idcolor = kwargs['idcolor'] if 'idcolor' in kwargs else 'k'
    alphahex = kwargs['alphahex'] if 'alphahex' in kwargs else 0.7
    conn_ecol = kwargs['conn_ecol'] if 'conn_ecol' in kwargs else 'None'

    if logScale:
        hex_values = np.ma.array(np.log10(hex_values), mask=(~np.isfinite(np.log10(hex_values))))

    # get grids and fill with values
    if 'hexdict' in kwargs:
        hex_dict = kwargs['hexdict']

    else:
        hex_dict = get_hexgrid(kshape, hw_hex=hw_hex)
        # print len(hex_values)
        # print hex_dict.keys()
        for ii in sorted(hex_dict.keys()): hex_dict[ii] += [hex_values[ii]]

    if 'conn_ref' in kwargs:
        conn_ref = kwargs['conn_ref']
    else:
        conn_ref = hex_values[None, :]  # just take the distance between the hexvalues then!

    # print conn_ref.min(),conn_ref.max()

    if showConn:
        conn_dict = get_conngrid(kshape, hw_hex=hw_hex, hw_conn=hw_conn)
        # append data to the conngrid
        for jj in sorted(conn_dict.keys()):
            p1, p2 = conn_dict[jj][-1]
            pair_dists = np.linalg.norm(
                conn_ref[:, p1] - conn_ref[:, p2])  # euclidean distance between the two hexagons the connector connects
            conn_dict[jj] += [pair_dists]

            # pair_dists = np.abs(hex_values[p2]-hex_values[p1])  # euclidean distance between the two hexagons the connector connects
            # conn_dict[jj]+=[pair_dists]

        allpairs = np.array([conn_dict[key][-1] for key in sorted(conn_dict.keys())])

    xcents, ycents = np.array([hex_dict[key][0] for key in sorted(hex_dict.keys())]).T

    # set up the colormaps and draw
    if 'vminmax' in kwargs:
        vmin, vmax = kwargs['vminmax']
    else:
        vmin, vmax = np.nanmin(hex_values), np.nanmax(hex_values)
    cmap_hex = get_cmap(hexcol)
    cNorm_hex = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap_hex = cmx.ScalarMappable(norm=cNorm_hex, cmap=cmap_hex)

    if show_mask_str:
        if type(hex_values) == np.ma.MaskedArray:
            for hexid in np.where(hex_values.mask)[0]:
                center, temp1, temp2 = hex_dict[hexid]
                ax.text(center[0], center[1], '--', fontsize=15, color='k', fontweight='bold', ha='center', \
                        va='bottom')

    for hexid, hexparams in list(hex_dict.items()):
        center, verts, hexval = hexparams
        if 'quality_map' in kwargs:
            if np.isnan(hexval): mycol_h ='w'
            else: mycol_h = kwargs['quality_map'][hexval.astype('int')]
        else:
            mycol_h = scalarMap_hex.to_rgba(hexval)
        if type(hex_values) == np.ma.core.MaskedArray:
            if hex_values.mask[hexid]:mycol_h = maskcol
        if 'ec' in kwargs:
            ec_col = kwargs['ec']
        else:
            ec_col = mycol_h
        ax.add_patch(Polygon(verts, fc=mycol_h, ec=ec_col, alpha=alphahex))
        if showHexId == True:
            ax.text(center[0] - 0.3 * hw_hex, center[1] - 0.3 * hw_hex, str(hexid), fontsize=8, ha='center',
                    va='center', color=idcolor)

    if showConn == True:
        cmap_conn = get_cmap(conncol)
        cNorm_conn = colors.Normalize(vmin=np.min(allpairs), vmax=np.max(allpairs))
        scalarMap_conn = cmx.ScalarMappable(norm=cNorm_conn, cmap=cmap_conn)
        for connid, connparams in list(conn_dict.items()):
            center, mode, verts, pair, pair_dist = connparams
            mycol_c = scalarMap_conn.to_rgba(pair_dist)
            ax.add_patch(Polygon(verts, fc=mycol_c, ec=conn_ecol, alpha=1.))

    if not scalefree:
        if labelsOn == True:
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

        else:
            ax.set_axis_off()

        ax.set_xlim([-1.5 * hw_hex, kshape[0] - 1 + 1.5 * hw_hex])
        ax.set_ylim([-1.5 * hw_hex, kshape[1] - 1 + 1.5 * hw_hex])

        ax.set_aspect('equal', adjustable='datalim', anchor='SW')

        # ax.set_xlim([-1.5*hw_hex,kshape[0]-1+1.5*hw_hex])
        # ax.set_ylim([-1.5*hw_hex,kshape[1]-1+1.5*hw_hex])

        # ax.set_xlim([-2*hw_hex,kshape[0]-1+2*hw_hex])
        # ax.set_ylim([-3*hw_hex,kshape[1]-1+3*hw_hex])

    if 'hex_ax' in kwargs:
        hex_ax = kwargs['hex_ax']
        hex_cb = colorbar.ColorbarBase(hex_ax, cmap=cmap_hex,
                                       norm=cNorm_hex,
                                       orientation='vertical')
        # hex_cb.set_ticks([])
        hightext, lowtext = str(np.around(hex_values.max(), 1)), str(np.around(hex_values.min(), 1))
        if logScale: hightext, lowtext = '10^%s' % (hightext), '10^%s' % (lowtext)
        hex_ax.text(0.5, 0.9, hightext, rotation=-90, fontsize=15, color='w', ha='center', va='center',
                    transform=hex_ax.transAxes)
        hex_ax.text(0.5, 0.1, lowtext, rotation=-90, fontsize=15, color='k', ha='center', va='center',
                    transform=hex_ax.transAxes)
        hex_ax.set_axis_off()
        # hex_cb.set_ticks([np.around(hex_values.min(),2), np.around(hex_values.max(),2)])
        # setp(hex_ax.get_yticklabels(), rotation=-90, color='k')

    if 'conn_ax' in kwargs:
        conn_ax = kwargs['conn_ax']

        conn_cb = colorbar.ColorbarBase(conn_ax, cmap=cmap_conn,
                                        norm=cNorm_conn,
                                        orientation='vertical')
        # conn_cb.set_ticks([])
        conn_ax.text(0.5, 0.9, str(np.around(allpairs.max(), 1)), rotation=-90, fontsize=15, color='w', ha='center',
                     va='center', transform=conn_ax.transAxes)
        conn_ax.text(0.5, 0.1, str(np.around(allpairs.min(), 1)), rotation=-90, fontsize=15, color='k', ha='center',
                     va='center', transform=conn_ax.transAxes)
        conn_ax.set_axis_off()
        # setp(conn_ax.get_yticklabels(), rotation=-90, color='k')
    if return_scale:
        return {'hexint': [hex_values.min(), hex_values.max()], 'connint': [allpairs.min(), allpairs.max()]}

    ax.set_aspect('equal')


def plot_addText(ax, kshape, textvalues, hw_hex=0.5, **kwargs):
    hex_dict = get_hexgrid(kshape, hw_hex=hw_hex)
    textstrs = [str(val) for val in textvalues]
    for hexid, hexparams in list(hex_dict.items()):
        center, verts = hexparams
        if textstrs[hexid] != '':
            # print textstrs[hexid]
            # print center
            ax.text(center[0], center[1], textstrs[hexid], va='center', ha='center', \
                    **kwargs)


def get_figureparams(kshape,hw_hex=0.75,sizefac=1):

    t_h,b_h,l_w,r_w = [0.6*sizefac]+[0.2*sizefac]*3
    hex_dict =  get_hexgrid(kshape,hw_hex=hw_hex*sizefac)
    xmax,ymax = np.max(hex_dict[np.prod(kshape)-1][1],axis=0)
    xmin,ymin = np.min(hex_dict[0][1],axis=0)
    if np.mod(kshape[1],2.) == 1:xmax = xmax+hw_hex*sizefac
    pan_h, pan_w = ymax-ymin,xmax-xmin
    fheight = t_h+pan_h+b_h
    fwidth = l_w+pan_w+r_w
    return [fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax]


def make_fig_with_cbar(fwidth,fheight,l_w,r_w,b_h,t_h,xmin,xmax,ymin,ymax,add_width=1.,width_ratios=[ 1.,0.035]):
    f, axarr = plt.subplots(1, 2, figsize=(fwidth + add_width, fheight), gridspec_kw={'width_ratios': width_ratios})
    f.subplots_adjust(left=l_w / fwidth, right=1. - r_w / fwidth-0.1, bottom=b_h / fheight,
                      top=1. - t_h / fheight-0.05, wspace=0.05)  # -tspace/float(fheight)
    ax,cax = axarr
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_axis_off()
    return [f, ax,cax]

def plot_cmap(cax,cmapstr,norm,**kwargs):
    cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.cm.get_cmap(cmapstr), norm=norm, orientation='vertical',**kwargs)
    cax.tick_params(tickdir='out', labelsize=10, width=1., pad=0.3, length=3)
    return cb
