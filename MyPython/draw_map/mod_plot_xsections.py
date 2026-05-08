"""
  Plot vertical sections of T, S, ...
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import importlib
from copy import copy
import matplotlib.colors as colors
from matplotlib.patches import Polygon

PPTHN = os.environ.get('PPTHN', '').strip()
if len(PPTHN) == 0:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  ppthn_candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
  if os.path.isdir(os.path.join(ppthn_candidate, 'MyPython')):
    PPTHN = ppthn_candidate
  else:
    cwd = os.getcwd()
    parts = [pp for pp in cwd.split('/') if pp]
    if 'python' in parts:
      idx = parts.index('python')
      PPTHN = '/' + os.path.join(*parts[:idx+1])
    else:
      PPTHN = ppthn_candidate
sys.path.append(PPTHN + '/MyPython/hycom_utils')
sys.path.append(PPTHN + '/MyPython/draw_map')
sys.path.append(PPTHN + '/MyPython/mom6_utils')

import mod_read_hycom
from mod_read_hycom import read_grid_topo, read_hycom, read_topo
from mod_read_hycom import zz_zm_fromDP
from mod_utils_fig import bottom_text


def find_indx_lonlat(x0,y0,LON,LAT,xsct="none"):
  """
  Find closest grid point to lon/lat coordinate
  """
  if x0 > 180.:
    x0 = x0-360.

  dmm = np.sqrt((LON-x0)**2+(LAT-y0)**2)
  jj0, ii0 = np.where(dmm == np.min(dmm)) # global indices
#
# If section provided, indices for this section:
  ip0 = -1
  jp0 = -1

  if not xsct=="none":
    SCT = Jsections()
    aa  = SCT.get(xsct)
    i1  = aa[0]
    i2  = aa[1]
    j1  = aa[2]
    ip0 = ii0[0]-i1
    jp0 = jj0[0]-j1 

  return ii0[0], jj0[0], ip0, jp0

#
def minmax_clrmap(dmm,pmin=10,pmax=90,cpnt=0.01,fsym=False):
  """
  Find min/max limits for colormap 
  discarding pmin and 1-pmax min/max values
  cpnt - decimals to leave
  """
  dmm = dmm[~np.isnan(dmm)]
  a1  = np.percentile(dmm,pmin)
  a2  = np.percentile(dmm,pmax)
  cff = 1./cpnt
  rmin = cpnt*(int(a1*cff))
  rmax = cpnt*(int(a2*cff))

  if fsym and (rmin<0. and rmax>0.) :
    dmm = max([abs(rmin),abs(rmax)])
    rmin = -dmm
    rmax = dmm

  return rmin,rmax

def plot_2Dmap(A2d,HH,fgnmb=1,clrmp=[],rmin=0.,rmax=50.,\
               ctl="2D map",Ipnt=[],Jpnt=[],btx=" "):
  """
  Plot 2 D map
  Ipnt,Jpnt - locations to plot on the map
  """
  print('Plotting  '+ctl+'...')
  if not clrmp:
    clrmp = copy(plt.cm.afmhot_r)

  clrmp.set_bad(color=[0.7,0.7,0.7])
  plt.ion()
  fig1 = plt.figure(fgnmb,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])
  im1 = plt.pcolormesh(A2d,shading='flat',\
                       vmin=rmin, vmax=rmax, cmap=clrmp)
  plt.contour(HH,[0.0],colors=[(0.5,0.5,0.5)],linewidths=1)

  if len(Ipnt)>0:
    plt.plot(Ipnt,Jpnt,linestyle='',marker=".",markersize=5,color="m")

  ax2 = fig1.add_axes([ax1.get_position().x1+0.02,
               ax1.get_position().y0,0.02,
               ax1.get_position().height])
  if rmin == 0:
    clb = plt.colorbar(im1, cax=ax2, extend='max')
  elif rmax == 0:
    clb = plt.colorbar(im1, cax=ax2, extend='min')
  else:
    clb = plt.colorbar(im1, cax=ax2, extend='both')

  ax2.set_yticklabels(ax2.get_yticks())
  ticklabs = clb.ax.get_yticklabels()
  clb.ax.set_yticklabels(ticklabs,fontsize=12)

  plt.sca(ax1)
  ax1.set_title(ctl)

  bottom_text(btx)

# Print several locations of points:
  if len(Ipnt):
    ax3 = plt.axes([0.1,0.3,0.25,0.4])
    nmax = min(20,Ipnt.shape[0])
    ss1  = 'Some locations > eps \n  I      J\n'
    for kk in range(nmax):
      ss1 = ss1+'{0}  {1}\n'.format(Ipnt[kk],Jpnt[kk])
    ax3.text(0.1,0.1,ss1)
    ax3.axis('off')

  return

def plot_1prof(B,Z,ctl='Profile'):
  plt.ion()
  print('Plotting '+ctl)
  fig1 = plt.figure(1,figsize=(7,8),constrained_layout=False)
  plt.clf()
  plt.plot(B,Z)
  plt.title(ctl)

  return

def plot_xsection(A2d, X, Z, Hbtm, Xbtm, clrmp, rmin, rmax, \
                  xl1, xl2, sttl='Section', stxt='', **kwargs):
  """
    2D vertical section
  """
  from matplotlib.patches import Polygon

# Plot map with section:
  fmap = False
  btx  = ''
  fgnmb = 1
  patch_btm = True
  draw_cntrs = False
  for key, value in kwargs.items():
    if key == 'plot_map':
      HH = value
      fmap = True
    if key == 'I_indx':
      II = value
    if key == 'J_indx':
      JJ = value
    if key == 'btx':
      btx = value
    if key == 'fgnmb':
      fgnmb = value
    if key == 'patch_btm':
      patch_btm = value
    if key == 'contours':
      cntrs = value
      draw_cntrs = True

  yl1 = np.ceil(np.min(Hbtm))

  plt.ion()
  fig1 = plt.figure(fgnmb,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.25, 0.8, 0.7])
  im1 = ax1.pcolormesh(X, Z, A2d, \
                 cmap=clrmp,\
                 vmin=rmin, \
                 vmax=rmax)

  if draw_cntrs:
    CS = ax1.contour(X, Z, A2d, cntrs, linestyles='solid', 
                colors=[(0.,0.,0.)], linewidths=1.0)
    ax1.clabel(CS, inline=True, fontsize=8)

  # Patch bottom:
  if patch_btm:
    verts = [(np.min(Xbtm),-8000),*zip(Xbtm,Hbtm),(np.max(Xbtm),-8000)]
    poly = Polygon(verts, facecolor='0.6', edgecolor='0.6', zorder=5)
    ax1.add_patch(poly)

  ax1.set_xlim([xl1, xl2])
  ax1.set_ylim([yl1, 0])


  ax2 = fig1.add_axes([ax1.get_position().x1+0.02,
               ax1.get_position().y0,0.02,
               ax1.get_position().height])
  clb = plt.colorbar(im1, cax=ax2, extend='both')
  ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
  ax2.set_yticklabels(ax2.get_yticks())
  ticklabs = clb.ax.get_yticklabels()
  clb.ax.set_yticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
  clb.ax.tick_params(direction='in', length=12)
  plt.sca(ax1)

  ax1.set_title(sttl)

  if len(stxt) > 0:
    ax3 = plt.axes([0.1, 0.2, 0.8, 0.1])
    ax3.text(0, 0.01, stxt)
    ax3.axis('off')

# Plot map:
  if fmap: 
    import mod_utils as mutil
    LMSK = HH.copy()
    LMSK = np.where(LMSK<0.,0.,1.)
    lcmp = mutil.clrmp_lmask(2, clr_land=[0.3,0.3,0.3])
    ax3 = plt.axes([0.75, 0.02, 0.2, 0.2])
    ax3.pcolormesh(LMSK, shading='flat',\
                    cmap=lcmp, \
                    vmin=0, \
                    vmax=1)
    if np.min(HH) < -1000:
      ax3.contour(HH, [-5000,-4000,-3000,-2000,-1000], \
                  colors=[(0.9,0.9,0.9)], linestyles='solid',linewidths=1)
    ax3.plot(II,JJ,'-',color=[1.,0.2,0], linewidth=1)
    il1 = 0
    il2 = HH.shape[1]
    jl1 = 0
    jl2 = HH.shape[0]

    ax3.axis('scaled')
    ax3.set_xlim([il1,il2])
    ax3.set_ylim([jl1,jl2])
  # Set y, x axis not visible
  #ahd = plt.gca()
    xax = ax3.axes.get_xaxis()
    xax = xax.set_visible(False)
    yax = ax3.axes.get_yaxis()
    yax = yax.set_visible(False)

  if len(btx) > 0:
    bottom_text(btx,pos=[0.02, 0.03])

  return


