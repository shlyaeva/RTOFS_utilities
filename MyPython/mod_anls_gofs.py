import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import time
import timeit
import pickle
#from netCDF4 import Dataset as ncFile
from copy import copy
import matplotlib.colors as colors
import matplotlib.mlab as mlab

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
sys.path.append(PPTHN + '/MyPython')
sys.path.append(PPTHN + '/MyPython/mom6_utils')

from mod_utils_fig import bottom_text
import mod_time as mtime
import mod_utils as mutil
import mod_read_hycom as mhycom
import mod_colormaps as mcmp
import mod_mom6 as mom6util

def find_westcoast(HH, LON, LAT):
  """
    Find coastline West coast 
    for gofs3.1 topo 11
    Topo gile is subsampled - smaller domain
  """
  II = np.array([228, 207, 200, 183, 180, 161, 144, 101,  62,  57])
  JJ = np.array([17,  38,  56,  70,  83, 100, 149, 167, 230, 425])

  return II, JJ

def find_NWP_isobath(HH, LON, LAT, DX, z0):
  """
    Find indices of the z0 isobath along the NW Pacific coast
  """
  import mod_misc1 as mmisc

  IV, JV = find_westcoast(HH, LON, LAT)
  II, JJ = mmisc.polysegm_indx(IV,JV)

# Search coast line then go off shore until z=z0
#  z0   = -400.
  nI   = len(II)
  INDX = []
  JNDX = []
  DL   = np.array([])
  for ikk in range(nI):
    i0   = II[ikk]
    j0   = JJ[ikk]
    if ikk>0 and j0 == JJ[ikk-1]:
      continue
    aa    = HH[j0,:i0]
    inn   = max(np.argwhere(~np.isnan(aa)))[0]+1
    bb    = aa[:inn]
    izb0  = max(np.argwhere(bb < z0))[0]
    dmean = np.mean(DX[j0,izb0:inn])
    if len(INDX) == 0:
      INDX = np.array([[izb0+1,inn]])
      JNDX = np.array([j0])
    else:
      INDX = np.append(INDX, np.array([[izb0+1,inn]]), axis=0)
      JNDX = np.append(JNDX, np.array([j0]))
    DL    = np.append(DL, dmean)

    INDX = INDX.astype('int')

  return INDX, JNDX, DL


