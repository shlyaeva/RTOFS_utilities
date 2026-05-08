"""
  Subroutine for preparing SIS2 relaxation files
"""
import xarray
import os
import importlib
import numpy as np
import sys
import matplotlib.pyplot as plt
from yaml import safe_load

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
sys.path.append('./seasonal-workflow')

import mod_misc1 as mmisc1
import mod_mom6 as mom6util
importlib.reload(mom6util)
from mod_utils_fig import bottom_text


def interp2Dfld(A2d, IMOM, JMOM, INDX, JNDX, LMsk, LON0, LAT, hlon, hlat, \
                eps_err=1.e-2, land_mask=False, info_step=10000):
  """
    Interpolate A2d (2D field) from the donor grid (e.g. PIOMAS) onto target (MOM6) grid
    IMOM, JMOM - MOM6 indices where fields need to be interpolated
    INDX, JNDX - n x 4 arrays of PIOMAS grid points (gmapi) for bilinear interpolation
    LMsk - land/ocean mask of MOM6 grid
    LON, LAT - original grid of the data
    hlon, hlat - grid MOM6 - where data are being interpolated to
    land_mask - False: do not mask land with nans, keep filled with ocean values
  """
  import mod_utils_ob as muob
  import time
  import mod_bilinear as mblnr
  importlib.reload(mblnr)

  # Find basis functions for a reference rectangle:
  phi1,phi2,phi3,phi4 = mblnr.basisFn_RectRef()
  phi_basis           = np.array([phi1, phi2, phi3, phi4]).transpose() # basis funs in columns

  npnts = INDX.shape[0]
  jdm, idm = LMsk.shape
  assert np.max(LMsk) == 1 and np.min(LMsk) == 0,\
    f"LMsk should be 0 and 1, given: np.min(LMsk) and np.max(LMsk)"

  print(f'Interpolating 2D {npnts} pnts ...')
  # Make sure that gmapi is for the right section:
  assert npnts==len(IMOM), "INDX and IMOM mismatch in length"

  Ai = np.zeros((jdm,idm))
  if land_mask:
    Ai = np.where(LMsk==0,np.nan, Ai)

  start_time = time.perf_counter()
  end_time = start_time
  for ikk in range(npnts):
    if ikk%info_step == 0:
      end_time = time.perf_counter()
      dlt_time = (end_time - start_time) / 60.
      print(f'Elapsed time={dlt_time:.4f} min  {ikk/npnts*100.:.2f}% done ...')
      #start_time = time.perf_counter()
      
    imom = IMOM[ikk]
    jmom = JMOM[ikk]
    #print(f'ikk={ikk}') 
    if LMsk[jmom,imom] == 0:
      continue
    x0   = hlon[jmom, imom]
    y0   = hlat[jmom, imom]
    # Normalize x0
    x0 = (x0 + 360) % 360

    # Use cartesian coordinates for mapping
    # To avoid the wrapping discontinuity
    #  -180/180 or 360/0 discontinuity 
    # of the box vertix coordinates (e.g, xx = 359, 0.5, 0.5, 359)
    # and x0 coordinate wrt to box vertices e.g. x0 = -0.2, xx=359, 0.5, 0.5, 359
    # shift all coordinates to -180,180 wrt to x0
    #LON = mblnr.shift_longitudes(LON0, ref_lon=x0) # <-- can be slow for large LON
                                                 # instead: shift 4 longitudes xx
    II = np.squeeze(INDX[ikk,:])
    JJ = np.squeeze(JNDX[ikk,:])

    xx0 = LON0[JJ,II]
    xx = mblnr.shift_longitudes(xx0, ref_lon=x0)
    yy = LAT[JJ,II]
    # Avoid N. Pole - errors in spehrical distance calculation
    yy[yy > 89.999] = 89.999

    # Make sure the vertices are close enough:
    if np.max(abs(np.diff(xx))) > 90. or np.max(abs(xx-x0)) > 90.:
      DD = mmisc1.dist_sphcrd(y0, x0, yy, xx)
      DE = mmisc1.dist_sphcrd(yy[0], xx[0], yy, xx)
      cell_diag = np.max(DE)
      # Sanity check: target pnt should not be farther from any vertex 
      # than the farthest vertex is from reference vertex
      assert np.max(DD) <= cell_diag * 1.01, \
        f"ikk={ikk} Check lat/lon box for x0={x0:.4f}, y0={y0:.4f} \n" \
        f"(max pnt dist={max_target_dist:.2f} m, max Cell diag={cell_diag:.2f} m)"

    f_repeated= muob.check_repeated_vertices(xx,yy)
    if f_repeated:
      print(f"Bad box with coninciding vertices ikk={ikk}, approximate interpolation")
      xht = yht = 1.e-3
    else:
      # This is robust and works near the poles and long discontinuities:
      # Build transformer for projecting to local tangent plane
      # for accurate conversion lon/lat --> x,y on Cartesian grid
      # East-North-Up (ENU)
      #
      #xref, yref = mmisc1.polygon_centroid(xx,yy)
      xref, yref = xx[0], yy[0]  # reference in the 1st grid pnt
      transf_enu = mblnr.make_lonlat2xy_transformer(xref, yref) # projection centered @(xref,yref)
      XV, YV     = transf_enu.transform(xx, yy)    # convert 4 vertices
      x0c, y0c   = transf_enu.transform(x0,y0)     # convert the target pnt
      #XV, YV   = mblnr.lonlat2xy_enu(xx, yy, xref, yref)
      #x0c, y0c = mblnr.lonlat2xy_enu(x0,y0, xref, yref)

      # For rotated grid boxes, mapping may give singular matrix AA
      # Try to rotate the quadrilateral to orient sides with X and Y axis:
      XVr, YVr, x0r, y0r  = muob.rotate_box(XV, YV, x0c, y0c)
      #xht, yht = mblnr.map_x2xhat(XV, YV, x0c, y0c)   
      xht, yht = mblnr.map_x2xhat(XVr, YVr, x0r, y0r)  # map to reference coordinates

    # Fix round off errors for points on the side of the ref. square that are close to +/-1:
    if 0 < abs(xht)-1. < eps_err:
      xht = np.round(xht)
    if 0 < abs(yht)-1. < eps_err:
      yht = np.round(yht)
    if abs(xht) > 1. or abs(yht) > 1.:
      # If nothing works, interpolate into the center
      # these should be very rare for locations on land where I / J axes converge 
      print(f"Fixing by rotating ref BOX failed ikk={ikk} x0={x0:.2f} y0={y0:.2f} " +\
            f"xht={xht:8.5f} yht={yht:8.5f}, use xhy, yht as middle pnt")
      xht = yht = 1.e-3

    HT = A2d[JJ,II]
    # Typically, land values should be filled
    # in case, they have not:  Get rid off nans
    nnans = len(np.where(np.isnan(HT))[0])
    if nnans == len(HT):
      Ai[jmom,imom] = np.nan
      continue
    else:
      HT = np.where(np.isnan(HT), np.nanmean(HT), HT)

    hintp  = mblnr.bilin_interp(phi1, phi2, phi3, phi4, xht, yht, HT)

# Check: abs. values of interpolated values <= original data
    mxHT = np.max(abs(HT))
    if mxHT == 0: 
      mxHT = 1.e-20
    dmx  = abs(hintp)/mxHT
    if dmx > 1.1:
      print(f"!!! segm{nsgm} {varnm} Min/Max test violated: ikk={ikk} dlt: {dmx}")
      if dmx > 1.5:
        raise Exception("MinMax test: Interp error Check interpolation")

    Ai[jmom,imom] = hintp

  return Ai

def check_gmapi_index(ikk, A2d, IMOM, JMOM, INDX, JNDX, LMsk, LON0, LAT, hlon, hlat, \
                eps_err=1.e-2, land_mask=False):
  """
    Check gmapi indices for a point 
  """
  import mod_utils_ob as muob
  import time
  import mod_bilinear as mblnr
  importlib.reload(mblnr)
    
  # Find basis functions for a reference rectangle:
  phi1,phi2,phi3,phi4 = mblnr.basisFn_RectRef()  
  phi_basis           = np.array([phi1, phi2, phi3, phi4]).transpose() # basis funs in columns

  npnts = INDX.shape[0]
  jdm, idm = LMsk.shape
  assert np.max(LMsk) == 1 and np.min(LMsk) == 0,\
    f"LMsk should be 0 and 1, given: np.min(LMsk) and np.max(LMsk)"

  # Make sure that gmapi is for the right section:
  assert npnts==len(IMOM), "INDX and IMOM mismatch in length"

  imom = IMOM[ikk]
  jmom = JMOM[ikk]
  #print(f'ikk={ikk}') 
  assert LMsk[jmom,imom] > 0, f"Land point: ikk={ikk} imom={imom} jmom={jmom}"
    
  x0   = hlon[jmom, imom]
  y0   = hlat[jmom, imom]
  # Normalize x0
  x0 = (x0 + 360) % 360

  II = np.squeeze(INDX[ikk,:])
  JJ = np.squeeze(JNDX[ikk,:])

  xx0 = LON0[JJ,II]
  xx = mblnr.shift_longitudes(xx0, ref_lon=x0)
  yy = LAT[JJ,II]
  # Avoid N. Pole - errors in spehrical distance calculation
  yy[yy > 89.999] = 89.999

  f_repeated= muob.check_repeated_vertices(xx,yy)
  if f_repeated:
    print(f"Bad box with coninciding vertices ikk={ikk}, approximate interpolation")
    raise RuntimeError("check repeated vertices xx={xx} yy={yy}")
    
  xref, yref = xx[0], yy[0]  # reference in the 1st grid pnt
  transf_enu = mblnr.make_lonlat2xy_transformer(xref, yref) # projection centered @(xref,yref)
  XV, YV     = transf_enu.transform(xx, yy)    # convert 4 vertices
  x0c, y0c   = transf_enu.transform(x0,y0)     # convert the target pnt

  # Try to rotate the quadrilateral to orient sides with X and Y axis:
  XVr, YVr, x0r, y0r  = muob.rotate_box(XV, YV, x0c, y0c)
  xht, yht = mblnr.map_x2xhat(XVr, YVr, x0r, y0r)  # map to reference coordinates

  # Fix round off errors for points on the side of the ref. square that are close to +/-1:
  if 0 < abs(xht)-1. < eps_err:
    print(f"abs xht > 1: {xht}")
    xht = np.round(xht)
  if 0 < abs(yht)-1. < eps_err:
    print(f"abs yht > 1: {yht}")
    yht = np.round(yht)
  if abs(xht) > 1. or abs(yht) > 1.:
    print(f"Fixing by rotating ref BOX failed ikk={ikk} x0={x0:.2f} y0={y0:.2f} " +\
          f"xht={xht:8.5f} yht={yht:8.5f}, use xhy, yht as middle pnt")
    xht = yht = 1.e-3

  HT = A2d[JJ,II]
  # Typically, land values should be filled
  # in case, they have not:  Get rid off nans
  nnans = len(np.where(np.isnan(HT))[0])
  if nnans == len(HT):
    print(f"Values at 4 points are all nans")
  else:
    HT = np.where(np.isnan(HT), np.nanmean(HT), HT)

  hintp  = mblnr.bilin_interp(phi1, phi2, phi3, phi4, xht, yht, HT)


  plt.ion()
  fig1 = plt.figure(1,figsize=(9,9))
  plt.clf()
  ax1 = plt.axes([0.1, 0.15, 0.8, 0.8])
  ax1.plot(x0, y0, 'r*')
  ax1.plot(xx, yy, '.-')
  ax1.plot(xx, yy, '.-')
  ax1.plot([xx[0],xx[-1]],[yy[0],yy[-1]],'b-')
  ax1.set_title(f"ikk={ikk} imom={imom} jmom={jmom} x0={x0:.4f} y0={y0:.4f}") 

  return

def read_PIOMAS(yr0, mm0, dfpiomas, varnm):
  """
  Derive thikness or conc. fields for yr0, mm0 
  dfpiomas = dir + filename

  monthly fields
  1901 - 2010
  https://psc.apl.uw.edu/research/projects/piomas-20c/

  PIOMAS-20C is a sea ice thickness reconstruction covering the period 1901-2010. 
  It is constructed using a coupled ice-ocean model using atmospheric forcing data from 
  the ECMWF ERA-20C reanalysis to provide atmospheric forcing. Sea ice concentrations 
  from the Hadley Center HadISST v2.0 data set are assimilated to constrain the model 
  at the ice-edge. 

  """
  import mod_time as mtime

  ds_piomas = xarray.open_dataset(dfpiomas)
  varconc = 'sic'
  varthck = 'sit'

  print(f'Reading PIOMAS-reconstruct {yr0}/{mm0} {dfpiomas}')

  if not varnm=='sic' and not varnm=='sit':
    raise Exception (f'PIOMAS variables are sic and sit, requested {varnm}')


  # Find record #: days since 1901-01-01 = day=1, index=0
  dnmb0 = mtime.datenum([yr0,mm0,1])
  dnmbR = mtime.datenum([1901,1,1])
  ndays = int(dnmb0-dnmbR) + 1
  #Time  = dset['time'].data  # np datetime array
  Month = ds_piomas['month'].data
  Year  = ds_piomas['year'].data
  D     = np.sqrt((Month-mm0)**2 + (Year-yr0)**2)
  tindx = np.argmin(D)
  A2d   = ds_piomas[varnm].data[tindx,:].squeeze()  # thikness, m

  return A2d

def read_PIOMASv21(yr0, mm0, dfpiomas, varnm):
  """
  Derive thikness or conc. fields for yr0, mm0 
  dfpiomas = dir + filename

  monthly fields from PIOMAS v2.1 reanalysis
  1979-present

  monthly fields
  1979-present
  https://pscfiles.apl.washington.edu/zhang/PIOMAS/data/v2.1/

  PIOMASv2.1  is a sea ice reanalysis
  sea ice concentration (edge) is assimilated using sat. ice conc. 
  """
  import mod_time as mtime
  varthck = 'heff'
  varconc = 'area'

  if varnm == 'iconc':
    varnm = varconc
  elif varnm == 'ithkn':
    varnm = varthck

  print(f'Reading PIOMASv2.1 {yr0}/{mm0} {dfpiomas}')

  if not varnm==varconc and not varnm==varthck:
    raise Exception (f'PIOMAS variables are {varconc} and {varthck}, requested {varnm}')

  ds_piomas = xarray.open_dataset(dfpiomas)
  Month = ds_piomas['month'].data
  Year  = ds_piomas['year'].data
  D     = np.sqrt((Month-mm0)**2 + (Year-yr0)**2)
  tindx = np.argmin(D)
  assert(D[tindx]==0), f"Requested {yr0}/{mm0} not found in {dflthkn}"
  A2d   = ds_piomas[varnm].data[tindx,:].squeeze()  # thikness, m

  return A2d

def read_relax_piomas(dnmb0, pthsis, varnm):
  """
   Read PIOMAS target relaxation fields from PIOMASv2.1
   The fields are on NEP10k grid
  """
  import mod_time as mtime
  import mod_misc1 as mmisc

  YR0, MM0, DD0 = mtime.datevec(dnmb0)[:3]
  flthck  = f'piomas_heff{YR0}_v21.nc'
  varthck = 'heff'
  flconc  = f'piomas_area{YR0}_v21.nc'
  varconc = 'area'

  # Read saved relax. fields:
  YR1 = YR0
  YR2 = YR0+1
  flout = f'PIOMASv21_ithkn_iconc_{YR1}_{YR2}_monthly.nc'
  diclim = os.path.join(pthsis, flout)
  print(f'Reading relax fields from {diclim}')
  ds_rlx = xarray.open_dataset(diclim)
  Time = ds_rlx['time'].data
  TM = mmisc.convert_nptime_to_datenum(Time)
  #dnmb0 = mtime.datenum([YR0,MM0,15,12])
  D = abs(TM-dnmb0)
  itime = np.argmin(D)
  dv0 = mtime.datevec(TM[itime])
  assert dv0[0]==YR0, f'Requested YR={YR0}, year in rlx file={dv0[0]}'
  assert dv0[1]==MM0, f'Requested month={MM0}, month in rlx file={dv0[1]}'

  match varnm:
    case('ithkn'):
      ifld = 'ithkn'
    case('iconc'):
      ifld = 'iarea'

  A2dS = ds_rlx[ifld].isel(time=itime).data

  return A2dS, flout


def read_sis2_testrun(dnmb0, pthtest, prfx, varnm, use_mnth):
  """
    Read output from the ice relaxation test runs
    use_mnth - read monthly mean data
               if not, read day 15 for this month
               to compare with monthly mean PIOMAS
  """
  import mod_time as mtime
  dv0  = mtime.datevec(dnmb0)
  YR0, MM0, DD0 = dv0[:3]
  dref = dnmb0 - mtime.datenum([1993,1,1])

  print(f'{pthtest} Plot date: {YR0}/{MM0}/{DD0}')

  if use_mnth:
    floutp = 'ice_month.nc'
  else:
    floutp = 'ice_daily.nc'

  if len(prfx) > 0:
    flice_name = f'{prfx}.{floutp}'
  else:
    flice_name  = floutp

  dfsis2 = os.path.join(pthtest, flice_name)

  print(f'Reading {dfsis2}')

  dset   = xarray.open_dataset(dfsis2, decode_times=False)

  TIME = dset['time'].data
  DTM = np.abs(TIME-dref)
  itime = np.argmin(DTM)
  assert DTM[itime] < 15, f'Given date: {YR0}/{MM0}/{DD0} - Check dates in the arch file {dfsis2}'

  HIce = dset['sithick'].isel(time=itime).data
  CIce = dset['siconc'].isel(time=itime).data
  if varnm == 'iconc':
    A2d = CIce
  elif varnm == 'ithkn':
    A2d = CIce*HIce

  return A2d


def linear_distr1D(A1d, nav=3):
  """
    Spread out a value over adjacent cells
    using linear distribution function (averaging)
    input: 1D array
  """
  Afltr = A1d.copy()
  npnts = len(A1d)
  nav_hlf = int(np.floor(nav/2))
  assert nav<npnts, f"Number of averaged grid points {nav} should be < {npnts}"
  for ik in range(npnts):
    i1 = ik-nav_hlf
    i2 = ik+nav_hlf
    i1 = max([0,i1])
    i2 = min([i2,npnts-1])+1
    a_mn = np.mean(A1d[i1:i2])
    Afltr[ik] = a_mn

  return Afltr

def gauss_distr1D(A1d, icat0, sgm=1.3, conserve=True):
  """
    Spread out a value over adjacent cells
    using gaussian distribution function (averaging)
    input: 1D array
    sgm - controlls the spread of the gaussian filter
    icat0 - ice category where the initial ice is given
  """
  Afltr = A1d.copy()
  npnts = len(A1d)
  aa0 = A1d[icat0]
  XX = np.arange(npnts)
  Afltr = aa0*np.exp(-(XX-icat0)**2/(sgm**2))

  if conserve:
  # Conserve grid ice concentration:
    Afltr = aa0*Afltr/np.sum(Afltr)

  return Afltr

def adjust_hcat(ICAT, hcat, ccat, ncat, hice, eps0=1.e-6, verbose=False):
  """
    Try to Adjust ice thkns in all cats
    Dumping all extra ice into the last cat
    This algorithm works mostly when hcat[k] > hlim[k]
    ICAT - ice cat min/max thicknesses 
    Last category - max ice thickness is not bounded
  """
  chcat = hcat*ccat
  for kk in range(ncat):
    hmin = ICAT[kk]
    if kk<ncat-1:
      hmax = ICAT[kk+1] - eps0
    else:
      hmax = 1.e3

    hcat_k = chcat[kk]/ccat[kk]
    if hcat_k < hmin:
    # increase hcat and decrease ccat
      hcat_k = hmin + eps0
    elif hcat_k > hmax:
    # decrease hcat and increase ccat
      hcat_k = hmax - eps0
    hcat[kk] = hcat_k

    ctot = np.sum(ccat)
    htot = np.sum(ccat*hcat)
    if verbose:
      print(f"ice cat={kk+1} ctot={ctot:.2f} htot={htot:.2f}")

  chcat = hcat*ccat
  htot = np.sum(chcat)
  ctot = np.sum(ccat)
  dlt_chcat = htot - hice  # want dlt_chcat = 0.

  # Adjust last cat to keep hice and cice conserved:
  if abs(dlt_chcat) > eps0:
    dlt_h = dlt_chcat/ccat[ncat-1]
    hcat[ncat-1] = hcat[ncat-1] - dlt_h

  # Check ice cat limits:
  if verbose:
    dlt_hice = np.zeros((ncat))
    for kk in range (ncat):
      hmin = ICAT[kk]
      hmax = ICAT[kk+1] - eps0
      if hcat[kk] < hmin:
        dlt_hice[kk] = hcat[kk]-hmin
      elif hcat[kk] > hmax:
        dlt_hice[kk] = hcat[kk]-hmax
      print(f"Up swap: icat={kk+1} h[k] exceeds ice limits by = {dlt_hice[kk]:.6f}")
      print(f"icat={kk+1} hmin={hmin:.3f}/hmax={hmax:.3f} hcat={hcat[kk]:.3f}")

  return hcat, ccat

def adjust_high_thkn(hcat, ccat, ICAT, ncat, hice, eps0, verbose=False):
  """
    Reduce ice in thikest cat
  """
  chcat = hcat*ccat  # preserve ice mass
  hmin = ICAT[ncat-1]
  hmax = ICAT[ncat] - eps0
  hcat_k = hcat[ncat-1]
  adj_high = hcat_k > hmax # adjust excesss ice in thick. cat.

  if not adj_high:
    print(f"no high thkn found, H({ncat})={hcat_k:.3f}")

  icc = 0
  niter = 50
  while adj_high:
    icc += 1
    if icc > niter:
      break
   
    hk_old = hcat[ncat-1]
    print(f"adj_high: iter={icc} Adjusting thick cat {ncat}: hcat[k]={hcat_k:.3f}")
    if hcat_k > hmax:
      # decrease hcat and increase ccat
      hcat_k = hmax - eps0
    hcat[ncat-1] = hcat_k
    #dlt_ccat = chcat[ncat-1]/hcat_k - ccat[ncat-1]
    ccat[ncat-1] = chcat[ncat-1]/hcat_k

    # Adjust ccat to conserve ctot damping or taking ice from the thickest cats
    ctot = np.sum(ccat)
    dlt_cice = ctot-cice
    wght = ccat/np.sum(ccat)
    ccat = ccat - wght*dlt_cice
    ccat = np.where(ccat<=eps0, eps0, ccat)
    hcat = chcat/ccat

    hcat, ccat = adjust_hcat(ICAT, hcat, ccat, ncat, hice, verbose=False)

    hcat_k = hcat[ncat-1]
    dlt_hk = hcat_k - hk_old
    adj_high = hcat_k > hmax

    print(f"adj_high: iter={icc} new h({ncat})={hcat_k:.3f}, dlt = {dlt_hk:.6f}")
    if abs(dlt_hk)<1.e-6:
      break
    
  return hcat, ccat

def adjust_low_thkn(hcat, ccat, ICAT, ncat, hice, eps0, verbose=False):
  """
    Increase ice in thikest cat
    redistribute ice from lower cats
  """
  chcat = hcat*ccat  # preserve ice mass
  ithk = ncat-1
  hmin_thk = ICAT[ithk] + eps0
  hmax_thk = ICAT[ithk+1] - eps0
  hcat_thk = hcat[ithk]
  adj_low = hcat_thk < hmin_thk # adjust excesss ice in thick. cat.

  if not adj_low:
    print(f"no low thkn found, H({ncat})={hcat_thk:.3f}")

  # Estimate how much ice needs to be added for min thkn and conc:
  cmax_thk  = hice*cice/hmin_thk    # max concentration if all ice keep in thkst cat
  cmin_thk  = 1.e-3*cmax_thk        # some small conc >> eps0
  chmin_thk = hmin_thk*cmin_thk     # min ice mass in thkst cat to have required conc and thkn
  dch_thk   = chmin_thk-chcat[ithk] # change of ice mass in thkst cat

  for kk in range(ithk): 
    # go from 1st to the next to last cat:
    hmin = ICAT[kk] + eps0
    hmax = ICAT[kk+1] - eps0
    cmin = eps0
    hk_old = hcat[kk]
    ck_old = ccat[kk]
    chk_old = chcat[kk] 
    # Check if there is enough ice to adjust the thick. cat
    # Then find how much ice needs to be redistributed from this cat
    if (chk_old-hmin*cmin) >= dlt_ch:
      chk_new = chk_old-dlt_ch
      wgt = chk_new/chk_old
      hk_new = hk_old*wgt
      hk_new = np.max([hmin, hk_new])
      ccat[kk] = chk_new/hk_new
      hcat[kk] = hk_new 
    else:
    # Insufficient ice in this cat, take all leaving min ice mass 
      hcat[kk] = hmin
      ccat[kk] = cmin
      dlt_ch_k = chk_old - hcat[kk]*ccat[kk]
      assert dlt_ch_k > 0, f"dlt_chk should be >0 dlt_chk={dlt_ch_k:.4f}"
      dlt_ch = dlt_ch - dlt_ch_k

    # Adjust thickest cat:
    ccat[ithk]  = cice - np.sum(ccat[:ithk])
    chcat[ithk] = chcat[ithk] + dlt_ch
    hcat[ithk]  = chcat[ithk]/ccat[ithk]
    dlt_ch = ch_min-chcat[ithk]
    if dlt_ch <= eps0:
      break

  return hcat, ccat

def check_hcice(hcat, ccat, hice, cice, eps0=1.e-6, verb=False):
  """
    CHeck cice and hice conserved
  """
  chcat = hcat*ccat
  err_cice = False
  err_hice = False
  if abs(np.sum(chcat)-hice) > eps0:
    err_hice = True
    if verb:
      print(f"hice {hice:.3f} not conserved: {np.sum(chcat):.3f}")

  if abs(np.sum(ccat)-cice) > eps0:
    err_cice = True
    if verb:
      print(f"cice {cice:.3f} not conserved: {np.sum(cice):.3f}")

  return err_hice, err_cice

def check_hcat(hcat, ccat, ICAT, ncat, eps0, icat=-1):
  """
    Check if hcat[k] is within the cat thkn limits
    specify icat to check 1 cat
    returns adj_hcat:
    <0 - below the hmin
    >0 - exceeds hmax
    0 - within the limits
  """
  adj_hcat = 0
  if icat<-1:
    ii1=0
    ii2=ncat
  else:
    ii1=icat
    ii2=ii1+1

  for kk in range(ii0,ii1):
    if ccat[kk] < eps0:
    # No ice
      continue
    hmin = ICAT[kk]
    if kk<ncat-1:
      hmax = ICAT[kk+1] - eps0
    else:
      hmax = 1.e3

    if hcat[kk] < hmin:
      adj_hcat = hcat[kk]-hmin
    elif hcat[kk] > hmax:
      adj_hcat = hcat[kk]-hmax

  return adj_hcat 

def redistribute_hice(hice, cice, ICAT=[], eps0=1.e-6, ck_min=1.e-3, verbose=False):
  """
    Redistribute ice by ice cats
    such that the mean grid 
    ice thickness is conserved, i.e.
    htot = sum(i_cat)(h_cat(i)*conc_cat(i)) = hice
    ctot = sum(i)(conc_cat(i)) = cice
    eps0 - close to 0 value used to check errors and "0"
    ck_min = 1.e-3  ice conc. in lower cats,  some small value >> eps0

    ice thkn in the thikest cat may not match the limits for this cat !

    Start with dumping all ice into the thickest cat 
    such that hmin(k)<hice < hmax(k)
    and then redistribute 
    into lower cats
  """
  if len(ICAT) == 0:
    ICAT = np.array([1.0e-10, 0.1, 0.3, 0.7, 1.1])
  ncat  = len(ICAT)
  ithk  = ncat-1

  err_lim = 1.e-3  # allow higher error for ITD adjustment to avoid iteration errors

  hcat = np.zeros((ncat))
  ccat = np.zeros((ncat))

  #itmp = ICAT.copy()*0.
  if hice < ICAT[0]:
  # open water
    return hcat, ccat
  if cice < eps0:
  # open water
    return hcat, ccat

  # Check that there is enough ice to be i
  # distributed over the cats for min cice and hice:
  #ck_min = 1.e-3  # some small value >> eps0
  htot_min = np.sum(ck_min*ICAT)
  if hice < htot_min or cice < ck_min*ncat:
    hcat = np.zeros((ncat))+eps0
    ccat = np.zeros((ncat))+eps0
    ccat[ithk] = cice - np.sum(ccat[:ithk]) 
    hcat[ithk] = hice/ccat[ithk]    # can be some high values due to low cice
    chcat = ccat*hcat
    # Limit max hcat  aand adjust ccat allowing to be not exact?

    print(f"Not enough ice")
    return hcat, ccat

  # Find primary ice cat. where grid cell mean hice falls in:
  icat = 1e3
  hcat_k = hice/cice     # ice thickness in a category: m3/m2 --> m
  #print(f'hcat_k={hcat_k:.2f}')
  for kk in range(ncat):
    hbnd = ICAT[kk]
    if kk < ncat-1:
      hbnd_up = ICAT[kk+1]
    else:
      hbnd_up = 1.e3

    #print(f'kk={kk} hcat_k={hcat_k:.2f} hbnd={hbnd} hbnd_up={hbnd_up}')
    if hcat_k >= hbnd and hcat_k < hbnd_up:
      icat = kk
      break
   
  assert icat < ncat, f"Could not find ice cat for {hice}"
  print(f"hice={hice:.2f}m --> cat={icat+1}:  {hbnd:.2f}/{hbnd_up:.2f}")
 
  hcat = np.zeros((ncat))
  ccat = np.zeros((ncat))
  ccat[icat]  = cice
  hcat[icat]  = hice/cice 
  chcat = ccat*hcat

  err_hice, err_cice  = check_hcice(hcat, ccat, hice, cice)
  assert not err_hice, f"1. error hice not conserved"
  assert not err_cice, f"1. error cice not conserved"
    
  for kk in range(icat):
    hmin = ICAT[kk]
    if kk<ncat-1:
      hmax = ICAT[kk+1] - eps0
    else:
      hmax = 1.e3

    ccat_k = ck_min
    hcat_k = ICAT[kk] + eps0
    dch_k = ccat_k*hcat_k
    if dch_k >= chcat[icat]:
      # cannot redistribute ice, not enough in thickest cat
      print(f"icat={kk+1} cannot redistribute ice, not enough in thickest cat")
      break

    ccat[kk] = ccat_k
    hcat[kk] = hcat_k
    chcat[kk] = dch_k

    # Update the cat with initial ice:
    cnew = ccat[icat]-ccat[kk]
    cnew = np.max([ck_min, cnew])
    ccat[icat] = cnew
    chcat[icat] = chcat[icat] - dch_k 
    hcat[icat] = chcat[icat]/ccat[icat]

    # Check if ice thkn is within the limits 
    #adj_hcat = check_hcat(hcat,ccat, ICAT, ncat, eps0, icat=icat)
 
  ctot = np.sum(ccat)
  htot = np.sum(ccat*hcat)
  print(f"redistr: ctot={ctot:.3f} cice={cice:.3f}, htot={htot:.3f} hice={hice:.3f}")
  err_hice, err_cice  = check_hcice(hcat, ccat, hice, cice)
  assert not err_hice, f"1. error hice not conserved"
  assert not err_cice, f"1. error cice not conserved"

  return hcat, ccat

def redistribute_hice_v0(hice, cice, ICAT0=[], eps0 = 1.e-6, \
    itd_method='equal', verbose=False):
  """
    The script does not work well for wierd hice/cice values, e.g.
    hice=3, cice=0.3 - thickest cat may have wrong hcat[k]
    or small hice: hice=0.1, cice=0.8 - has problems with distributing
    across all cats 

    Redistribute ice thickness (grid cell mean), 1 pnt, 
    by ice categories such that the mean grid 
    ice thickness is conserved, i.e.
    htot = sum(i_cat)(h_cat(i)*conc_cat(i)) = hice
    ctot = sum(i)(conc_cat(i)) = cice
    eps0 - close to 0 value used to check errors and "0"

    ice thickn. distribution methods for initial thkn and conc distr. by cats:
    - simple = place all ice in the category that corresponds hice (grid cell mean value)
    - gauss = start with ice concentration aplying Gaussian filter on cice in the cat. = hice
    - equal = start with ice concentration equally distributed over the cats.

  """
  if len(ICAT0) == 0:
    ICAT0 = np.array([1.0e-10, 0.1, 0.3, 0.7, 1.1])
  ncat  = len(ICAT0)

  err_lim = 1.e-3  # allow higher error for ITD adjustment to avoid iteration errors

  # Add upper bound for the last cat:
  hi_max = 75.
  ICAT = np.append(ICAT0,[hi_max])

  hcat = np.zeros((ncat))
  ccat = np.zeros((ncat))

  #itmp = ICAT.copy()*0.
  if hice < ICAT[0]:
  # open water
    return hcat, ccat
  if cice < eps0: 
  # open water
    return hcat, ccat

# Find ice cat. where grid cell mean hice falls in:
  indx_cat = 999
  for kk in range(ncat):
    hbnd = ICAT[kk]
    if kk < ncat-1:
      hbnd_up = ICAT[kk+1]
    else:
      hbnd_up = 100.

    if hice >= hbnd and hice < hbnd_up:
      indx_cat = kk
      break    

  assert indx_cat < ncat, f"Could not find ice cat for {hice}"
  print(f"hice={hice:.2f} m, assigned cat={indx_cat+1} hlim={ICAT[indx_cat]:.2f}/{ICAT[indx_cat+1]:.2f}")
  hcat = np.zeros((ncat))
  ccat = np.zeros((ncat))
  ccat[indx_cat] = cice
  #chice = hice*cice      # m3/m2 or [m] - vol/m2 used in CICE chice*rho_ice = kg/m2
  # chice_cat array = hcat(k)*ccat(k) and sum(chice_cat)=sum(hcat(k)*ccat(k)) = hice !
  # so, when only 1 cat. is non-zero: chice_cat[k]=hice 
  chice_cat = np.zeros((ncat))
  chice_cat[indx_cat] = hice  # m3/m2 or [m] - vol/m2 used in CICE chice*rho_ice = kg/m2
 
  if hice/cice > np.max(ICAT):
  # ice will be too thick if redistribute by cats, try simple ITD:
    print(f"hice {hice:.3f} and cice {cice:.3f} will result in ice > {ICAT[-1]:.3f}")
    print(f"Simple ITD")
    hcat[indx_cat] = hice/cice
    htot = np.sum(hcat*ccat)
    assert abs(htot-hice) < eps0, f"Failed to keep hice Simple ITD: htot={htot:.6f}"
    return  hcat, ccat

  match itd_method:
    case('simple'):
      chcat = chice_cat.copy()
      #hcat[indx_cat] = hice/cice
      ccat = np.where(ccat<eps0, eps0, ccat)
    case('gauss'):
      chcat = gauss_distr1D(chice_cat, indx_cat, sgm=1.8) 
      ccat = gauss_distr1D(ccat, indx_cat, sgm=2.2)
    case('equal'):
      ccat = np.zeros((ncat)) + cice/ncat
    #case('weighted'):
    #  ccat = weighted_distr1D(chice_cat, indx_cat)

  # no 0 in the ccat:
  ccat = np.where(ccat<eps0, eps0, ccat)

  # First guess of h ice by categories given (hcat(k)*ccat(k)) and sum(ccat(k))=cice
  # Make sure sum(ccat)=cice
  cfctr = np.sum(ccat)/cice  # reduction factor to adjust concentration:
  ccat = ccat/cfctr
  hcat = chcat/ccat  # hice[k] may not be within ice thkn cat. limits, adjust

  # adjust hcat to keep thkn values within the ice categories:
  # readjust ccat to keep chcat[k] unchanged
  #ccat_min = 1.e-10  # min concentration to keep in each category

  # Adjust hice for all cats except for the last one
  hcat, ccat = adjust_hcat(ICAT, hcat, ccat, ncat, hice, verbose=verbose)
   
  chcat = hcat*ccat
  htot = np.sum(chcat)
  ctot = np.sum(ccat)
  
  print(f"Up swap: ctot={ctot:.4f} htot={htot:.4f}")

  # Expected that tot ice and cice are conserved here:
  assert abs(htot-hice) < eps0, f"1. htot={htot} hice={hice} do not match"
  assert abs(np.sum(ccat)-cice) < eps0, f"1. ={htot} hice={hice} do not match"

  # Check if thickiest ice needs adjustment
  # other cats should be good
  # if hcat is too low - move from lower cats
  # if hcat is too high - increase conc, reduce hcat and adjust conc everywhere to keep cice
  hmin = ICAT[ncat-1]
  hmax = ICAT[ncat] - eps0
  hcat_k = hcat[ncat-1]
  adj_high = hcat_k > hmax # adjust excesss ice in thick. cat.
  adj_low = hcat_k < hmin  # adjust too low (and negative) thkn 
  if adj_high:
    hcat, ccat = adjust_high_thkn(hcat, ccat, ICAT, ncat, hice, eps0, verbose=True)
  elif adj_low:
    hcat, ccat = adjust_high_thkn(hcat, ccat, ICAT, ncat, hice, eps0, verbose=True)

  # Check final distribution: 
  htot = np.sum(ccat*hcat) # should be conserved 
  ctot = np.sum(ccat)
 
  print(f"Input:          hice={hice:.3f}, cice={cice:.3f}")
  print(f"After redistr:  htot={htot:.3f}, ctot={ctot:.3f}")
#  assert abs(htot-hice) < err_lim, f"3. htot={htot} hice={hice} do not match"
#  assert abs(ctot-cice) < err_lim, f"3. ctot={ctot} cice={cice} do not match"

  return hcat, ccat
  
def fcast_mo_to_cal(MMI, MF):
  """
    Find calendar month corresponding to the forecast month=MF initialized in MMI
  """
  mfcast = np.arange(MMI,MMI+12)
  mfcast = np.where(mfcast > 12, mfcast-12, mfcast)
  mcal = mfcast[MF-1]

  return mcal

def cal_mo_to_fcast(MMI, MM):
  """
    Find forecast month (lead time) corresponding to the calend. mo MM for the
    forecast initialized in MMI
  """
  mfcast = np.arange(MMI,MMI+12)
  mfcast = np.where(mfcast > 12, mfcast-12, mfcast)
  if np.any(mfcast == MM):
    imo = (mfcast == MM).argmax()
  else:
    raise Exception(f"Could not find fcast month for calend mo={MMI}")
  mf = imo+1  # f/cast month number

  return mf

def cal_months_forecast(MMI, YRI=1, nyrs=1):
  """
    Create a list of calendar months for a forecast
    initialized on month MMI
    if YRI > 0 - also return a list of years
  """
  MM = np.arange(MMI,MMI+12)
  YY = MM.copy()*0 + YRI
  YY = np.where(MM>12, YY+1, YY)
  MM = np.where(MM>12, MM-12, MM)
  MCAL = MM.copy()
  YCAL = YY.copy()
  for kk in range(2, nyrs+1):
    MCAL = np.append(MCAL,MM)
    YCAL = np.append(YCAL,YY+kk-1)

  return YCAL, MCAL

def read_SPEAR_iconc_clim_interp(YR, MMI, ens_nmb, nyrs_clim=5):
  """
    Read monthly ice conc. clim from SPEAR init = MMI
    for a given year YR
  """
  import xarray
  if nyrs_clim == 5:
    ICLIM=[[1993,1997],[1995,1999],[2000,2004],[2005,2009],[2010, 2014],[2015,2019],[2016,2020]]

  ICLIM=np.array(ICLIM)
  iclm = np.where((ICLIM[:,0] <= YR) & (ICLIM[:,1] >= YR))[0][0]
  YRS,YRE = ICLIM[iclm,:]

  pthpkl = '/work/Dmitry.Dukhovskoy/anls_output/spear_ice'
  fclim = f'spear_interpNEP_siconc_clim_{YRS}_{YRE}_MI{MMI:02d}e{ens_nmb:02d}.nc'
  dfclim = os.path.join(pthpkl,fclim)
  print(f'Opening {dfclim}')
  dset = xarray.open_dataset(dfclim)
  
  return dset
   
def read_NSIDC_iconc_clim_interp(YR, nyrs_clim=5):
  """
    Read monthly ice conc. clim from NSIDC interpolated to NEP fields
    for a given year YR
  """
  import xarray
  if nyrs_clim == 5:
    ICLIM=[[1993,1997],[1995,1999],[2000,2004],[2005,2009],[2010, 2014],[2015,2019],[2016,2020]]

  ICLIM=np.array(ICLIM)
  iclm = np.where((ICLIM[:,0] <= YR) & (ICLIM[:,1] >= YR))[0][0]
  YRS,YRE = ICLIM[iclm,:]

  # NEP grid:
  fyaml = 'paths_seasfcst.yaml'
  with open(fyaml) as ff:
    pthseas = safe_load(ff)

  pthclim = pthseas['ALL']['dirnsidc_clim']
  flclim  = f'NSIDC_NRT_interpNEP_iconc_clim_{YRS}_{YRE}.nc'
  dflclim = os.path.join(pthclim,flclim)

  print(f'Opening {dflclim}')
  dset = xarray.open_dataset(dflclim)

  return dset

def calc_iconc_ithkn_mnthmean(dnmb0,pthtest,varnm, prfx='', ndav=5, outfld='icem'):
  """
    From N-day av. (ndav) fields compute monthly mean fields of ice conc 
    ice thickness (ice volume/m2)
    From SIS2 simulations
    dnmb0 - any date in the month
  """
  import mod_time as mtime
  import mod_anls_seas as manseas
  dvR   = mtime.datevec(dnmb0)
  YYR   = dvR[0]
  MMR   = dvR[1]
  mday1 = int(mtime.datenum([YYR,MMR,1]))
  mday2 = mday1 + int(mtime.month_days(MMR,YYR))-1

  Asum = None
  icc  = 0
  for dnmbR in range(mday1+1,mday2+1,ndav):
    # Find closest output:
    YR0, jday0, dnmb0, flname_out = manseas.find_closest_output(pthtest, dnmbR, fld=outfld)
    dv0  = mtime.datevec(dnmb0)
    YR0, MM0, DD0 = dv0[:3]
    jday0   = int(mtime.date2jday([YR0,MM0,DD0]))

    if MM0 != MMR:
      print(f' found month {MM0} requested {MMR}, skipping ...')
      continue

    if len(prfx) > 0:
      flice_name = f'{prfx}.icem_{YR0}_{jday0:03d}.nc'
    else:
      flice_name  = f'icem_{YR0}_{jday0:03d}.nc'
    dfsis2 = os.path.join(pthtest, flice_name)

    print(f'Reading {YR0}/{MM0:02d}/{DD0:02d}: {dfsis2}')

    dset   = xarray.open_dataset(dfsis2)

    HIce = dset['sithick'].isel(time=0).data
    CIce = dset['siconc'].isel(time=0).data
    if varnm == 'iconc':
      A2d = CIce
    elif varnm == 'ithkn':
      A2d = CIce*HIce

    if Asum is None:
      Asum = A2d.copy()
    else:
      Asum = Asum + A2d

    icc += 1

  A2d = None
  if icc>1:
    A2d = Asum / icc
  else:
    A2d = Asum.copy()

  return A2d

def cell_ithkn_sis2(dcice, imo, variconc='siconc', varimass='simass', rho_ice=905.0):
  """
    Estimate cell-mean ice thickness from sea ice mass:
    hi_cell = Mice / rho_ice * aice
    rho_ice is from SIS_input:
    RHO_ICE = 905.0  
    imo - time index of the ice field to extract = 0, ...

    Still probably not the right approach
    ice mass given in SIS output is kg (ice)/m2(ice area)
    
  """
  with xarray.open_dataset(dcice) as ds:
    M2d = ds[varimass].isel(time=imo).data.squeeze()
    C2d = ds[variconc].isel(time=imo).data.squeeze()

  hice_cell = M2d/rho_ice * C2d 

  return hice_cell

def mnthly_PIOMAS_linear_daily(diclim,dnmb0,varnm):
  """
    For more accurate comparison with simulated ice fields
    that use time-wieghted values of A(Month-1):A(Month+1) 
    Rederive monthly PIOMAS target iconc and ithkn using
    saved 2-yr fields used for irlx

    Use picewise linear Lagrange cardinal basis for interpolation
    between time (t-1) and (t+1)
  """
  import mod_misc1 as mmisc
  import mod_time as mtime

  YR0, MM0, DD0 = mtime.datevec(dnmb0)[:3]
  print(f"Deriving daily-weighted {varnm} for {YR0}/{MM0} {diclim}")
  ds_rlx = xarray.open_dataset(diclim) 
  Time = ds_rlx['time'].data
  TM = mmisc.convert_nptime_to_datenum(Time)
  D = abs(TM-dnmb0)
  itime = np.argmin(D)
  dv0 = mtime.datevec(TM[itime])
  assert dv0[0]==YR0, f'Requested YR={YR0}, year in rlx file={dv0[0]}'
  assert dv0[1]==MM0, f'Requested month={MM0}, month in rlx file={dv0[1]}'
  assert itime > 0, f'Cannot interp: No previous record to Requested time in {diclim}'
  assert itime < len(TM)-1, f'Cannot interp: No record passed Requested time in {diclim}' 
 
  match varnm:
    case('ithkn'):
      ifld = 'ithkn'
    case('iconc'):
      ifld = 'iarea'
    case _:
      raise ValueError(f"Unexpected variable name: {varnm}")

  tm_prv = np.floor(TM[itime-1])  # Time of the previous data point, i.e. previous month day 15
  tm_nxt = np.floor(TM[itime+1])  # Time of the following data point, i.e. next month
  tm0    = np.floor(TM[itime])    # Time for the current month
  Aprv   = ds_rlx[ifld].isel(time=itime-1).data.squeeze()
  Anxt   = ds_rlx[ifld].isel(time=itime+1).data.squeeze()
  A0     = ds_rlx[ifld].isel(time=itime).data.squeeze()

  dayS = int(mtime.datenum([YR0,MM0,1]))
  mdays0 = mtime.month_days(MM0,YR0)
  dayE = int(mtime.datenum([YR0,MM0,mdays0]))

  Asum = np.zeros((Anxt.shape))
  icnt = 0
  for tt in range(dayS,dayE+1):
    Phi_i = Phi_ip1 = 0.
    if tt <= tm0:
      # Cardinal basis:
      Phi_i   = (tt-tm0)/(tm_prv-tm0)
      Phi_ip1 = (tt-tm_prv)/(tm0-tm_prv)
      # Node values:
      Ai   = Aprv.copy()
      Aip1 = A0.copy() 
    else:
      Phi_i = (tt-tm_nxt)/(tm0-tm_nxt)
      Phi_ip1 = (tt-tm0)/(tm_nxt-tm0)
      # Nodal values:
      Ai   = A0.copy()
      Aip1 = Anxt.copy()
 
    Aintrp = Ai*Phi_i + Aip1*Phi_ip1
    Asum = Asum + Aintrp
    icnt += 1

  Amnth = Asum / icnt

  ds_rlx.close()

  return Amnth


def derive_iconc_contour(A2d, ic0=0.15, npmin=10):
  """
    Derive ice conc contour 
    A2d - ice conc 2D fields
    npmin - the min # of points in the contour to keep
  """

  plt.ioff()
  figA = plt.figure(10, figsize=(8,8))
  plt.clf()

  ny, nx = A2d.shape
  x = np.arange(nx)
  y = np.arange(ny)
  X, Y = np.meshgrid(x, y)

  ax0 = plt.axes([0.1,0.1,0.8,0.8])
  #ax0.contour(X, Y, HH, linestyles='solid', levels=[0], colors=[(0.5, 0.5, 0.5)])
  CS = ax0.contour(X,Y,A2d, levels=[ic0], colors=[(0,0.5,1)])
  #ax0.axis('scaled')

  # For NEP:
  xl1 = 24
  xl2 = 342
  yl1 = 565
  yl2 = 816
  ax0.set_xlim([xl1,xl2])
  ax0.set_ylim([yl1,yl2])

  SGS  = CS.allsegs[0]  # should be only 1 contoured value
  nsgs = len(SGS)

# Delete all small segments:
  CNTR = []
  for isg in range(nsgs):
    XY = SGS[isg]
    X  = XY[:,0]
    Y  = XY[:,1]
    if len(X) < npmin:
      continue

    CNTR.append(XY)

# Arranage all segments in order
  nC  = len(CNTR)
  TCNT = []

  if nC > 1:
    for ii in range(nC):
      if ii == 0:
        cntr0 = CNTR[0]
        x0   = cntr0[0,0]
        y0   = cntr0[0,1]
        dltD = 500.
      else:
        x0   = TCNT[-1,0]
        y0   = TCNT[-1,1]
        dltD = 200.

      xsgm, ysgm, imin, jmin = arange_1segm(CNTR,x0,y0, dltD=dltD)

      if len(xsgm) == 0:
        continue

      # Remove selected segment:
      CNTR.pop(imin)

      if ii == 0:
        TCNT = np.transpose(np.array((xsgm,ysgm)))
      else:
        aa   = np.transpose(np.array((xsgm,ysgm)))
        TCNT = np.append(TCNT, aa, axis=0)

  else:
    TCNT = np.array(CNTR).squeeze()


  #X = TCNT[:,0]
  #Y = TCNT[:,1]
  #axA1.plot(X,Y,'.-') 
  plt.close(figA)
  plt.ion()

  return TCNT

def derive_iconc_contour_ARC(A2d, ic0=0.15, npmin=20):
  """
    Derive ice conc contour 
    A2d - ice conc 2D fields
    npmin - the min # of points in the contour to keep
  """

  plt.ioff()
  figA = plt.figure(10, figsize=(8,8))
  plt.clf()

  ny, nx = A2d.shape
  x = np.arange(nx)
  y = np.arange(ny)
  X, Y = np.meshgrid(x, y)

  ax0 = plt.axes([0.1,0.1,0.8,0.8])
  #ax0.contour(X, Y, HH, linestyles='solid', levels=[0], colors=[(0.5, 0.5, 0.5)])
  CS = ax0.contour(X,Y,A2d, levels=[ic0], colors=[(0,0.5,1)])
  #ax0.axis('scaled')

  SGS  = CS.allsegs[0]  # should be only 1 contoured value
  nsgs = len(SGS)

# Delete all small segments:
  CNTR = []
  for isg in range(nsgs):
    XY = SGS[isg]
    X  = XY[:,0]
    Y  = XY[:,1]
    if len(X) < npmin:
      continue

    CNTR.append(XY)

# Arranage all segments in order
  nC  = len(CNTR)
  TCNT = []

  if nC > 1:
    for ii in range(nC):
      if ii == 0:
        cntr0 = CNTR[0]
        x0   = cntr0[0,0]
        y0   = cntr0[0,1]
        dltD = 500.
      else:
        x0   = TCNT[-1,0]
        y0   = TCNT[-1,1]
        dltD = 200.

      xsgm, ysgm, imin, jmin = arange_1segm(CNTR,x0,y0, dltD=1e6)

      if len(xsgm) == 0:
        continue

      # Remove selected segment:
      CNTR.pop(imin)

      if ii == 0:
        TCNT = np.transpose(np.array((xsgm,ysgm)))
      else:
        aa   = np.transpose(np.array((xsgm,ysgm)))
        TCNT = np.append(TCNT, aa, axis=0)

  else:
    TCNT = np.array(CNTR).squeeze()


  #X = TCNT[:,0]
  #Y = TCNT[:,1]
  #axA1.plot(X,Y,'.-') 
  plt.close(figA)
  plt.ion()

  return TCNT

def arange_1segm(CNTR, x0, y0, dltD=50.):
  """
    Find segment closest to x0, y0 
    arange the orientation of the segment
    so that it starts from the pnt closest to x0, y0
    CNTR - list with segments X,Y as np arrays
    Ignore contours that are > dltD points from the previous segment
  """
  nC  = len(CNTR)
  DFS = np.zeros((nC,2))*np.nan
  for isg in range(nC):
    XY = CNTR[isg]
    X  = XY[:,0]
    Y  = XY[:,1]

    d1 = np.sqrt((X[0]-x0)**2+(Y[0]-y0)**2)
    d2 = np.sqrt((X[-1]-x0)**2+(Y[-1]-y0)**2)

    DFS[isg,0] = d1
    DFS[isg,1] = d2

  imin = np.argmin(np.min(DFS, axis=1))
  jmin = np.argmin(np.min(DFS, axis=0))

  xsgm = CNTR[imin][:,0]
  ysgm = CNTR[imin][:,1]

  if jmin == 1:
    xsgm = np.flip(xsgm)
    ysgm = np.flip(ysgm)

# Disconnected segment - ignore:
  if np.min(DFS > dltD):
    xsgm = []
    ysgm = []
    imin = []
    jmin = []

  return xsgm, ysgm, imin, jmin

def mask_NEP10k_BerArc(HH,hlat):
  """
    Defines regional masks to include
    only Bering Sea or Arctic/Chukchi portion of NEP10k
  """
  jdm, idm = HH.shape
  # Check that this is NEP10k:
  assert jdm == 816 and idm ==342, f'Check domain dimensions, NEP10k j/i: {jdm}/{idm}'

  # Bering Sea - Chukchi Sea :
  hsh = -5000.
  LMsk = np.where((HH>=hsh) & (HH<0), 1, 0)
  # Mask out southern lats:
  LMsk = np.where(hlat<55.,0,LMsk)
  LMsk[:567,:] = 0
  LMsk[:,:39] = 0
  LMsk[:595,177:] = 0
  LMsk[:579,:129] = 0
  LMsk[:575,:143] = 0
  #LMsk[748:,:143] = 0

  # Remove near-boundary points:
  LMsk[810:,:] = 0
  LMsk[:,338:] = 0

  # 
  # Mask for Bering Sea
  # Bounded by the Bering Strait 
  BMsk = LMsk.copy()
  BMsk = np.where(hlat>66,0,BMsk)
  #JB,IB = np.where(BMsk==1)
  # Mask for the Arctic Oc. part of the domain:
  # Ber. Str. + S. Chukchi Shelf
  AMsk = LMsk.copy()
  AMsk = np.where(BMsk==1, 0, AMsk)
  AMsk[:,:192] = 0

  return BMsk, AMsk



