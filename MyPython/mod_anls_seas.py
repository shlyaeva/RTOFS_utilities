"""
  modules/ functions for analysis of seasonal runs
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray
#import yaml
from yaml import safe_load
import importlib

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
import mod_mom6 as mmom6
import mod_misc1 as mmisc

def xsct_segments_woa(sctnm, lonW, latW, fyaml='paths_seasfcst.yaml', \
                      fyaml_paths='pypaths_gfdlpub.yaml'):
  """
    Find end points of the xsection segments
    similar to segments defined for NEP MOM6
    WOA grid is Mercator
  """
  with open(fyaml) as ff:
    pthseas = safe_load(ff)

  Is  = pthseas['ANLS_NEP'][sctnm]['II']
  Js  = pthseas['ANLS_NEP'][sctnm]['JJ']

  nlegs   = len(Is) - 1
  IJ      = np.zeros((nlegs+1,2))
  IJ[:,0] = Is
  IJ[:,1] = Js

  lonW = np.where(lonW < 0, lonW+360., lonW)
# Hgrid lon. lat:
  with open(fyaml_paths) as ff:
    gridfls = safe_load(ff)

  pthtopo    = gridfls['MOM6_NEP']['seasonal_fcst']['pthgrid']
  fgrid      = gridfls['MOM6_NEP']['seasonal_fcst']['fgrid']
  ftopo_mom  = gridfls["MOM6_NEP"]["seasonal_fcst"]["ftopo"]
  hgrid      = xarray.open_dataset(os.path.join(pthtopo,fgrid))
  hmask      = xarray.open_dataset(os.path.join(pthtopo, 'ocean_mask.nc'))
  dstopo_nep = xarray.open_dataset(os.path.join(pthtopo, ftopo_mom))
  dfgrid_mom = os.path.join(pthtopo, fgrid)
  hlon, hlat = mmom6.read_mom6grid(dfgrid_mom, grdpnt='hgrid')
  HH         = dstopo_nep['depth'].data
  HH         = np.where(HH < 1.e-20, np.nan, HH)
  HH         = -HH
  HH         = np.where(np.isnan(HH), 1., HH)
  DX, DY     = mmom6.dx_dy(hlon,hlat)
  SGMT       = mmisc.define_segments(IJ, DX, DY, curve_ornt='positive',\
                                     check_pole=False)
  II         = SGMT.I_indx
  JJ         = SGMT.J_indx
  nLeg       = SGMT.Leg_number
  hLsgm1     = SGMT.half_Lsgm1
  hLsgm2     = SGMT.half_Lsgm2
  XX         = hlon[JJ,II]
  YY         = hlat[JJ,II]
  Hbtm       = HH[JJ,II]
  LSgm       = np.zeros((len(II)))  # total segment length = half1 + half2

  for ik in range(len(II)):
     LSgm[ik] = hLsgm1[ik] + hLsgm2[ik]

# Distance along the section
# normalize by the total distance
  Lsection = mmisc.dist_sphcrd(YY[-1],XX[-1],YY[0],XX[0]) # total length section, m
  Xdist = np.cumsum(LSgm)
  Xdist = Xdist-Xdist[0]
  Xdist = Xdist/Xdist[-1]*Lsection*1.e-3  # normalized, km

# Find end points of the segments for WOA:
# For latitudinal sections - simply follow the lat:
  nlat = sctnm.split("_")[1]
  if nlat[-1] == 'N':
    lat1 = float(nlat[:2])
    lat2 = lat1
    lon1 = XX[1]
    lon2 = XX[-1]
    Xs = np.array([lon1, lon2])
    Ys = np.array([lat1, lat2])
  else: 
# Not a latitud. section:
    Xs = np.zeros((len(Is)))
    Ys = np.zeros((len(Is)))
    for ii in range(len(Is)):
      i0 = Is[ii]
      j0 = Js[ii]
      Xs[ii] = hlon[j0,i0]
      Ys[ii] = hlat[j0,i0]    

# Find WOA indices:
  IsWOA = np.zeros((len(Xs)), dtype='int') - 999
  JsWOA = np.zeros((len(Xs)), dtype='int') - 999
  for ii in range(len(Xs)):
    x0 = Xs[ii]
    y0 = Ys[ii]
    dlat = abs(latW - y0)
    ilat = np.argmin(dlat)
    lat0 = latW[ilat]
    LAT  = np.zeros((len(lonW))) + lat0
    DD = mmisc.dist_sphcrd(LAT, lonW, y0, x0)
    ilon = np.argmin(DD)

    IsWOA[ii] = ilon
    JsWOA[ii] = ilat

# MOM6 info for plotting:
  indxsct   = np.arange(len(Xdist))
  dim_name  = "sect_indx"
  darr_btm  = xarray.DataArray(Hbtm, dims=(dim_name), \
            coords={dim_name: indxsct})
  darr_dist = xarray.DataArray(Xdist, dims=(dim_name), \
            coords={dim_name: indxsct})
  darr_lon  = xarray.DataArray(XX, dims=(dim_name), \
            coords={dim_name: indxsct})
  darr_lat  = xarray.DataArray(YY, dims=(dim_name), \
            coords={dim_name: indxsct})
  dsetBtm   = xarray.Dataset({
              "Hbtm_section": darr_btm, \
              "Dist_section": darr_dist, \
              "Lon_section": darr_lon, \
              "Lat_section": darr_lat
              }) 

  return IsWOA, JsWOA, dsetBtm

def season_decade_woa(YR, MM, month2season=True, decadal_clim=True):
  """
    Find WOA season and year span for decadal averages
    File naming convention:
    woa23_[DECA]_[v][tp][ft][gr].[form_end] - all formats, except NetCDF
    woa23_[DECA]_[v][tp]_[gr].[form_end] - NetCDF format
    where:
    [DECA] - decade
    [v] - variable
    [tp] - time period
    [ft] - field type
    [gr] - grid
    [form_end] - file name extention
    Note: '.dat' - ASCII; '.csv' - comma separated value; 
          '.dbf', '.shp', '.shx' - ArcGIS shape files; '.nc' - netCDF files

    Some monthly climatologies are all-time periods 
    Seasonal climatologies are by decades
    Find seasons for given month MM if month2season
    month2season = True : convert MM to the season to find decadal clim
                   False: need montly overall climatology, not decadal mean

    decadal_clim = True: read decadal climatology
                   False: -"- -"- all-years mean climatology
  """
  SEAS = {"1" : 13,
          "2" : 13,
          "3" : 13, 
          "4" : 14,
          "5" : 14, 
          "6" : 14,
          "7" : 15, 
          "8" : 15,
          "9" : 15, 
          "10": 16,
          "11": 16, 
          "12": 16}

  DECA = np.array([[1955, 1964],
                   [1965, 1974],
                   [1975, 1984],
                   [1985, 1994],
                   [1995, 2004],
                   [2005, 2014],
                   [2015, 2022]])

  if MM > 12:
    seas = 0    # annual
  else:
    seas = SEAS[f"{MM}"]

  if not month2season:
    seas = MM

  if not decadal_clim:
    yr1 = 1900
    yr2 = 2023
    decade = 'decav'  # overall mean climatology
  else:
    if YR > np.max(DECA) or YR < np.min(DECA):
      raise Exception(f"{YR} is not in the time range for WOA")
    ii = np.where((DECA[:,0]<=YR) & (DECA[:,1]>=YR))[0][0]
    yr1, yr2 = DECA[ii,:]
    yr1_end = f"{yr1}"[2:5]
    yr2_end = f"{yr2}"[2:5]
    if YR < 1995:
      decade = yr1_end+yr2_end
    elif YR >= 1995 and YR < 2005:
      decade = "95A4"
    elif YR >= 2005 and YR < 2015:
      decade = "A5B4"
    elif YR >= 2015 and YR < 2023:
      decade = "B5C2"

  return seas, decade, yr1, yr2


def calc_N2(T, S, ZZ, latW):
  """ 
  Input T, S - 2D or 3D arrays
#  Calculate N2 = -g/rho*d2(rho)/dz2 
#  Using in situ T and S - calculate rho relative to the
#  mid-grid depth - following Chelton, 1996
# Use the neutral density gradient method, Chelton et al., 1996

  Note for model output, T, S are in the mid-depths,
  then rho's are in the interfaces, surface rho is missing - BC

  """
  import mod_swstate
  #importlib.reload(mod_swstate)
  from mod_swstate import sw_press
  from mod_swstate import adiab_Tgrad
  from mod_swstate import sw_ptmp
  from mod_swstate import sw_dens0
  from mod_swstate import sw_smow
  from mod_swstate import sw_seck
  from mod_swstate import sw_dens

#  print('Calculating pressure at midpoints ...')
  kdm, idm = T.shape    
  grav   = 9.81
  rho0   = 1025.0
  Z_phi  = np.zeros(kdm-1)  # depths for e/function Phi
  N2     = np.zeros((kdm-1, idm))
  n2fill = 1.e-8     # missing values

  for kk in range(kdm-1):
#    print(' Layer {0}'.format(kk))
    z1 = ZZ[kk]
# To avoid negative P at the surface:
    if abs(z1) < 1.e-30: z1 = 0.
    z2 = ZZ[kk+1]
    t1 = T[kk,:].squeeze()
    t2 = T[kk+1,:].squeeze()
    s1 = S[kk,:].squeeze()
    s2 = S[kk+1,:].squeeze()
    Z1 = z1*np.ones((idm))
    Z2 = z2*np.ones((idm))
    p1_db, p1_pa = sw_press(Z1,latW)  # pressure upper interface
    p2_db, p2_pa = sw_press(Z2,latW)  # pressure bottom interface
  #
  # Find mid-point of the layers
    p0_db = 0.5*(p1_db+p2_db)
    z0    = 0.5*(z1+z2)
    Z_phi[kk] = z0
#    print(f'z1={z1:5.1f} z2={z2:5.2f} p1={np.nanmin(p1_db):5.1f}'  +\
#           f' p2={np.nanmin(p2_db):5.1f} z0={z0:5.1f}')
  #
  # Calculate rho(z1--->z0) with depth reference at midpoint
  # and rho(z2--->z0) at midpoint 
    t_z1z0 = sw_ptmp(s1,t1,p1_db,p0_db)
    t_z2z0 = sw_ptmp(s2,t2,p2_db,p0_db)
    rho_z1z0 = sw_dens(s1,t_z1z0,p0_db)
    rho_z2z0 = sw_dens(s2,t_z2z0,p0_db)

  # Calculate d(rho)/dz for z0 - center-difference
    drho_dz  = (rho_z1z0 - rho_z2z0)/(z1 - z2)
    N2z0     = -grav/rho0*drho_dz
    N2[kk,:] = N2z0
#    print(f'z1={z1} z2={z2} z0={z0}')
#    print(f'k={kk}, z={z0}, min/max N2: {np.nanmin(N2z0)}, {np.nanmax(N2z0)}')
#
# If N2 < 0 - density inversion happens in some ~homogeneous layers
# when parcels are brought down from z1 and up from z2
# replace with above N2 or below of surface layer
  print('Fixing N2<0')
  for kk in range(kdm-1):
    N2z0=N2[kk,:].squeeze()
    if kk > 0:
      dmm = N2[kk-1,:].squeeze()
      N2z0 = np.where(N2z0 < 0., dmm, N2z0)
    else:
      dmm = N2[kk+1,:].squeeze()
      N2z0 = np.where(N2z0 < 0., dmm, N2z0)

    N2z0     = np.where(N2z0<0, n2fill, N2z0)
    N2[kk,:] = N2z0

#    print(f'k={kk}, z={z0}, min/max N2: {np.nanmin(N2z0)}, {np.nanmax(N2z0)}')

  return N2, Z_phi

def solve_SturmLiouville(N2z, Hb0, Z_phi, latj, mode=1):
  """

  # Numerically Solve Sturm-Liouville e/value problem 

  """
  import mod_solver as msolv
  importlib.reload(msolv)
 
  omg = 7.29e-5 # Earth angular velocity
  #tic = timeit.default_timer()
  #ticR = timeit.default_timer()
#  print(f'Solving Strum-Liouville, Hbtm={Hb0:4.1f}, requested mode={mode}')

  if np.isnan(N2z[0]): 
    print(f'N2 profile is nan, land point? Skipping')
    return np.nan, np.nan, np.nan 

  k1 = np.where(np.isnan(N2z))[0]
  if k1.size:   
    kbtm = k1[0]-1
  else:
# Bottom deeper than the last layer
# Check if this is so
    kbtm = N2z.shape[0] - 1
    if abs(Hb0) < abs(Z_phi[-1]):
      raise Exception(f"Bottom={Hb0} last z={Z_phi[-1]} expected Z_phi[-1]>Hb0")

  zbtm = Hb0
  # Create Matrix A with Dk, Dk+1 for 2nd derivative of N2
  AA = msolv.form_mtrxA(Z_phi, kbtm, zbtm)

  # Form 1/N2*AA:
  # The matrix is cutoff by the bottom depth
  ka, na = AA.shape
  N2A = np.zeros((ka,na))
  for kk in range(ka):
  #  n2 = N2zF[kk]
    n2 = N2z[kk]  
    if n2 == 0:
      n2=1.e-12
    N2A[kk,:] = 1./n2*AA[kk,:]

# For now: use python eig function, for this
# form Matrix A unfolding
# the 3 -elemnts form and find eigenvalues/vectors
# W - eigenvalues, not sorted out by magnitude!
# V- corresponding eigenvectors (in columns)
    W, V = msolv.eig_unfoldA(kbtm, N2A)

# Calculate
# Choose lmbd1 = as min(abs(W)) and lmbd1<0
  absW = np.abs(W)
  Indx = np.argsort(absW)
  im   = Indx[mode-1]    # 1st mode
#  im2  = Indx[1]    # 2nd mode
#  im   = np.argmin(absW)
  if W[im] > 0.:
    print('ERROR: W[im] >0: im={0}, W[im]={1}, zbtm={4:6.1f}, ii={2}, jj={3}'.\
           format(im, W[im], ii, jj, zbtm))

#  latj = latW[ik]
  Rearth = 6371.e3  # Earth R
  fcor = 2.*omg*np.sin(np.deg2rad(latj))
  betta = (2.*omg*np.cos(np.deg2rad(latj)))/Rearth

  if abs(latj) >= 5.0:
    lmbd = np.sqrt(-1./(W[im]*fcor**2))
  else:
    lmbd = (-1./(4.*(betta**2)*(W[im])))**(1./4.)

  RsbNum = lmbd*1.e-3  # km
  Phi    = V[:,im]   # eigenfunction
  mu     = W[im]     # e/value ( <0)
  Cphs   = np.sqrt(-1./mu)  # m-th mode phase speed m/s

# Insert BCs' for Phi:
# Surface:
  Phi = np.insert(Phi, 0, 0.)
# Add phi at the bottom
  Phi = np.append(Phi, 0.)

  return Phi, RsbNum, Cphs 

def derive_dP_WOA(ZZ, A3d):
  """
  Estimate layer thicknesses for WOA Z layers
  Use any 3D field to find bottom
  """
  import mod_mom6 as mmom6
  kdm, jdm, idm = A3d.shape
  dP = A3d.copy()*0.
  ZM = mmom6.zz2zm(ZZ)
  ZM = np.append(ZM, ZZ[-1]) 

  for kk in range(kdm):
    a2d = A3d[kk,:].squeeze()
    if kk == 0: 
      dP[kk,:] = abs(ZM[0])
# Keep land mask to avoid land points
      J,I = np.where(np.isnan(a2d))
      dP[kk,J,I] = np.nan
    else:
      dP[kk,:] = abs(ZM[kk]-ZM[kk-1])
      J,I = np.where(np.isnan(a2d))
      dP[kk,J,I] = 0.
      dP[kk,:] = np.where(np.isnan(dP[kk-1,:]), np.nan, dP[kk,:])  # keep land mask

  return dP

def derive_dP_from_ZZtopo(ZZ, HH):
  """
  Derive layer thickness array from interface depths ZZ and bottom topography HH
  dP below bottom = 0.
  ZZ can be 1D or 3D, HH = 2D
  ZZ and HH use negative depths
  """
  if np.min(HH) > 0.:
    raise Exception("HH topography depths has to be <0")
  if np.min(ZZ) > 0:
    print(f"WARNING: ZZ > 0, converting to negative depths")
    ZZ = -ZZ

  ndim = len(ZZ.shape)
  jdm, idm = HH.shape
  kdm = ZZ.shape[0]-1
  if ndim == 1:
    Z3d   = np.tile(ZZ, idm*jdm).reshape((idm,jdm,kdm+1))
    Z3d   = np.transpose(Z3d, (2, 1, 0))
  elif ndim == 3:
    Z3d = ZZ

  dP3d = abs(np.diff(Z3d, axis=0))

# Make 0-thicknesses
  for kk in range(kdm):
    zup  = Z3d[kk,:].squeeze()
    zbtm = Z3d[kk+1,:].squeeze()
    dmm  = dP3d[kk,:].squeeze()
    dmm  = np.where(zup < HH, 0., dmm)
    dmm  = np.where( (zup > HH) & (zbtm < HH), abs(HH-zup), dmm)
    dP3d[kk,:] = dmm

  return dP3d

def find_closest_output(pthoutp, dnmb0, fld='oceanm', days_err=10):
  """
    Find closest output file to given date
    MOM/HYCOM file naming assumed: fld_YYYY_DAY.nc
    days_err - max dlt days between requested and found file 
  """
  import os
  import mod_time as mtime

  if not os.path.isdir(pthoutp):
    print(f'not exist: {pthoutp}')
    return 0, 0, 0, 'none'

# Find 1st ouptut date:
  LF = os.listdir(pthoutp)
#  print(LF)
  if len(LF) == 0:
    raise Exception(f'No files in {pthoutp}')
#  YR0 = np.zeros((len(LF)))
#  JD0 = np.zeros((len(LF)))
  icc = -1
  for fls in LF:
    #print(fls)
    bsname = fls.split(".")[0]
    fld_name =  bsname.split("_")[0]
    if not fld_name == fld: continue
    try: 
      yrf    = int(bsname.split("_")[1])
      jday   = int(bsname.split("_")[2])
    except:
    # Skip files that do match file naming pattern
      #print(f"skipping {fls}")
      continue
    icc += 1
    if icc == 0:
      YR0 = np.array([yrf])
      JD0 = np.array([jday])
    else:
      YR0 = np.append(YR0,yrf)
      JD0 = np.append(JD0,jday)
#    print(f'icc={icc} yr={yrf} jday={jday}')

  if icc < 0:
    print(f'ERR: no files found {fld}* in {pthoutp}')
    print(LF)
    raise Exception(f'Check fld={fld} ???')
 
  DNMB  = mtime.jday2dnmb(YR0, JD0)
  D     = np.sqrt((DNMB-dnmb0)**2)
  ii    = np.argmin(D)
  year  = YR0[ii]
  jday  = JD0[ii]
#  print(f'Min ii={ii} year={year} jday={jday} dnmb={mtime.jday2dnmb(1994,58)}')
  dnmb  = mtime.jday2dnmb(year, jday)
  flname= LF[ii]

  if D[ii] > days_err:
    print(f'Closest match is {D[ii]} days apart from requested day, increase days_err to override')
    raise Exception(f' Could not find closest file within {days_err} days ')

  return int(year), int(jday), dnmb, flname

def derive_ice_contour(AA, tz0=0.15, nmin=10):
  """
    Deduce coordinates of the ice edge contour, AA 2D field with 0 - 1 ice conc
    nmin - min # of points to keep the contour
  """
  plt.ioff()
  figA = plt.figure(10,figsize=(8,8))
  plt.clf()

  axA1 = plt.axes([0.1, 0.2, 0.7, 0.7])
  CS   = axA1.contour(AA,[tz0])

  axA1.axis('equal')
#  axA1.set_xlim([xl1,xl2])
#  axA1.set_ylim([yl1,yl2])

  SGS  = CS.allsegs[0]  # should be only 1 contoured value
  nsgs = len(SGS)
  
# Delete all closed contours and small contours
  CNTR = []
  for isg in range(nsgs):
    XY = SGS[isg]
    X  = XY[:,0]
    Y  = XY[:,1]

    if len(X) <= nmin: continue

    dEnd = np.sqrt((X[0]-X[-1])**2+(Y[0]-Y[-1])**2)
    if dEnd < 1.:
      continue

    CNTR.append(XY)
  
  plt.close(figA)

  plt.ion()

  return CNTR

def monthly_avrg_vertxsect(pthfcst, yrR, moR, JJ, II, varnm):
  """
    Compute monthly average fields from n-daily mean output
    for 2D vertical sections
  """
  LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
  if len(LOUTP) == 0:
    print(f'No output found in {pthfcst}')
    return []
 
  print(f'Computing mean {varnm} for {yrR}/{moR} nrec={len(LOUTP)}')
  cc = 0
  for ifl in range(len(LOUTP)):
    flocn_name = LOUTP[ifl]
    dfmom6 = os.path.join(pthfcst, flocn_name)
    dset   = xarray.open_dataset(dfmom6)
    if varnm == 'temp' or varm == 'potT':
      A2d = dset['potT'].data[0,:,JJ,II].squeeze()
    elif varnm == 'salin' or varnm == 'salt':
      A2d = dset['salt'].data[0,:,JJ,II].squeeze()
    A2d = np.transpose(A2d)

    cc += 1
    if cc == 1:
      Asum = A2d.copy()
    else:
      Asum = Asum + A2d

  Asum = Asum/float(cc)

  return Asum

def derive_bottom_temp(T3d, dP, dpmin=1.e-1):
  """
    Derive bottom temperature / salinity / etc  from 3D T/S/U/... field
    Bottom layers defined on min layer thickness dpmin
    dP - layer thicknesses
  """
  kdm, jdm, idm = T3d.shape
  Tbtm = np.zeros((jdm,idm))*np.nan
  for ik in range(1,kdm):
    dpup  = dP[ik-1,:].squeeze()
    dpbtm = dP[ik,:].squeeze()
    tz    = T3d[ik-1,:]
    if ik < kdm-1:
      Jb, Ib = np.where( (dpup > dpmin) & (dpbtm <= dpmin) )
    else:
  # Deep layers include all left:
      Jb, Ib = np.where( dpup > dpmin )
    if len(Jb) == 0: continue
    Tbtm[Jb, Ib] = tz[Jb, Ib]

  return Tbtm

def yrmo_seasonal_fcst(yr_init, mo_init, nmo_fcst=12):
  """
    Find calendar years / months for a seasonal f/cast that starts on yr_init / mo_init
    Returns an array of YRS/Months for the forecast period
  """
  dstrt = mtime.datenum([yr_init, mo_init,15])
  dold = dstrt - 32

  Time = np.zeros((nmo_fcst, 2)).astype(int)
  for imo in range(nmo_fcst):
    dnew = dold + 32
    dv_new = mtime.datevec(dnew)
    dnew = mtime.datenum([dv_new[0], dv_new[1], 1])
    dold = dnew
    Time[imo,0] = int(dv_new[0])
    Time[imo,1] = int(dv_new[1])

  return Time

def mocalend_from_mofcast(YYI, MMI, MMF):
  """
    Find calendar month & year corresponding to the forecast lead month MMF
    for the f/cast run initializaed on YYI/MMI
  """
  mtot = MMI+MMF-1
  nyr  = (mtot-1) // 12
  YYF  = YYI + nyr
  MMF  = mtot%12
  if MMF == 0:
    MMF = 12

  return YYF, MMF
  

def mofcst_from_mocalend(yr_init,mo_init,MM):
  """
    Find forecast month # wrt to init date yr_init/mo_init  given calendar month
  """
  if MM < mo_init:
    MM = MM + 12
  mo_fcst = MM-mo_init+1

  return mo_fcst

def yr_init_fcst_from_datenum(dnmb, MMI):
  """
    Find init year of the forecast given init month=MMI that includes
     date = datenum (day number dnmb) 
    For year-long f/casts, i.e. the f/fcasts initiliazed on MMI/1 run for 12 months 
  """
  yr0, mm0, dd0 = mtime.datevec(dnmb)[:3]
  dnmbF_start  = mtime.datenum([yr0,MMI,1])
  dmm = dnmbF_start + 367
  yy, mm, dd = mtime.datevec(dmm)[:3]
  dmm = mtime.datenum([yy,mm,1])
  dnmbF_end = dmm - 1 
  if dnmb >= dnmbF_start and dnmb <= dnmbF_end:
    yr_init = yr0
  elif dnmb < dnmbF_start:
    yr_init = yr0-1
  else:
    raise Exception(f"Could not find year init for {yr0}/{mm0}/{dd0}")

  return yr_init

def monthly_mean_from_Ndaily_ocean2D(pthfcst0, yr_init, mo_init, varnm, ocnfld, vlr, MAVRG=[1]):
  """
    From N-day average ocean 3D fields: 
    for 1 v. layer only
    compute monthly average 2D field for given variable and model layer(s)
    Field is averaged over MAVRG months (1 = Jan, 2 - Feb, etc)
    Months will be related to the years of the forecast, i.e.
    Jan for the f/cast that starts on 2013/10 will be Jan-2014
  """
  import pandas as pd

  mo_fcsts = yrmo_seasonal_fcst(yr_init, mo_init)
# Time-average fields:
  icc = 0
  ilr = vlr-1
  Time = []
  for imo in MAVRG:
    ix = np.where(mo_fcsts[:,1] == imo)[0][0]
    yr_fcst = mo_fcsts[ix,0]
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{yr_fcst}{imo:02d}')
    print(f'Avergaing {varnm} {yr_fcst}/{imo:02d}')

    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      continue

    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      if varnm == 'temp' or varnm == 'potT':
        A2d = dset['potT'].isel(time=0, zl=ilr).data.squeeze()
      elif varnm == 'salin' or varnm == 'salt':
        A2d = dset['salt'].isel(time=0, zl=ilr).data.squeeze()
      elif varnm == 'ssh':
        A2d = dset['ssh'].isel(time=0).data.squeeze()

      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]
      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

      if icc == 0:
        Asum = A2d.copy()
      else:
        Asum = Asum + A2d

      icc += 1

  Asum = Asum / icc
  print(f'N time records={icc}, min/max = {np.nanmin(Asum)} / {np.nanmax(Asum)}')  

  return Asum, Time

def monthly_mean_from_Ndaily_ocean3D(pthfcst0, yr_init, mo_init, varnm, \
                                     ocnfld, MAVRG=[1], mnth='calendar'):
  """
    From N-day average ocean 3D fields: 
    for all v. layers
    compute monthly average 2D field for given variable and model layer(s)

    Default (mnth='calendar'):
    Field is averaged over MAVRG months (1 = Jan, 2 - Feb, etc)
    Months will be related to the years of the forecast, i.e.
    Jan for the f/cast that starts on 2013/10 will be Jan-2014

    mnth='fcst':
     Month are counted from the initial time (i.e. lead time)
     mnth=1 - 1st month of the f/cast, etc. 
  """
  import pandas as pd

  mo_fcsts = yrmo_seasonal_fcst(yr_init, mo_init)
# Time-average fields:
  icc = 0
  Time = []
  for imo in MAVRG:
    if mnth=='calendar':
      ix = np.where(mo_fcsts[:,1] == imo)[0][0]
    elif mnth=='fcst':
      ix = imo-1
    yr_fcst   = mo_fcsts[ix,0]
    mnth_fcst = mo_fcsts[ix,1]
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{yr_fcst}{mnth_fcst:02d}')
    print(f'Avergaing {varnm} {yr_fcst}/{mnth_fcst:02d}')

    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      continue

    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      if varnm == 'temp' or varnm == 'potT':
        A3d = dset['potT'].isel(time=0).data.squeeze()
      elif varnm == 'salin' or varnm == 'salt':
        A3d = dset['salt'].isel(time=0).data.squeeze()

      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]
      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

      if icc == 0:
        Asum = A3d.copy()
      else:
        Asum = Asum + A3d

      icc += 1

  Asum = Asum / icc
  print(f'N time records={icc}, min/max = {np.nanmin(Asum)} / {np.nanmax(Asum)}')  

  return Asum, Time

def monthly_TSbtm_Ndaily_ocean3D(pthfcst0, yr_init, mo_init, varnm, \
                                     ocnfld, MAVRG=[1], mnth='calendar'):
  """
    From N-day derive T/S bottom statistics from 3D fields: 

    Default (mnth='calendar'):
    Field is averaged over MAVRG months (1 = Jan, 2 - Feb, etc)
    Months will be related to the years of the forecast, i.e.
    Jan for the f/cast that starts on 2013/10 will be Jan-2014

    mnth='fcst':
     Month are counted from the initial time (i.e. lead time)
     mnth=1 - 1st month of the f/cast, etc. 
  """
  import pandas as pd

  mo_fcsts = yrmo_seasonal_fcst(yr_init, mo_init)
# Time-average fields:
  icc = 0
  Time = []
  for imo in MAVRG:
    if mnth=='calendar':
      ix = np.where(mo_fcsts[:,1] == imo)[0][0]
    elif mnth=='fcst':
      ix = imo-1
    yr_fcst   = mo_fcsts[ix,0]
    mnth_fcst = mo_fcsts[ix,1]
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{yr_fcst}{mnth_fcst:02d}')
    print(f'Avergaing {varnm} {yr_fcst}/{mnth_fcst:02d}')

    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      continue

    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      if varnm == 'temp' or varnm == 'potT':
        A3d = dset['potT'].isel(time=0).data.squeeze()
      elif varnm == 'salin' or varnm == 'salt':
        A3d = dset['salt'].isel(time=0).data.squeeze()

      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]
      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

      # Read dP only once
      if icc == 0:
        dP = dset['h'].data.squeeze()
        dP = np.where(dP < 1.e-3, 0., dP)

      Abtm = derive_bottom_temp(A3d, dP)
      Abtm = np.expand_dims(Abtm, axis=0)

      if icc == 0:
        AA = Abtm.copy()
      else:
        AA = np.append(AA, Abtm, axis=0)

      icc += 1

  # Get statistics:
  Amean = np.mean(AA, axis=0)
  Astd  = np.std(AA, axis=0)

  print(f'N time records={icc}, min/max mean = {np.nanmin(Amean):.3f} / {np.nanmax(Amean):.3f}')  

  return Amean, Astd, Time

def monthly_depth_mean_from_Ndaily3D(pthfcst0, yr_init, mo_init, varnm, ocnfld, vlr1, vlr2, MAVRG=[1]):
  """
    From N-day average ocean 3D fields: 
    average over v. layers and 
    compute monthly average 2D field for given variable and model layer(s)
    Field is averaged over MAVRG months (1 = Jan, 2 - Feb, etc)
    Months will be related to the years of the forecast, i.e.
    Jan for the f/cast that starts on 2013/10 will be Jan-2014
  """
  import pandas as pd

  mo_fcsts = yrmo_seasonal_fcst(yr_init, mo_init)
# Time-average fields:
  icc = 0
  Time = []
  for imo in MAVRG:
    ix = np.where(mo_fcsts[:,1] == imo)[0][0]
    yr_fcst = mo_fcsts[ix,0]
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{yr_fcst}{imo:02d}')
    print(f'Avergaing {varnm} {yr_fcst}/{imo:02d}')

    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      continue

    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      izz = 0
      Zsum = []
      for ilr in range(vlr1-1, vlr2):
        Z2d = dset[varnm].isel(time=0, zl=ilr).data.squeeze()
        if izz == 0:
          Zsum = Z2d.copy()
        else:
          Zsum = Zsum + Z2d
        izz += 1
      A2d = Zsum / izz

      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]
      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

      if icc == 0:
        Asum = A2d.copy()
      else:
        Asum = Asum + A2d

      icc += 1

  Asum = Asum / icc
  print(f'N time records={icc}, min/max = {np.nanmin(Asum)} / {np.nanmax(Asum)}')  

  return Asum, Time

def monthly_vsect_mean_from_Ndaily3D(pthfcst0, yr_init, mo_init, varnm, ocnfld, \
                                     Isct, Jsct, MAVRG=[1], nlrs=75):
  """
    From N-day average ocean 3D fields: 
    monthly average along transect(Isct, Jsct)
    compute monthly average 2D field for given variable
    U and V components are collocated at the transect grid points !

    Field is averaged over MAVRG months (1 = Jan, 2 - Feb, etc)
    Months will be related to the years of the forecast, i.e.
    Jan for the f/cast that starts on 2013/10 will be Jan-2014
  """
  import pandas as pd

  Isct = Isct.astype(int)
  Jsct = Jsct.astype(int)

  mo_fcsts = yrmo_seasonal_fcst(yr_init, mo_init)
# Time-average fields:
  icc = 0
  Time = []
  for imo in MAVRG:
    ix = np.where(mo_fcsts[:,1] == imo)[0][0]
    yr_fcst = mo_fcsts[ix,0]
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{yr_fcst}{imo:02d}')
    print(f'Avergaing {varnm} {yr_fcst}/{imo:02d}')

    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      continue

    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      A2d = np.zeros((nlrs,len(Isct))) + 1.e9
      for ilr in range(nlrs):
        Z2d = dset[varnm].isel(time=0, zl=ilr).data.squeeze()

      # Collocate U/V:
        if varnm == 'u' or varnm == 'U':
          Z2c = mmom6.collocateU2H(Z2d, 'symmetr', f_land0 = True)
        elif varnm == 'v' or varnm == 'V':
          Z2c = mmom6.collocateV2H(Z2d, 'symmetr', f_land0 = True)
        else:
          Z2c = Z2d
        A2d[ilr,:] = Z2c[Jsct,Isct].squeeze()

      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]
      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

      if icc == 0:
        Asum = A2d.copy()
      else:
        Asum = Asum + A2d

      icc += 1

  Asum = Asum / icc
  print(f'N time records={icc}, min/max = {np.nanmin(Asum)} / {np.nanmax(Asum)}')  

  return Asum, Time

def read_oceanm3D_field(pthfcst, flname, fld_read, notime=True):
  """
    Read a 2D or 3D variable from N-daily 3D output archive files oceanm_*.nc
    Seasonal forecasts
  """
  dflread = os.path.join(pthfcst,flname)
  print(f"Reading {fld_read} <-- {dflread}")
  dset   = xarray.open_dataset(dflread)
  if notime:
    AA = dset[fld_read].isel(time=0).data.squeeze()
  else:
    AA = dset[fld_read].data.squeeze()

  return AA

def list_oceanice_files(pthfcst, prefix='oceanm', subdir=''):
  """
    Create list of all oceanm or icem output files for specific run
    subdir = subdirectory inside pthfcst where oceanm, icem to look
    e.g. subdir = oceanm_201905
    otherwise all output files from all subdirs will be given
  """
  if len(subdir) == 0:
    SUBDIR = [drr for drr in os.listdir(pthfcst) if os.path.isdir(os.path.join(pthfcst, drr))]
  else:
    SUBDIR = list([subdir])

  lchar = len(prefix)
  list_files = []
  for dir_out in SUBDIR:
#    print(f'{dir_out}')
    if not dir_out[:lchar] == prefix:
      continue

    pthfull = os.path.join(pthfcst, dir_out)
    LOUTP = [fl for fl in os.listdir(pthfull) if os.path.isfile(os.path.join(pthfull, fl))]
    list_files = list_files + LOUTP
    
  return list_files

def timeser_spatavrg(pthfcst0, yr_init, mo_init, JJ, II, lr, varnm, nens, ocnfld, nmo=12):
  """
    Compute spatially averaged fields from n-daily mean output
    seasonal forecasts
  """
  import pandas as pd

  dstrt = mtime.datenum([yr_init, mo_init,15])
  dold = dstrt - 32

  Tts = []
  Time = []
  for imo in range(nmo):
    dnew = dold + 32
    dv_new = mtime.datevec(dnew)
    dnew = mtime.datenum([dv_new[0], dv_new[1], 1])
    dold = dnew
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{dv_new[0]}{dv_new[1]:02d}')
    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]

    print(f'Reading {pthfcst} nrec={len(LOUTP)}')
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      return []
   
    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      if varnm == 'temp' or varnm == 'potT':
        A2d = dset['potT'].data[0,lr,:,:].squeeze()
      elif varnm == 'salin' or varnm == 'salt':
        A2d = dset['salt'].data[0,lr,:,:].squeeze()
      Asub = A2d[JJ[0]:JJ[1],II[0]:II[1]]
      amn  = np.nanmean(Asub)
      Tts.append(amn)
      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]

      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

  Tts = np.array(Tts)
  Time = np.array(Time)

  return Tts, Time

def timeser_spatavrg_stdoutp(pthfcst, archv_fl, varnm, lr, MSKBS, Acell):
  """
    Compute spatially averaged fields from stdoutput
    seasonal forecasts

    For standard output files (GFDL -type ouptut fields)
    ocean_daily.nc, ocean_month.nc etc
    lr = 1, ... vertical laeyr # for 3D output fields
    lr <= 0 for 2D fields
    
  """
  import pandas as pd

  if archv_fl == 'ocean_daily.nc': lr=-1

  dfmom6 = os.path.join(pthfcst, archv_fl)
  dset   = xarray.open_dataset(dfmom6)
  if lr <= 0:
    A2d = dset[varnm].data.squeeze()
  else:
    A2d = dset[varnm].data[0,lr-1,:,:].squeeze()
  tm = dset['time'].data
  tmP = pd.to_datetime(tm)

  JBS, IBS = np.where(MSKBS == 1)
  areg = np.nansum(Acell*MSKBS)

  Tts = []
  Time = []
  ntimes = A2d.shape[0]
  print(f"Reading {varnm} from {archv_fl}, {ntimes} records ...")
  for itime in range(ntimes):
    aa = A2d[itime,:].squeeze()
    amean = np.nansum(aa[JBS,IBS]*Acell[JBS,IBS])/areg
    Tts.append(amean)
    yr0 = tmP.year[itime]
    mo0 = tmP.month[itime]
    dd0 = tmP.day[itime]
    dnmb0 = mtime.datenum([yr0,mo0,dd0])
    Time.append(dnmb0)

  Tts = np.array(Tts)
  Time = np.array(Time)

  return Tts, Time


def timeser_spatavrg_dayoutp(pthfcst0, yr_init, mo_init, varnm, lr, MSKBS, \
                             Acell, ocnfld='oceanm', nmo=12):
  """
    Compute spatially averaged fields from n-daily mean output
    seasonal forecasts
    oceanm_*.nc

    Return time series for nmo months

    lr = 1, ... vertical laeyr # for 3D output fields
    lr >= max N of v. layers - means bottom layer
    if lr > local depth - NaN mask applied
    lr <= 0 for 2D fields
  """
  import pandas as pd

  dstrt = mtime.datenum([yr_init, mo_init,15])
  dold = dstrt - 32

  print(f'Computing spatial average {varnm} layer={lr}')
  Tts = []
  Time = []
  for imo in range(nmo):
    dnew = dold + 32
    dv_new = mtime.datevec(dnew)
    dnew = mtime.datenum([dv_new[0], dv_new[1], 1])
    dold = dnew
    pthfcst = os.path.join(pthfcst0,f'{ocnfld}_{dv_new[0]}{dv_new[1]:02d}')
    LOUTP = [fl for fl in os.listdir(pthfcst) if os.path.isfile(os.path.join(pthfcst, fl))]

    print(f'Reading {pthfcst} nrec={len(LOUTP)}')
    if len(LOUTP) == 0:
      print(f'No output found in {pthfcst}')
      return []
   
    for ifl in range(len(LOUTP)):
      flocn_name = LOUTP[ifl]
      dfmom6 = os.path.join(pthfcst, flocn_name)
      dset   = xarray.open_dataset(dfmom6)
      ZM = dset['zl'].data
      nlyrs = ZM.shape[0]

      if lr == 1:
        # surface values 3D fields:
        A2d = dset[varnm].data[0,lr,:,:].squeeze()
      elif lr >= nlyrs:
        # bottom values:
        dP   = dset['h'].data[0,:].squeeze()
        dP   = np.where(dP < 1.e-3, 0., dP)
        A3d  = dset[varnm].data[0,:].squeeze()
        A2d  = derive_bottom_temp(A3d, dP)
      elif lr < 0:
        # 2D fields
        A2d = dset[varnm].data[0,:,:].squeeze()

      JBS, IBS = np.where(MSKBS == 1)
      areg = np.nansum(Acell*MSKBS)
      amean = np.nansum(A2d[JBS,IBS]*Acell[JBS,IBS])/areg
      print(f'Min/max {varnm} = {np.nanmin(amean):12.6f}/{np.nanmax(amean):12.6f}')
      Tts.append(amean)
      tm = dset['time'].data
      tmP = pd.to_datetime(tm)
      yr0 = tmP.year[0]
      mo0 = tmP.month[0]
      dd0 = tmP.day[0]

      dnmb0 = mtime.datenum([yr0,mo0,dd0])
      Time.append(dnmb0)

  Tts = np.array(Tts)
  Time = np.array(Time)

  return Tts, Time

def timeser_spatavrg_GLORYS(pthglorys, yr_init, mo_init, varnm, lr, MSKBS, Acell, ndays=365):
  """
    Compute spatially averaged fields from GLORYS reanalysis 
    region extraceted for the NEP

    Return time series for ndays

    lr = 1, ... vertical laeyr # for 3D output fields
    lr >= max N of v. layers - means bottom layer
    if lr > local depth - NaN mask applied
    lr <= 0 for 2D fields
  """
  import pandas as pd

  dstrt = mtime.datenum([yr_init, mo_init,1])

  print(f'Computing spatial average {varnm} layer={lr}')
  Tts = []
  Time = []
  for iday in range(ndays):
    dnmb  = dstrt + iday
    dv    = mtime.datevec(dnmb)
    yr, mo, mday = dv[:3]
    pthfcst = os.path.join(pthglorys,f'{yr}','nep_10')
    flnm  = f'GLORYS_REANALYSIS_NEP_{yr}-{mo:02d}-{mday:02d}.nc'
    dfglorys = os.path.join(pthfcst, flnm) 
    dset  = xarray.open_dataset(dfglorys)
    ZM  = dset['depth'].data
    nlyrs = len(ZM)
    if lr == 1:
      # surface values 3D fields:
      A2d = dset[varnm].data[0,lr,:,:].squeeze()
    elif lr >= nlyrs:
      # bottom values:
      dP   = dset['h'].data[0,:].squeeze()
      dP   = np.where(dP < 1.e-3, 0., dP)
      A3d  = dset[varnm].data[0,:].squeeze()
      A2d  = derive_bottom_temp(A3d, dP)
    elif lr < 0:
      # 2D fields
      A2d = dset[varnm].data[0,:,:].squeeze()
  
    JBS, IBS = np.where(MSKBS == 1)
    areg = np.nansum(Acell*MSKBS)
    amean = np.nansum(A2d[JBS,IBS]*Acell[JBS,IBS])/areg
#    print(f'Min/max {varnm} = {np.nanmin(amean):12.6f}/{np.nanmax(amean):12.6f}')
    print(f'{varnm} mean={amean:12.6f}')
    Tts.append(amean)
    tm = dset['time'].data
    tmP = pd.to_datetime(tm)
    yr0 = tmP.year[0]
    mo0 = tmP.month[0]
    dd0 = tmP.day[0]

    dnmb0 = mtime.datenum([yr,mo,mday])
    Time.append(dnmb0)

  Tts = np.array(Tts)
  Time = np.array(Time)

  return Tts, Time


def plot2D_CalCur(A2d, clrmp, rmin, rmax, xlim1, xlim2, ylim1, ylim2, \
                  fgnmb=1, btx='', tscntrs=[], tslabels=[], sttl="CalCur region", \
                  hlon=[], hlat=[], HH=[]):
  """
    Template 2D figure of T, S, etc fields for California Current region - southern part of NEP
  """
  cntr_clr = [0.3, 0.3, 0.3]

  plt.ion()

  fig1 = plt.figure(fgnmb,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])
  im1 = ax1.pcolormesh(A2d, cmap=clrmp, vmin=rmin, vmax=rmax)
  ax1.axis('scaled')
  ax1.set_xlim([xlim1, xlim2])
  ax1.set_ylim([ylim1, ylim2])
  if len(hlon) > 0 and len(hlat) > 0:
    ax1.contour(hlon, [x for x in range(200, 360, 10)], colors=[(0.8, 0.8, 0.8)], linestyles='solid', linewidths=1)
    ax1.contour(hlat, [x for x in range(0, 89, 10)], colors=[(0.8, 0.8, 0.8)], linestyles='solid', linewidths=1)

  if len(HH) > 0:
    ax1.contour(HH,[0], colors=[(1, 1, 1)], linestyles='solid', linewidths=1)

  if len(tscntrs) > 0:
    CS = ax1.contour(A2d, tscntrs, colors=[cntr_clr], linestyles='solid', linewidths=1)
    if len(tslabels) > 0:
      ax1.clabel(CS, tslabels, inline=1, fontsize=10)

  ax1.set_title(sttl)

  ax2 = fig1.add_axes([ax1.get_position().x1+0.025, ax1.get_position().y0,
                     0.02, ax1.get_position().height])
  # extend: min, max, both
  clb = plt.colorbar(im1, cax=ax2, orientation='vertical', extend='both')
  ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
  ax2.set_yticklabels(ax2.get_yticks())
  ticklabs = clb.ax.get_yticklabels()
  #  clb.ax.set_yticklabels(ticklabs,fontsize=10)
  clb.ax.set_yticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
  clb.ax.tick_params(direction='in', length=12)


  if len(btx) > 0:
    bottom_text(btx, fsz=8, pos=[0.05, 0.03])

  return

def plot_stereogr_axis(fig1, m, xR, yR, A2d, clrmp, rmin, rmax, \
                       btx=[], tscntrs=[], tslabels=[], clrbar=True, sttl='stereogr proj'):
  """
    Plot 2D field in stereogrpahic projection
    mapping function = m
    axis = ax1
  """
  
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])

  m.drawcoastlines(color='w')
  m.drawparallels(np.arange(-90.,120.,10.))
  m.drawmeridians(np.arange(-180.,180.,10.))
  cntr_clr = [0.3, 0.3, 0.3] 

  img = m.pcolormesh(xR, yR, A2d, cmap=clrmp, vmin=rmin, vmax=rmax)

  if len(tscntrs) > 0:
    CS = m.contour(xR, yR, A2d, tscntrs, colors=[cntr_clr], linestyles='solid', linewidths=1)
    if len(tslabels) > 0:
      ax1.clabel(CS, tslabels, inline=1, fontsize=10)

  ax1.set_title(sttl)

  if clrbar:
    ax2 = fig1.add_axes([ax1.get_position().x1+0.025, ax1.get_position().y0,
                       0.02, ax1.get_position().height])
    # extend: min, max, both
    clb = plt.colorbar(img, cax=ax2, orientation='vertical', extend='both')
    ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
    ax2.set_yticklabels(ax2.get_yticks())
    ticklabs = clb.ax.get_yticklabels()
    #  clb.ax.set_yticklabels(ticklabs,fontsize=10)
    clb.ax.set_yticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
    clb.ax.tick_params(direction='in', length=12)

  if len(btx) > 0:
    bottom_text(btx, fsz=8, pos=[0.05, 0.03])

  return ax1


def colormap_params(regn_name, varnm, zz0=-1.):
  """
    Define parameters to plot T, S fields for different regions / depths
  """

  match regn_name:
    case 'CalCur':
      if varnm == 'salin' or varnm == 'salt' or varnm == 'so':
        if zz0 >= -50.:
          rmin = 30.0
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x for x in range(32,36)]
        elif zz0 < -50. and zz0 >= -150:
          rmin = 33.0
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        elif zz0 < -150. and zz0 >= -500:
          rmin = 33.4
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        else: 
          rmin = 33.0
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]

      if varnm == 'temp' or varnm == 'potT' or varnm == 'thetao':
        if zz0 >= -50.:
          rmin = 10.0
          rmax = 28.0
          tscntrs = [x for x in range(10,38,1)]
          tslabels = [x for x in range(10,38,2)]
        elif zz0 < -50. and zz0 >= -150:
          rmin = 8.0
          rmax = 22.0
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -150. and zz0 >= -250:
          rmin = 8.0
          rmax = 18.0
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -250. and zz0 >= -500:
          rmin = 5.0
          rmax = 15.0
          tscntrs = [x for x in range(10,24,1)]
          tslabels = [x for x in range(10,24,1)]
        else: 
          rmin = 2.0
          rmax = 12.0
          tscntrs = [x/10 for x in range(20,140,5)]
          tslabels = [x/10 for x in range(20,140,20)]

      if varnm == 'UV' or varnm == 'Uspeed':
        if zz0 >= -50.:
          rmin = 0
          rmax = 0.2
          tscntrs = [x/100 for x in range(0,50,5)]
          tslabels = [x/100 for x in range(0,50,10)]
        else:
          rmin = 0
          rmax = 0.1
          tscntrs = [x/100 for x in range(0,40,2)]
          tslabels = [x/100 for x in range(0,40,4)]

      if varnm == 'o2':
        if zz0 >= -100.:
          rmin = 150.
          rmax = 300.
        else:
          rmin = 100.  # micro-moles/kg !
          rmax = 250.
        tscntrs = [x for x in range(100,500,25)]
        tslabels = [x for x in range(100,500,25)]
        
      if varnm == 'po4':
        if zz0 >= -50.:
          rmin = 0.
          rmax = 2.5
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 2.5
        tscntrs = [x/10 for x in range(5,50,5)]
        tslabels = [x/10 for x in range(5,50,5)]

      if varnm == 'sio4':
        if zz0 >= -80.:
          rmin = 0.
          rmax = 30.
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 40.
        tscntrs = [x for x in range(0,50,5)]
        tslabels = [x for x in range(0,50,5)]

      if varnm == 'no3':
        # For log natural transformed data:
        if zz0 >= -20.:
          rmin = -7.
          rmax = 3.
        elif -60 <= zz0 < -20:
          rmin = -10.
          rmax = 5.
        else:
          rmin = -1.  # micro-moles/kg !
          rmax = 4.
        tscntrs = [x for x in range(0,50,5)]
        tslabels = [x for x in range(0,50,5)]

    case 'BeringChuk':
      if varnm == 'salin' or varnm == 'salt' or varnm == 'so':
        if zz0 >= -25.:
          rmin = 28.
          rmax = 33.
          tscntrs = [x/10 for x in range(240,354,5)]
          tslabels = [x/10 for x in range(240,350,10)]
        elif zz0 < -25. and zz0 >= -100:
          rmin = 31
          rmax = 33.5
          tscntrs = [x/10 for x in range(290,350,5)]
          tslabels = [x/10 for x in range(290,350,10)]
        elif zz0 < -100. and zz0 >= -150:
          rmin = 32.0
          rmax = 33.5
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        elif zz0 < -150. and zz0 >= -500:
          rmin = 33.4
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        else: 
          rmin = 33.0
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]

      if varnm == 'temp' or varnm == 'potT' or varnm == 'thetao':
        if zz0 >= -25.:
          rmin = -1.8
          rmax = 14.
          tscntrs = [x for x in range(-2,20,1)]
          tslabels = [x for x in range(-2,20,2)]
        elif zz0 < -25. and zz0>= -51.:
          rmin = -1.8
          rmax = 8.
          tscntrs = [x for x in range(-2,20,1)]
          tslabels = [x for x in range(-2,20,2)]
        elif zz0 < -51. and zz0 >= -150:
          rmin = -1.8
          rmax = 6.
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -150. and zz0 >= -250:
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -250. and zz0 >= -500:
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x for x in range(10,24,1)]
          tslabels = [x for x in range(10,24,1)]
        else: 
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x/10 for x in range(20,140,5)]
          tslabels = [x/10 for x in range(20,140,20)]

      if varnm == 'UV' or varnm == 'Uspeed':
        if zz0 >= -50.:
          rmin = 0
          rmax = 0.2
          tscntrs = [x/100 for x in range(0,50,5)]
          tslabels = [x/100 for x in range(0,50,10)]
        else:
          rmin = 0
          rmax = 0.1
          tscntrs = [x/100 for x in range(0,40,2)]
          tslabels = [x/100 for x in range(0,40,4)]
              
      if varnm == 'o2':
        if zz0 >= -50.:
          rmin = 150.
          rmax = 400.
        else:
          rmin = 120.  # micro-moles/kg !
          rmax = 340.
        tscntrs = [x for x in range(100,500,50)]
        tslabels = [x for x in range(100,500,50)]

      if varnm == 'po4':
        if zz0 >= -50.:
          rmin = 0.
          rmax = 2.5
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 2.5
        tscntrs = [x/10 for x in range(5,50,5)]
        tslabels = [x/10 for x in range(5,50,5)]

      if varnm == 'sio4':
        if zz0 >= -100.:
          rmin = 0.
          rmax = 65.
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 80.
        tscntrs = [x for x in range(0,50,5)]
        tslabels = [x for x in range(0,50,5)]

      if varnm == 'no3':
        # For log natural transformed data:
        if zz0 >= -20.:
          rmin = -7.
          rmax = 3.
        elif -100 <= zz0 < -20:
          rmin = 0.
          rmax = 3.5
        else:
          rmin = 1.  # micro-moles/kg !
          rmax = 4.
        tscntrs = [x/10 for x in range(0,50,5)]
        tslabels = [x/10 for x in range(0,50,5)]

    case 'GulfAlaska':
      if varnm == 'salin' or varnm == 'salt' or varnm == 'so':
        if zz0 >= -20.:
          rmin = 27.
          rmax = 33.
          tscntrs = [x/10 for x in range(240,354,5)]
          tslabels = [x/10 for x in range(240,350,10)]
        elif zz0 < -20. and zz0 >= -100:
          rmin = 29.0
          rmax = 33.4
          tscntrs = [x/10 for x in range(290,350,5)]
          tslabels = [x/10 for x in range(290,350,10)]
        elif zz0 < -100. and zz0 >= -150:
          rmin = 32
          rmax = 34.
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        elif zz0 < -150. and zz0 >= -500:
          rmin = 33.4
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]
        else: 
          rmin = 33.0
          rmax = 35.0
          tscntrs = [x/10 for x in range(320,360,2)]
          tslabels = [x/10 for x in range(320,360,2)]

      if varnm == 'temp' or varnm == 'potT' or varnm == 'thetao':
        if zz0 >= -30.:
          rmin = -1.8
          rmax = 18.
          tscntrs = [x for x in range(-2,20,1)]
          tslabels = [x for x in range(-2,20,2)]
        elif zz0 < -30. and zz0>= -51.:
          rmin = -1.8
          rmax = 14.
          tscntrs = [x for x in range(-2,20,1)]
          tslabels = [x for x in range(-2,20,2)]
        elif zz0 < -51. and zz0 >= -150:
          rmin = -1.8
          rmax = 12.
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -150. and zz0 >= -250:
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x for x in range(5,24,1)]
          tslabels = [x for x in range(5,24,1)]
        elif zz0 < -250. and zz0 >= -500:
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x for x in range(10,24,1)]
          tslabels = [x for x in range(10,24,1)]
        else: 
          rmin = -1.8
          rmax = 6.0
          tscntrs = [x/10 for x in range(20,140,5)]
          tslabels = [x/10 for x in range(20,140,20)]

      if varnm == 'UV' or varnm == 'Uspeed':
        if zz0 >= -50.:
          rmin = 0
          rmax = 0.2
          tscntrs = [x/100 for x in range(0,50,5)]
          tslabels = [x/100 for x in range(0,50,10)]
        else:
          rmin = 0
          rmax = 0.1
          tscntrs = [x/100 for x in range(0,40,2)]
          tslabels = [x/100 for x in range(0,40,4)]
              
      if varnm == 'o2':
        if zz0 >= -100.:
          rmin = 180.
          rmax = 350.
        else:
          rmin = 100.  # micro-moles/kg !
          rmax = 300.
        tscntrs = [x for x in range(100,500,25)]
        tslabels = [x for x in range(100,500,25)]

      if varnm == 'po4':
        if zz0 >= -20.:
          rmin = 0.
          rmax = 2.2
        elif -100 < zz0 < -20.:
          rmin = 0.
          rmax = 2.5
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 2.8
        tscntrs = [x/10 for x in range(5,50,5)]
        tslabels = [x/10 for x in range(5,50,5)]

      if varnm == 'sio4':
        if zz0 >= -100.:
          rmin = 0.
          rmax = 65.
        else:
          rmin = 0.  # micro-moles/kg !
          rmax = 80.
        tscntrs = [x for x in range(0,50,5)]
        tslabels = [x for x in range(0,50,5)]

      if varnm == 'no3':
        # For log natural transformed data:
        if zz0 >= -20.:
          rmin = -7.
          rmax = 3.
        elif -100 <= zz0 < -20:
          rmin = -6.
          rmax = 4.
        else:
          rmin = 1.  # micro-moles/kg !
          rmax = 4.
        tscntrs = [x/10 for x in range(0,50,5)]
        tslabels = [x/10 for x in range(0,50,5)]

  return rmin, rmax, tscntrs, tslabels


def stereogr_params_regions(regn_name):
  """
    Parameters for sterographic projections
    for some regions NEP10k
  """
  match regn_name:
    case 'CalCur':
      width  = 4000*1.e3
      height = 4000*1.e3
      lat0   = 33.5
      lon0   = -128.
    case 'BeringChuk':
      width  = 3300*1.e3
      height = 3700*1.e3
      lat0   = 65.
      lon0   = -175.
    case 'GulfAlaska':
      width  = 3900*1.e3
      height = 3200*1.e3
      lat0   = 52.
      lon0   = -149.
    case _:
      raise ValueError("undefined region: {regn_name}")

  return lon0, lat0, height, width

def arrange_1segm(CNTR, x0, y0, dltD=50.):
  """
    Find segment closest to x0, y0 
    arrange the orientation of the segment
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

def derive_NEPcoastline(HH, hcntr=0., nmin=100, xFS=334, yFS=1, indx_int=False, ignore_closed=True):
  """
    Derive indices of the coastline or any other isobath = hcntr for the NEP domain
    The longest continuous contour closest to xFS yFS point 
     is only kept, no islands
    returned coordinates are floats (for accurate plotting / data interpolation)
    indx_int = True --> return integer not repeating indices, may be not following exactly HH=hcntr 
  """
  plt.ioff()
  figA = plt.figure(10,figsize=(8,8))
  plt.clf()
  axA1 = plt.axes([0.1, 0.2, 0.7, 0.7])
  CS   = axA1.contour(HH,[hcntr])

  axA1.axis('equal')
#  axA1.set_xlim([xl1,xl2])
#  axA1.set_ylim([yl1,yl2])

  SGS  = CS.allsegs[0]  # should be only 1 contoured value
  nsgs = len(SGS)

# Delete all closed contours
  CNTR = []
  for isg in range(nsgs):
    XY = SGS[isg]
    X  = XY[:,0]
    Y  = XY[:,1]

    dEnd = np.sqrt((X[0]-X[-1])**2+(Y[0]-Y[-1])**2)
    if ignore_closed and dEnd <= 1.:
      continue

    # ignore short contours:
    if len(X) < nmin:
      continue

    CNTR.append(XY)

# Arranage all segments in order
# First find segment that starts at xFS, yFS
  nC  = len(CNTR)
  TCNT = []

  for ii in range(nC):
    if ii == 0:
      x0   = xFS
      y0   = yFS
      dltD = 100.
    else:
      x0   = TCNT[-1,0]
      y0   = TCNT[-1,1]
      dltD = 10.

    xsgm, ysgm, imin, jmin = arrange_1segm(CNTR,x0,y0, dltD=dltD)

    if len(xsgm) == 0:
      continue

# Remove selected segment:
    CNTR.pop(imin)

    if ii == 0:
      TCNT = np.transpose(np.array((xsgm,ysgm)))
    else:
      aa   = np.transpose(np.array((xsgm,ysgm)))
      TCNT = np.append(TCNT, aa, axis=0)


  plt.close(figA)
  plt.ion()

  if indx_int:
    print('Rounding coast indices to integers, removing repeatitions ...')
    TCNT = TCNT.astype(int)

# Check for repeating indices:
    nn = TCNT.shape[0]
    dT = np.diff(TCNT, axis=0)
    DD = np.sqrt(dT[:,0]**2 + dT[:,1]**2)
    Irpt = np.where(DD==0)[0]
    TCNT = np.delete(TCNT, Irpt, axis=0)

  return TCNT

def derive_closed_contour(HH, xC, yC, hcntr=[0.], dlt_imin=5, nmin=4, indx_int=False):
  """
    Derive indices of a (semi-)closed contour that contains pnt xC, yC 
    e.g., center of cyclonic circulaiton
    hcntr = list of contours, the closeset to xC yC as its interior point will be picked up

    dlt-imin - required minimum distance between the endpoints of the contour 
    nmin - ignore very small contours <nmin pnts
    indx_int = True --> return integer not repeating indices, may be not following exactly HH=hcntr 
  """
  import mod_misc1 as mmisc

  plt.ioff()
  figA = plt.figure(10,figsize=(8,8))
  plt.clf()
  axA1 = plt.axes([0.1, 0.2, 0.7, 0.7])
  CS   = axA1.contour(HH, hcntr)

  axA1.axis('equal')
#  axA1.set_xlim([xl1,xl2])
#  axA1.set_ylim([yl1,yl2])

# Find all closed contours
  CNTR = []
  DST  = []
  for SGS  in CS.allsegs[:]:  # multiple or 1 contour for specified [hcntr]
    nsgs = len(SGS)

    for isg in range(nsgs):
      XY = SGS[isg]
      X  = XY[:,0]
      Y  = XY[:,1]

      dEnd = np.sqrt((X[0]-X[-1])**2+(Y[0]-Y[-1])**2)
      if dEnd > dlt_imin:
        continue

      if not mmisc.inpolygon_1pnt(xC,yC, X,Y):
        continue

      # ignore short contours:
      if len(X) < nmin:
        continue

      d2pnt = np.min(np.sqrt((X-xC)**2+(Y-yC)**2))
      CNTR.append(XY)
      DST.append(d2pnt)

# All selected contour have xC yC inside, next:
# Find contour closest to Xc Yc
  iCntr = np.argmin(DST)
  TCNT = CNTR[iCntr]

  plt.close(figA)
  plt.ion()

  if indx_int:
    print('Rounding coast indices to integers, removing repeatitions ...')
    TCNT = TCNT.astype(int)

# Check for repeating indices:
    nn = TCNT.shape[0]
    dT = np.diff(TCNT, axis=0)
    DD = np.sqrt(dT[:,0]**2 + dT[:,1]**2)
    Irpt = np.where(DD==0)[0]
    TCNT = np.delete(TCNT, Irpt, axis=0)

  return TCNT

def chop_contour(IJcntr, IJstart, IJend, dd_max=50):
  """
    Chop the contour from IJstart to IJend
    Closest to IJ's points on the contour are used
  """
  ddS  = np.sqrt((IJstart[0]-IJcntr[:,0])**2 + (IJstart[1]-IJcntr[:,1])**2)
  iiS = np.argmin(ddS)
  ddE  = np.sqrt((IJend[0]-IJcntr[:,0])**2 + (IJend[1]-IJcntr[:,1])**2)
  iiE = np.argmin(ddE)

  assert ddS[iiS] <= dd_max, f'min dist to Start pnt={ddS[iiS]} - too far?'
  assert ddE[iiE] <= dd_max, f'min dist to End pnt={ddE[iiE]} - too far?'
  
  # Chop the end segments ttaking into account correct orientation:
  if iiS < iiE:
    IJcntr = IJcntr[iiS:iiE,:]
  else:
    IJcntr = IJcntr[iiE:iiS,:]

  return IJcntr

def smooth_coastline(TCNT, npnts=11, indx_int=False):
  """
    Smooth coastline TCNT[I,J] using running mean
    Returned indices are floats
    indx_int True --> indices are integers, repetead indices eliminated
  """
  from mod_solver import runmn
  IC = TCNT[:,0]
  JC = TCNT[:,1]
  Npp = len(IC)
  XC = np.arange(0,Npp+1) # extra point for running mean script
  ICf = runmn(IC, XC, mnwnd=npnts)
  JCf = runmn(JC, XC, mnwnd=npnts)

  TCNTf = np.stack((ICf,JCf), axis=1)

  if indx_int:
#    print('Rounding coast indices to integers, removing repeatitions ...')
    TCNTf = TCNTf.astype(int)

# Check for repeating indices:
    nn = TCNTf.shape[0]
    dT = np.diff(TCNTf, axis=0)
    DD = np.sqrt(dT[:,0]**2 + dT[:,1]**2)
    Irpt = np.where(DD==0)[0]
    TCNTf = np.delete(TCNTf, Irpt, axis=0)

  return TCNTf  

def costaline_ocean_points(SHf, HH):
  """
    Given coastline and topo, adjust coastline points to be on ocean pnts
    closest to the coastline
  """
  Ish = SHf[:,0]
  Jsh = SHf[:,1]
  Hb  = HH[Jsh,Ish] 
  Ilnd = np.where(Hb >= 0)[0]
# Exclude existing coastline:
  HHb = HH.copy()
  HHb[Jsh,Ish] = 999.
  JJ_ocn, II_ocn = np.where(HHb < 0)
  if Ilnd.size == 0:
    print('All points are ocean points, no adjustment needed ...')
    return SHf

  for ixx in Ilnd:
    i0  = Ish[ixx]
    j0  = Jsh[ixx]
    idel = []
    im1=jm1=ip1=jp1=-999
    # if end points are on land - remove them:
    if ixx == 0:
      idel.append(ixx)
      continue
    elif ixx == len(Ish):
      idel.append(ixx)
      continue

    im1 = Ish[ixx-1]
    jm1 = Jsh[ixx-1]    
    ip1 = Ish[ixx+1]
    jp1 = Jsh[ixx+1]    
    DD = np.sqrt((II_ocn-i0)**2 + (JJ_ocn-j0)**2)
    kk = np.argmin(DD) 
    jocn = JJ_ocn[kk]
    iocn = II_ocn[kk]

    # Make sure that the new point is within the two neighboring coastline points:
    # simple check a better approach is possible:
    # Both angle between next and previous pnts and the new pnt +/- 90 from the direction
    # connecting prvious and next pnt
#    alf = np.arctan2((jp1-jm1),(ip1-im1))*180./np.pi  # direction from previos to next coast point
#    alf_old = np.arctan2((j0-jm1),(i0-im1))*180./np.pi # dir to the old coast pnt
#    alf_new = np.arctan2((jocn-jm1),(iocn-im1))*180./np.pi # dir to the new coast pnt that is now ocean
#    dltA_old = abs(alf-alf_old)
#    dltA_new = abs(alf-alf_new)
#    bet = np.arctan2((jm1-jp1),(im1-ip1))*180./np.pi 
#    bet_old = np.arctan2((j0-jp1),(i0-ip1))*180./np.pi # dir to the old coast pnt
#    bet_new = np.arctan2((jocn-jp1),(iocn-ip1))*180./np.pi # dir to the new coast pnt that is now ocean
#    dltB_new = abs(bet-bet_new)
#    if dltA_new > 90. or dltB_new > 90.:
#      print(f'New coast point not btw neighb. points: Old i0={i0}, j0={j0}, new: i={iocn}, j={jocn}')
#      print(f'Neighb pnts, i/j: {im1}/{jm1}, {ip1}/{jp1}')
#      print(f'ixx={ixx}, Deleting this point')
#      idel.append(ixx)
    Ish[ixx] = iocn
    Jsh[ixx] = jocn
 
  if idel:
    Ish = np.delete(Ish,idel)
    Jsh = np.delete(Jsh,idel)

  Shf_ocn = np.stack((Ish,Jsh), axis=1)

  return Shf_ocn

def projectU_to_coastline(U2d, V2d, Ish, Jsh, Icst, Jcst, navrg=5):
  """
  2D vertical sections of U,V components project onto axes 
  with Y - along coast and X- normal to coastline
  Ucst - normal vel, Vcst - along-coast component
  navrg - # of points +/- from the closest pnt on the coastline to determine
          the local coastline direction
  """
  import math
  r2dgr = 180./np.pi

  jdim, idim = U2d.shape
  ncst = len(Icst)
  U2dR = V2dR = np.zeros((jdim, idim))
  for ii in range(idim):
    i0 = Ish[ii]
    j0 = Jsh[ii]
    dd = np.sqrt((Icst-i0)**2 + (Jcst-j0)**2)
    imin = np.argmin(dd)
    ic0 = Icst[imin]
    jc0 = Jcst[imin]
    im1 = np.max([0,imin-navrg])
    ip1 = np.min([ncst, imin+navrg])
    icS = Icst[im1]
    jcS = Jcst[im1]
    icE = Icst[ip1]
    jcE = Jcst[ip1]

    ycst = jcE-jcS
    xcst = icE-icS 
    Lcst = np.sqrt(xcst**2 + ycst**2)
    xcst = xcst/Lcst
    ycst = ycst/Lcst

    # Angle from coastline to Y-axis in math sense:
    alf = math.atan2(ycst,xcst) - np.pi/2.

    U1d = U2d[:,ii]
    V1d = V2d[:,ii]
    # Rotation matrix to rotate axes by angle alf (or vector by -alf):
    U1r = U1d*np.cos(alf) + V1d*np.sin(alf)
    V1r = -U1d*np.sin(alf) + V1d*np.cos(alf)

    U2dR[:,ii] = U1r
    V2dR[:,ii] = V1r


  return U2dR, V2dR

def projectU1d_to_coastline(U1d, V1d, Ish, Jsh, Icst, Jcst, navrg=5):
  """
  1D arrays of U,V components project onto axes aligned with the coast line
  with Y - along coast and X- normal to coastline
  Ucst - normal vel, Vcst - along-coast component
  Ish, Jsh - grid points of U vectors
  Icst, Jcst - grid points of the coastline
  navrg - # of points +/- from the closest pnt on the coastline to determine
          the local coastline direction
  """
  import math
  r2dgr = 180./np.pi

  idim = U1d.shape[0]
  ncst = len(Icst)
  U1dR = V1dR = np.zeros((idim))
  for ii in range(idim):
    i0 = Ish[ii]
    j0 = Jsh[ii]
    dd = np.sqrt((Icst-i0)**2 + (Jcst-j0)**2)
    imin = np.argmin(dd)
    ic0 = Icst[imin]
    jc0 = Jcst[imin]
    im1 = np.max([0,imin-navrg])
    ip1 = np.min([ncst, imin+navrg])
    icS = Icst[im1]
    jcS = Jcst[im1]
    icE = Icst[ip1]
    jcE = Jcst[ip1]

    ycst = jcE-jcS
    xcst = icE-icS 
    Lcst = np.sqrt(xcst**2 + ycst**2)
    xcst = xcst/Lcst
    ycst = ycst/Lcst

    # Angle from coastline to Y-axis in math sense:
    alf = math.atan2(ycst,xcst) - np.pi/2.

    uu = U1d[ii]
    vv = V1d[ii]
    # Rotation matrix to rotate axes by angle alf (or vector by -alf):
    ur =  uu*np.cos(alf) + vv*np.sin(alf)
    vr = -uu*np.sin(alf) + vv*np.cos(alf)

    U1dR[ii] = ur
    V1dR[ii] = vr

  return U1dR, V1dR


def distance_segments(DX, DY, Ish, Jsh, cff=1.e-3):
  """
  Find distance of the segments along a contour
  DX, DY - grid spacing
  """
  # Distance along the line:
  nnw   = np.shape(Ish)[0]
  LDX   = np.zeros((nnw))
  for ii in range(nnw-1):
    i0 = Ish[ii]
    j0 = Jsh[ii]
    i1 = Ish[ii+1]
    j1 = Jsh[ii+1]
    ld = DX[j0,i0]*abs(i1-i0) + DY[j0,i0]*abs(j1-j0)
    LDX[ii+1] = ld*cff   # m --> km

  DIST = np.cumsum(LDX)
  LDX[0] = LDX[1] 

  return DIST, LDX


def plot_sect_map(HH, hlat, hlon, Xsh, Ysh, Ldist_sh, fgnmb, sctnm, \
                  proj='ortho', btx='mod_anls_seas.py:plot_sect_map'):
  """
  Plot map showing the transect
  """
  import mod_colormaps as mclrmp
  from mpl_toolkits.basemap import Basemap, cm


  # Add extra row/col for plotting
  # Add extra row/col for plotting with pcolormesh
  lonw = hlon.copy()
  latw = hlat.copy()
  lonw = np.insert(lonw, -1, lonw[:,-1]+0.01, axis=1)
  lonw = np.insert(lonw, -1, lonw[-1,:]+0.01, axis=0)
  latw = np.insert(latw, -1, latw[-1,:]+0.01, axis=0)
  latw = np.insert(latw, -1, latw[:,-1]+0.01, axis=1)

  res  = 'l'
  match proj:
    case 'ortho':
      if np.max(Ysh) < 50:
        lon0 = 220.
        lat0 = 40.
      else:
        lon0 = 220.
        lat0 = 60.
      m = Basemap(projection='ortho', lon_0=lon0, lat_0=lat0, resolution=res)
    case 'stere':
      lat0 = np.mean(Ysh)
      lon0 = np.mean(Xsh)
      width = abs(max(Xsh)-min(Xsh))*130.e3
      height = abs(max(Ysh)-min(Ysh))*130.e3
      m = Basemap(width=width, height=height, resolution='l',\
            projection='stere', lat_ts=35, lat_0=lat0, lon_0=lon0)

    
  xR, yR = m(lonw, latw)
  PMsk = ( (xR > 1e20) | (yR > 1e20) )
  AA = HH.copy()
  AA = np.insert(AA, 0, AA[:,-1], axis=1)
  AA = np.insert(AA, -1, AA[-1,:], axis=0)
  AA[PMsk] = np.nan
  xR[PMsk]   = 1.e30
  yR[PMsk]   = 1.e30

  ny, nx = HH.shape
  AA = AA[0:ny, 0:nx]
  AA = np.where(HH >= 0., np.nan, AA)

  # Find section in orth coordinates:
  xSCT = []
  ySCT = []
  xSCT, ySCT = m(Xsh, Ysh)


  clrmp_name = 'winter'
  clr_ramp   = [1, 1, 1]   # add white color at the end of the colormap
  clrmp = mclrmp.addendclr_colormap(clrmp_name, clr_ramp, nramp=0.1, ramp_start=False)
  clrmp.set_bad(color=[0.5, 0.5, 0.5])
  rmin = -7000.
  rmax = 0.

  plt.ion()

  fig1 = plt.figure(fgnmb,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])
  m.drawcoastlines()
  im1 = m.pcolormesh(xR, yR, AA, cmap=clrmp, vmin=rmin, vmax=rmax)

  m.drawparallels(np.arange(-90.,120.,10.))
  m.drawmeridians(np.arange(-180.,180.,10.))

  # Plot section with distance markers:
  m.plot(xSCT,ySCT,'r-',linewidth=2)
  nn = len(Ldist_sh)
  dx = 200.
  xpnt = 0.
  for ii in range(nn):
    msz=3
    if Ldist_sh[ii] >= xpnt:
      if xpnt == 0.:
        m.plot(xSCT[ii],ySCT[ii], marker='o', markersize=msz, color=(0.,0.,1))
      else:
        m.plot(xSCT[ii],ySCT[ii], marker='o', markersize=msz, color=(0.,0.,0.))
      xpnt = xpnt + dx

  sttl = f"{sctnm} X markers dx={dx:4.0f} Max(X)={np.max(Ldist_sh):5.0f}km"
  ax1.set_title(sttl)

  ax2 = fig1.add_axes([ax1.get_position().x1+0.025, ax1.get_position().y0,
                     0.02, ax1.get_position().height])
  # extend: min, max, both
  clb = plt.colorbar(im1, cax=ax2, orientation='vertical', extend='min')
  ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
  ax2.set_yticklabels(ax2.get_yticks())
  ticklabs = clb.ax.get_yticklabels()
  #  clb.ax.set_yticklabels(ticklabs,fontsize=10)
  clb.ax.set_yticklabels(["{:.0f}".format(i) for i in clb.get_ticks()], fontsize=10)
  clb.ax.tick_params(direction='in', length=12)

  #btx = 'plot_sections_ortho.py'
  bottom_text(btx, fsz=7)

def derive_conservTbtm_from_T3d(T3d, S3d, PR, dP, hlon, hlat):
  """
    Derive conservative T bottom from 3D T and S fields
    PR - 3D array of pressure at T depths (derived from ZM arrays)
    dP = layer thickness, m, 3D array
  """
  PPTHN = '/home/Dmitry.Dukhovskoy/python'
  sys.path.append(PPTHN + '/TEOS_10/gsw')
  sys.path.append(PPTHN + '/TEOS_10/gsw/gibbs')
  sys.path.append(PPTHN + '/TEOS_10/gsw/utilities')
  import mod_swstate as msw
  import conversions as gsw
  # Compute absolute salinity from practical S:
  print('Computing absolute S')
  kdm, jdm, idm = T3d.shape
  SA = gsw.SA_from_SP(S3d, PR, hlon, hlat)

  # Compute conservative T from potential T
  print('Computing conservative T')
  CT3d = gsw.CT_from_pt(SA, T3d)

  # Derive bottom T:
  Tbtm = np.zeros((jdm,idm))*np.nan
  dpmin = 1.e-1
  for ik in range(1,kdm):
    dpup  = dP[ik-1,:].squeeze()
    dpbtm = dP[ik,:].squeeze()
    tz    = CT3d[ik-1,:]
    if ik < kdm-1:
      Jb, Ib = np.where( (dpup > dpmin) & (dpbtm <= dpmin) )
    else:
  # Deep layers include all left:
      Jb, Ib = np.where( dpup > dpmin )
    if len(Jb) == 0: continue
    Tbtm[Jb, Ib] = tz[Jb, Ib]

  return Tbtm

def avrg_cice_NSIDC(YR1, YR2, MM1, MM2, get_coord=True):
  """
    Average over years/months monthly ice concentration fields
    fields are downloaded from the Near-Real-Time NOAA/NSIDC 
    Climate Data Record of Passive Microwave Sea Ice Concentration 
    https://nsidc.org/data/g10016
    Use script: /home/Dmitry.Dukhovskoy/scripts/data_process/get_NRT_seaconc.sh

    for seasonal averaging, assumed that MM1<MM2, i.e. 
    MM1 = 4, MM2=12
    if winter season MM1=12, MM2=2 - need to update the code
  """
  fyaml = 'paths_seasfcst.yaml'
  with open(fyaml) as ff:
    pthseas = safe_load(ff)

  LON = []
  LAT = []
  jcc = 0
  for YR in range(YR1,YR2+1):
    pthnsidc=pthseas["NRT_NSIDC"]['pthmnth'].format(YR=YR)
    for MM in range(MM1,MM2+1):
      fsfx = 'f11'
      if (YR == 1995 and MM >= 10) or (YR > 1995 and YR <2008):
        fsfx = 'f13'
      if (YR >= 2008):
        fsfx = 'f17'
    
      flnsidc  = f'seaice_conc_monthly_nh_{YR}{MM:02d}_{fsfx}_v04r00.nc'

      drflnsidc = os.path.join(pthnsidc, flnsidc)
      print(f'Reading NTR NSIDC ice conc: {drflnsidc}')
      dset_nsidc = xarray.open_dataset(drflnsidc)

      ICnrt = dset_nsidc['nsidc_nt_seaice_conc_monthly'].data[0,:].squeeze()
      ICnrt = np.where(ICnrt>1., np.nan, ICnrt)
      # Flip NSIDC grid:
      ICnrt = np.flipud(ICnrt)

      if get_coord:
        Xnrt  = dset_nsidc['xgrid'].data
        Ynrt  = dset_nsidc['ygrid'].data
        # Flip grid:
        Ynrt  = np.flipud(Ynrt)

      # Convert Polar Coordinates to Geostatic coordinates (lon/lat)
        import mod_misc1 as mmisc
        XX, YY = np.meshgrid(Xnrt, Ynrt, indexing='xy')
        LON, LAT = mmisc.convert_polarXY_lonlat(XX,YY)

      # Make lon 0, 360 to match NEP grid
        LON = np.where(LON<0, LON+360., LON)

        get_coord = False


      if jcc == 0:
        ICM = ICnrt
      else:
        ICM = ICM + ICnrt

      jcc += 1        
      dset_nsidc.close()

  ICM = ICM.squeeze()/jcc

  return ICM, LON, LAT

def mask_BeringSea_NEPgrid(HH,hlon,hlat):
  """
    Return mask of the Bering Sea for NEP10k domain
  """
  LMsk = np.where(HH<0, 1, 0)
  LMsk = np.where(hlat<53.,0,LMsk)
  LMsk[:567,:] = 0
  LMsk[:,:39] = 0
  LMsk[:595,177:] = 0
  # Mask for Bering Sea + Ber. Str. + S. Chukchi Shelf
  BMsk = LMsk.copy()
  BMsk[750:,189:] = 0
  BMsk[:750,239:] = 0

  return BMsk

def mask_Arctic_NEPgrid(HH,hlon,hlat):
  """
    Return mask of the Arctic portion of the NEP10k domain
  """
  LMsk = np.where(HH<0, 1, 0)
  LMsk = np.where(hlat<53.,0,LMsk)
  LMsk[:567,:] = 0
  LMsk[:,:39] = 0
  LMsk[:595,177:] = 0
  # Mask for Bering Sea + Ber. Str. + S. Chukchi Shelf
  BMsk = LMsk.copy()
  BMsk[750:,189:] = 0
  BMsk[:750,239:] = 0

  AMsk = LMsk.copy()
  AMsk = np.where(BMsk==1, 0, AMsk)

  return AMsk


