"""
  Functions for extracting and writing
  OB file for regional MOM6
"""
import xarray
import os
import importlib
import numpy as np
import sys
import matplotlib.pyplot as plt
from yaml import safe_load

#import pickle
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

def load_var(spear_dir, varnm, yr1=1993, yr2=2020, mstart=1, fzint=False):
  """
    Extract SPEAR monthly climatology
    fzint = True - include z layer interface depths into returned dataset
  """
  flname = f'spear_moclim.mstart{mstart:02d}.{yr1}-{yr2}.{varnm}.nc'
  dflinp = os.path.join(spear_dir, flname)

  if not varnm == 'ssh' and not varnm == 'SSH':
    spear_var = xarray.open_dataset(dflinp).rename({'z_l': 'z'})
  else:
    spear_var = xarray.open_dataset(dflinp)
    fzint = False

# Make sure missing/fill value was properly used
  assert np.abs(spear_var).max() < 1e5

  if fzint:
    zzi = spear_var.z_i
    dset_out = xarray.Dataset({f"{varnm}": spear_var[varnm], "z_interf": zzi})
  else:
    dset_out = xarray.Dataset({f"{varnm}":spear_var[varnm]})

  return dset_out

def read_spear_output(pthspear, varnm, flnm, fzint=False):
  """
    Read SPEAR output fields
    These could be from regional subset fields
    fzint = True - include z layer interface depths into returned dataset
  """
  dflinp = os.path.join(pthspear,flnm)
  if not varnm == 'ssh' and not varnm == 'SSH':
    spear_var = xarray.open_dataset(dflinp).rename({'z_l': 'z'})
  else:
    spear_var = xarray.open_dataset(dflinp)
    fzint = False

# Make sure missing/fill value was properly used
#  assert np.abs(spear_var).max() < 1e5

  if fzint:
    zzi = spear_var.z_i
    dset_out = xarray.Dataset({f"{varnm}": spear_var[varnm], "z_interf": zzi})
  else:
    dset_out = xarray.Dataset({f"{varnm}":spear_var[varnm]})

  return dset_out


def add_valid_time(ds):
  months = ds.lead.values.astype(int) + mon
  dates = [dt.datetime(1993, m, 1) if m <= 12 else dt.datetime(1994, m-12, 1) for m in months]
  ds['time'] = (('lead', ), dates)
  ds = ds.swap_dims({'lead': 'time'})
  return ds


def add_coords(da):
  xcoord = [str(d) for d in da.coords if d in ['xh', 'xq']]
  ycoord = [str(d) for d in da.coords if d in ['yh', 'yq']]
  assert len(xcoord) == len(ycoord) == 1
  xc = xcoord[0]
  yc = ycoord[0]
  coord_tuple = (yc, xc)
  coord_sel = {xc: da[xc], yc: da[yc]}
  lat = next(static[v] for v in ['geolat', 'geolat_u', 'geolat_v'] if xc in static[v].coords and yc in static[v].coords)
  lon = next(static[v] for v in ['geolon', 'geolon_u', 'geolon_v'] if xc in static[v].coords and yc in static[v].coords)
  da['lat'] = (coord_tuple, lat.sel(**coord_sel).data)
  da['lon'] = (coord_tuple, lon.sel(**coord_sel).data)
  return da

def spear2momOB_gmapi(segments, lon_spear, lat_spear, dflout, fsave=True):
  """
   For interpolation find mapping indices: 4 grid points on SPEAR grid around
   a MOM grid point along the OB segment
   SPEAR lon/ lat - 2D array of u, v, or h-points
  """
  import mod_regmom as mrmom

  dset = xarray.Dataset()
  nsgm = len(segments)
  for ii in range(nsgm):
    isgm = ii+1
#    print(f"Segm = {isgm}")
    dset_segm = segments[ii]
    xOB   = dset_segm.coords.lon.data
    yOB   = dset_segm.coords.lat.data
    npnts = len(xOB)

    INDX = np.zeros((npnts,4)).astype(int)-999
    JNDX = np.zeros((npnts,4)).astype(int)-999
    for ikk in range(npnts):
      print(f"segment={isgm} ikk = {ikk}")
      x0 = xOB[ikk]
      y0 = yOB[ikk]
      ixx, jxx = mrmom.find_gridpnts_box(x0, y0, lon_spear, lat_spear, dhstep=1.) 
      INDX[ikk,:] = ixx
      JNDX[ikk,:] = jxx

    darri = xarray.DataArray(INDX, dims=(f"npnts_{isgm:03d}","vertices"), \
                  coords={f"npnts_{isgm:03d}": np.arange(npnts), \
                  "vertices": np.arange(1,5)})
    darrj = xarray.DataArray(JNDX, dims=(f"npnts_{isgm:03d}","vertices"), \
                  coords={f"npnts_{isgm:03d}": np.arange(npnts), \
                  "vertices": np.arange(1,5)})

    if isgm == 1:
      dset = xarray.Dataset({f"indx_segm{isgm:03d}": darri, f"jndx_segm{isgm:03d}": darrj})
    else:
      dstmp = xarray.Dataset({f"indx_segm{isgm:03d}": darri, \
                  f"jndx_segm{isgm:03d}": darrj})
      dset = xarray.merge([dset, dstmp])
     
    f_plt = False
    if f_plt:
      plt.ion()

      fig1 = plt.figure(1,figsize=(9,8))
      plt.clf()
      ax1 = plt.axes([0.1, 0.24, 0.8, 0.7])

      if min(xOB) > 0. and min(lon_spear) < 0.:
        lon_spear = np.where(lon_spear<0, lon_spear+360, lon_spear)

      ax1.plot(xOB, yOB,'r*')
      for ikk in range(npnts):
#      ax1.plot(x0,y0,'r*')
        ixx = INDX[ikk,:]
        jxx = JNDX[ikk,:]
        xv  = np.append(lon_spear[jxx,ixx], lon_spear[jxx[0],ixx[0]])
        yv  = np.append(lat_spear[jxx,ixx], lat_spear[jxx[0],ixx[0]])
        ax1.plot(xv, yv, 'o-')

      ax1.axis('scaled')
      ax1.set_title('mod_utils_ob.py: MOM6 NEP OB segment 1, SPEAR gmapi')

  if fsave:
    print(f"Saving gmapi indices --> {dflout}")
    dset.to_netcdf(dflout, format='NETCDF3_64BIT', engine='netcdf4')

  return

def derive_obsegm_lonlat(hgrid, segments, isgm):
  """
    Derive lon lat and construct data sets
  """
  nsgm   = isgm+1 
  dset_segm = segments[isgm]
  xOB    = dset_segm.coords.lon.data
  yOB    = dset_segm.coords.lat.data
  npnts  = len(xOB)
  nx     = dset_segm.nx
  ny     = dset_segm.ny
  bndary = dset_segm.border 

  if bndary == 'west':
    lon_segm = hgrid['x'].isel(nxp=0).data
    lat_segm = hgrid['y'].isel(nxp=0).data
  elif bndary == 'north':
    lon_segm = hgrid['x'].isel(nyp=-1).data
    lat_segm = hgrid['y'].isel(nyp=-1).data
  elif bndary == 'east':
    lon_segm = hgrid['x'].isel(nxp=-1).data
    lat_segm = hgrid['y'].isel(nxp=-1).data
  elif bndary == 'south':
    lon_segm = hgrid['x'].isel(nyp=0).data
    lat_segm = hgrid['y'].isel(nyp=0).data

  nxsgm = [x for x in range(0,nx)]
  nysgm = [x for x in range(0,ny)]
  xdim  = f'nx_segment_{nsgm:03d}'
  ydim  = f'ny_segment_{nsgm:03d}'
  if ny == 1:
    darr_lon = xarray.DataArray(lon_segm, dims=(xdim),
                              coords={xdim: nxsgm})
    darr_lat = xarray.DataArray(lat_segm, dims=(xdim),
                              coords={xdim: nxsgm})
  else:
    darr_lon = xarray.DataArray(lon_segm, dims=(ydim),
                              coords={ydim: nysgm})
    darr_lat = xarray.DataArray(lat_segm, dims=(ydim),
                              coords={ydim: nysgm})

  dset_segm = xarray.Dataset({f"lon_segment_{nsgm:03d}": darr_lon, \
           f"lat_segment_{nsgm:03d}": darr_lat})

  return dset_segm

 

def derive_obsegm_3D(hgrid, ds, segments, isgm, varnm, INDX, JNDX, time_steps=[]):
  """
    Derive data set for an OB segment = isgm for variable = varnm
    Scalar (h-point) or u/v 3D fields (+ time dim)  --> 2D vertical sections 
    ds - xarray of climatology field for 1 variable, subset to the region
    hgrid - topo/grid for the climatology field (MOM6 NEP grid)
    segments - xarray with information for OB segments created in boundary.Segment
    isgm - segment # [0, 1, ...]
    time_steps - array of time (days, or hours, or sec) since .... 

    INDX, JNDX - n x 4 arrays of grid points (gmapi) for bilinear interpolation

 Use bilinear interpolation 
 Points are mapped onto a reference square for interpolation then 
 Perform interpolation on reference rectangle, that is 
 similar to interp on actual rectangle
 The advantage of using a reference quadrilateral is that basis
 functions computed once and used for all grid points - saved lots of 
 computational time
     See p. 83-90, Gockenbach, "Understanding Finite-Element Methods" textbook

  """
  import mod_bilinear as mblnr
  importlib.reload(mblnr)

# Find basis functions for a reference rectangle:
  phi1,phi2,phi3,phi4 = mblnr.basisFn_RectRef()
  phi_basis           = np.array([phi1, phi2, phi3, phi4]).transpose() # basis funs in columns

  nsgm  = isgm+1 
  dset_segm = segments[isgm]
  xOB   = dset_segm.coords.lon.data
  yOB   = dset_segm.coords.lat.data
  npnts = len(xOB)
  nx    = dset_segm.nx
  ny    = dset_segm.ny

# Make sure that gmapi is for the right section:
  assert npnts==len(INDX), "INDX and OB segments mismatch in length"

# Read SPEAR grid, topo
#  HHS, LONs, LATs = subset_spear_topo()
  LONs, LATs = subset_spear_coord('h')
# Make Longitudes same range:
  xOB = np.where(xOB<0., xOB+360., xOB)
  LONs = np.where(LONs<0., LONs+360., LONs)

  dtm   = ds.time.data
  AA    = ds[f'{varnm}'].data
  ntime = len(dtm) 

# interface depths and dz
  zzi  = ds.z_interf.data
  dz   = np.diff(zzi)
  if ny == 1:
    dz2d = np.tile(dz,(nx,1)).transpose() 
  else:
    dz2d = np.tile(dz,(ny,1)).transpose() 

  dz3d = np.tile(dz2d,(ntime,1,1))

# Assuming 3D arrays and time dimension:
  ndim = len(AA.shape)
  if ndim < 4:
    raise Exception(f'4D array (time, depth, y, x) is assumed, check array dim')

# No vertical interpolation, so use n levels from input data:
  nzlev = AA.shape[1]
  Ai = np.zeros((ntime,nzlev,ny,nx))+1.e30

  for itime in range(ntime):
    print(f'OBsegment = {nsgm} Time = {itime+1}')
    A   = AA[itime,:]
# Fill missing values (bottom/land):
    dmm =  mom6util.fill_land3d(A)
    assert np.max(abs(dmm)) < 1.e20, "Bottom/land not filled correctly"
    assert not np.isnan(np.max(abs(dmm))), "Bottom/land not filled correctly"
    A3d = dmm.copy()

# Horizontal interpolation - bilinear
    for ikk in range(npnts):
      x0 = xOB[ikk]
      y0 = yOB[ikk]
      II = np.squeeze(INDX[ikk,:])
      JJ = np.squeeze(JNDX[ikk,:])
#      II, JJ = mblnr.sort_gridcell_indx(II,JJ)
      ii1, ii2, ii3, ii4 = II
      jj1, jj2, jj3, jj4 = JJ
#      ii1, jj1 = II[0], JJ[0]
#      ii2, jj2 = II[1], JJ[1]
#      ii3, jj3 = II[2], JJ[2]
#      ii4, jj4 = II[3], JJ[3]

      xx = LONs[JJ,II]
      yy = LATs[JJ,II]

# Use cartesian coordinates for mapping
      if x0 < 0.: x0 = x0+360.
      xx = np.where(xx<0., xx+360., xx)
      f_repeated= check_repeated_vertices(xx,yy)
      if f_repeated:
        print(f"Bad box with coninciding vertices ikk={ikk}, approximate interpolation")
        xht = 1.e-3
        yht = 1.e-3
      else:
        xref     = x0-0.1
        yref     = y0-0.1
        XV, YV   = mblnr.lonlat2xy_wrtX0(xx, yy, xref, yref)
        x0c, y0c = mblnr.lonlat2xy_pnt(x0, y0, xref, yref)
        xht, yht = mblnr.map_x2xhat(XV, YV, x0c, y0c)   # cartesian coord
  #      INp      = mmisc1.inpolygon_1pnt(x0c, y0c, XV, YV)
  #      INp2    = mmisc1.inpolygon_1pnt(x0, y0, XX, YY)
  # If the grid point inside the box, then mapping is the problem
  # if not - then these are a few cases near singularities of SPEAR I/J axes
  # over land where bounding boxes could not be located 
  # In a few cases, box coordinates may repeat due to SPEAR singular I/J grid
  # that does not change coordinates for I/J near singular regions (boxes are triangulars)
  # THis results in singular matrix A for mappring
  # For very thin  rotated, skewed quadrilaterals mapping does not work well
  # Try to rotate the quadrilateral
        if abs(xht) > 1 or abs(yht) > 1:
          xht0 = xht
          yht0 = yht
          XVr, YVr, x0r, y0r  = rotate_box(XV, YV, x0c, y0c)
          xht, yht = mblnr.map_x2xhat(XVr, YVr, x0r, y0r)
#        print(f"ERR: ikk={ikk} Mapping ref box xht={xht0:5.1f} yht={yht0:5.1f}, fixed " +\
#              f" xht={xht:5.1f}, yht={yht:5.1f}")

        if abs(xht) > 1. or abs(yht) > 1.:
# If nothing works, interpolate into the center
# these should be very rare for locations on land where I / J axes converge 
          print(f"Fixing by rotating ref BOX failed ikk={ikk} " +\
                f"xht={xht:5.1f} yht={yht:5.1f}, approximate xhy, yht as middle pnt")
          xht = 1.e-3
          yht = 1.e-3
        
      aa1 = np.squeeze(A3d[:,jj1,ii1])
      aa2 = np.squeeze(A3d[:,jj2,ii2])
      aa3 = np.squeeze(A3d[:,jj3,ii3])
      aa4 = np.squeeze(A3d[:,jj4,ii4])

      HT  = np.array([aa1, aa2, aa3, aa4]).transpose()
      hintp  = mblnr.bilin_interp1D(phi1, phi2, phi3, phi4, xht, yht, HT)

# Check: abs. values of interpolated values <= original data
      mxHT = np.max(abs(HT), axis=1)
      dmx  = abs(hintp)/mxHT
      if max(dmx-1.) > 0.1:
        print(f"!!! segm{nsgm} {varnm} Min/Max test violated: ikk={ikk} dlt: {np.max(dmx)}")
        if max(dmx-1.) > 1.:
          raise Exception("MinMax test: Interp error Check interpolation")

      if ny == 1:
        Ai[itime, :, 0, ikk] = hintp
      else:
        Ai[itime, :, ikk, 0] = hintp
    
# Construct data set:
  if ny == 1:
    dz3d = np.expand_dims(dz3d, axis=2)
  else:
    dz3d = np.expand_dims(dz3d, axis=3)
  nxsgm = [x for x in range(0,nx)]
  nysgm = [x for x in range(0,ny)]
  if len(time_steps) == 0:
    stime = [x for x in range(1,ntime+1)]
  else:
    stime = time_steps
  nzsgm = [x for x in range(0,nzlev)]
  sgmnm = f"segment_{nsgm:03d}"
  dim1  = "time"
  dim2  = f"nz_{sgmnm}"
  dim3  = f"ny_{sgmnm}"
  dim4  = f"nx_{sgmnm}"
  darr_var = xarray.DataArray(Ai, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
  darr_dz  = xarray.DataArray(dz3d, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
#  f"{varnm}_segment_{nsgm:03d}"
  dset_segm = xarray.Dataset({f"{varnm}_segment_{nsgm:03d}": darr_var, \
           f"dz_{varnm}_segment_{nsgm:03d}": darr_dz})

  return dset_segm
   
 
def derive_obsegm_ssh(hgrid, ds, segments, isgm, INDX, JNDX, time_steps=[], varnm='ssh', \
                      calendar='julian', dnmb0=0):
  """
    Derive data set for an OB segment = isgm for variable = varnm
    SSH fields at h-points 2D fields --> 1D OB segment 
    ds - xarray of climatology field for 1 variable, subset to the region
    hgrid - topo/grid for the climatology field
    segments - xarray with information for OB segments created in boundary.Segment
    isgm - segment # [0, 1, ...]

    INDX, JNDX - n x 4 arrays of grid points (gmapi) for bilinear interpolation

   Use bilinear interpolation 

  """
  import mod_bilinear as mblnr
#  importlib.reload(mblnr)
  import mod_time as mtime

# Find basis functions for a reference rectangle:
  phi1,phi2,phi3,phi4 = mblnr.basisFn_RectRef()
  phi_basis           = np.array([phi1, phi2, phi3, phi4]).transpose() # basis funs in columns

  nsgm  = isgm+1 
  dset_segm = segments[isgm]
  xOB   = dset_segm.coords.lon.data
  yOB   = dset_segm.coords.lat.data
  npnts = len(xOB)
  nx    = dset_segm.nx
  ny    = dset_segm.ny

# Make sure that gmapi is for the right section:
  assert npnts==len(INDX), "INDX and OB segments mismatch in length"

# Read SPEAR grid, topo
#  HHS, LONs, LATs = subset_spear_topo()
  LONs, LATs = subset_spear_coord('h')
# Make Longitudes same range:
  xOB = np.where(xOB<0., xOB+360., xOB)
  LONs = np.where(LONs<0., LONs+360., LONs)

  dtm   = ds.time.data
#  AA    = ds['ssh'].data
  ntime = len(dtm) 

# 2D array and time dimension:
#  ndim = len(AA.shape)
#  if ndim != 3:
#    raise Exception(f'3D array (time, y, x) is assumed for ssh, check array dim')

  Ai = np.zeros((ntime,ny,nx))+1.e30

  for itime in range(ntime):
    print(f'interpolating ssh, segm={nsgm}, Time = {itime+1}')
#    A   = AA[itime,:]
    A = ds[varnm].isel(time=itime).data.squeeze()
# Fill missing values (bottom/land):
    dmm =  mom6util.fill_land3d(A)
    assert np.max(abs(dmm)) < 1.e20, "Bottom/land not filled correctly: No huge values are allowed"
    assert not np.isnan(np.max(abs(dmm))), "Bottom/land not filled correctly: no NaN values are allowed"
    A2d = dmm.copy()

# Horizontal interpolation - bilinear
    for ikk in range(npnts):
      x0 = xOB[ikk]
      y0 = yOB[ikk]
      II = np.squeeze(INDX[ikk,:])
      JJ = np.squeeze(JNDX[ikk,:])
#      II, JJ = mblnr.sort_gridcell_indx(II,JJ)
      ii1, ii2, ii3, ii4 = II
      jj1, jj2, jj3, jj4 = JJ

      xx = LONs[JJ,II]
      yy = LATs[JJ,II]

# Use cartesian coordinates for mapping
      if x0 < 0.: x0 = x0+360.
      xx = np.where(xx<0., xx+360., xx)
      xref     = x0-0.1
      yref     = y0-0.1
      XV, YV   = mblnr.lonlat2xy_wrtX0(xx, yy, xref, yref)
      x0c, y0c = mblnr.lonlat2xy_pnt(x0, y0, xref, yref)
      xht, yht = mblnr.map_x2xhat(XV, YV, x0c, y0c)   # cartesian coord
# if mapping does not work Try to rotate the quadrilateral
      if abs(xht) > 1 or abs(yht) > 1:
        xht0 = xht
        yht0 = yht
        XVr, YVr, x0r, y0r  = rotate_box(XV, YV, x0c, y0c)
        xht, yht = mblnr.map_x2xhat(XVr, YVr, x0r, y0r)
#        print(f"ERR: ikk={ikk} Mapping ref box xht={xht0:5.1f} yht={yht0:5.1f}, fixed " +\
#              f" xht={xht:5.1f}, yht={yht:5.1f}")

        if abs(xht) > 1. or abs(yht) > 1.:
# If nothing works, interpolate into the center
# these should be very rare for locations on land where I / J axes converge 
          print(f"Fixing by rotating ref BOX failed ikk={ikk} " +\
                f"xht={xht:5.1f} yht={yht:5.1f}, approximate xhy, yht as middle pnt")
          xht = 1.e-3
          yht = 1.e-3

      aa1 = np.squeeze(A2d[jj1,ii1])
      aa2 = np.squeeze(A2d[jj2,ii2])
      aa3 = np.squeeze(A2d[jj3,ii3])
      aa4 = np.squeeze(A2d[jj4,ii4])

      HT  = np.array([aa1, aa2, aa3, aa4]).transpose()
      hintp  = mblnr.bilin_interp(phi1, phi2, phi3, phi4, xht, yht, HT)

# Check: abs. values of interpolated values <= original data
      mxHT = np.max(abs(HT))
      if mxHT == 0:
        mxHT = 1.e-20
      dmx  = abs(hintp)/mxHT
      if (dmx-1.) > 0.1:
        print(f"!!! segm{nsgm} ssh Min/Max test violated: ikk={ikk} dlt: {np.max(dmx)}")
        if (dmx-1.) > 1.:
          raise Exception("MinMax test: Interp error Check interpolation")

      if ny == 1:
        Ai[itime, 0, ikk] = hintp
      else:
        Ai[itime, ikk, 0] = hintp
    
# Construct data set:
  nxsgm = [x for x in range(0,nx)]
  nysgm = [x for x in range(0,ny)]
  if len(time_steps) > ntime:
# If requested time is > ntime, assume time padding is required at the end
    npad = len(time_steps) - ntime
    print(f"Given time_steps {len(time_steps)} >  N of records {ntime}, padding extra time {npad} days")
    for ipd in range(npad):
      dmm = np.expand_dims(Ai[-1,:], axis=0)
      Ai = np.append(Ai, dmm, axis=0)
    stime = time_steps
  elif len(time_steps) < ntime:
    print(f"Provided time_steps len={len(time_steps)} <  N of records {ntime}, use {ntime} for Time")
    stime = [x for x in range(1,ntime+1)]
  else:
    stime = time_steps

  if calendar.lower() == 'gregorian':
    yrS, jdayS = mtime.dnmb2jday(dnmb0)
    dnmbE = dnmb0 + len(stime)-1
    yrE, jdayE = mtime.dnmb2jday(dnmbE)
    stime = mtime.npdatetime_year(yrS, yrE=yrE, day_start=jdayS, day_end=jdayE)

  sgmnm = f"segment_{nsgm:03d}"
  dim1  = "time"
  dim2  = f"ny_{sgmnm}"
  dim3  = f"nx_{sgmnm}"
  darr_var = xarray.DataArray(Ai, dims=(dim1, dim2, dim3),
             coords={dim1: stime, dim2: nysgm, dim3: nxsgm})
  dset_segm = xarray.Dataset({f"zos_segment_{nsgm:03d}": darr_var})

  return dset_segm

def derive_obsegm_uv(hgrid, ds_uo, ds_vo, segments, isgm, theta_rot, varnm, \
                     INDX, JNDX, time_steps=[]):
  """
    Derive data set for an OB segment = isgm for U and V components
    Scalar (h-point) or u/v 3D fields (+ time dim)  --> 2D vertical sections 
    dsu / dsv - xarray of climatology field for U/V variable, subset to the region
    hgrid - topo/grid for the climatology field (MOM6 NEP grid)
    segments - xarray with information for OB segments created in boundary.Segment
    isgm - segment # [0, 1, ...]

    INDXu, JNDXu - n x 4 arrays of grid points (gmapi) for bilinear interpolation
                   for u-points on SPEAR
    similar for v-components INDXv, JNDXv

    icegrid - spear static file with grid info, need rotation angles for
              U-vectors rotation onto MOM NEP grid from SPEAR
              alternatively, rotation angle rot_angle is provided
              angle is from J-axis to true North dir (math convention + is c/clockwise)

   Use bilinear interpolation 

  """
  import mod_bilinear as mblnr
  importlib.reload(mblnr)

# Find basis functions for a reference rectangle:
  phi1,phi2,phi3,phi4 = mblnr.basisFn_RectRef()
  phi_basis           = np.array([phi1, phi2, phi3, phi4]).transpose() # basis funs in columns

  nsgm   = isgm+1 
  dset_segm = segments[isgm]
  xOB    = dset_segm.coords.lon.data
  yOB    = dset_segm.coords.lat.data
  npnts  = len(xOB)
  nx     = dset_segm.nx
  ny     = dset_segm.ny
  bndary = dset_segm.border  

# Make sure that gmapi is for the right section:
  assert npnts==len(INDX), "INDX-U and OB segments mismatch in length"
  assert npnts==len(JNDX), "JNDX-V and OB segments mismatch in length" 

# SPEAR grid coords:
# SPEAR grid coords:
  if varnm == 'u':
    LONs, LATs = subset_spear_coord('u')
    ds    = ds_uo
  elif varnm == 'v':
    LONs, LATs = subset_spear_coord('v')
    ds    = ds_vo

# Make Longitudes same range:
  xOB = np.where(xOB<0., xOB+360., xOB)
  LONs = np.where(LONs<0., LONs+360., LONs)

  dtm   = ds.time.data
  ntime = len(dtm) 
  U4D   = ds_uo['uo'].data
  V4D   = ds_vo['vo'].data

# interface depths and dz
  zzi  = ds.z_interf.data
  dz   = np.diff(zzi)
  if ny == 1:
    dz2d = np.tile(dz,(nx,1)).transpose() 
  else:
    dz2d = np.tile(dz,(ny,1)).transpose() 

  dz3d = np.tile(dz2d,(ntime,1,1))

# Assuming 3D arrays and time dimenstion:
  ndim = len(U4D.shape)
  if ndim < 4:
    raise Exception(f'4D array (time, depth, y, x) is assumed, check array dim')

# No vertical interpolation, so use n levels from input data:
  nzlev = U4D.shape[1]
  Ai = np.zeros((ntime,nzlev,ny,nx))+1.e30

# For vector rotation, make sure the angle sign matches math convention
# For NEP hgrid saves angle positive clockwise (from axis to true N/E)
# *(-1) make it in math sense: positive c/clockwise axis J/I to true N/E
# also can derive angles at the OB: theta=dset_segm.coords.angle.data
# use plot_rotangleNEP(hgrid, hmask, nfig=1) to plot angles for the whole domain
  if bndary == 'west':
    thetaNEP = -hgrid['angle_dx'].isel(nxp=0).data  
  elif bndary == 'north':
    thetaNEP = -hgrid['angle_dx'].isel(nyp=-1).data  
  elif bndary == 'east':
    thetaNEP = -hgrid['angle_dx'].isel(nxp=-1).data  
  elif bndary == 'south':
    thetaNEP = -hgrid['angle_dx'].isel(nyp=0).data  

  for itime in range(ntime):
    print(f'{varnm} Time = {itime+1}')
    A   = U4D[itime,:]
# Fill missing values (bottom/land):
    dmm =  mom6util.fill_land3d(A)
    assert np.max(abs(dmm)) < 1.e20, "Bottom/land not filled correctly"
    assert not np.isnan(np.max(abs(dmm))), "Bottom/land not filled correctly"
    U3d = dmm.copy()

    A   = V4D[itime,:]
    dmm =  mom6util.fill_land3d(A)
    assert np.max(abs(dmm)) < 1.e20, "Bottom/land not filled correctly"
    assert not np.isnan(np.max(abs(dmm))), "Bottom/land not filled correctly"
    V3d = dmm.copy()

# Horizontal interpolation - bilinear
    for ikk in range(npnts):
      x0 = xOB[ikk]
      y0 = yOB[ikk]
      II = np.squeeze(INDX[ikk,:])
      JJ = np.squeeze(JNDX[ikk,:])
#      II, JJ = mblnr.sort_gridcell_indx(II,JJ)
      ii1, ii2, ii3, ii4 = II
      jj1, jj2, jj3, jj4 = JJ

      xx = LONs[JJ,II]
      yy = LATs[JJ,II]

# Use cartesian coordinates for mapping
      if x0 < 0.: x0 = x0+360.
      xx = np.where(xx<0., xx+360., xx)
      f_repeated= check_repeated_vertices(xx,yy)
      if f_repeated: 
        print(f"Bad box with coninciding vertices ikk={ikk}, approximate interpolation")
        xht = 1.e-3
        yht = 1.e-3
      else:
        xref     = x0-0.1
        yref     = y0-0.1
        XV, YV   = mblnr.lonlat2xy_wrtX0(xx, yy, xref, yref)
        x0c, y0c = mblnr.lonlat2xy_pnt(x0, y0, xref, yref)
        xht, yht = mblnr.map_x2xhat(XV, YV, x0c, y0c)   # cartesian coord
  #      INp      = mmisc1.inpolygon_1pnt(x0c, y0c, XV, YV)
  #      INp2    = mmisc1.inpolygon_1pnt(x0, y0, XX, YY)
  # If the grid point inside the box, then mapping is the problem
  # if not - then these are a few cases near singularities of SPEAR I/J axes
  # over land where bounding boxes could not be located 
  # In a few cases, box coordinates may repeat due to SPEAR singular I/J grid
  # that does not change coordinates for I/J near singular regions (boxes are triangulars)
  # THis results in singular matrix A for mappring
  # For very thin  rotated, skewed quadrilaterals mapping does not work well
  # Try to rotate the quadrilateral
        if abs(xht) > 1 or abs(yht) > 1:
          xht0 = xht
          yht0 = yht
          XVr, YVr, x0r, y0r  = rotate_box(XV, YV, x0c, y0c)
          xht, yht = mblnr.map_x2xhat(XVr, YVr, x0r, y0r)
#        print(f"ERR: ikk={ikk} Mapping ref box xht={xht0:5.1f} yht={yht0:5.1f}, fixed " +\
#              f" xht={xht:5.1f}, yht={yht:5.1f}")

        if abs(xht) > 1. or abs(yht) > 1.:
# If nothing works, interpolate into the center
# these should be very rare for locations on land where I / J axes converge 
          print(f"Fixing by rotating ref BOX failed ikk={ikk} " +\
                f"xht={xht:5.1f} yht={yht:5.1f}, approximate xhy, yht as middle pnt")
          xht = 1.e-3
          yht = 1.e-3

# Angle in radians!
      tht1 = theta_rot[jj1,ii1]
      tht2 = theta_rot[jj2,ii2]
      tht3 = theta_rot[jj3,ii3]
      tht4 = theta_rot[jj4,ii4]
      uu1  = np.squeeze(U3d[:,jj1,ii1])
      uu2  = np.squeeze(U3d[:,jj2,ii2])
      uu3  = np.squeeze(U3d[:,jj3,ii3])
      uu4  = np.squeeze(U3d[:,jj4,ii4])
      vv1  = np.squeeze(V3d[:,jj1,ii1])
      vv2  = np.squeeze(V3d[:,jj2,ii2])
      vv3  = np.squeeze(V3d[:,jj3,ii3])
      vv4  = np.squeeze(V3d[:,jj4,ii4])

# Rotate onto N/E grid
      uNE1, vNE1 = rotate_uv(uu1, vv1, tht1)
      uNE2, vNE2 = rotate_uv(uu2, vv2, tht2)
      uNE3, vNE3 = rotate_uv(uu3, vv3, tht3)
      uNE4, vNE4 = rotate_uv(uu4, vv4, tht4)

      f_plt = False
      if f_plt:
        u1 = uu1[0]*100
        v1 = vv1[0]*100
        ur = uNE1[0]*100
        vr = vNE1[0]*100
        plot_vectors(u1, v1, ur, vr, nfig=1)


# Interpolate U V components (on true N/E grid) to NEP OB segments
# then rotate onto NEP I/J grid
# NEP angle is J/I to N/E not in math sense but positive clockwise!!!
# i.e. this is N/E to J/I angle in math sense
      UT  = np.array([uu1, uu2, uu3, uu4]).transpose()
      VT  = np.array([vv1, vv2, vv3, vv4]).transpose()
      Uintp = mblnr.bilin_interp1D(phi1, phi2, phi3, phi4, xht, yht, UT)
      Vintp = mblnr.bilin_interp1D(phi1, phi2, phi3, phi4, xht, yht, VT)
    
      tht_nep = thetaNEP[ikk]
      UR, VR = rotate_uv(Uintp, Vintp, tht_nep, ij2NE=False)
      #plot_vectors(Uintp[0]*100, Vintp[0]*100, UR[0]*100, VR[0]*100, nfig=1)

# Check: abs. values of interpolated values <= original data
      mxHT = np.max(abs(UT), axis=1)
      dmx  = (abs(Uintp))/mxHT
      if max(dmx-1.) > 0.1:
        print(f"!!! segm{nsgm} {varnm} Min/Max test violated: ikk={ikk} dlt: {np.max(dmx)}")
        if max(dmx-1.) > 1.:
          raise Exception("MinMax test: Interp error Check interpolation")

      if varnm == 'u':
        hintp = UR
      else:
        hintp = VR

      if ny == 1:
        Ai[itime, :, 0, ikk] = hintp
      else:
        Ai[itime, :, ikk, 0] = hintp
    
# Construct data set:
  if ny == 1:
    dz3d = np.expand_dims(dz3d, axis=2)
  else:
    dz3d = np.expand_dims(dz3d, axis=3)
  nxsgm = [x for x in range(0,nx)]
  nysgm = [x for x in range(0,ny)]
  if len(time_steps) == 0:
    stime = [x for x in range(1,ntime+1)]
  else:
    stime = time_steps
  nzsgm = [x for x in range(0,nzlev)]
  sgmnm = f"segment_{nsgm:03d}"
  dim1  = "time"
  dim2  = f"nz_{sgmnm}"
  dim3  = f"ny_{sgmnm}"
  dim4  = f"nx_{sgmnm}"
  darr_var = xarray.DataArray(Ai, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
  darr_dz  = xarray.DataArray(dz3d, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
#  f"{varnm}_segment_{nsgm:03d}"
  dset_segm = xarray.Dataset({f"{varnm}_segment_{nsgm:03d}": darr_var, \
           f"dz_{varnm}_segment_{nsgm:03d}": darr_dz})

  return dset_segm
  
def rotate_uv(uu, vv, tht, ij2NE=True):
  """
    Rotate OLD axes to NEW axes by angle tht (positive in math sense), i.e.
    rotate vector to (-tht) vector U={uu,vv} ---> Ur={ur,vr}

    angle tht (in radians): angle between I/J axis and N/E true directions 
    positive / negative in math sense
    e.g. J-axis to true North angle tht > 0 
         means positive rotation of J to N
    ij2NE=False:         to rotate back from true North/East to I/J axes

    Use rotation matrix for (-tht):  cos(tht) sin(tht)
                                    -sin(tht) cos(tht)
  """
  if ij2NE:
    ur = uu*np.cos(tht) + vv*np.sin(tht)
    vr = uu*(-np.sin(tht)) + vv*np.cos(tht)
  else:
    ur = uu*np.cos(tht) - vv*np.sin(tht)
    vr = uu*np.sin(tht) + vv*np.cos(tht)

  return ur, vr

def get_rotangle(icegrid_spear, fconfig, grid_spear):
  """
    Derive angle between the model IJ-axes and true North-East
    from ice.static.nc file
    saved are components of the rotation matrix for vector rotation from
    the model I/J grid onto true N/E axes
    U_trueNE = R*U
    Derive angle as:
    arcsin(SINROT) ---> angle between J-axis and N (positive in math sense, J to N)
    arccos(COSROT) ---> angle between I-axis and E (sign of angle is not preserved)
  """
  from yaml import safe_load
  import matplotlib.pyplot as plt
  from matplotlib import colormaps

  with open(fconfig) as ff:
    config = safe_load(ff)
  spear_dir = os.path.join(config['filesystem']['spear_month_ens'], 'monthly_clim')

  # Domain lon/lat boundaries:
  lonW = config['domain']['west_lon']
  lonE = config['domain']['east_lon']
  latN = config['domain']['north_lat']
  latS = config['domain']['south_lat']

  LON = icegrid_spear['GEOLON'].data
  LAT = icegrid_spear['GEOLAT'].data
  if lonW > np.max(LON):
    lonW = lonW-360.
  elif lonW < np.min(LON):
    lonW = lonW+360.

  if lonE > np.max(LON):
    lonE = lonE-360.
  elif lonE < np.min(LON):
    lonE = lonE+360.

  lon_slice = slice(lonW, lonE)
  lat_slice = slice(latS, latN)

  dset    = icegrid_spear.sel(yT=lat_slice, xT=lon_slice) 
  cosrot  = dset['COSROT'].data
  sinrot  = dset['SINROT'].data
  lon_sub = dset['GEOLON'].data
  lat_sub = dset['GEOLAT'].data
  Lmask   = grid_spear['wet'].sel(yh=lat_slice, xh=lon_slice).data

  theta_I2E = np.arccos(cosrot) # does not preserve the sign of the angle
  theta_J2N = np.arcsin(sinrot) 
  rad2deg   = 180./np.pi
  theta     = theta_J2N

  f_chck = False
  if f_chck:
    plt.ion()
    fig = plt.figure(1,figsize=(9,8))
    plt.clf()
    #ax1.cla()
    ax1 = plt.axes([0.1, 0.55, 0.8, 0.32])
    im=ax1.pcolormesh(theta_I2E*rad2deg, cmap='seismic', vmin=-60, vmax=60)
    ax1.contour(Lmask, [0.01], colors=[(0,0,0)])
  #  ax1.contour(cosrot, [x/10 for x in range(0,10)])
    ax1.contour(cosrot, [0.99], colors=[(0,0,1)])
#    ax1.contour(lon_sub,[x for x in range(-205,-100,5)], linestyles='solid', 
#                linewidths=0.5, colors=[(0.7,0.7,0.7)])
    ax1.contour(lat_sub,[x for x in range(10,88,5)], linestyles='solid', 
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
    ax1.axis('scaled')
    if lon_slice is None: ax1.set_ylim([150, 319])

    ax2 = plt.axes([0.91, 0.55, 0.03, 0.32])
    fig.colorbar(im, cax=ax2)
    ax1.set_title('Angle I to East from COSROT')

    ax3 = plt.axes([0.1, 0.08, 0.8, 0.32])
    im2=ax3.pcolormesh(theta_J2N*rad2deg, cmap='seismic', vmin=-60, vmax=60)
    ax3.contour(Lmask, [0.01], colors=[(0,0,0)])
    ax3.contour(sinrot, [0.01], colors=[(0,0,1)])
#    ax3.contour(lon_sub,[x for x in range(-205,-100,5)], linestyles='solid', 
#                linewidths=0.5, colors=[(0.7,0.7,0.7)])
    ax3.contour(lon_sub,[x for x in range(-300,-1,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
    ax3.contour(lon_sub,[x for x in range(0,100,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
    ax3.axis('scaled')
    if lon_slice is None: ax3.set_ylim([150, 319])

    ax4 = plt.axes([0.91, 0.08, 0.03, 0.32])
    fig.colorbar(im2, cax=ax4)
    ax3.set_title('Angle J to North from SINROT')

    btx = 'mod_utils_ob.py'
    bottom_text(btx)


  return theta, cosrot, sinrot 

def segm_topo(nsegm, HHM, hgrid):
  """
    Derive bottom profile, lon, lat along the OB segment
  """
  import mod_misc1 as mmsc1

  if nsegm == 1:
    bndry = 'north'
  elif nsegm == 2:
    bndry = 'east'
  elif nsegm == 3:
    bndry = 'south'
  elif nsegm == 4:
    bndry = 'west'
  else:
    raise Exception(f"N of boundaries should be 4 for NEP, nsegm={nsegm}")

  if bndry == 'west':
    lon_segm = hgrid['x'].isel(nxp=0).data
    lat_segm = hgrid['y'].isel(nxp=0).data
    Hbtm     = HHM[:,0].squeeze()
  elif bndry == 'north':
    lon_segm = hgrid['x'].isel(nyp=-1).data
    lat_segm = hgrid['y'].isel(nyp=-1).data
    Hbtm     = HHM[-1,:].squeeze()
  elif bndry == 'east':
    lon_segm = hgrid['x'].isel(nxp=-1).data
    lat_segm = hgrid['y'].isel(nxp=-1).data
    Hbtm     = HHM[:,-1].squeeze()
  elif bndry == 'south':
    lon_segm = hgrid['x'].isel(nyp=0).data
    lat_segm = hgrid['y'].isel(nyp=0).data
    Hbtm     = HHM[0,:].squeeze()

# Calculate distance along the OB segment:
  npnts = len(lat_segm)
  dL = np.zeros((npnts))
  for ikk in range(npnts-1):
    dltX = mmsc1.dist_sphcrd(lat_segm[ikk], lon_segm[ikk], lat_segm[ikk+1], lon_segm[ikk+1])*1.e-3
    dL[ikk+1] = dltX
  dist_segm = np.cumsum(dL)
#  dist_segm = mmsc1.dist_sphcrd(lat_segm[0], lon_segm[0], lat_segm, lon_segm)*1.e-3
  dist_btm  = dist_segm[0:-1:2].copy()
  if dist_btm[-1] < dist_segm[-1]:
    dist_btm[-1] = dist_segm[-1]

  nbpnts    = len(lon_segm)  # supergrid
  nhpnts    = len(Hbtm)
  dim1_name = 'supergrid_pnts'
  dim2_name = 'grid_pnts'
  nxsegm    = [x for x in range(nbpnts)]
  nxbotm    = [x for x in range(nhpnts)]
  darr_topo = xarray.DataArray(Hbtm, dims=(dim2_name), coords={dim2_name: nxbotm})
  darr_lon  = xarray.DataArray(lon_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_lat  = xarray.DataArray(lat_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_dist = xarray.DataArray(dist_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_bdx  = xarray.DataArray(dist_btm, dims=(dim2_name), coords={dim2_name: nxbotm})
  darr_name = xarray.DataArray([bndry], dims=("NL"), coords={"NL": [1]})

  dset_segm = xarray.Dataset({"segm_name": darr_name, \
                              "topo_segm": darr_topo, \
                              "lon_segm": darr_lon, \
                              "lat_segm": darr_lat, \
                              "dist_supergrid": darr_dist, \
                              "dist_grid": darr_bdx})

  return dset_segm

def calculate_dist_section(lon_segm, lat_segm, f_km=True):
  """
    Calculate distance (m / km) along a transect / segment
    given geog. coordinates along the transect
  """
  import mod_misc1 as mmsc1

  coeff = 1.0
  if f_km: coeff = 1.e-3

  npnts = len(lon_segm)
  dL = np.zeros((npnts))
  for ikk in range(npnts-1):
    dltX = mmsc1.dist_sphcrd(lat_segm[ikk], lon_segm[ikk], lat_segm[ikk+1], lon_segm[ikk+1])*coeff
    dL[ikk+1] = dltX
  dist_segm = np.cumsum(dL)

  return dist_segm, dL

def discard_repeated_indx(II, JJ, keep_last=False):
  """
    Remove repeated indices, i.e. eliminate same grid points
  """
  npnts = len(II)
  dL = np.zeros((npnts))
  for ikk in range(npnts-1):
    dltX = np.sqrt((II[ikk]-II[ikk+1])**2 + (JJ[ikk]-JJ[ikk+1])**2)
    dL[ikk+1] = dltX

# Eliminate same points:
  dL[0] = 1.e-6
  Ikeep = np.where(dL > 1.e-20)[0]  
  Isub  = II[Ikeep]
  Jsub  = JJ[Ikeep]
  if keep_last:
    if Isub[-1] != II[-1] and Jsub[-1] != JJ[-1]:
      Isub = np.append(Isub, II[-1])
      Jsub = np.append(Jsub, JJ[-1])

  return Isub, Jsub


def segm_topo_spear(II, JJ, lon_spear, lat_spear, HHS, nsegm=None):
  """
    Derive bottom profile, lon, lat along a  segment for SPEAR
    nsegm - NEP OB segment # if needed
    HHS - spear topo
    lon_spear, lat_spear - lon/ lat of the domain (typically 2D arrays)
    assumed to be at h-points - similar to topography
  """
  import mod_misc1 as mmsc1

  if nsegm == 1:
    bndry = 'north'
  elif nsegm == 2:
    bndry = 'east'
  elif nsegm == 3:
    bndry = 'south'
  elif nsegm == 4:
    bndry = 'west'
  else:
    bndry = 'SPEAR segment'
 
  if len(lon_spear.shape) == 1:
    lon_segm = lon_spear[II]
    lat_segm = lat_spear[JJ]
  else: 
    lon_segm = lon_spear[JJ,II]
    lat_segm = lat_spear[JJ,II]
  Hbtm     = HHS[JJ,II]

# Calculate distance along the OB segment:
  npnts = len(lat_segm)
  dL = np.zeros((npnts))
  for ikk in range(npnts-1):
    dltX = mmsc1.dist_sphcrd(lat_segm[ikk], lon_segm[ikk], lat_segm[ikk+1], lon_segm[ikk+1])*1.e-3
    dL[ikk+1] = dltX

# Eliminate same points:
#  dL[0] = 1.e-6
#  Ikeep = np.where(dL > 1.e-20)[0]  

  dist_segm = np.cumsum(dL)
#  dist_segm = mmsc1.dist_sphcrd(lat_segm[0], lon_segm[0], lat_segm, lon_segm)*1.e-3
  dist_btm  = dist_segm.copy()
  if dist_btm[-1] < dist_segm[-1]:
    dist_btm[-1] = dist_segm[-1]

  nbpnts    = len(lon_segm)  #
  nhpnts    = len(Hbtm)
  dim1_name = 'var_pnts'
  dim2_name = 'hgrid_pnts'
  nxsegm    = [x for x in range(nbpnts)]
  nxbotm    = [x for x in range(nhpnts)]
  darr_topo = xarray.DataArray(Hbtm, dims=(dim2_name), coords={dim2_name: nxbotm})
  darr_lon  = xarray.DataArray(lon_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_lat  = xarray.DataArray(lat_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_dist = xarray.DataArray(dist_segm, dims=(dim1_name), coords={dim1_name: nxsegm})
  darr_bdx  = xarray.DataArray(dist_btm, dims=(dim2_name), coords={dim2_name: nxbotm})
  darr_name = xarray.DataArray([bndry], dims=("NL"), coords={"NL": [1]})

  dset_segm = xarray.Dataset({"segm_name": darr_name, \
                              "topo_segm": darr_topo, \
                              "lon_segm": darr_lon, \
                              "lat_segm": darr_lat, \
                              "dist_supergrid": darr_dist, \
                              "dist_grid": darr_bdx})
  return dset_segm

def plot_rotangle(grid_spear, icegrid_spear, lon_slice=None, lat_slice=None):
  """
    Plot rot angles (cos/sine) from ice grid file
  """
  import matplotlib.pyplot as plt
  from matplotlib import colormaps
  from mod_utils_fig import bottom_text

  if not lon_slice is None:
    Lmask = grid_spear['wet'].sel(yh=lat_slice, xh=lon_slice).data
    cosrot = icegrid_spear['COSROT'].sel(yT=lat_slice, xT=lon_slice).data
    sinrot = icegrid_spear['SINROT'].sel(yT=lat_slice, xT=lon_slice).data
    lon_sub = icegrid_spear['GEOLON'].sel(yT=lat_slice, xT=lon_slice).data
    lat_sub = icegrid_spear['GEOLAT'].sel(yT=lat_slice, xT=lon_slice).data
  else:
    Lmask = grid_spear['wet'].data
    cosrot = icegrid_spear['COSROT'].data
    sinrot = icegrid_spear['SINROT'].data
    lon_sub = icegrid_spear['GEOLON'].data
    lat_sub = icegrid_spear['GEOLAT'].data


  theta = np.arcsin(sinrot)*180./np.pi  # degrees

  plt.ion()
  fig = plt.figure(1,figsize=(9,8))
  plt.clf()
  #ax1.cla()
  ax1 = plt.axes([0.1, 0.55, 0.8, 0.32])
  im=ax1.pcolormesh(cosrot, cmap='plasma', vmin=0.0, vmax=1)
  ax1.contour(Lmask, [0.01], colors=[(0,0,0)])
#  ax1.contour(cosrot, [x/10 for x in range(0,10)])
  ax1.contour(cosrot, [0.99], colors=[(0,0,1)])
  ax1.contour(lat_sub,[x for x in range(10,88,5)], linestyles='solid',     
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax1.axis('scaled')
  if lon_slice is None: ax1.set_ylim([150, 319])

  ax2 = plt.axes([0.91, 0.55, 0.03, 0.32])
  fig.colorbar(im, cax=ax2)
  ax1.set_title('COSROT')


  ax3 = plt.axes([0.1, 0.08, 0.8, 0.32])
  im2=ax3.pcolormesh(sinrot, cmap='seismic', vmin=-1, vmax=1)
  ax3.contour(Lmask, [0.01], colors=[(0,0,0)])
  ax3.contour(sinrot, [0.01], colors=[(0,0,1)])
  ax3.contour(lon_sub,[x for x in range(-300,-1,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax3.contour(lon_sub,[x for x in range(0,100,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax3.axis('scaled')
  if lon_slice is None: ax3.set_ylim([150, 319])

  ax4 = plt.axes([0.91, 0.08, 0.03, 0.32])
  fig.colorbar(im2, cax=ax4)
  ax3.set_title('SINROT')

  btx = 'mod_utils_ob: plot_rotangle'
  bottom_text(btx)

  fig2 = plt.figure(2,figsize=(9,8))
  plt.clf()
  ax21 = plt.axes([0.1, 0.4, 0.8, 0.5])
  im2  = ax21.pcolormesh(theta, cmap='turbo', vmin=-90, vmax=90)
  ax21.contour(Lmask, [0.01], colors=[(0,0,0)])
  ax21.contour(sinrot, [0.01], colors=[(0,0,1)])
  ax21.contour(lat_sub,[x for x in range(10,88,5)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax21.contour(lon_sub,[x for x in range(-300,-1,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax21.contour(lon_sub,[x for x in range(0,100,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax21.axis('scaled')
  if lon_slice is None: ax21.set_ylim([150, 319])

  ax22 = plt.axes([0.91, 0.4, 0.03, 0.5])
  fig2.colorbar(im2, cax=ax22)
  ax21.set_title('Angle: Axes to true N/E, degrees')

  bottom_text(btx)

def plot_rotangleNEP(hgrid, hmask, nfig=1):
  """
    Plot rot angles for NEP domain
  """
  import matplotlib.pyplot as plt
  from matplotlib import colormaps
  from mod_utils_fig import bottom_text

  Lmask = hmask['mask'].data
  theta = hgrid['angle_dx'].data*180./np.pi  # degrees
  ny_mask, nx_mask = Lmask.shape
  xmask = [x for x in range(0,2*nx_mask,2)]
  ymask = [x for x in range(0,2*ny_mask,2)]
  lon_nep = hgrid['x'].data
  lat_nep = hgrid['y'].data

  plt.ion()
  fig2 = plt.figure(nfig,figsize=(9,8))
  plt.clf()
  ax21 = plt.axes([0.1, 0.1, 0.8, 0.8])
  im2  = ax21.pcolormesh(theta, cmap='turbo', vmin=0, vmax=90)
#  ax21.contour(Lmask, [0.01], colors=[(0,0,0)])
  ax21.contour(lat_nep,[x for x in range(5,88,5)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax21.contour(lon_nep,[x for x in range(120,359,10)], linestyles='solid',
                linewidths=0.5, colors=[(0.7,0.7,0.7)])
  ax21.axis('scaled')
#  if lon_slice is None: ax21.set_ylim([150, 319])

  ax22 = plt.axes([0.91, 0.4, 0.03, 0.5])
  fig2.colorbar(im2, cax=ax22)
  ax21.set_title('NEP Angle: Axes to true N/E, degrees')

  btx = 'mod_utils_ob.py: plot_rotangleNEP'
  bottom_text(btx)

def plot_vectors(u1, v1, ur, vr, nfig=1):
  """
    Quick plot 2 vectors to check rotation
  """
  plt.ion()
  fig = plt.figure(nfig,figsize=(9,8))
  fig.clf()
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])
  ax1.quiver(u1, v1, angles='xy', scale_units='xy', scale=1)
  ax1.quiver(ur, vr, angles='xy', scale_units='xy', scale=1, color='r')
  vmax = np.max(abs(np.array([u1,v1,ur,vr])))
  ax1.set_xlim([-vmax, vmax])
  ax1.set_ylim([-vmax, vmax])
  ax1.set_title('Original: black, Rotated: Red')

  return

def subset_spear_topo():
  """
    Subset SPEAR topo grid to NEP domain
  """
  fyaml = 'pypaths_gfdlpub.yaml'
  with open(fyaml) as ff:
    gridfls = safe_load(ff)

  fconfig = 'config_nep.yaml'
  with open(fconfig) as ff:
    config = safe_load(ff)
  spear_dir = os.path.join(config['filesystem']['spear_month_ens'], 'monthly_clim')
  spear_topo_dir = config['filesystem']['spear_topo_grid']
  ftopo_spear = os.path.join(spear_topo_dir,'ocean_z.static.nc')
  spear_topo = xarray.open_dataset(ftopo_spear)
  lonW = config['domain']['west_lon']
  lonE = config['domain']['east_lon']
  LONh = xarray.open_dataset(ftopo_spear).variables['geolon'].data

  if lonW > np.max(LONh):
    lonW = lonW-360.
  elif lonW < np.min(LONh):
    lonW = lonW+360.

  if lonE > np.max(LONh):
    lonE = lonE-360.
  elif lonE < np.min(LONh):
    lonE = lonE+360.

  lon_slice = slice(lonW, lonE)
  lat_slice = slice(config['domain']['south_lat'], config['domain']['north_lat'])
  ds_grid_spear  = xarray.open_dataset(ftopo_spear).sel(xh=lon_slice, xq=lon_slice, \
                   yh=lat_slice, yq=lat_slice)
  HHS = ds_grid_spear.variables['depth_ocean'].data
  HHS = -abs(HHS)
  HHS = np.where(np.isnan(HHS), 0.5, HHS)
  LONs = ds_grid_spear.variables['geolon'].data
  LATs = ds_grid_spear.variables['geolat'].data
 
  return HHS, LONs, LATs

def rotate_box(XV, YV, x0c, y0c):
  """
    For tilted, rotated quadrilaterals mapping does not work well
    Try to rotate the box to align it with X/Y axes
    XV, YV coordinates for 4 vertices
    x0c, y0c - coordinates of the grid point inside the box
    rotate around the 1st vertex
  """
  XR = XV.copy()
  YR = YV.copy()
#  x00c = x0c
#  y00c = y0c
  x0c = x0c-XV[0]
  y0c = y0c-YV[0]  
  XV = XV-XV[0]
  YV = YV-YV[0]

  x1, x2, x3, x4 = XV
  y1, y2, y3, y4 = YV
  Av  = mmisc1.construct_vector([x1,y1],[x2, y2])
  Bv  = mmisc1.construct_vector([x1,y1],[x1+1.,y1]) # horizontal axis
  prA, cosT, tht = mmisc1.vector_projection(Av, Bv) 
  if Av[1,0] < 0.: tht = -tht

  tht_rad = tht*np.pi/180.
  RR = np.array([[np.cos(tht_rad), np.sin(tht_rad)],[np.sin(-tht_rad), np.cos(tht_rad)]])

  for ii in range(4):
    U  = np.array([[XV[ii]],[YV[ii]]])
    Ur = np.matmul(RR,U).squeeze()
    XR[ii] = Ur[0]
    YR[ii] = Ur[1]

  U = np.array([[x0c],[y0c]])
  Ur = np.matmul(RR,U).squeeze()
  x0r = Ur[0]
  y0r = Ur[1]

  return XR, YR, x0r, y0r 

def check_repeated_vertices(xx,yy):
  """
    Fix repeated vertices with identical x,y coordinates
  """
  DD = np.zeros((4))
  for ii in range(4):
    x1 = xx[ii]
    y1 = yy[ii]
    if ii == 3:
      x2 = xx[0]
      y2 = yy[0]
    else:
      x2 = xx[ii+1]
      y2 = yy[ii+1]
    DD[ii]   = mmisc1.dist_sphcrd(y1, x1, y2, x2)
 
  f_repeat = False
  if np.min(DD) < 1.e-3: f_repeat = True
   
  return f_repeat
  
def subset_spear_coord(grid_point):
  """
    Subset SPEAR geogr coord to NEP domain
    grid_point = 'h','u','v' for staggerd grid
  """
  fyaml = 'pypaths_gfdlpub.yaml'
  with open(fyaml) as ff:
    gridfls = safe_load(ff)

  fconfig = 'config_nep.yaml'
  with open(fconfig) as ff:
    config = safe_load(ff)
  spear_dir = os.path.join(config['filesystem']['spear_month_ens'], 'monthly_clim')
  spear_topo_dir = config['filesystem']['spear_topo_grid']
  ftopo_spear = os.path.join(spear_topo_dir,'ocean_z.static.nc')
  spear_topo = xarray.open_dataset(ftopo_spear)
  lonW = config['domain']['west_lon']
  lonE = config['domain']['east_lon']
  LONh = xarray.open_dataset(ftopo_spear).variables['geolon'].data

  if lonW > np.max(LONh):
    lonW = lonW-360.
  elif lonW < np.min(LONh):
    lonW = lonW+360.

  if lonE > np.max(LONh):
    lonE = lonE-360.
  elif lonE < np.min(LONh):
    lonE = lonE+360.

  lon_slice = slice(lonW, lonE)
  lat_slice = slice(config['domain']['south_lat'], config['domain']['north_lat'])
  ds_grid_spear  = xarray.open_dataset(ftopo_spear).sel(xh=lon_slice, xq=lon_slice, \
                   yh=lat_slice, yq=lat_slice)
  if grid_point == 'h':
    LONs = ds_grid_spear.variables['geolon'].data
    LATs = ds_grid_spear.variables['geolat'].data
  elif grid_point == 'u':
    LONs = ds_grid_spear.variables['geolon_u'].data
    LATs = ds_grid_spear.variables['geolat_u'].data
  elif grid_point == 'v':
    LONs = ds_grid_spear.variables['geolon_v'].data
    LATs = ds_grid_spear.variables['geolat_v'].data
 
  return LONs, LATs

def interp_OBsegm_mnth2daily(dset, ndays, nsgm, varnm, npol=3, calendar='julian', dnmb0=0):
  """
    Interpolate in time OB segment data from monthly --> daily
    npol - degree of the interpolating polynomial where enough data points
    assumed time of the monthly fields = day 15 of each month
    time before the 1st data pnt and after the last data point = keep constant 
    calendar: julian - just day 3 since the 1st day of the record
              gregorian - np datetime, actual dates
  """
  import mod_interp1D as minterp
  import mod_time as mtime

  print(f'Time interpolation: {varnm} OB={nsgm} ndays={ndays}')

  Nnodes = npol+1
  Nlft = int(np.floor(Nnodes/2))
  Nrht = int(Nnodes - Nlft)
  time_mo    = dset.time.data   # year days for monthly data - mid of the months
  time_steps = [x for x in range(1,ndays+1)] #Interpolation time points - 
                                         # days since day 0 of the forecast 
                                         # Day 1 = start of the f/cast
  huge = 1.e20   # nan values
  assert npol < 5, "Polynomials of a degree > 4 are not supported"

#  time_day = np.array([x for x in range(1,ndays+1)])
  fldnm = f'{varnm}_segment_{nsgm:03d}'  
  xdim  = f'nx_segment_{nsgm:03d}'
  ydim  = f'ny_segment_{nsgm:03d}'
  zdim  = f'nz_segment_{nsgm:03d}'
  nx    = len(dset[xdim])
  ny    = len(dset[ydim])
  nz    = len(dset[zdim])
  npnts = max([nx,ny])
  npmin = min([nx,ny])
# Check these are 2D sections:
  assert npmin == 1, f"Expected 2D OB sections: check nx={nx} ny={ny}"
  if ny == 1:
    Lx = True
  else:
    Lx = False

#  Ai = np.zeros((ndays,nz,ny,nx))*np.nan
  Ai = np.zeros((ndays,nz,npnts))*np.nan 
  for ii in range(ndays):
    jday0 = ii+1
    if jday0 <= time_mo[0]:
      if Lx: 
#        Ai[ii,:] = dset[fldnm].isel(time=0, f'{ydim}'=0).data
        Ai[ii,:] = dset[fldnm].data[0,:,0,:]
      else:
        Ai[ii,:] = dset[fldnm].data[0,:,:,0]
    elif jday0 >= time_mo[-1]:
      if Lx: 
        Ai[ii,:] = dset[fldnm].data[-1,:,0,:]
      else:
        Ai[ii,:] = dset[fldnm].data[-1,:,:,0]
    else:
# Do interpolation
      kmin = np.where(time_mo <= jday0)[0] 
      kmax = np.where(time_mo > jday0)[0]
      Indx = np.zeros((Nnodes))
      if len(kmin) >= Nlft and len(kmax) >= Nrht:
        Indx = np.array([x for x in range(kmin[-Nlft],kmin[-Nlft]+Nnodes)])
      elif len(kmin) < Nlft:
        Indx = np.array([x for x in range(kmin[0], kmin[0]+Nnodes)])
      elif len(kmax) < Nrht:
        Indx = np.array([x for x in reversed(range(kmax[-1], kmax[-1]-Nnodes, -1))])
      else:
        raise Exception(f'polyn order {npol}, Nlft={Nlft}, Nrht={Nrht}, Check kmin kmax')

      A2d = dset[fldnm].data[Indx[0],:,:,:].squeeze()
      A2d = np.where(A2d > 0.1*huge, np.nan, A2d)
      J,I = np.where(~np.isnan(A2d))
      nj = len(J)
      Y = np.zeros((nj,Nnodes))    # node values at Nnodes time points
      X = np.zeros((nj,Nnodes))    # time points
      for inode in range(Nnodes):
        X[:,inode] = X[:,inode] + time_mo[Indx[inode]] # time nodes
        Y[:,inode] = dset[fldnm].data[Indx[inode],:,:,:].squeeze()[J,I]

      Yi = minterp.lagr_polynom2D(X, Y, jday0, axis_interp=1)
      A2d[J,I] = Yi
      Ai[ii,:] = A2d.copy()

  dz2d  = dset[f"dz_{varnm}_segment_{nsgm:03d}"].isel(time=0).data.squeeze()
  if Lx: 
    Ai = np.expand_dims(Ai, axis=2)
    dz2d = np.expand_dims(dz2d, axis=1)
  else:
    Ai = np.expand_dims(Ai, axis=3)
    dz2d = np.expand_dims(dz2d, axis=2)
  dz3d = np.tile(dz2d,(ndays,1,1,1))

  print(f'Time interpolation end: min/max {np.nanmin(Ai):8.3f}/{np.nanmax(Ai):8.3f}')

# Replace nans:
  Ai = np.where(np.isnan(Ai), huge, Ai)

# Construct data set of time-interpolated OB sections:
  stime = time_steps
  if calendar.lower() == 'gregorian':
    yrS, jdayS = mtime.dnmb2jday(dnmb0)
    dnmbE = dnmb0 + len(time_steps)-1
    yrE, jdayE = mtime.dnmb2jday(dnmbE)
    
    stime = mtime.npdatetime_year(yrS, yrE=yrE, day_start=jdayS, day_end=jdayE)

  nxsgm = [x for x in range(0,nx)]
  nysgm = [x for x in range(0,ny)]
  nzsgm = [x for x in range(0,nz)]
  sgmnm = f"segment_{nsgm:03d}"
  dim1  = "time"
  dim2  = f"nz_{sgmnm}"
  dim3  = f"ny_{sgmnm}"
  dim4  = f"nx_{sgmnm}"
  darr_var = xarray.DataArray(Ai, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
  darr_dz  = xarray.DataArray(dz3d, dims=(dim1, dim2, dim3, dim4),
             coords={dim1: stime, dim2: nzsgm, dim3: nysgm, dim4: nxsgm})
#  f"{varnm}_segment_{nsgm:03d}"
  dset_segmTI = xarray.Dataset({f"{varnm}_segment_{nsgm:03d}": darr_var, \
           f"dz_{varnm}_segment_{nsgm:03d}": darr_dz})

  return dset_segmTI

def monthlyOB_from_SPEAR_ensmbls(dnmb_start, varnm, run_name, ENSR ):
  """
    Derive OB from monthly SPEAR output of 1 yr f/casts for specified ensembles
    and interpolate onto 
    regional supergrid
    One variable only - varnm
    Note SPEAR monthly fields should be subset to a smaller regions
    gmapi indices should correspond to the smaller subdomain
    Assumed domain = NEP with 4 OBs, need to change for a different domain / OBs
  """
  from boundary import Segment
  import mod_time as mtime
  import mod_mom6 as mmom6

  dv_start   = mtime.datevec(dnmb_start)

  FMONTHS = [x for x in range(1,13)]
  nmonths = len(FMONTHS)
  TMM  = np.zeros((nmonths,4), dtype=int)  # f/cast time: year, month, Ndays in month
  #dnmb = dnmb_start-15
  dnmb = dnmb_start
  nmdays = 0
  for imo in range(nmonths):
    dnmb = dnmb + nmdays
    dv = mtime.datevec(dnmb)
    nmdays = mtime.month_days(dv[1],dv[0])
    dnmb0 = int(mtime.datenum([dv[0],dv[1],15]))
    TMM[imo,:2] = dv[:2]
    TMM[imo,2] = int(nmdays)
    TMM[imo,3] = dnmb0
  ndays = np.sum(TMM[:,2])

  fyaml = 'pypaths_gfdlpub.yaml'
  with open(fyaml) as ff:
    gridfls = safe_load(ff)

  # MOM6 NEP topo/grid:
  pthtopo    = gridfls['MOM6_NEP'][run_name]['pthgrid']
  fgrid      = gridfls['MOM6_NEP'][run_name]['fgrid']
  ftopo_mom  = gridfls["MOM6_NEP"][run_name]["ftopo"]
  outdir     = gridfls['MOM6_NEP'][run_name]['pthoutp']
  hgrid      = xarray.open_dataset(os.path.join(pthtopo,fgrid))
  hmask      = xarray.open_dataset(os.path.join(pthtopo, 'ocean_mask.nc'))
  dstopo_nep = xarray.open_dataset(os.path.join(pthtopo, ftopo_mom))
  dfgrid_mom  = os.path.join(pthtopo, fgrid)

  segments = [ Segment(1, 'north', hgrid, output_dir=outdir),
               Segment(2, 'east',  hgrid, output_dir=outdir),
               Segment(3, 'south', hgrid, output_dir=outdir),
               Segment(4, 'west',  hgrid, output_dir=outdir)]

  nOB = len(segments)
  # Time array for monthly clim:
  time_days = np.zeros((12))
  for imo in range(1,13):
    dnmb = TMM[imo-1,3]
    time_days[imo-1] = dnmb - dnmb_start + 1

  #static = xarray.open_dataset('/work/acr/spear/analysis/ocean_z.static.nc')
  grid_spear = xarray.open_dataset('/work/Dmitry.Dukhovskoy/data/SPEAR/ocean_z.static.nc')
  icegrid_spear = xarray.open_dataset('/work/Dmitry.Dukhovskoy/data/SPEAR/ice.static.nc')

  fconfig = 'config_nep.yaml'
  with open(fconfig) as ff:
    config = safe_load(ff)

  # Check if mapping indices exist, gmapi:
  dirgmapi = config['filesystem']['spear_mom_gmapi']
  flgmaph  = f'spear2mom_NEP_OB_gmapi_hpnt.nc'
  flgmapu  = f'spear2mom_NEP_OB_gmapi_upnt.nc'
  flgmapv  = f'spear2mom_NEP_OB_gmapi_vpnt.nc'
  dflgmaph = os.path.join(dirgmapi, flgmaph)
  dflgmapu = os.path.join(dirgmapi, flgmapu)
  dflgmapv = os.path.join(dirgmapi, flgmapv)
  # h-point indices
  dsh = xarray.open_dataset(dflgmaph)
  # u-point indices
  dsu = xarray.open_dataset(dflgmapu)
  # v-point indices
  dsv = xarray.open_dataset(dflgmapv)

  nens = len(ENSR)

  icc = 0
  dsetOB = xarray.Dataset()
  for isgm in range(nOB):
    nsgm = isgm+1
    print(f'Processing lon/lat OB segment={nsgm}')
    dset   = derive_obsegm_lonlat(hgrid, segments, isgm)
    dsetOB = xarray.merge([dsetOB, dset])

  for iens in range(nens):
    ens = ENSR[iens]
    spear_dir = config['filesystem']['nep_spear_subset'].\
                     format(year=dv_start[0], ens=ens)

    if varnm == 'thetao' or varnm == 'so':
      # Load monthly SPEAR data subset for NEP
      flnm_spear = f'NEP_spear_{dv_start[0]}{dv_start[1]:02d}.{varnm}.nc'
      ds = read_spear_output(spear_dir, varnm, flnm_spear, fzint=True)

      for isgm in range(nOB):
        nsgm   = isgm+1
        print(f'Processing {varnm} OB segment={nsgm} ens={ens:02d}')
        INDX   = dsh[f'indx_segm{nsgm:03d}'].data
        JNDX   = dsh[f'jndx_segm{nsgm:03d}'].data
        # Interpolate onto NEP OB supergrid:
        dset   = derive_obsegm_3D(hgrid, ds, segments, isgm, varnm,
                                  INDX, JNDX, time_steps=time_days)
    #    dset   = xarray.Dataset({f"{varnm}_segment_{nsgm:03d}": darr})
        dsetOB = xarray.merge([dsetOB, dset])
        fldnm_old = f'{varnm}_segment_{nsgm:03d}'
        fldnm_new = f'{varnm}_e{ens:02d}_segment_{nsgm:03d}'
        dsetOB = dsetOB.rename({fldnm_old: fldnm_new})

  # UV fields
    elif varnm == 'u' or varnm == 'v':
      # Derive rotation angle, rad and components of the rotation matrix
      # for rotating vectors onto true N/E grid from SPEAR
      r2d = 180./np.pi
      theta_rot, cosrot, sinrot = get_rotangle(icegrid_spear, fconfig, grid_spear)
      print(f"Rotation angle for SPEAR min/max: {np.min(theta_rot)*r2d:6.2f} " + \
            f"/ {np.max(theta_rot)*r2d:6.2f}")

      # Load monthly SPEAR data subset for NEP
      flnmu_spear = f'NEP_spear_{dv_start[0]}{dv_start[1]:02d}.uo.nc'
      flnmv_spear = f'NEP_spear_{dv_start[0]}{dv_start[1]:02d}.vo.nc'
      ds_uo = read_spear_output(spear_dir, 'uo', flnmu_spear, fzint=True)
      ds_vo = read_spear_output(spear_dir, 'vo', flnmv_spear, fzint=True)

      for isgm in range(nOB):
        nsgm   = isgm+1
        print(f'Processing {varnm} OB segment={nsgm}')

        if varnm == 'u':
          INDX  = dsu[f'indx_segm{nsgm:03d}'].data
          JNDX  = dsu[f'jndx_segm{nsgm:03d}'].data
        elif varnm == 'v':
          INDX  = dsv[f'indx_segm{nsgm:03d}'].data
          JNDX  = dsv[f'jndx_segm{nsgm:03d}'].data

        dset   = derive_obsegm_uv(hgrid, ds_uo, ds_vo, segments, isgm, theta_rot,\
                                        varnm, INDX, JNDX, time_steps=time_days)
        dsetOB = xarray.merge([dsetOB, dset])
        fldnm_old = f'{varnm}_segment_{nsgm:03d}'
        fldnm_new = f'{varnm}_e{ens:02d}_segment_{nsgm:03d}'
        dsetOB = dsetOB.rename({fldnm_old: fldnm_new})

    for segm in [1,2,3,4]:
  #    vv  = f"{varnm}_segment_{segm:03d}"
      vdz = f"dz_{varnm}_segment_{segm:03d}"
      dm1 = f'lat_segment_{segm:03d}'
      dm2 = f'lon_segment_{segm:03d}'
      dsetOB[fldnm_new].attrs["coordinates"] = f"{dm1} {dm2}"
      dsetOB[vdz].attrs["coordinates"] = f"{dm1} {dm2}"

  return dsetOB
    

def minmax_clrmap(dmm, pmin=10, pmax=90, cpnt=0.01, fsym=False):
  """
  Find min/max limits for colormap 
  discarding pmin and 1-pmax min/max values
  cpnt - decimals to leave, if cpnt=0 - round to the nearest integer
  """
  dmm = dmm[~np.isnan(dmm)]
  a1  = np.percentile(dmm,pmin)
  a2  = np.percentile(dmm,pmax)
  if cpnt == 0:
   rmin = np.floor(a1)
   rmax = np.ceil(a2)
  else:
    cff = 1./cpnt
    rmin = cpnt*(int(a1*cff))
    rmax = cpnt*(int(a2*cff))

  if fsym and (rmin<0. and rmax>0.) :
    dmm = max([abs(rmin),abs(rmax)])
    rmin = -dmm
    rmax = dmm

  return rmin,rmax

def plot_xsection(A2d, X, Z, Hbtm, Xbtm, clrmp, rmin, rmax, \
                  xl1, xl2, sttl='OB section', stxt='', fgnmb=1, \
                  btx='', fclrbar=True, ax1=[], fig1=[]):
  """
    2D vertical section
  """
  from matplotlib.patches import Polygon

  yl1 = np.ceil(np.min(Hbtm))

  if isinstance(ax1, list) and len(ax1) == 0:
    plt.ion()
    fig1 = plt.figure(fgnmb,figsize=(9,8))
    plt.clf()
    ax1 = plt.axes([0.1, 0.25, 0.8, 0.7])

  im1 = ax1.pcolormesh(X, Z, A2d, \
                 cmap=clrmp,\
                 vmin=rmin, \
                 vmax=rmax)

  # Patch bottom:
  verts = [(np.min(Xbtm),-8000),*zip(Xbtm,Hbtm),(np.max(Xbtm),-8000)]
  poly = Polygon(verts, facecolor='0.6', edgecolor='0.6')
  ax1.add_patch(poly)

  ax1.set_xlim([xl1, xl2])
  ax1.set_ylim([yl1, 0])

  if fclrbar:
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

  if len(btx) > 0:  bottom_text(btx)

  return 

def plot_Nxsections(A2d, X, Z, Hbtm, Xbtm, clrmp, rmin, rmax, \
                  xl1, xl2, fig1, ax1, sttl='OB section', stxt='', fgnmb=1, \
                  btx='', clrb_pos=[], clrb_ornt='horizontal', \
                  txt_pos=[0.1, 0.2, 0.8, 0.1]):
  """
    2D vertical sections - plot each section in separate axes on same figure pane
    plot 1 colorbar specified position
  """
  from matplotlib.patches import Polygon

  yl1 = np.ceil(np.min(Hbtm))

  if isinstance(ax1, list) and len(ax1) == 0:
    plt.ion()
    fig1 = plt.figure(fgnmb,figsize=(9,8))
    plt.clf()
    ax1 = plt.axes([0.1, 0.25, 0.8, 0.7])

  im1 = ax1.pcolormesh(X, Z, A2d, \
                 cmap=clrmp,\
                 vmin=rmin, \
                 vmax=rmax)

  # Patch bottom:
  verts = [(np.min(Xbtm),-8000),*zip(Xbtm,Hbtm),(np.max(Xbtm),-8000)]
  poly = Polygon(verts, facecolor='0.6', edgecolor='0.6')
  ax1.add_patch(poly)

  ax1.set_xlim([xl1, xl2])
  ax1.set_ylim([yl1, 0])

  if len(clrb_pos) > 0:
    ax2 = fig1.add_axes(clrb_pos)
    if clrb_ornt == 'horiz':
      clb = plt.colorbar(im1, cax=ax2, orientation='horizontal', extend='both')
      ax2.xaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
      ax2.set_xticklabels(ax2.get_xticks())
      ticklabs = clb.ax.get_xticklabels()
      clb.ax.set_xticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
    else:
      clb = plt.colorbar(im1, cax=ax2, extend='both')
      ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
      ax2.set_yticklabels(ax2.get_yticks())
      ticklabs = clb.ax.get_yticklabels()
      clb.ax.set_yticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
    clb.ax.tick_params(direction='in', length=12)
    plt.sca(ax1)
  
  ax1.set_title(sttl)

  if len(stxt) > 0:
    ax3 = plt.axes(txt_pos)
    ax3.text(0, 0.01, stxt)
    ax3.axis('off')

  if len(btx) > 0:  bottom_text(btx, pos=[0.05, 0.03])

  return 


def plot_OBpoints(xOB, yOB, INDX, JNDX, LONs, LATs, lon_topo, lat_topo, \
                  hlon, hlat, HHM, HHS, fgnmb=1):
  """
    Plot OB segments NEP and SPEAR grid points for interp
  """
  xOB = np.where(xOB<0., xOB+360., xOB)
  LONs = np.where(LONs<0., LONs+360., LONs)

  plt.ion()
  fig1 = plt.figure(fgnmb,figsize=(9,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.25, 0.8, 0.7])

  ax1.plot(xOB, yOB, '.-')
  hcntr = [x for x in range(-5000,-10, 500)]
  ax1.contour(hlon, hlat, HHM,[0], colors=[(0,0,0)])
  ax1.contour(hlon, hlat, HHM, hcntr, linestyles='solid', colors=[(0.7,0.7,0.7)])
  ax1.contour(lon_topo, lat_topo, HHS, [0], colors=[(1,0,0)])

  npp, _ = INDX.shape
  for ii in range(npp):
    I = INDX[ii,:]
    J = JNDX[ii,:]
# Discard same grid points:
    D = 0.1
    if ii > 0:
      D = np.sqrt(np.sum((INDX[ii,:]-INDX[ii-1,:])**2) + np.sum((JNDX[ii,:]-JNDX[ii-1,:])**2))
    if D < 1.e-20: continue

    I = np.append(I, I[0])
    J = np.append(J, J[0])
    ax1.plot(LONs[J,I], LATs[J,I], '.-')

  ax1.axis('scaled')

  ax1.set_title('OB segment NEP and SPEAR boudning boxes for interpolation\n' + \
                'Red: SPEAR coastline, Black: NEP coastline')
  btx = 'mod_utils_ob.py:plot_OBpoints'
  bottom_text(btx)
  
  return

def OBsegm_clrmp_rminrmax(segm_nm, varnm, Xbtm=[]):
  """
    Select colormap, min max limits for plotting NEP OB sections
  """
  import mod_utils as mutil
  if varnm == 'so':
    clrmp = mutil.colormap_salin(clr_ramp=[1,0.85,1])
    clrmp.set_bad(color=[0.2, 0.2, 0.2])
  elif varnm == 'thetao':
    clrmp = mutil.colormap_temp(clr_ramp=[0.9,0.8,1])
    clrmp.set_bad(color=[1,1,1])

  if segm_nm == 'north':
    xl1 = 1900.
    if varnm == 'thetao':
      rmin = -2.
      rmax = 1.
    if varnm == 'so':
      rmin = 29.
      rmax = 35.
  elif segm_nm == 'east':
    xl1 = 5800
    if varnm == 'thetao':
      rmin = -2.
      rmax = 2.
    if varnm == 'so':
      rmin = 29.
      rmax = 35.
  elif segm_nm == 'south':
    xl1 = 0
    if varnm == 'so':
      rmin = 34.0
      rmax = 34.9
    else:
      rmin = 0.
      rmax = 28.
  elif segm_nm == 'west':
    xl1 = 0
    if varnm == 'so':
      rmin = 34.0
      rmax = 34.9
    else:
      rmin = 0.
      rmax = 25.0
  else:
    xl1 = 0

  if len(Xbtm) > 0:
    xl2  = max(Xbtm)
  else:
    xl2 = 1.e9

  return clrmp, rmin, rmax, xl1, xl2 

def find_NEPoutput_day(pthoutp, dnmbR, ndav, fld='oceanm'):
  """
    In the output dir within saved files find output date closest to
    desired date
    file naming assumed: fld_YYYY_DAY.nc
  """
  import os
  import mod_time as mtime

# Find 1st ouptut date:
  LF = os.listdir(pthoutp)
  YR0 = np.zeros((len(LF)))
  JD0 = np.zeros((len(LF)))
  icc = -1
  for fls in LF:
    icc += 1
    bsname = fls.split(".")[0]
    fld_name =  bsname.split("_")[0]
    if not fld_name == fld: continue
    yrf    = int(bsname.split("_")[1])
    jday   = int(bsname.split("_")[2])
    YR0[icc] = yrf
    JD0[icc] = jday
  
  D     = np.sqrt(YR0**2 + JD0**2)   
  ii    = np.argmin(D) 
  yr1   = YR0[ii]
  jday1 = JD0[ii]
  dnmb1 = mtime.jday2dnmb(yr1,jday1)
  DOUTP = np.arange(dnmb1, dnmb1+367, ndav)
  D     = np.abs(DOUTP - dnmbR)
  ii    = np.argmin(D)
  dnmb_out  = DOUTP[ii]
  DVout = mtime.datevec(dnmb_out)
  yrout, mout, dout = DVout[:3]
  _, jdayout = mtime.dnmb2jday(dnmb_out)

  return int(yrout), int(mout), int(dout), int(jdayout)

