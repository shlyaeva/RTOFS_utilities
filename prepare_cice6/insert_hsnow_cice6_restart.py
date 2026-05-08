"""
  Modify snow depth on sea ice in the CICE6 restart file
  by direct insertion of snow depth fields

  Assumed that all non-nan non-zero values are inserted 
  to the grid values where aice > 0

  Snow is distributed across the thikn. categories proportional 
  to the aice (ice partial area)

  Here, snow depth climatology (1998-2007) from NASA SSM/I gridded fields
  are used

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import matplotlib
import xarray
from copy import copy
import matplotlib.colors as colors
from yaml import safe_load
from mpl_toolkits.basemap import Basemap, cm
import argparse
from pathlib import Path

PPTHN = os.environ.get("PPTHN")
if not PPTHN:
  # Repository-local fallback (works independent of current working directory)
  # .../RTOFS_utilities/prepare_cice6/<script>.py -> .../RTOFS_utilities
  PPTHN = str(Path(__file__).resolve().parents[1])

sys.path.extend([
    os.path.join(PPTHN, 'MyPython', 'hycom_utils'),
    os.path.join(PPTHN, 'MyPython', 'draw_map'),
    os.path.join(PPTHN, 'MyPython'),
    os.path.join(PPTHN, 'MyPython', 'mom6_utils')
])

from mod_utils_fig import bottom_text
import mod_time as mtime
import mod_swstate as msws
import mod_cice6_utils as mc6util
importlib.reload(mc6util)

rest_date = 20250103
rest_hr   = 0
hunits    = 'cm'
regn = 'south'

yrR = mmR = ddR = hrR = None
yrN = mmN = ddN = hrN = None

parser = argparse.ArgumentParser()
parser.add_argument("--rdate", help=f"restart date input file, default={rest_date}", type=int)
parser.add_argument("--rhr", help=f"input file, restart hour = 0, ..., 23, default={rest_hr}", type=int)
parser.add_argument("--rdate_out", help="output file, restart date if different from input", type=int)
parser.add_argument("--rhr_out", help="output file, restart hour if date is different from input", type=int)
parser.add_argument("--flrst_in", help="rest file in, otherwise name constructed from rest_date", type=str)
parser.add_argument("--flrst_out", help="new rest file, otherwise name constructed from rdate_out", type=str)
parser.add_argument("--regn", help=f"where icon incerted: south, north, global, default={regn}", type=str)
parser.add_argument("--pth_in", help="input restart directory, default keeps script behavior", type=str)
parser.add_argument("--pth_out", help="output restart directory, default uses input restart directory", type=str)
parser.add_argument("--hsnow_file", help="optional custom interpolated snow depth file", type=str)
parser.add_argument("--hsnow_var", help="variable name in --hsnow_file (default: snow_depth)",
                    type=str, default="snow_depth")
args = parser.parse_args()

flrst_in  = args.flrst_in if args.flrst_in else None
flrst_out = args.flrst_out if args.flrst_out else None
pth_in = args.pth_in if args.pth_in else None
pth_out = args.pth_out if args.pth_out else None
hsnow_file = args.hsnow_file if args.hsnow_file else None
hsnow_var = args.hsnow_var if args.hsnow_var else "snow_depth"
regn = args.regn if args.regn else regn

# if rest_date and rest_date_out are provided
# Derive dates assuming file nameing is cice_restart.res.YYYYMMDD.XX[XXX]
if flrst_in is not None:
  try:
    yrR, mmR, ddR, hrR, mintR = mc6util.get_date_filename(flrst_in)
    rest_date = int(yrR*1e4 + mmR*100 + ddR)
    rest_hr = hrR
  except ValueError:
    if args.rdate is None:
      raise ValueError(
        f"Could not parse date from --flrst_in='{flrst_in}'. "
        "Provide --rdate (and optionally --rhr) or use a filename containing YYYYMMDD."
      )
    rest_date = args.rdate
    rest_hr = args.rhr if args.rhr is not None else rest_hr
else:
  rest_date = args.rdate if args.rdate else rest_date
  rest_hr   = args.rhr if args.rhr else rest_hr

if flrst_out is not None:
  try:
    yrN, mmN, ddN, hrN, mintN = mc6util.get_date_filename(flrst_out)
    rest_date_out = int(yrN*1e4 + mmN*100 + ddN)
    rest_hr_out = hrN
  except ValueError:
    if args.rdate_out is None and args.rdate is None:
      raise ValueError(
        f"Could not parse date from --flrst_out='{flrst_out}'. "
        "Provide --rdate_out/--rhr_out (or --rdate/--rhr) or use a filename containing YYYYMMDD."
      )
    rest_date_out = args.rdate_out if args.rdate_out is not None else rest_date
    if args.rhr_out is not None:
      rest_hr_out = args.rhr_out
    elif args.rhr is not None:
      rest_hr_out = args.rhr
    else:
      rest_hr_out = rest_hr
else:
  rest_date_out = args.rdate_out if args.rdate_out else rest_date
  rest_hr_out   = args.rhr_out if args.rhr_out else rest_hr
  
print(f"Restart date input:  {rest_date}:{rest_hr}")
print(f"Restart date output: {rest_date_out}:{rest_hr_out}")

change_rest_time = (rest_date != rest_date_out) or (rest_hr != rest_hr_out)

# Get date numbers:
# Input restart file
dnmbR = mtime.rdate2datenum(rest_date*100+rest_hr)  # restart day nmb
if yrR is None:
  yrR,mmR,ddR,hrR = mtime.datevec(dnmbR, round_hrs=True)[:4]
nsecR = hrR*3600

# Dates of the output fields in the new restart:
dnmbN = mtime.rdate2datenum(rest_date_out*100+rest_hr_out)
if yrN is None:
  yrN, mmN, ddN, hrN = mtime.datevec(dnmbN, round_hrs=True)[:4]
nsecN = hrN*3600
 
syst_info = os.uname()
machine = syst_info.nodename
    
if 'dtn' in machine:
  print("Running on DTN node:", machine)
  node_nm = "dtn"
elif 'gaea' in machine:
  print("Running on Gaea compute node:", machine)
  node_nm = "gaea"
elif 'an' in machine:
  print("Running on PPAN node:", machine)  
  node_nm = "ppan"
else:
  print("Unknown machine:", machine)

fyaml = 'paths_ufs.yaml'
with open(fyaml) as ff:
  pths_ufs = safe_load(ff)


def extract_suffix(fname):
  parts = fname.split('.')
  # must end with .nc, so suffix is the part before that
  if len(parts) > 2 and parts[-1] == 'nc':
    suffix = parts[-2]
    if not suffix.isdigit():
      return suffix
  return None

if pth_in is None:
  pthrest = '/gpfs/f6/sfs-emc/proj-shared/Dmitry.Dukhovskoy/RUNDIRS/restart_sfs_C192mx025/ice/Neil_IC/GFS/mem008'
else:
  pthrest = pth_in
pthrest_out = pth_out if pth_out is not None else pthrest

pthdata = pths_ufs[node_nm]["MOM6"]["pthdata"]

# CICE parameters:
puny      = 1.e-11
c0        = 0.0
c1        = 1.0
c2        = 2.0
p5        = 0.5
Lsub      = 2.835e6    # latent heat sublimation fw (J/kg)
Lvap      = 2.501e6    # latent heat vaporization fw (J/kg)
Lfresh    = Lsub - Lvap # latent heat of melting of fresh ice (J/kg)
cp_ice    = 2106.       # specific heat of fresh ice (J/ kg/K)
rhos      = 330.        # density of snow (kg/m3)
hs_min    = 1.e-4       # min snow thickness for computing Tsno (m)
nsal      = 0.407
msal      = 0.573
min_salin = 0.1      # threshold for brine pocket treatment
saltmax   = 3.2        # max S at ice base
hg        = 1.e20    # bad values, land mask, etc.
nslyr     = 1   # snow layers
Tmin      = -100.   # minimum snow T
# Check ice_in what ITD is used
# 0.00, 0.64, 1.39, 2.47, 4.57
hicat = np.array([0., 0.64, 1.39, 2.47, 4.57, 50.])

# in CICE6, snow surf max T = 0C
# Set surface snow T < 0 to prevent rapid snow melt during the first time steps
# This mainly applies for summer months
# Tsfc = Tsnow in the 1 layer --> change qsnon(1)
Tsfc_max = -1.0   
rho_ice  = 917. 
hsnow_max = 500.   # to avoid very thick hsnow / ice_area which will cause picard iteration crush
#rho_ocean = msws.sw_dens0(32.,-1.8)  # take lower S to guarantee snow-ice interf above sea level
rho_ocean = 1025.

# Snow depth climatology, Interpolated fields mesh025:
if hsnow_file is None:
  if regn == 'global':
    raise ValueError("For regn='global', provide --hsnow_file (and --hsnow_var if needed)")
  pthsnow = os.path.join(pthdata,'snow_nasa','monthly_clim')
  flhsn = 'SSMI_hsnow_mnthclim_1998_2007_mesh025_1440x1080_south.nc'
  dflhsn = os.path.join(pthsnow,flhsn)
else:
  dflhsn = hsnow_file
print(f"Reading interpolated hsnow {dflhsn}")
with xarray.open_dataset(dflhsn) as ds_snow:
  if hsnow_var not in ds_snow:
    raise KeyError(f"Variable '{hsnow_var}' not found in {dflhsn}")
  HSi = ds_snow[hsnow_var].isel(time=mmN-1).data.squeeze()
  LON = ds_snow['lon'].data
  LAT = ds_snow['lat'].data
  units = ds_snow[hsnow_var].attrs.get('units', None)
  if units is not None:
    print(f"'snow_depth' units: {units}")
    hunits = units
  else:
    print("No 'units' attribute found for 'snow_depth', use default: {hunits}")

units_m = hunits == 'm'
#print(f"snow depth units = {hunits}")
#print(f"units_m={units_m}")
#A = STOP

# Restart from a GFS17 rt13  run:
#flrst_in = f"cice_model.res.{yrR}{mmR:02d}{ddR:02d}.{nsecR:06d}.nc"
# Restart with inserted iconc from NSIDC NRT:
if flrst_in is None:
  flrst_in = f"cice_model.res.{yrR}{mmR:02d}{ddR:02d}.{hrR:02d}.iconc.nc"

dflrst_in = os.path.join(pthrest, flrst_in)
print(f"Reading restart: {dflrst_in}")
ds_in = xarray.open_dataset(dflrst_in)
ds_out = ds_in.copy(deep=True)

# Checks:
assert nslyr == 1, f"Code needs to be modified for nslyr>1, nslyr={nslyr}"
assert Tsfc_max <= 0., f"Tsfc_max={Tsfc_max} has to be <=0"

# Input values:
aicen  = ds_in['aicen'].data  # partial area by cats
vsnon  = ds_in['vsnon'].data  # snow vol per m2 of ice area
qsnon  = ds_in['qsno001'].data  # snow enthalpy by cats for 1 snow layer
vicen  = ds_in['vicen'].data   # ice vol per unit area of grid cell m3/m2
tsfcn  = ds_in['Tsfcn'].data   # snow/ice surface T
qicen1 = ds_in['qice001'].data # ice enthalpy, lr 1 surface
sicen1 = ds_in['sice001'].data # ice S, layer 1
apndn  = ds_in['apnd'].data    # the fraction of the pond of ice area, for each cat
hpndn  = ds_in['hpnd'].data    # depth of the ponds in a cell, by cats
ncat, jdim, idim = vsnon.shape
ds_in.close()

# Aggregated ice partial area:
aice = np.sum(aicen, axis=0).squeeze()

# Select points to insert:
Jins, Iins = np.where((HSi > puny) & (~np.isnan(HSi)) & (aice > puny))
Xins = LON[Jins,Iins]
Yins = LAT[Jins,Iins]
npnts = len(Jins)

print(f"Found {npnts} points for insertion, min/max lat={np.min(Yins):.1f}/{np.max(Yins):.1f}"
       f" lon={np.min(Xins):.1f}/{np.max(Xins):.1f}")

# Note qsnon, qice < 0 !
aicen_new  = aicen.astype(ds_out['aicen'].dtype).copy()
vsnon_new  = vsnon.astype(ds_out['vsnon'].dtype).copy()
qsnon_new  = qsnon.astype(ds_out['qsno001'].dtype).copy()
qicen1_new = qicen1.astype(ds_out['qice001'].dtype).copy()
apndn_new  = apndn.astype(ds_out['apnd'].dtype).copy()
hpndn_new  = hpndn.astype(ds_out['hpnd'].dtype).copy()
tsfcn_new  = tsfcn.astype(ds_out['Tsfcn'].dtype).copy()
vicen_new  = vicen.astype(ds_out['vicen'].dtype).copy()

dvol_sum = 0.
print("Snow depth insertion ...")
for ipp in range(npnts):
  if ipp%10000 == 0:
    print(f"   {ipp/npnts*100.:.2f}% done ...")
  j0 = Jins[ipp]
  i0 = Iins[ipp]

  ai  = aice[j0,i0]           # aggregated ice partial area 
  ain = aicen[:,j0,i0]        # partial areas by cats
  vin = vicen[:,j0,i0]        # ice volume per unit grid cell area by cats
  vsn = vsnon[:,j0,i0]        # snow volume per unit grid-cell area m2
  apn = apndn[:,j0,i0]        # pond fractional area of ice (level) area
  hpn = hpndn[:,j0,i0]        # pond depth
  tsn = tsfcn[:,j0,i0]        # snow/ice surface T by cats

  # New ice snow thickness over sea ice:
  if units_m:
    hsn_new = HSi[j0,i0]        # m of snow over sea ice
  else:
    hsn_new = HSi[j0,i0]*0.01   # m of snow over sea ice 

  # This should not happen (target HSi assumed to be m3/m2_ice) just to make sure
  # there is no very thick snow --> possible crash in picard iteration for very thick snow or ice
  if hsn_new > hsnow_max:
    hsn_new = hsnow_max

  # ice conc should not change except for a few locaitons to adjust snow load across cats:
  ain_new = ain.copy()
  
  # Note hsn_new = sum(vsn) / aice for aice > 0, m3/m2_ice ==> mean snow thickn over ice 
  # sum(vsn) = vstot_new = HSi[j0,i0] * aice, obs. gridded data assume 100% iconc 
  # Distribute new snow depth evenly by cats in snow vol m3/m2:
  vstot_new = hsn_new * ai
  if hsn_new <= hs_min or ai < puny:
    vsn_new = np.zeros_like(ain)
  else:
    # Distribute across cats proportionally to iconc:
    wts = ain/ai
    vsn_new = vstot_new * wts

  # Update snow enthalpy: J/m3  
  # see icepack_therm_vertical.F90 in icepack
  #
  # snow enthalpy should be: qsn_min <= qsn <= qsn_max
  # In theory, qsn_max = -rhos*Lfresh (latent heat of metling at 0C)
  # Make it a little lower to keep snow from melting right away
  # In general, snow enth. = enth(Tsfcn) if Tsfcn <=0
  #hsn_new = vsn_new / ain
  qsn = qsnon[:,j0,i0]        # enthalpy, J/m3 < 0
  qsn_min = -rhos * Lfresh + (Tmin + 0.01) * cp_ice * rhos  # enth. of the coldest possible snow
  qsn_max = -rhos * Lfresh + Tsfc_max * cp_ice * rhos  # Tsfc_max <= 0
  qsn_tsf = -rhos * Lfresh + tsn * cp_ice * rhos  # enth. for surf temp
  qT0 = -Lfresh*rhos      # enth. of pure snow at 0C

  # Update enthalpy of snow and enforce physical constraints
  # Limit qsn to [qsn_min, qsn_max]
  qsn_new = np.clip(qsn, qsn_min, qsn_max)

  # It should not exceed enthalpy implied by surface temperature (qsn </= qsn_tsf)
  # This is true for 1 snow layer
  qsn_new = np.minimum(qsn_new, qsn_tsf)

  # No ice --> no snow enthalpy
  qsn_new = np.where(ain <= puny, 0., qsn_new)

  # Zero snow volume --> zero enthalpy
  qsn_new = np.where(vsn_new <= 0., 0., qsn_new)

  # Tsfcn should match snow enthalpy in layer 1
  # Update Tsfcn if needed:
  # for dry snow (T<=0):
  tsn_new = (qsn_new + rhos*Lfresh) / (cp_ice*rhos)
  tsn_new = np.where(abs(qsn_new) < puny, 0., tsn_new)
  for ik in range(ncat):
    if abs(qsn_new[ik]) < puny:
      continue
    assert tsn_new[ik] <= 0., f"check qsn_new={qsn_new[ik]:.4e} --> tsn_new={tsn_new[ik]}"

  # Update ice enthalpy in the surface layer to prevent rapid snow melt
  # if qice > qsnow, this is particularly important for 
  # no snow --> snow cases during summer
  sin = sicen1[:,j0,i0]
  qin = qicen1[:,j0,i0] 
  qin_new = mc6util.ice_enthalpy_BL99(tsn_new, sin)
  qin_new = np.minimum(qin_new, qin)
  # Check ice:
  #Tice = mc6util.ice_enthalpy_to_temp(qin,sin)
  Tice_new = mc6util.ice_enthalpy_to_temp(qin_new,sin)
  mu_ice = 0.054  # liquidus ratio btw frz T and salinity of brine, [deg/ppt] BL99
  Tice_melt = -mu_ice * sin
  if np.any(Tice_new >= Tice_melt):
    print(f"j0={j0}, i0={i0}, ice T exceeds melting T")
    for kcat in range(len(Tice_new)):
      print(f"cat={kcat+1}: Tice_new={Tice_new[kcat]:.4f}  Tmelt={Tice_melt[kcat]:.4f}")
    raise Exception("ERR Updating ice enthalpy layer 1")

  vtot_init = np.nansum(vsn)
  vtot_new  = np.nansum(vsn_new)
  #print(f"tot vsnon change = {vtot_new-vtot_init}") 

  # Pond fractional area and depth
  # For now, make it 0 to prevent rapid snow melting when apnd >> 0
  # Note if apnd is changed and > 0, need to adjust hpnd to maintain nonnegative freeboard
  # see: icepack_meltpond_lvl.F90 as an example
  apn_new = apn * 0.
  hpn_new = hpn * 0.

  # Snow-ice interface should be at or above the sea level
  # if it is below, snow will be melted instantly to bring the interface to sea level
  # First, if needed - try to redistribute excess snow across ice thikn. cats:
  vsn_new = mc6util.adjust_snow_freeboard(vin, vsn_new, ain, rho_ice=rho_ice, \
                            rho_snow=rhos, rho_ocean=rho_ocean)

  # Check if the snow-ice interface in all cats is above the sea level
  # if not, try to slightly modify ice in the cat where needed
  # Note this will slightly change ice volume (ice thickness)
  # skip this step if ice vol has to be preserved
  ice_frb = mc6util.snow_ice_freeboard(vin, vsn_new, ain, \
                    rho_ice=rho_ice, rho_snow=rhos, rho_ocean=rho_ocean)  
  if np.any(ice_frb < 0.):
    vin_new, ain_new, vsn_new = mc6util.adjust_ice_freeboard(vin, vsn_new, ain, hicat, \
                    rho_ice=rho_ice, rho_snow=rhos, rho_ocean=rho_ocean, fdebug=True) 
  else:
    vin_new = vin.copy()

  # Update:
  dvol_sum = dvol_sum + (vtot_new-vtot_init)
  aicen_new[:,j0,i0]  = ain_new
  vsnon_new[:,j0,i0]  = vsn_new
  qsnon_new[:,j0,i0]  = qsn_new
  qicen1_new[:,j0,i0] = qin_new
  apndn_new[:,j0,i0]  = apn_new
  hpndn_new[:,j0,i0]  = hpn_new
  tsfcn_new[:,j0,i0]  = tsn_new
  vicen_new[:,j0,i0]  = vin_new

  diff = np.nansum(vsn_new - vsnon[:, j0, i0])
  diff2 = np.nansum(vsn_new -vsn)
  diff3 = np.nansum(vsnon[:,j0,i0] - vsnon_new[:,j0,i0])
  if diff == 0 and abs(diff2) > 0:
    print(f"No change at {j0},{i0}, expected diff={diff2}")

  if diff3 == 0 and abs(diff2) > 0:
    print(f"No change in the arrays at {j0},{i0}, expected diff={diff2}")

# Checking:
print(f"Snow vol change: dvol_sum = {dvol_sum}")
total_vsnon_init = np.nansum(vsnon)
total_vsnon_new  = np.nansum(vsnon_new)
print(f"Tot snow volume change (m3/m2): {total_vsnon_new - total_vsnon_init}")

# Update data set:
#ds_out = ds_out.assign(vsnon=vsnon_new, qsno001=qsnon_new)
ds_out['aicen'].values[:]   = aicen_new
ds_out['vsnon'].values[:]   = vsnon_new
ds_out['qsno001'].values[:] = qsnon_new
ds_out['qice001'].values[:] = qicen1_new
ds_out['apnd'].values[:]    = apndn_new
ds_out['hpnd'].values[:]    = hpndn_new
ds_out['Tsfcn'].values[:]   = tsfcn_new
ds_out['vicen'].values[:]   = vicen_new

# Sanity checking:
assert "vsnon" in ds_out and "qsno001" in ds_out, "Missing updated snow fields vsnon and qsno001"
assert ds_out["vsnon"].shape == vsnon_new.shape, "Check shape of vsnon "
assert ds_out["qsno001"].shape == qsnon_new.shape, "Check shape of qsnon "

#A = STOP

# Check hice(n) as it is caclulated in icepack_therm_vertical.F90
# hice(n) = vice(n) / aice(n) 
print(' =========  ICE  =========')
for k in range(1,ncat+1):
  aice_n = aicen[k-1,:].squeeze()
  vice_n = vicen[k-1,:].squeeze()
  hice_n = np.divide(vice_n, aice_n, out=np.zeros_like(aice), where=aice_n != 0)
  jmin, imin = np.unravel_index(hice_n.argmin(), hice_n.shape)
  jmax, imax = np.unravel_index(hice_n.argmax(), hice_n.shape)
  print(f"Cat {k}, j={jmin}, i={imin}, min hice(n): {np.nanmin(hice_n)}, "+\
        f"aice(n): {aice_n[jmin,imin]}, vice(n): {vice_n[jmin,imin]}")
  print(f"  j={jmax}, i={imax}, max hice(n): {np.nanmax(hice_n)}, "+\
        f"aice(n): {aice_n[jmax,imax]}, vice(n): {vice_n[jmax,imax]}")

print(' =========  SNOW =========')
for k in range(1,ncat+1):
  aice_n = aicen[k-1,:].squeeze()
  vsno_n = vsnon_new[k-1,:].squeeze()
  hsno_n = np.divide(vsno_n, aice_n, out=np.zeros_like(aice), where=aice_n != 0)
  jmin, imin = np.unravel_index(hsno_n.argmin(), hsno_n.shape)
  jmax, imax = np.unravel_index(hsno_n.argmax(), hsno_n.shape)
  print(f"Cat {k}, j={jmin}, i={imin}, min hsnow(n): {np.nanmin(hsno_n)}, "+\
        f"aice(n): {aice_n[jmin,imin]}, vsno(n): {vsno_n[jmin,imin]}")
  print(f"  j={jmax}, i={imax}, max hsnow(n): {np.nanmax(hsno_n)}, "+\
        f"aice(n): {aice_n[jmax,imax]}, vsno(n): {vsno_n[jmax,imax]}")

print(' ======== snow enthalpy =======')
for k in range(1,ncat+1):
  qtot_old = np.nansum(qsnon[k-1,:])
  qtot_new = np.nansum(qsnon_new[k-1,:])
  print(f"Cat {k}, old snow enth={qtot_old:.4e} new snow enth={qtot_new:.4e}")

print(" ")

# Attributes:
from datetime import datetime
istep1_val = ds_out.attrs.get('istep1', None)
ds_out.attrs.update({
    "title": "CICE6 restart with inserted hsnow from SSM/I NASA gridded fields for S. Ocean",
    "source": "insert_hsnow_cice6_restart.py",
    "istep1": np.int32(istep1_val) if istep1_val is not None else np.int32(0), 
    "myear": np.int32(yrN),
    "mmonth": np.int32(mmN),
    "mday": np.int32(ddN),
    "msec": np.int32(nsecN),
    "history": f"Modified {datetime.now().isoformat()}",
})

# Save:flrst_in
# Construct output file name if not provided:
if flrst_out is None:
  sfx = extract_suffix(flrst_in)
  if sfx is None:
    #flrst_out = f"cice_model.res.{yrN}{mmN:02d}{ddN:02d}.{nsecN:06d}.newhsnow.nc"
    flrst_out = f"cice_model.res.{yrN}{mmN:02d}{ddN:02d}.{hrN:02d}.snow.nc"
  else:
    flrst_out = f"cice_model.res.{yrN}{mmN:02d}{ddN:02d}.{hrN:02d}.{sfx}.snow.nc"
dflrst_out = os.path.join(pthrest_out,flrst_out)
print(f"Saving CICE restart --> {dflrst_out}")
ds_out.to_netcdf(dflrst_out, encoding={var: {'_FillValue': None} for var in ds_out.data_vars}, format='NETCDF3_64BIT')
ds_out.close()


f_chck = False
if f_chck:
  plt.ion()

  units = 'm'
  clrmp = mclrmps.colormap_uv()
  rmin = -0.5
  rmax = 0.5
  clrmp.set_bad(color=[0.2, 0.2, 0.2])

  sttl = 'Restart vsno m3/m2_ice: diff restart vs SSMI' 

  # m3(snow)/m2_ice
  vsno_ice = np.nansum(vsnon_new, axis=0).squeeze()
  # Aggregated ice partial area:
  aice = np.sum(aicen_new, axis=0).squeeze()
  vsno_cell = np.divide(vsno_ice, aice, out=np.zeros_like(aice), where=aice > 0)


  # New ice snow thickness over sea ice:
  if not units_m:
    AAi = HSi * 0.01        # m of snow over sea ice
  else:
    AAi = HSi.copy()

  dHS = vsno_cell - AAi
 
  # S. Ocean:
  m = Basemap(projection='spstere',boundinglat=-50,lon_0=180,resolution='l')
  #lons, lats = m.makegrid(idim, jdim) # get lat/lons of ny by nx evenly spaced grid.
  #x, y = m(lons, lats) # compute map proj coordinates.
  xh, yh = m(TLON,TLAT) # CICE6 coordinates

  if regn == 'south':
    xl1 = -8.e6
    xl2 = -1.2e6
    yl1 = xl1
    yl2 = xl2

  fig1 = plt.figure(1,figsize=(9,9))
  plt.clf()
  ax1 = plt.axes([0.08, 0.1, 0.8, 0.8])
  m.drawcoastlines()

  # draw parallels.
  parallels = np.arange(-80,-10,10.)
  m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
  # draw meridians
  meridians = np.arange(-360,359.,45.)
  m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

  img = ax1.pcolormesh(xh, yh, dHS, cmap=clrmp, vmin=rmin, vmax=rmax, shading='auto')
  ax1.contour(xh, yh, Aice, [0.15], linestyles='solid', colors=[(0.2,0.9,0.2)], linewidths=1)

  ax1.set_xlim([xl1, xl2])
  ax1.set_ylim([yl1, yl2])
  ax1.invert_yaxis()
  ax1.invert_xaxis()

  ax1.set_title(sttl)

  ax2 = fig1.add_axes([ax1.get_position().x1+0.025, ax1.get_position().y0,
                     0.02, ax1.get_position().height])
  if rmin < 0:
    clb = plt.colorbar(img, cax=ax2, orientation='vertical', extend='both')
  else:
    clb = plt.colorbar(img, cax=ax2, orientation='vertical', extend='max')

  ax2.yaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
  ax2.set_yticklabels(ax2.get_yticks())
  ticklabs = clb.ax.get_yticklabels()
  #  clb.ax.set_yticklabels(ticklabs,fontsize=10)
  clb.ax.set_yticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
  clb.ax.tick_params(direction='in', length=12)

  ax3 = fig1.add_axes([0.02, 0.03, 0.8, 0.06])
  ax3.text(0, 0, sinfo, fontsize=8)
  ax3.axis('off')

  btx = 'insert_hsnow_cice6_restart.py'
  bottom_text(btx, pos=[0.2, 0.01])





