"""
  Insert ice concentration (sea ice partial area) and 
  (optionally) ice thickness 
  into CICE6 aice fields 
  Using NSIDC interpoalted fields for iconc
  and CryoSat for ice thickness

  if ice thickness is opted out, this should be idential to
  insert_iconc_cice6_restart.py

  See python/prepare_cice6/interp_NSIDC_iconc_mesh025.py
                    interp_CryoSat_ithkn_antarct_mesh025.py

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
import mod_utils as mutil
import mod_colormaps as mclrmps
import mod_anls_seas as manseas
import mod_utils_ob as mutob
import mod_mom6 as mmom6
import mod_misc1 as mmisc
import mod_cice6_utils as mc6util
importlib.reload(mc6util)

def find_varnm(dflithkn, var_opt):
  with xarray.open_dataset(dflithkn) as ds_ithkn:
    for varnm in var_opt:
      if varnm in ds_ithkn.data_vars:
        #print(f"Using variable {varnm}")
        return varnm
        
  raise KeyError("No ice thickness variable name found, check file")
    
  return

rest_date = 20250103
rest_hr = 0
regn = 'south'
yrR = mmR = ddR = hrR = None
yrN = mmN = ddN = hrN = None

parser = argparse.ArgumentParser()
parser.add_argument("--ithkn", type=int, default=1,
    help="insert ice thickn climatology, 0=no, 1=yes (default 1)", choices=[0,1])
parser.add_argument("--rdate", help=f"restart date input file, default={rest_date}", type=int)
parser.add_argument("--rhr", help=f"input file, restart hour = 0, ..., 23, default={rest_hr}", type=int)
parser.add_argument("--rdate_out", help="output file, restart date if different from input", type=int)
parser.add_argument("--rhr_out", help="output file, restart hour if date is different from input", type=int)
parser.add_argument("--pth_in", help="input restart directory with original file, default=None", type=str)
parser.add_argument("--flrst_in", help="rest file in, otherwise name constructed from rest_date", type=str)
parser.add_argument("--pth_out", help="output restart directory where new file be dumped, default=None", type=str)
parser.add_argument("--flrst_out", help="new rest file, otherwise name constructed from rdate_out", type=str)
parser.add_argument("--regn", help=f"where icon incerted", 
                    choices=['north','south','global'], type=str)
parser.add_argument("--iconc_file", help="optional custom interpolated ice concentration file", type=str)
parser.add_argument("--iconc_var", help="variable name in --iconc_file (default: ice_conc)", type=str,
                    default="ice_conc")
parser.add_argument("--ithkn_file", help="optional custom interpolated ice thickness file", type=str)
parser.add_argument("--ithkn_var", help="optional variable name in --ithkn_file", type=str)
args = parser.parse_args()

ins_thkn = bool(args.ithkn)
flrst_in  = args.flrst_in  if args.flrst_in  else None
flrst_out = args.flrst_out if args.flrst_out else None
regn      = args.regn if args.regn else regn
# if rest_date and rest_date_out are provided
# Derive dates assuming file nameing is cice_restart.res.YYYYMMDD.XX[XXX]
# or YYYYMMDD.<time>.---.nc
if flrst_in is not None:
  yrR, mmR, ddR, hrR, mintR = mc6util.get_date_filename(flrst_in)
  rest_date = int(yrR*1e4 + mmR*100 + ddR)
  rest_hr = hrR
else:
  rest_date = args.rdate if args.rdate else rest_date
  rest_hr   = args.rhr if args.rhr else rest_hr

if flrst_out is not None:
  yrN, mmN, ddN, hrN, mintN = mc6util.get_date_filename(flrst_out)
  rest_date_out = int(yrN*1e4 + mmN*100 + ddN)
  rest_hr_out = hrN  
else:
  rest_date_out = args.rdate_out if args.rdate_out else rest_date
  rest_hr_out   = args.rhr_out if args.rhr_out else rest_hr

pth_in = args.pth_in if args.pth_in else None
pth_out = args.pth_out if args.pth_out else None
iconc_file = args.iconc_file if args.iconc_file else None
iconc_var = args.iconc_var if args.iconc_var else "ice_conc"
ithkn_file = args.ithkn_file if args.ithkn_file else None
ithkn_var = args.ithkn_var if args.ithkn_var else None

print(f"Restart date input:  {rest_date}:{rest_hr}")
print(f"Restart date output: {rest_date_out}:{rest_hr_out}")
if ins_thkn:
  print("Insert NSDIC NRT ice concenatraion + CryoSat ice thickness climatology into CICE restart\n")
else:
  print("Insert NSDIC NRT ice concenatraion, NO ice thickness\n")

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

if pth_in is None:
  pthrest = os.path.join(pths_ufs[node_nm]["MOM6"]["pthrest"],'new')
else:
  pthrest = pth_in
#pthrest = '/gpfs/f6/sfs-emc/proj-shared/Dmitry.Dukhovskoy/RUNDIRS/restart_da'

# Output dir for new restart:
if pth_out is None:
  pthrest_out = os.path.join(pths_ufs[node_nm]["MOM6"]["pthrest"],'new')
else:
  pthrest_out = pth_out

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

# Get MOM6 grid
pthgrid = pths_ufs[node_nm]["MOM6"]["pthgrid"]
dfgrid_mom = os.path.join(pthgrid, "ocean_hgrid.1440x1080.nc")
dftopo_mom = os.path.join(pthgrid, "ocean_topog.1440x1080.nc")

with xarray.open_dataset(dftopo_mom) as dstopo:
  HH = dstopo['depth'].data.squeeze()

HH = np.where(HH < 1.e-20, np.nan, HH)
HH = -HH
HH = np.where(np.isnan(HH), 1., HH)
jdm, idm = HH.shape

LON, LAT = mmom6.read_mom6grid(dfgrid_mom, grdpnt='hgrid')

# Interpolated NSIDC ice conc:
pthnsidc = os.path.join(pthdata,f"NRT_NOAA_NSIDC_seaconc/{yrN}")
RMsk = np.where(HH>=0, 0, 1)
if regn == 'south':
  RMsk[LAT > -60.] = 0
elif regn == 'north':
  RMsk[LAT < 50.] = 0
elif regn == 'global':
  pass
else:
  raise Exception(f"Unrecognized region {regn}")

print(f"old restart: {yrR}/{mmR:02d}/{ddR:02d}:{hrR:02d}")
print(f"new restart: {yrN}/{mmN:02d}/{ddN:02d}:{hrN:02d}")

if iconc_file is None:
  if regn == 'global':
    raise ValueError("For regn='global', provide --iconc_file (and --iconc_var if needed)")
  pthnsidc, fliconc = mc6util.pathfname_icesnow_mesh025(
      fyaml, node_nm, "iconc_NSIDC", YR=yrN, MM=mmN, regn=regn)
  dfliconc = os.path.join(pthnsidc, fliconc)
else:
  dfliconc = iconc_file
print(f'Loading interpolated ice conc {dfliconc}')
with xarray.open_dataset(dfliconc) as dsint:
  if iconc_var not in dsint:
    raise KeyError(f"Variable '{iconc_var}' not found in {dfliconc}")
  AICEint = dsint[iconc_var].isel(time=ddN-1).squeeze()

AICEint = np.where(RMsk == 0, np.nan, AICEint)

# Read ice thickness data:
if ins_thkn:
  var_opt = ['ice_thkn', 'ithkn', 'hi', 'ice_thickness']
  if ithkn_file is None:
    if regn == 'global':
      raise ValueError("For regn='global' with --ithkn 1, provide --ithkn_file")
    pthithkn, flithkn = mc6util.pathfname_icesnow_mesh025(fyaml, node_nm, 'ithkn_clim', regn=regn)
    dflithkn = os.path.join(pthithkn, flithkn)
  else:
    dflithkn = ithkn_file
  
  #pthithkn = os.path.join(pthdata,'CryoSat2_antarctic_ice_snow_thkn','clim')
  #flithkn = 'CryoSat_hice_mnthclim_2011_2020_mesh025_1440x1080_south.nc'
  #dflithkn = os.path.join(pthithkn,flithkn)
  ithkn_varnm = ithkn_var if ithkn_var else find_varnm(dflithkn, var_opt)
  print(f"Reading ice thickn varnm='{ithkn_varnm}' for month {mmN} from {dflithkn}")
  with xarray.open_dataset(dflithkn) as ds_ithkn:
    if ithkn_varnm not in ds_ithkn:
      raise KeyError(f"Variable '{ithkn_varnm}' not found in {dflithkn}")
    ITHKN = ds_ithkn[ithkn_varnm].isel(time=mmN-1).data
else:
  ITHKN = np.full_like(HH, np.nan)

if flrst_in is None:
  flrst_in = f"cice_model.res.{yrR}{mmR:02d}{ddR:02d}.{nsecR:06d}.nc"

dflrst_in = os.path.join(pthrest, flrst_in)
print(f"Reading restart: {dflrst_in}")
ds_in = xarray.open_dataset(dflrst_in)
ds_out = ds_in.copy(deep=True)
ds_in.close()


# Insert aice and distribute/reduce proportionally over ice cats.:
assert nslyr == 1, f"Code needs to be modified for nslyr>1, nslyr={nslyr}"
aicen = ds_out['aicen'].data  # partial area by cats
vicen = ds_out['vicen'].data  # ice vol per m2 of grid cell by cats
vsnon = ds_out['vsnon'].data  # snow vol per m2 of ice area
qsnon = ds_out['qsno001'].data  # snow enthalpy by cats for 1 snow layer
Tsfcn = ds_out['Tsfcn'].data  # surface T in each cat.
ncat, jdim, idim = vsnon.shape

sice = {}
qice = {}
nilrs = 7   # ice layers
for i in range(1, nilrs+1):
  varnum = f"{i:03d}" 
  sice[varnum] = ds_out[f"sice{varnum}"].data
  qice[varnum] = ds_out[f"qice{varnum}"].data

# Aggregated ice partial area:
aice = np.sum(aicen, axis=0).squeeze()

# Select points to insert:
Jins, Iins = np.where((RMsk > 0) & (~np.isnan(AICEint)))
Xins = LON[Jins,Iins]
Yins = LAT[Jins,Iins]
npnts = len(Jins)

npnts_ithkn = 0
if ins_thkn:
  mask_ithkn = (RMsk > 0) & np.isfinite(ITHKN)
  npnts_ithkn = np.count_nonzero(mask_ithkn)

print(f"iconc insertion: {npnts} pnts, min/max lat={np.min(Yins):.1f}/{np.max(Yins):.1f}"
       f" lon={np.min(Xins):.1f}/{np.max(Xins):.1f}")
if ins_thkn:
  print(f"ithkn insertion: {npnts_ithkn} pnts ")


def find_adj_icepnts(dlti, aice, i0, j0):
  jdm, idm = aice.shape
  i1 = max(i0 - dlti, 0)
  i2 = min(i0 + dlti, idm - 1)
  j1 = max(j0 - dlti, 0)
  j2 = min(j0 + dlti, jdm - 1)

  A = aice[j1:j2+1, i1:i2+1]
  jmm, imm = np.where(A > puny)

  # No grid points with ice found:
  if jmm.size == 0:
    return np.array([], dtype=int), np.array([], dtype=int)

  jice = jmm + j1
  iice = imm + i1

  return iice, jice

def find_adj_ocnpnts(dlti, aice, i0, j0):
  jdm, idm = aice.shape
  i1 = max(i0 - dlti, 0)
  i2 = min(i0 + dlti, idm - 1)
  j1 = max(j0 - dlti, 0)
  j2 = min(j0 + dlti, jdm - 1)

  A = aice[j1:j2+1, i1:i2+1]
  jmm, imm = np.where(A <= puny)

  # No grid points with no ice found:
  if jmm.size == 0:
    return np.array([], dtype=int), np.array([], dtype=int)

  jocn = jmm + j1
  iocn = imm + i1

  return iocn, jocn

# Ice thickness cats
# for kitd = 1 - linear remapping
# and kcatbound = 0 , the lower cat. thickness values:
# 0.00, 0.64, 1.39, 2.47, 4.57
hicat = np.array([0., 0.64, 1.39, 2.47, 4.57, 50.])
dhi_min = 0.01  # min diff between cat ice thicknesses from 2 adjacent cats
hcat_indx = np.arange(1,ncat+1)

# Note qsnon, qice < 0 !
vsnon_new = vsnon.astype(ds_out['vsnon'].dtype).copy()
qsnon_new = qsnon.astype(ds_out['qsno001'].dtype).copy()
aicen_new = aicen.astype(ds_out['aicen'].dtype).copy()
vicen_new = vicen.astype(ds_out['vicen'].dtype).copy()
Tsfcn_new = Tsfcn.astype(ds_out['Tsfcn'].dtype).copy()
qicen_new = {}
sicen_new = {}
for i in range(1, nilrs+1):
  varnum = f"{i:03d}"
  qicen_new[varnum] = qice[varnum].astype(ds_out[f"qice{varnum}"].dtype).copy()
  sicen_new[varnum] = sice[varnum].astype(ds_out[f"sice{varnum}"].dtype).copy()

Tfrz = -1.86243522  # ocean freez. T
Tsfc_max = -0.1  # max surf temp
dvol_sum = 0.
dlti = 2         # N of +/- i , j indices to search for ice pnts around i0,j0
vitot_min = 0.05    # min total ice vol m3/m2_grid, when ai_old = 0 --> ai_new > 0
hitot_min = 0.1    # mean ice thickness over ice area: = sum(hice(n)*aice(n)) / sum(aice(n)) 
hsnow_min = 0.01   # min snow thickness for noice --> ice case, this is m3/m2_ice 
hsnow_max = 500.   # to avoid very thick hsnow / ice_area which will cause picard iteration crush
print("Ice concentration insertion ...")
for ipp in range(npnts):
  if ipp%10000 == 0:
    print(f"   {ipp/npnts*100.:.2f}% done ...")
  j0 = Jins[ipp]
  i0 = Iins[ipp]

  # Note hsnow = vsn / aice for aice > 0
  # for cat n: vsn(n) = hsnow(n) * aice(n) 
  ai_old  = aice[j0,i0]       # aggreageted ice partial area 
  ain_old = aicen[:,j0,i0]    # partial areas by cats
  vsn_old = vsnon[:,j0,i0]    # snow volume per unit grid-cell area m2 in cats
  vin_old = vicen[:,j0,i0]    # ice volume per unit grid-cell area m2 in cats
  #hin_old = np.divide(vin_old, ain_old, out=np.zeros_like(vin_old), where=ain_old != 0) # ice thkn or m3/m2_ice
  tsfcn_old = Tsfcn[:,j0,i0]  # surf T
  #ai_old = np.sum(ain_old)   # aggreageted ice partial area 

  # Distribute new iconc proportionally by cats in snow vol m3/m2:
  ai_new = AICEint[j0,i0]
  if ai_new <= puny:
    ai_new = 0.
 
  if ai_new > 1.:
    ai_new = 1.

  # iconc change:
  if ai_new < puny:
    ain_new = ain_old * 0.
  else:
    if ai_old < puny:
      # all new iconc in cat 1:
      ain_new = ain_old * 0.
      ain_new[0] = ai_new
    else:
      cff = ai_new/ai_old
      ain_new = ain_old * cff
      
  # Adjust small truncation errors:
  sum_ain = np.sum(ain_new)
  if (sum_ain > 1.0) and (sum_ain - 1.0 < 1e-12):
    ain_new = ain_new / sum_ain - 1.e-12
  ain_new = np.where(ain_new < puny, 0., ain_new)
 
  assert np.sum(ain_new) <= 1., f"Check ain_new: sum>1: {np.sum(ain_new)}"
  assert np.min(ain_new) >= 0., f"Check ain_new: min val < 0 {np.min(ain_new)}"


  # Update surface T and ice vol / ice thickness
  tsf_new = None
  iice = jice = iocn = jocn = None
  vin_new = None
  aice_case = None
  if ai_old <= puny and ai_new > puny:
    # Case: no ice --> ice, created ice in the grid cell
    aice_case = "noice2ice"

    # Update Tsfcn
    # Find N closest ice points:
    iice, jice = find_adj_icepnts(dlti, aice, i0, j0) 
    if len(iice) == 0:
      # no ice pnt adjacent to j0,i0:
      tsf_new = Tsfcn[:,j0,i0]*0.0 + Tsfc_max
      vin_new = hitot_min * ain_new
    else: 
      # where ice - <= Tmax, where no ice = Tfrz
      Tsf_adj = Tsfcn[:,jice,iice]
      tsf_new  = np.nanmean(Tsf_adj, axis=1)
      tsf_new  = np.where(tsf_new > Tsfc_max, Tsfc_max, tsf_new)
      tsf_new  = np.where(ain_new < puny, Tfrz, tsf_new) 

      Vice_adj = vicen[:,jice,iice]
      vin_new = np.nanmean(Vice_adj, axis=1)
      vitot_new = np.max([np.sum(vin_new), vitot_min]) # total ice vol/grid area m3/m2 =grid mean ice thkn, m
      wt = ain_new / np.sum(ain_new)
      vin_new = vitot_new * wt

  elif ai_old > puny and ai_new > puny:
    # Case: ice --> updated ice conc
    aice_case = "ice2ice"
    tsf_new = np.where(tsfcn_old > Tsfc_max, Tsfc_max, tsfcn_old)
    tsf_new  = np.where(ain_new < puny, Tfrz, tsf_new)           # ocean freezing T where no ice

    # Try to preserve mean ice thkn over ice:
    vitot_old = np.sum(vin_old)                          # m3/m2_cell or mean ice thkn over grid cell
    hitot_old = vitot_old / ai_old                       # m3/m2_ice, mean ice thkn over ice
    vin_new   = np.max([hitot_old,hitot_min]) * ain_new  # m3/m2_cell or grid-cell mean ice thickness, m

  elif ai_old > puny and ai_new < puny:
    # Case: ice --> no ice
    aice_case = "ice2noice"
    # Copy tsfc from adj grid cells
    iocn, jocn = find_adj_ocnpnts(dlti, aice, i0,j0)
    if len(iocn) == 0:
      # no ocn pnts:
      tsf_new = Tsfcn[:,j0,i0]*0.0 + Tfrz
    else:
      Tsf_adj = Tsfcn[:,jocn,iocn]
      tsf_new = np.nanmean(Tsf_adj, axis=1)
      tsf_new = np.where(tsf_new > Tsfc_max, Tsfc_max, tsf_new)
          
    vin_new = vin_old*0.0

  elif ai_old < puny and ai_new < puny:
    # Case: no ice --> no ice
    aice_case = "noice2noice"
    tsf_new = Tsfcn[:,j0,i0]
    vin_new = vin_old*0.0
  
  else:
    raise Exception(f"Unexpected case for ai_old={ai_old} and ai_new={ai_new}")  

  # Should not happen but Checking if any NaN occur:
  if np.isnan(tsf_new).any():
    tsf_new = np.where(np.isnan(tsf_new), Tsfc_max, tsf_new)
  if np.isnan(vin_new).any():
    vin_new = np.where(np.isnan(vin_new), 0., vin_new)

  # Check that ice vol is correctly distributed across the ice thickn. cats:
  # If not - distribute across ice cats conserving aice and vice 
  # and matching ice cats
  ain_min = 1.e-8    # lower bound of ain(n) to avoid zeros
  vice_clim = 0.     # ice vol / m2_cell from clim (CryoSat is cell mean ice thickn)
  vice_old = 0.      # ice vol / m2 _cell from old restart
  if ins_thkn:
    vice_clim = ITHKN[j0,i0] 
    if np.isnan(vice_clim):
      vice_clim = 0.

  vice_old = np.sum(vin_old)
  vice_new = np.sum(vin_new)

  if vice_clim > vitot_min:
    vtot_target = vice_clim
  else:
    # Case when ithkn is turned off but also
    # this ignores hice = 0 or thin ice in clim fields when ithkn is turned on
    vtot_target = np.max([vice_old, vice_new])

  ain_new, vin_new = mc6util.adjust_thkncats_aice(ain_new, vin_new, vtot_target, \
                         hicat, dhi_min,  bnd_min=ain_min)
  
  ain_new = np.where(ain_new <= ain_min, 0., ain_new)
  vin_new = np.where(ain_new <= ain_min, 0., vin_new)

  assert np.sum(ain_new) < (1.+1e-12), f"Ice conc > 1 {np.sum(ain_new)} j={j0} i={i0}"

  # Check ice cats:
  hin_new = np.divide(vin_new, ain_new, out=np.zeros_like(vin_new), where=ain_new != 0)
  cat_missed, hcat_new = mc6util.check_ithkn_cats(hicat, hin_new, ain_new)
  if cat_missed is not None:
    print(f"ipp={ipp} {aice_case} error in ice cats")
    raise Exception("Check ain_new, hin_new not in ice thkn cats")


  Tsfcn_new[:,j0,i0] = tsf_new
  vicen_new[:,j0,i0] = vin_new
  aicen_new[:,j0,i0] = ain_new

  # Update sice00?, qice00?
  for ilr in range(1, nilrs+1):
    varnum = f"{ilr:03d}"
    # Ice salinity by layers - compute S profile using BZ99 formulation:
    sice_lr = mc6util.sice_lr_cice4(ilr, nilrs, ain_new) 
    sice_old = sice[varnum][:,j0,i0]
    sice_new = np.where(sice_old < puny, sice_lr, sice_old)
    sice_new = np.where(ain_new < puny, 0., sice_new)

    sicen_new[varnum][:,j0,i0] = sice_new

    # ice enthalpy, use BL99
    # keep ice T below freezing T
    tice_max = sice_new*0 + Tfrz - 0.01
    qice_lr = mc6util.ice_enthalpy_BL99(tice_max, sice_new)
    qice_old = qice[varnum][:,j0,i0]
    qice_new = np.where(ain_old < puny, qice_lr, qice_old)  # fill gaps
    qice_new = np.where(ain_new < puny, 0., qice_new)       # no enth if no ice

    qicen_new[varnum][:,j0,i0] = qice_new

  # Check snow volume and ice_mean hsnow, add min snow if needed:
  vstot_old = np.sum(vsn_old)  # snow vol m3/m2_cell or cell mean snow thickn.
  if ai_new > puny:
    hsn_new = vstot_old / ai_new  # mean snow thickn over ice
    if hsn_new < hsnow_min:
      hsn_new = hsnow_min
      vstot_new = hsn_new * ai_new
    elif hsn_new > hsnow_max:
      hsn_new = hsnow_max
      vstot_new = hsn_new * ai_new
    else:
      vstot_new = vstot_old

    # Distribute evenly across cats:
    wts = ain_new/ai_new
    vsn_new = vstot_new * wts
  else:
    vsn_new = ain_new * 0.

  vsn_new = np.where(ain_new <= puny, 0., vsn_new)
  vsnon_new[:,j0,i0] = vsn_new

  # Update snow enthalpy: J/m3  
  # snow enthalpy should be: qsn_min <= qsn <= qsn_max
  # In theory, qsn_max = -rhos_Lfresh (latent heat of metling at 0C)
  # Make it a little lower to keep snow from melting right away
  #hsn_new = vsn_new / ain
  qsn = qsnon[:,j0,i0]        # enthalpy, J/kg < 0
  qsn_min = -rhos * Lfresh + (Tmin + 0.01) * cp_ice * rhos  # enth. of the coldest possible snow
  qsn_max = -rhos * Lfresh - 0.01 * cp_ice * rhos  # a little colder than 0C snow
  qsn_tsfc = -rhos * Lfresh + tsf_new * cp_ice * rhos # snow enth for surf. T
  qT0 = -Lfresh*rhos      # enth. of pure snow at 0C

  # Update new enthalpy of new snow:
  # Clip to min/max enthalpy, set to 0 where no snow:
  qsn_new = np.clip(qsn_tsfc, qsn_min, qsn_max)
  qsn_new = np.where(ain_new <= puny, 0., qsn_new)      # no ice
  qsn_new = np.where(vsn_new <= 0, 0., qsn_new)     # no snow
  qsnon_new[:,j0,i0] = qsn_new

  #vtot_old = np.nansum(vsn_old)
  #vtot_new  = np.nansum(vsn_new)
  #print(f"tot vsnon change = {vtot_new-vtot_init}") 


# Update data set:
#ds_out = ds_out.assign(vsnon=vsnon_new, qsno001=qsnon_new)
ds_out['vsnon'].values[:]   = vsnon_new
ds_out['qsno001'].values[:] = qsnon_new
ds_out['aicen'].values[:]   = aicen_new
ds_out['vicen'].values[:]   = vicen_new
ds_out['Tsfcn'].values[:]   = Tsfcn_new
for ilr in range(1, nilrs+1):
  varnum = f"{ilr:03d}"
  ds_out[f"sice{varnum}"].values[:] = sicen_new[f"{varnum}"]
  ds_out[f"qice{varnum}"].values[:] = qicen_new[f"{varnum}"]


# Sanity checking:
assert ds_out["vsnon"].shape == vsnon_new.shape, "Check shape of vsnon "
assert ds_out["qsno001"].shape == qsnon_new.shape, "Check shape of qsnon "

# Check hice(n) as it is caclulated in icepack_therm_vertical.F90
# hice(n) = vice(n) / aice(n) 
print(' =========  ICE  =========')
for k in range(1,ncat+1):
  aice_n = aicen_new[k-1,:].squeeze()
  vice_n = vicen_new[k-1,:].squeeze()
  hice_n = np.divide(vice_n, aice_n, out=np.zeros_like(aice), where=aice_n != 0)
  jmin, imin = np.unravel_index(hice_n.argmin(), hice_n.shape)
  jmax, imax = np.unravel_index(hice_n.argmax(), hice_n.shape)
  print(f"Cat {k}, j={jmin}, i={imin}, min hice(n): {np.nanmin(hice_n)}, "+\
        f"aice(n): {aice_n[jmin,imin]}, vice(n): {vice_n[jmin,imin]}")
  print(f"         j={jmax}, i={imax}, max hice(n): {np.nanmax(hice_n)}, "+\
        f"aice(n): {aice_n[jmax,imax]}, vice(n): {vice_n[jmax,imax]}")

print(' =========  SNOW =========')
for k in range(1,ncat+1):
  aice_n = aicen_new[k-1,:].squeeze()
  vsno_n = vsnon_new[k-1,:].squeeze()
  hsno_n = np.divide(vsno_n, aice_n, out=np.zeros_like(aice), where=aice_n != 0)
  jmin, imin = np.unravel_index(hsno_n.argmin(), hsno_n.shape)
  jmax, imax = np.unravel_index(hsno_n.argmax(), hsno_n.shape)
  print(f"Cat {k}, j={jmin}, i={imin}, min hsnow(n): {np.nanmin(hsno_n)}, "+\
        f"aice(n): {aice_n[jmin,imin]}, vsno(n): {vsno_n[jmin,imin]}")
  print(f"         j={jmax}, i={imax}, max hsnow(n): {np.nanmax(hsno_n)}, "+\
        f"aice(n): {aice_n[jmax,imax]}, vsno(n): {vsno_n[jmax,imax]}")


# Attributes:
from datetime import datetime
istep1_val = ds_out.attrs.get('istep1', None)
ds_out.attrs.update({
    "title": f"CICE6 restart with inserted ice concentration from NSIDC NRT {rest_date_out} ",
    "source": "insert_iconc_ithkn_cice6_restart.py",
    "istep1": np.int32(istep1_val) if istep1_val is not None else np.int32(0), 
    "myear": np.int32(yrN),
    "mmonth": np.int32(mmN),
    "mday": np.int32(ddN),
    "msec": np.int32(nsecN),
    "history": f"Modified {datetime.now().isoformat()}",
})

# Save:
if flrst_out is None:
  if ins_thkn:
    flrst_out = f"cice_model.res.{yrN}{mmN:02d}{ddN:02d}.{hrN:02d}.iconc_ithkn.nc"
  else:
    flrst_out = f"cice_model.res.{yrN}{mmN:02d}{ddN:02d}.{hrN:02d}.iconc.nc"

dflrst_out = os.path.join(pthrest_out,flrst_out)
print(f"Saving CICE restart --> {dflrst_out}")
ds_out.to_netcdf(dflrst_out, encoding={var: {'_FillValue': None} for var in ds_out.data_vars}, format='NETCDF3_64BIT')
ds_out.close()


f_plt = False
if f_plt:
  clrmp = mclrmps.colormap_conc()
  rmin = 0.
  rmax = 1.
  clrmp.set_bad(color=[0.2, 0.2, 0.2])
  hlon = LON
  hlat = LAT   
 
  if regn == 'south':
    m = Basemap(projection='spstere',boundinglat=-50,lon_0=180,resolution='l')
  #lons, lats = m.makegrid(idim, jdim) # get lat/lons of ny by nx evenly spaced grid.
  parallels = np.arange(-80,-10,10.)
  meridians = np.arange(-360,359.,45.)
  xl1 = -9e6
  xl2 = -0.8e6
  yl1 = xl1
  yl2 = xl2

  xh, yh = m(hlon,hlat) # GFS coords

  plt.ion()
  fig1 = plt.figure(1, figsize=(8,8))
  plt.clf()
  ax1 = plt.axes([0.1, 0.1, 0.8, 0.8])

  m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
  m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
  img1 = ax1.pcolormesh(xh,yh,aice, cmap=clrmp, vmin=rmin, vmax=rmax)

  ax1.contour(xh,yh,HH,[0], linestyles='solid', colors=[(0.,0.,0.)], linewidths=1)

  ax1.set_xlim([xl1, xl2])
  ax1.set_ylim([yl1, yl2])
  ax1.invert_yaxis()
  ax1.invert_xaxis()

  # Plot pnt:
  x0 = hlon[j0,i0]
  y0 = hlat[j0,i0]
  xh0 = xh[j0,i0]
  yh0 = yh[j0,i0]
  ax1.plot(xh0,yh0,'o')

  # Colorbars
  ax3 = fig1.add_axes([0.2, 0.05, 0.6, 0.02])
  clb = plt.colorbar(img1, cax=ax3, orientation='horizontal', extend='max')
  ax3.xaxis.set_ticks(list(np.linspace(rmin,rmax,11)))
  ax3.set_xticklabels(ax3.get_xticks())
  ticklabs = clb.ax.get_xticklabels()
  clb.ax.set_xticklabels(["{:.2f}".format(i) for i in clb.get_ticks()], fontsize=10)
  clb.ax.tick_params(direction='in', length=12)



