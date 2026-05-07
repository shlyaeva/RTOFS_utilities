# CICE6 restart updates (iconc + ithkn + hsnow) from one YAML

This documents:
- `run_cice6_restart_updates.py`
- `run_cice6_restart_updates.example.yaml`

The runner chains the existing scripts in order:
1. `insert_iconc_ithkn_cice6_restart.py`
2. `insert_hsnow_cice6_restart.py`

No scientific behavior is changed; this only centralizes configuration and I/O.

## Run

```bash
python RTOFS_utilities/prepare_cice6/run_cice6_restart_updates.py \
  RTOFS_utilities/prepare_cice6/run_cice6_restart_updates.example.yaml
```

## YAML schema

```yaml
paths:
  restart_in: /path/to/original_restart.nc
  restart_out: /path/to/output_restart.nc

options:                       # optional
  region: south                # south|north|global
  insert_ice_thickness: true   # stage-1 --ithkn (0/1)
  # rdate: 20250103
  # rhr: 0
  # rdate_out: 20250103
  # rhr_out: 0

target_total_ice_concentration:   # optional custom file for iconc
  file: /path/to/iconc_target.nc
  variable: ice_conc

target_ice_thickness:             # optional custom file for ithkn
  file: /path/to/ithkn_target.nc
  variable: ice_thkn

target_snow_depth:                # optional custom file for hsnow
  file: /path/to/hsnow_target.nc
  variable: snow_depth
```

## Required keys

- `paths.restart_in`
- `paths.restart_out`

Everything else is optional.

## How input/output files are used

- Stage 1 reads `paths.restart_in` and writes a temporary intermediate restart.
- Stage 2 reads that intermediate restart and writes `paths.restart_out`.

## Common pitfalls

- **Global update mode**
  - Set `options.region: global` to update without latitude boundaries.
  - In global mode, provide all target files explicitly:
    - `target_total_ice_concentration.file`
    - `target_ice_thickness.file` (if `insert_ice_thickness: true`)
    - `target_snow_depth.file`
  - This avoids relying on hemisphere-specific default target-file discovery.

- **Variable names**
  - Set `variable` explicitly if your file does not use defaults:
    - iconc: default `ice_conc`
    - hsnow: default `snow_depth`
    - ithkn: if `variable` is omitted, stage-1 auto-tries:
      `ice_thkn`, `ithkn`, `hi`, `ice_thickness`

- **Time indexing**
  - Both stage scripts select monthly/day slices internally (e.g., `.isel(time=...)`).
  - Target files should have expected `time` dimension/coverage for the configured date.

- **Grid compatibility**
  - Target fields are expected on the same mesh as used by the original scripts
    (typically mesh025 interpolated products).

- **Region mismatch**
  - Keep `options.region` consistent with the target datasets (`south` vs `north`).

- **Date/hour handling**
  - If omitted, scripts infer from filenames/defaults exactly as before.
  - Use `rdate/rhr/rdate_out/rhr_out` only when you need to override.

## Notes

- The runner passes through to existing scripts rather than reimplementing physics.
- The updated stage scripts now accept optional `--pth_in/--pth_out` and custom
  target file/variable arguments to support flexible workflows.
