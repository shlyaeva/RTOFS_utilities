#!/usr/bin/env python3
"""
YAML-driven driver to run CICE6 restart updates:
1) insert_iconc_ithkn_cice6_restart.py
2) insert_hsnow_cice6_restart.py

This script does not change scientific behavior of the underlying scripts.
It only provides flexible input/output and target-file configuration.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from yaml import safe_load


def _require(mapping, key_path):
    node = mapping
    for key in key_path:
        if key not in node:
            raise KeyError(f"Missing required YAML key: {'.'.join(key_path)}")
        node = node[key]
    return node


def _as_str(path_like):
    return str(Path(path_like))


def _resolve_path(path_like, base_dir):
    pth = Path(path_like)
    if pth.is_absolute():
        return pth
    return (base_dir / pth).resolve()


def _get_first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _infer_rdate_from_path(path_like):
    path_text = str(path_like)
    # Prefer path segment boundaries around YYYYMMDD-like tokens.
    match = re.search(r"(?:^|[^0-9])(\d{8})(?:[^0-9]|$)", path_text)
    if match:
        return int(match.group(1))
    return None


def _to_int_or_none(value):
    if value is None:
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Run CICE6 iconc/ithkn + hsnow updates from YAML")
    parser.add_argument("config", type=Path, help="Path to YAML configuration")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable for running sub-scripts")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = safe_load(f)

    config_dir = args.config.resolve().parent

    paths = _require(cfg, ["paths"])
    restart_in = _resolve_path(_require(paths, ["restart_in"]), config_dir)
    restart_out = _resolve_path(_require(paths, ["restart_out"]), config_dir)

    options = cfg.get("options", {})
    regn = options.get("region", "south")
    ithkn_enable = int(bool(options.get("insert_ice_thickness", True)))

    stage1 = cfg.get("target_total_ice_concentration", {})
    stage2 = cfg.get("target_ice_thickness", {})
    stage3 = cfg.get("target_snow_depth", {})

    script_dir = Path(__file__).resolve().parent
    script_iconc = script_dir / "insert_iconc_ithkn_cice6_restart.py"
    script_hsnow = script_dir / "insert_hsnow_cice6_restart.py"

    if not script_iconc.exists() or not script_hsnow.exists():
        raise FileNotFoundError("Could not find required prepare_cice6 scripts")

    restart_in_dir = restart_in.parent
    restart_in_name = restart_in.name
    restart_out_dir = restart_out.parent
    restart_out_name = restart_out.name

    mom6_data_dir = _get_first_present(paths, ["mom6_data_dir", "data_dir"])
    mom6_grid_dir = _get_first_present(paths, ["mom6_grid_dir", "grid_dir"])
    mom6_hgrid_file = _get_first_present(paths, ["mom6_hgrid_file", "hgrid_file"])
    mom6_topo_file = _get_first_present(paths, ["mom6_topo_file", "topo_file"])
    if mom6_data_dir is not None:
        mom6_data_dir = _resolve_path(mom6_data_dir, config_dir)
    if mom6_grid_dir is not None:
        mom6_grid_dir = _resolve_path(mom6_grid_dir, config_dir)
    if mom6_hgrid_file is not None:
        mom6_hgrid_file = _resolve_path(mom6_hgrid_file, config_dir)
    if mom6_topo_file is not None:
        mom6_topo_file = _resolve_path(mom6_topo_file, config_dir)

    # Resolve restart date/hour from common key aliases first,
    # then infer date from path tokens if needed.
    rdate = _to_int_or_none(_get_first_present(options, ["rdate", "restart_date", "date", "rdate_in"]))
    rhr = _to_int_or_none(_get_first_present(options, ["rhr", "restart_hour", "hour", "rhr_in"], default=0))
    rdate_out = _to_int_or_none(_get_first_present(options, ["rdate_out", "restart_date_out", "date_out"]))
    rhr_out = _to_int_or_none(_get_first_present(options, ["rhr_out", "restart_hour_out", "hour_out"]))

    if rdate is None:
        rdate = _infer_rdate_from_path(restart_in)
    if rdate_out is None:
        rdate_out = _infer_rdate_from_path(restart_out)
    if rdate_out is None:
        rdate_out = rdate
    if rhr_out is None:
        rhr_out = rhr

    with tempfile.TemporaryDirectory(prefix="cice6_update_") as td:
        temp_dir = Path(td)
        temp_stage1_name = "stage1.iconc_ithkn.nc"

        cmd1 = [
            args.python,
            _as_str(script_iconc),
            "--ithkn", str(ithkn_enable),
            "--regn", str(regn),
            "--pth_in", _as_str(restart_in_dir),
            "--flrst_in", restart_in_name,
            "--pth_out", _as_str(temp_dir),
            "--flrst_out", temp_stage1_name,
        ]

        if rdate is not None:
            cmd1 += ["--rdate", str(rdate)]
        if rhr is not None:
            cmd1 += ["--rhr", str(rhr)]
        if rdate_out is not None:
            cmd1 += ["--rdate_out", str(rdate_out)]
        if rhr_out is not None:
            cmd1 += ["--rhr_out", str(rhr_out)]

        if "file" in stage1:
            cmd1 += ["--iconc_file", _as_str(_resolve_path(stage1["file"], config_dir))]
            if "variable" in stage1:
                cmd1 += ["--iconc_var", str(stage1["variable"])]

        if mom6_data_dir is not None:
            cmd1 += ["--mom6_data_dir", _as_str(mom6_data_dir)]
        if mom6_grid_dir is not None:
            cmd1 += ["--mom6_grid_dir", _as_str(mom6_grid_dir)]
        if mom6_hgrid_file is not None:
            cmd1 += ["--mom6_hgrid_file", _as_str(mom6_hgrid_file)]
        if mom6_topo_file is not None:
            cmd1 += ["--mom6_topo_file", _as_str(mom6_topo_file)]

        if "file" in stage2:
            cmd1 += ["--ithkn_file", _as_str(_resolve_path(stage2["file"], config_dir))]
            if "variable" in stage2:
                cmd1 += ["--ithkn_var", str(stage2["variable"])]

        print("[1/2] Running iconc/ithkn insertion...")
        subprocess.run(cmd1, check=True)

        cmd2 = [
            args.python,
            _as_str(script_hsnow),
            "--regn", str(regn),
            "--pth_in", _as_str(temp_dir),
            "--flrst_in", temp_stage1_name,
            "--pth_out", _as_str(restart_out_dir),
            "--flrst_out", restart_out_name,
        ]

        if rdate is not None:
            cmd2 += ["--rdate", str(rdate)]
        if rhr is not None:
            cmd2 += ["--rhr", str(rhr)]
        if rdate_out is not None:
            cmd2 += ["--rdate_out", str(rdate_out)]
        if rhr_out is not None:
            cmd2 += ["--rhr_out", str(rhr_out)]

        if "file" in stage3:
            cmd2 += ["--hsnow_file", _as_str(_resolve_path(stage3["file"], config_dir))]
            if "variable" in stage3:
                cmd2 += ["--hsnow_var", str(stage3["variable"])]

        if mom6_data_dir is not None:
            cmd2 += ["--mom6_data_dir", _as_str(mom6_data_dir)]
        if mom6_grid_dir is not None:
            cmd2 += ["--mom6_grid_dir", _as_str(mom6_grid_dir)]
        if mom6_hgrid_file is not None:
            cmd2 += ["--mom6_hgrid_file", _as_str(mom6_hgrid_file)]

        print("[2/2] Running hsnow insertion...")
        subprocess.run(cmd2, check=True)

    print(f"Done. Output restart: {restart_out}")


if __name__ == "__main__":
    main()
