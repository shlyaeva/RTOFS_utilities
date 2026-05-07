#!/usr/bin/env python3
"""
YAML-driven driver to run CICE6 restart updates:
1) insert_iconc_ithkn_cice6_restart.py
2) insert_hsnow_cice6_restart.py

This script does not change scientific behavior of the underlying scripts.
It only provides flexible input/output and target-file configuration.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run CICE6 iconc/ithkn + hsnow updates from YAML")
    parser.add_argument("config", type=Path, help="Path to YAML configuration")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable for running sub-scripts")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = safe_load(f)

    paths = _require(cfg, ["paths"])
    restart_in = Path(_require(paths, ["restart_in"]))
    restart_out = Path(_require(paths, ["restart_out"]))

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

        if "rdate" in options:
            cmd1 += ["--rdate", str(options["rdate"])]
        if "rhr" in options:
            cmd1 += ["--rhr", str(options["rhr"])]
        if "rdate_out" in options:
            cmd1 += ["--rdate_out", str(options["rdate_out"])]
        if "rhr_out" in options:
            cmd1 += ["--rhr_out", str(options["rhr_out"])]

        if "file" in stage1:
            cmd1 += ["--iconc_file", _as_str(stage1["file"])]
            if "variable" in stage1:
                cmd1 += ["--iconc_var", str(stage1["variable"])]

        if "file" in stage2:
            cmd1 += ["--ithkn_file", _as_str(stage2["file"])]
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

        if "rdate" in options:
            cmd2 += ["--rdate", str(options["rdate"])]
        if "rhr" in options:
            cmd2 += ["--rhr", str(options["rhr"])]
        if "rdate_out" in options:
            cmd2 += ["--rdate_out", str(options["rdate_out"])]
        if "rhr_out" in options:
            cmd2 += ["--rhr_out", str(options["rhr_out"])]

        if "file" in stage3:
            cmd2 += ["--hsnow_file", _as_str(stage3["file"])]
            if "variable" in stage3:
                cmd2 += ["--hsnow_var", str(stage3["variable"])]

        print("[2/2] Running hsnow insertion...")
        subprocess.run(cmd2, check=True)

    print(f"Done. Output restart: {restart_out}")


if __name__ == "__main__":
    main()
