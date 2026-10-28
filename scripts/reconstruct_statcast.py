#!/usr/bin/env python3
"""Starter reproduction of published *raw* Statcast X/Y/Z proxy estimates.

Inputs: a local pybaseball Statcast CSV or Parquet and a locally cloned copy of
https://github.com/jmaschino56/statcast-contact-point

Crucial: this does NOT reproduce the author's LightGBM whiff models or his
Savant-calibrated outputs. Whiff X/Z here use his published in-sample linear
candidates from reports/discovery_2025.json; Y uses the reported 2025 formula.

Example:
  python reconstruct_statcast.py \
      --input data/statcast_2025.parquet \
      --repo external/statcast-contact-point \
      --season 2025 \
      --output data/statcast_2025_deviations.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_NUMERIC = [
    "bat_speed", "launch_speed", "launch_angle", "miss_distance",
    "attack_angle", "attack_direction", "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z",
    "sz_top", "sz_bot",
]
REQUIRED_CATEGORICAL = ["stand", "description"]


def read_statcast(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError("Input must be .parquet, .csv, .pkl, or .pickle")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Existing pybaseball pitch-level export")
    p.add_argument("--repo", type=Path, required=True, help="Clone of jmaschino56/statcast-contact-point")
    p.add_argument("--season", type=int, required=True, help="One season at a time")
    p.add_argument("--output", type=Path, required=True, help="Output .parquet file")
    args = p.parse_args()

    root = args.repo.expanduser().resolve()
    report_path = root / "reports" / "discovery_2025.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing public fitted coefficients: {report_path}")
    if args.output.suffix.lower() != ".parquet":
        raise ValueError("Output must be a .parquet path")

    # Use the upstream author's implementation rather than recoding its formulas.
    sys.path.insert(0, str(root))
    from cp_lib import discover, geometry, join, reconstruct

    d = read_statcast(args.input)
    if d.empty:
        raise ValueError("Input has no rows")
    missing = sorted((set(REQUIRED_NUMERIC) | set(REQUIRED_CATEGORICAL)) - set(d.columns))
    if missing:
        raise ValueError("Input is missing fields:\n  " + "\n  ".join(missing))

    if "game_year" in d.columns:
        years = pd.to_numeric(d["game_year"], errors="coerce").dropna().unique().tolist()
        if years and (len(years) > 1 or int(years[0]) != args.season):
            raise ValueError(f"Expected only {args.season}; found game_year values {years}")
    if "game_type" in d.columns:
        d = d[d["game_type"].eq("R")].copy()

    for col in REQUIRED_NUMERIC:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype("float64")
    d = join.derive_columns(d, args.season)
    d = d[join.swing_filter(d, "bat_speed_nobunt")].reset_index(drop=True)
    if d.empty:
        raise ValueError("No tracked non-bunt swings after filtering")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    specs = {}
    for ax, verdict in report["verdicts"].items():
        b = verdict["best"]
        specs[ax] = reconstruct.FormulaSpec(
            status=verdict["status"],
            all_swing=b["all_swing"],
            coef=b["coef"],
            intercept=b["intercept"],
            by_stand=b["by_stand"],
            stand_models=b["stand_models"],
        )

    # Original contact-outcome inversions and recovered timing formula.
    out = reconstruct.reconstruct(
        d, args.season, specs,
        y_com_ft=float(report["y_com_ft"]),
        tilt_sign=float(report["tilt_sign"]),
    )

    # On whiffs, the original implementation expects trained LightGBM models
    # that are NOT released. Instead, use published best linear tail fits.
    features = geometry.candidate_features(
        d, args.season,
        y_com_ft=float(report["y_com_ft"]),
        tilt_sign=float(report["tilt_sign"]),
    )
    features["stand"] = d["stand"].to_numpy()
    # Essential: the published Z model uses miss_distance *directly* as a feature.
    # geometry.candidate_features creates z_from_miss but does not add miss_distance.
    features["miss_distance"] = d["miss_distance"].to_numpy(dtype=float)
    whiff = d["contact_type"].eq("whiff").to_numpy()
    for ax in ("x", "z"):
        best = discover.CandidateResult(**report["verdicts"][ax]["best"])
        estimate = discover.apply(best, features)
        good = whiff & np.isfinite(estimate)
        out.loc[good, f"{ax}_hat"] = estimate[good]
        out.loc[good, f"source_{ax}"] = "published_tail_linear"
        out.loc[whiff & ~good, f"source_{ax}"] = "unavailable"

    # Derive the three categorical bins from the published raw estimates.
    for ax in ("x", "y", "z"):
        out[f"{ax}_category_raw"] = discover.label_of(ax, out[f"{ax}_hat"].to_numpy(float))

    export = pd.concat([d, out], axis=1)
    export["reconstruction_version"] = "raw_author_contact_plus_published_tail_linear"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export.to_parquet(args.output, index=False)

    print(f"Rows exported: {len(export):,} -> {args.output}")
    print("\nCoverage (fraction with non-null raw estimate) by swing outcome:")
    report_cov = (
        export.assign(
            has_x=export["x_hat"].notna().astype(float),
            has_y=export["y_hat"].notna().astype(float),
            has_z=export["z_hat"].notna().astype(float),
        )
        .groupby("contact_type", dropna=False)
        .agg(swings=("contact_type", "size"), x_coverage=("has_x", "mean"),
             y_coverage=("has_y", "mean"), z_coverage=("has_z", "mean"))
    )
    print(report_cov.round(3).to_string())
    for ax in ("x", "z"):
        print(f"\n{ax.upper()} estimate sources:")
        print(export[f"source_{ax}"].value_counts(dropna=False).to_string())
    print("\nWARNING: contact X sign uses the input DataFrame's median intercept X; ")
    print("small or filtered samples do not reproduce a full-season league-wide sign decision.")
    print("WARNING: X/Z raw outputs are not Savant's exact physical contact coordinates.")
    print("WARNING: no Savant quantile calibration or LightGBM whiff predictions applied.")


if __name__ == "__main__":
    main()
