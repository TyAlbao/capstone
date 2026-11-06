from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ROOT, UPSTREAM, calibrate, reconstruct, validate
from .references import Config, KEEP, X, bootstrap_difference, calculate_references, predict_contact, prepare_season, original_swing_descriptions
OUT=ROOT/"data/savant_calibration"
DEVELOPMENT = 2025

METHODS = ("global", "batter", "ev50")

XCOLS = ("tied_up_percent", "centered_percent", "flailed_percent")

MEANS = ("avg_x_tied_up", "avg_x_flail")

CONFIG = Config()

def load_season(season):
    path = ROOT / "data/statcast_2024_2026.parquet"
    columns = pq.ParquetFile(path).schema.names
    # Do not use the legacy, cross-season normalized columns as predictors.
    keep = [c for c in KEEP if c in columns and not c.startswith("relative_intercept")]
    raw = pd.read_parquet(path, columns=keep, filters=[("game_year", "==", season)])
    raw["source_index"] = raw.index
    s = prepare_season(raw, season)
    s = s.loc[s.description.isin(original_swing_descriptions())].reset_index(drop=True)
    return s

def official(season, kind, contact_type):
    code = {"in_play":2, "foul":4, "whiff":9}[contact_type]
    path = OUT / f"{kind}_split_{season}.csv"
    frame = pd.read_csv(path)
    if not frame.year.eq(season).all(): raise ValueError("Wrong official season")
    d = frame.loc[frame.bat_contact_code.eq(code)].copy()
    if d.empty or d.id.isna().any() or d.id.duplicated().any(): raise ValueError("Invalid player IDs")
    values = d[list(XCOLS)].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError("Missing official rates")
    if np.allclose(values.sum(axis=1), 100, atol=.005):
        d[list(XCOLS)] /= 100
    elif not np.allclose(values.sum(axis=1), 1, atol=.0001):
        raise ValueError("Unexpected rate units or sums")
    if (d[list(XCOLS)].lt(0) | d[list(XCOLS)].gt(1)).any().any(): raise ValueError("Invalid rates")
    if not (d.n_swings.gt(0) & d.n_swings.eq(np.floor(d.n_swings))).all(): raise ValueError("Invalid denominators")
    d["id"] = d.id.astype(int)
    return d

def fit_maps(pred, swings, hist, development_truth):
    """Only development rows, development histograms, and development rates accepted."""
    if not swings.season.eq(DEVELOPMENT).all() or swings.empty:
        raise ValueError("Calibration predictions must be development-only")
    if not hist.season.eq(DEVELOPMENT).all(): raise ValueError("Holdout histogram forbidden")
    if not development_truth.year.eq(DEVELOPMENT).all(): raise ValueError("Holdout rates forbidden")
    if not pred.index.equals(swings.index): raise ValueError("Prediction row alignment mismatch")
    rates = {}
    for ct, code in (("in_play",2),("foul",4)):
        d = development_truth.loc[development_truth.bat_contact_code.eq(code)]
        if d.empty: continue
        rates[("x",ct)] = dict(zip(("lo","mid","hi"), np.average(d[list(XCOLS)], weights=d.n_swings, axis=0)))
    # Original upstream function; x-only histogram means y/z are skipped.
    return calibrate.fit_calibration(pred[["x_hat"]], swings, DEVELOPMENT,
        type_="pitcher", hist=hist, per_contact_type=True, rates=rates)

def apply_maps(values, swings, maps):
    p = pd.DataFrame({"x_hat":np.asarray(values,float), "y_hat":np.nan, "z_hat":np.nan}, index=swings.index)
    return calibrate.apply_calibration(p, swings, maps).x_cal.to_numpy()

def serialize_maps(maps):
    return {f"{axis}:{ct}":dict(src=m.src.tolist(), dst=m.dst.tolist()) for (axis,ct),m in maps.items()}

def map_fingerprint(maps):
    return hashlib.sha256(json.dumps(serialize_maps(maps), sort_keys=True).encode()).hexdigest()

def x_player_rates(swings, values, kind):
    """Use upstream categorization/aggregation; only X fields leave this wrapper."""
    cal = pd.DataFrame({"x_cal":np.asarray(values,float), "y_cal":np.nan, "z_cal":np.nan}, index=swings.index)
    cats = validate.categorize(cal, swings)
    r = validate.player_rates(swings, cats, kind, cal)
    return r[["id", "n_swings", *XCOLS, *MEANS, "unknown_x_percent"]]

def eligible_ids(swings, tracked, truth, kind, matched=True):
    n = swings.groupby(kind).size().rename("evaluated_n")
    tracked_n = tracked.groupby(kind).size().rename("tracked_n")
    d = truth[["id", "n_swings"]].merge(n, left_on="id", right_index=True, how="outer").merge(tracked_n, left_on="id", right_index=True, how="outer")
    d[["evaluated_n","tracked_n"]] = d[["evaluated_n","tracked_n"]].fillna(0).astype(int)
    d["evaluation_mismatch"] = (d.evaluated_n-d.n_swings).abs()/d.n_swings
    d["tracked_mismatch"] = (d.tracked_n-d.n_swings).abs()/d.n_swings
    d["qualifies"] = d.n_swings.ge(50) & d.evaluated_n.gt(0)
    if matched:
        d["qualifies"] &= d.evaluation_mismatch.le(.10) & d.tracked_mismatch.le(.10)
    return d

def compare(predictions, swings, tracked, truth, kind, metadata, matched=True, null_rates=None):
    """Every method is evaluated on exactly the same rows and player IDs."""
    den = eligible_ids(swings, tracked, truth, kind, matched)
    ids = den.loc[den.qualifies, "id"]
    truth = truth.loc[truth.id.isin(ids)]
    per_method, detailed, metrics = {}, [], []
    for method, values in predictions.items():
        ours = x_player_rates(swings, values, kind)
        per_method[method] = ours.loc[ours.id.isin(ids)].copy()
    if null_rates is not None:
        null = per_method["global"].copy()
        for col, rate in zip(XCOLS, null_rates): null[col] = rate
        null[list(MEANS)] = np.nan
        per_method["development_rate_null"] = null
    paired = []
    for method, ours in per_method.items():
        # This is the original comparison routine, including official n >= 50.
        with np.errstate(invalid="ignore", divide="ignore"):
            table = validate.compare_rates(ours, truth, min_swings=50)
        if table.empty: continue
        for col in XCOLS:
            if ours[col].nunique() < 2 or truth[col].nunique() < 2:
                table.loc[table.rate.eq(col), "pearson_r"] = np.nan
        table = table.loc[table.rate.isin((*XCOLS, *MEANS))].copy()
        table["mae_pp"] = np.where(table.rate.isin(XCOLS), table.mae*100, np.nan)
        detailed.append(table.assign(method=method, perspective=kind, **metadata))
        joined = ours.merge(truth[["id","name","n_swings",*XCOLS]], on="id", suffixes=("_ours","_official"), validate="one_to_one")
        joined["error_pp"] = np.mean(np.abs(joined[[c+"_ours" for c in XCOLS]].to_numpy()-joined[[c+"_official" for c in XCOLS]].to_numpy()), axis=1)*100
        joined["method"] = method
        paired.append(joined)
        core = table.set_index("rate")
        row = dict(**metadata, perspective=kind, method=method, players=len(joined), swings=int(ours.n_swings.sum()),
            combined_mae_pp=float(core.loc[list(XCOLS), "mae"].mean()*100),
            unknown_percent=float(np.average(ours.unknown_x_percent, weights=ours.n_swings)*100))
        for label, col in zip(("tied_up","centered","flail"), XCOLS):
            row[label+"_mae_pp"] = float(core.loc[col,"mae"]*100)
            row[label+"_pearson"] = core.loc[col,"pearson_r"]
        metrics.append(row)
    if not paired:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), den
    pairs = pd.concat(paired, ignore_index=True)
    reference = "global" if "global" in per_method else next(iter(per_method))
    baseline = pairs.loc[pairs.method.eq(reference), ["id","error_pp"]].rename(columns={"error_pp":"global_error_pp"})
    pairs = pairs.merge(baseline,on="id",validate="many_to_one")
    pairs["delta_mae_pp"] = pairs.error_pp-pairs.global_error_pp
    for row in metrics:
        delta = pairs.loc[pairs.method.eq(row["method"]),"delta_mae_pp"]
        row["delta_mae_pp"] = float(delta.mean())
        row["delta_ci_low"],row["delta_ci_high"] = bootstrap_difference(delta,CONFIG)
    pairs = pairs.assign(perspective=kind, **metadata)
    return pd.DataFrame(metrics), pd.concat(detailed,ignore_index=True), pairs, den

def references_for_target(dev_refs, target):
    """Unseen hitters get identified fallbacks; primary comparisons exclude them."""
    refs = dev_refs.reindex(dev_refs.index.union(pd.Index(target.batter.dropna().unique(),name="batter"))).copy()
    global_ref = float(dev_refs.ref_global.iloc[0])
    refs["ref_global"] = global_ref
    for m in ("batter","ev50"):
        refs[f"ref_{m}"] = refs[f"ref_{m}"].fillna(global_ref)
        refs[f"fallback_{m}"] = refs[f"fallback_{m}"].fillna(True).astype(bool)
    return refs

def common_predictions(swings, refs):
    return predict_contact(swings, refs, methods=METHODS)

def fit_comparison(swings, refs, hist, truth):
    p = common_predictions(swings, refs)
    finite = np.isfinite(p[[f"x_{m}" for m in METHODS]]).all(axis=1)
    p = p.loc[finite].reset_index(drop=True)
    maps = {}
    for m in METHODS:
        maps[m] = fit_maps(pd.DataFrame({"x_hat":p[f"x_{m}"]}), p, hist, truth)
    return maps, p
