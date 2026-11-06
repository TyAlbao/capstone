from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ROOT, UPSTREAM, OUT, discover, geometry, join, reconstruct
SOURCE = ROOT / "data/savant_whiff_tails/whiff_tail_matched.parquet"

FULL = ROOT / "data/statcast_2024_2026.parquet"

DISCOVERY = UPSTREAM / "reports/discovery_2025.json"

AXES = ("x", "y", "z")

CAT = {"x": "timing_x_tiedupflail", "y": "timing_y_earlylate", "z": "timing_z_overunder"}

UNITS = {"x": "inches", "y": "ms", "z": "inches"}

INPUTS = ["game_year", "stand", "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z",
          "attack_angle", "attack_direction", "swing_path_tilt", "bat_speed", "sz_top", "sz_bot",
          "intercept_ball_minus_batter_pos_x_inches", "intercept_ball_minus_batter_pos_y_inches", "miss_distance"]

GROUPS = {"FB": ["FF", "SI", "FC", "FA"], "BB": ["SL", "ST", "CU", "KC", "SV", "CS", "SC"],
          "CH": ["CH", "FS", "FO", "EP"]}

PITCH_MAP = {p: g for g, ps in GROUPS.items() for p in ps}

PRIMARY = {"x": "x_full_geometry", "y": "icpt_y_over_ballspeed", "z": "z_miss_decomp"}

PROXIES = {"x": {"icpt_x_body_proxy": "icpt_x_body", "along_bat_proxy": "along_bat_from_com"},
           "y": {}, "z": {"z_above_plane_raw": "z_above_plane", "z_from_miss_raw": "z_from_miss"}}

SEASON_ROLE = {2024: "backward_transfer", 2025: "development_reproduction", 2026: "forward_transfer"}

SEED = 20261104

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")

def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def extract_statcast(frame):
    missing = [c for c in INPUTS if f"statcast_{c}" not in frame]
    if missing:
        raise ValueError(f"Missing prefixed input columns: {missing}")
    return frame[[f"statcast_{c}" for c in INPUTS]].rename(columns=lambda c: c.removeprefix("statcast_")).copy()

def formulas():
    return reconstruct.load_verdicts(DISCOVERY)

def predict(frame):
    """Statcast-only prediction boundary: ignores all truth/provenance columns."""
    d = extract_statcast(frame)
    known_hand = d.stand.isin(["R", "L"])
    d["stand"] = d.stand.fillna("Unknown")
    specs, y_com, tilt = formulas()
    out = pd.DataFrame(index=frame.index)
    for year, g in d.groupby("game_year"):
        f = geometry.candidate_features(g, int(year), y_com_ft=y_com, tilt_sign=tilt)
        f["stand"] = g.stand
        f["miss_distance"] = g.miss_distance
        for ax in AXES:
            out.loc[g.index, f"pred_{ax}_{PRIMARY[ax]}"] = reconstruct._apply_formula(specs[ax], f)
            for name, field in PROXIES[ax].items():
                out.loc[g.index, f"pred_{ax}_{name}"] = f[field]
    out.loc[~known_hand] = np.nan
    return out

def boundary(axis, values):
    v = np.asarray(values, float)
    bounds = {"x": (-24, 36), "z": (-18, 12)}.get(axis, ())
    return np.logical_or.reduce([np.isclose(v, b, rtol=0, atol=1e-8) for b in bounds]) if bounds else np.zeros(len(v), bool)

def fit_baselines(dev):
    if set(dev.season.unique()) != {2025}:
        raise ValueError("Baselines must be fitted on 2025 only")
    out = {"fit_season": 2025, "axes": {}}
    for ax in AXES:
        g = dev.loc[dev[f"common_{ax}"]]
        if g.empty:
            raise ValueError(f"No eligible development observations for {ax}")
        y = g[f"sav_{ax}"].to_numpy(float)
        counts = g[CAT[ax]].value_counts().reindex(discover.LABELS[ax], fill_value=0)
        out["axes"][ax] = dict(n=len(g), mean=float(y.mean()), median=float(np.median(y)),
                               majority=str(counts.idxmax()), category_counts=counts.to_dict())
    return out

def numerical_metrics(actual, predicted, actual_categories, axis):
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    good = np.isfinite(y) & np.isfinite(p)
    y, p = y[good], p[good]
    cats = np.asarray(actual_categories)[good]
    result = dict(n=int(good.sum()), mae=np.nan, rmse=np.nan, r2=np.nan, pearson_r=np.nan,
                  bias=np.nan, sign_accuracy=np.nan, category_accuracy=np.nan,
                  actual_on_pred_slope=np.nan, actual_on_pred_intercept=np.nan)
    if not len(y):
        return result
    e = p - y
    result.update(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()),
                  sign_accuracy=float(np.mean(np.sign(y) == np.sign(p))),
                  category_accuracy=float(np.mean(discover.label_of(axis, p) == cats)))
    if np.var(y) > 0:
        result["r2"] = float(1 - np.sum(e**2) / np.sum((y - y.mean())**2))
    if np.var(p) > 1e-20 and np.var(y) > 1e-20:
        slope = float(np.mean((p-p.mean())*(y-y.mean())) / np.var(p))
        result.update(pearson_r=float(np.corrcoef(p, y)[0, 1]), actual_on_pred_slope=slope,
                      actual_on_pred_intercept=float(y.mean()-slope*p.mean()))
    return result

def methods(axis):
    return [PRIMARY[axis], *PROXIES[axis], "dev_mean", "dev_median", "dev_majority"]

def score(g, ax, method, params):
    total = len(g)
    g = g.loc[g[f"common_{ax}"]]
    if method == "dev_majority":
        r = numerical_metrics([], [], [], ax)
        r["category_accuracy"] = float(g[CAT[ax]].eq(params["axes"][ax]["majority"]).mean()) if len(g) else np.nan
        r["n"] = len(g)
    else:
        r = numerical_metrics(g[f"sav_{ax}"], g[f"pred_{ax}_{method}"], g[CAT[ax]], ax)
    r.update(axis=ax, method=method, stage="baseline" if method.startswith("dev_") else "raw",
             units=UNITS[ax], n_total=total, n_excluded=total-len(g), batters=int(g.batter.nunique()),
             pitchers=int(g.pitcher.nunique()), low_sample=len(g)<100 or g.batter.nunique()<10)
    return r

def table(d, params, group_columns=()):
    rows = []
    groups = d.groupby(list(group_columns), observed=True, dropna=False, sort=True) if group_columns else [((), d)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        tags = dict(zip(group_columns, key))
        for ax in AXES:
            for method in methods(ax):
                rows.append({**tags, **score(g, ax, method, params)})
    return pd.DataFrame(rows)

def validate_source(raw):
    if not raw.play_id.is_unique or len(raw) != 63092:
        raise ValueError("Unexpected source identities or row count")
    if raw.groupby("season").size().to_dict() != {2024:20549, 2025:21313, 2026:21230}:
        raise ValueError("Unexpected seasonal population")
    if not raw.valid_match.all() or not raw.season.eq(raw.statcast_game_year).all():
        raise ValueError("Invalid matches or season disagreement")
    for col in ("batter", "pitcher", "game_pk"):
        if not raw[col].eq(raw[f"statcast_{col}"]).all():
            raise ValueError(f"Identifier disagreement: {col}")

def prepare():
    raw = pd.read_parquet(SOURCE)
    validate_source(raw)
    cols = ["play_id", "season", "game_pk", "game_date", "batter", "pitcher", "sav_x", "sav_y", "sav_z",
            *CAT.values(), "selected_by_json", "lists", "n_selections", "n_selection_lists", "candidate_cap_flag"]
    d = raw[cols].copy()
    for c in ("stand", "p_throws", "pitch_type", "pitch_group", "miss_distance", "plate_x", "plate_z", "sz_top", "sz_bot",
              "at_bat_number", "pitch_number"):
        d[c] = raw[f"statcast_{c}"]
    d["savant_pitch_type"] = raw.pitch_type
    d["statcast_pitch_type"] = raw.statcast_pitch_type
    expected_group = d.pitch_type.map(PITCH_MAP)
    if not expected_group.fillna("Unknown").eq(d.pitch_group.fillna("Unknown")).all():
        raise ValueError("Existing pitch groups disagree with the inspected notebook mapping")
    d["pitch_group"] = expected_group.fillna("Unknown")
    d["savant_pitch_group"] = d.savant_pitch_type.map(PITCH_MAP).fillna("Unknown")
    d["pitch_group_disagreement"] = d.pitch_group.ne(d.savant_pitch_group)
    d["pitch_type_disagreement"] = d.pitch_type.ne(d.savant_pitch_type)
    d["season_role"] = d.season.map(SEASON_ROLE)
    pred = predict(raw)
    d = pd.concat([d, pred], axis=1)
    availability = []
    for ax in AXES:
        cols = [f"pred_{ax}_{m}" for m in methods(ax) if not m.startswith("dev_")]
        d[f"common_{ax}"] = np.isfinite(d[[f"sav_{ax}", *cols]].to_numpy(float)).all(axis=1)
        d[f"boundary_{ax}"] = boundary(ax, d[f"sav_{ax}"])
        if not (discover.label_of(ax, d[f"sav_{ax}"]) == d[CAT[ax]].to_numpy()).all():
            raise ValueError(f"Published category inconsistency on {ax}")
        for year, g in d.groupby("season"):
            for col in cols:
                availability.append(dict(season=year, axis=ax, method=col, rows=len(g), finite=int(np.isfinite(g[col]).sum()), common=int(g[f"common_{ax}"].sum())))
    params = fit_baselines(d.loc[d.season.eq(2025)])
    for ax in AXES:
        for method, stat in (("dev_mean", "mean"), ("dev_median", "median")):
            d[f"pred_{ax}_{method}"] = params["axes"][ax][stat]
        for method in methods(ax):
            if method == "dev_majority":
                d[f"category_{ax}_{method}"] = params["axes"][ax]["majority"]
            else:
                p = d[f"pred_{ax}_{method}"]
                d[f"error_{ax}_{method}"] = p - d[f"sav_{ax}"]
                d[f"category_{ax}_{method}"] = discover.label_of(ax, p)
    selections = pd.read_parquet(SOURCE.parent / "tail_selections.parquet")
    selections = selections.loc[selections.play_id.isin(d.play_id)]
    for label in ("miss_distance", "x", "y", "z"):
        ids = selections.loc[selections["list"].eq(label), "play_id"]
        d[f"selected_{label}"] = d.play_id.isin(ids)
    for perspective in ("batter", "pitcher"):
        ids = selections.loc[selections.perspective.eq(perspective), "play_id"]
        d[f"selected_{perspective}"] = d.play_id.isin(ids)
    nlists = selections.groupby("play_id")["list"].nunique()
    if not d.n_selection_lists.eq(d.play_id.map(nlists)).all():
        raise ValueError("Selection provenance inconsistency")
    d["n_perspective_lists"] = d.play_id.map(selections.drop_duplicates(["play_id", "perspective", "list"]).groupby("play_id").size())
    d["perspective_group"] = np.select([d.selected_batter & d.selected_pitcher, d.selected_batter], ["both", "batter_only"], default="pitcher_only")
    d["miss_bin"] = pd.cut(d.miss_distance, [0,1,2,4,8,16,np.inf], right=False,
                            labels=["0-1", "1-2", "2-4", "4-8", "8-16", "16+"])
    d["horizontal_bin"] = pd.cut(d.plate_x, [-np.inf,-.83,-.3,.3,.83,np.inf], right=False,
                                  labels=["far_negative", "negative", "middle", "positive", "far_positive"])
    d["zone_z"] = (d.plate_z-d.sz_bot)/(d.sz_top-d.sz_bot).where(d.sz_top.gt(d.sz_bot))
    d["vertical_bin"] = pd.cut(d.zone_z, [-np.inf,0,1/3,2/3,1,np.inf], right=False,
                                labels=["below_zone", "lower_third", "middle_third", "upper_third", "above_zone"])
    return d, params


def diagnostics(d, params):
    tables = {}
    overall = table(d, params)
    overall['population'] = 'pooled_mixed_development_and_transfer_not_independent'
    tables['whiff_overall_metrics'] = overall
    seasonal = table(d, params, ['season', 'season_role'])
    tables['whiff_season_metrics'] = seasonal
    tables['whiff_pitch_group_metrics'] = table(d, params, ['season', 'pitch_group'])
    strata = []
    for field in ('stand', 'p_throws', 'horizontal_bin', 'vertical_bin', 'miss_bin', 'n_selection_lists', 'n_perspective_lists', 'perspective_group'):
        t = table(d, params, ['season', field]).rename(columns={field: 'stratum'})
        t['stratum_type'] = field
        strata.append(t)
    tables['whiff_stratified_metrics'] = pd.concat(strata, ignore_index=True)
    selection = []
    for name in ('x', 'y', 'z', 'miss_distance', 'batter', 'pitcher'):
        t = table(d.loc[d[f'selected_{name}']], params, ['season'])
        t['selection'] = name
        t['overlapping'] = True
        selection.append(t)
    tables['whiff_selection_bias_metrics'] = pd.concat(selection, ignore_index=True)
    cap_rows = []
    for year, g in d.groupby('season'):
        for ax in ('x', 'z'):
            flagged = g[f'boundary_{ax}']
            for condition, sub in (('all', g), ('exclude_axis_boundary', g.loc[~flagged]), ('axis_boundary_only', g.loc[flagged])):
                for method in methods(ax):
                    cap_rows.append(dict(season=year, condition=condition, excluded=int(len(g) - len(sub)), boundary_total=int(flagged.sum()), **score(sub, ax, method, params)))
    tables['whiff_boundary_sensitivity'] = pd.DataFrame(cap_rows)
    confusion = []
    for year, g in d.groupby('season'):
        for ax in AXES:
            gax = g.loc[g[f'common_{ax}']]
            for method in methods(ax):
                counts = pd.crosstab(gax[CAT[ax]], gax[f'category_{ax}_{method}']).reindex(index=discover.LABELS[ax], columns=discover.LABELS[ax], fill_value=0)
                for actual in discover.LABELS[ax]:
                    for predicted in discover.LABELS[ax]:
                        confusion.append(dict(season=year, axis=ax, method=method, actual=actual, predicted=predicted, n=int(counts.loc[actual, predicted])))
    tables['whiff_confusion_matrices'] = pd.DataFrame(confusion)
    tables['whiff_pitch_type_group_audit'] = d.groupby(['season', 'statcast_pitch_type', 'savant_pitch_type', 'pitch_group', 'savant_pitch_group'], dropna=False).size().rename('n').reset_index()
    player_rows = []
    for key in ('batter', 'pitcher'):
        for (year, pid), g in d.groupby(['season', key]):
            for ax in AXES:
                r = score(g, ax, PRIMARY[ax], params)
                player_rows.append(dict(season=year, perspective=key, player_id=pid, eligible_player=len(g) >= 20, **r))
    players = pd.DataFrame(player_rows)
    tables['whiff_player_metrics'] = players
    summaries = []
    for keys, g in players.groupby(['season', 'perspective', 'axis']):
        use = g.loc[g.eligible_player]
        summaries.append(dict(zip(['season', 'perspective', 'axis'], keys)) | dict(players=len(g), eligible_players=len(use), macro_mae=float(use.mae.mean()), median_player_mae=float(use.mae.median()), p90_player_mae=float(use.mae.quantile(0.9)), macro_category_accuracy=float(use.category_accuracy.mean())))
    tables['whiff_player_summary'] = pd.DataFrame(summaries)
    return tables
