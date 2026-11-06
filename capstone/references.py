from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
import datetime as dt
import requests
from .data import ROOT, OUT, UPSTREAM, geometry, join, reconstruct, dump as dump_json
X="intercept_ball_minus_batter_pos_x_inches"
Y="intercept_ball_minus_batter_pos_y_inches"
METHODS = ("global", "batter", "ev50", "existing")

CATEGORIES = ("tied_up", "centered", "flail")

NUMERIC = ["bat_speed", "launch_speed", "launch_angle", "miss_distance", X, Y,
           "attack_angle", "attack_direction", "swing_path_tilt", "vx0", "vy0", "vz0",
           "ax", "ay", "az", "plate_x", "plate_z", "sz_top", "sz_bot"]

KEEP = ["game_year", "game_date", "game_type", "game_pk", "at_bat_number", "pitch_number",
        "batter", "pitcher", "stand", "p_throws", "pitch_type", "pitch_name", "pitch_group",
        "batter_name", "pitcher_name", "description", "is_swing", "relative_intercept_x",
        "relative_intercept_y", *NUMERIC]

@dataclass(frozen=True)
class Config:
    min_batter_swings: int = 50
    min_ev50_bip: int = 50
    denominator_tolerance: float = 0.10
    bootstrap_reps: int = 5000
    seed: int = 180

    def __post_init__(self):
        if min(self.min_batter_swings, self.min_ev50_bip, self.bootstrap_reps) < 1:
            raise ValueError("Sample thresholds and bootstrap reps must be positive")
        if not 0 <= self.denominator_tolerance < 1:
            raise ValueError("Denominator tolerance must lie in [0, 1)")

def prepare_season(data, season):
    """Author's exact bat_speed_nobunt rule, within one regular season."""
    d = data.loc[data.game_year.eq(season) & data.game_type.eq("R")].copy()
    for col in NUMERIC:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d = join.derive_columns(d, season)
    return d.loc[join.swing_filter(d, "bat_speed_nobunt")].reset_index(drop=True)

def calculate_references(swings, config=Config()):
    """Season-specific full tracked-swing references, with explicit global fallbacks.

    EV50 eligibility counts BIP with finite EV and X BEFORE upper-half selection.
    Existing relative columns are untouched; their implied pooled reference is
    exposed only as a separately labeled historical sensitivity comparison.
    """
    if swings.season.nunique() != 1:
        raise ValueError("References require exactly one season")
    valid_x = np.isfinite(swings[X])
    pool = swings.loc[valid_x & swings.batter.notna()]
    # Global includes every finite-X tracked swing, even if batter ID is missing.
    global_ref = float(swings.loc[valid_x, X].median())
    refs = pool.groupby("batter").agg(ref_batter=(X, "median"), n_batter=(X, "size"))
    bip = pool.loc[pool.description.eq("hit_into_play") & np.isfinite(pool.launch_speed)]
    med = bip.groupby("batter").launch_speed.transform("median")
    upper = bip.loc[bip.launch_speed.ge(med)]
    refs = refs.join(bip.groupby("batter").size().rename("n_ev50_bip"))
    refs = refs.join(upper.groupby("batter").agg(ref_ev50=(X, "median"), n_ev50_selected=(X, "size")))
    refs["n_ev50_bip"] = refs.n_ev50_bip.fillna(0).astype(int)
    refs["n_ev50_selected"] = refs.n_ev50_selected.fillna(0).astype(int)
    refs["fallback_batter"] = refs.n_batter.lt(config.min_batter_swings)
    refs["fallback_ev50"] = refs.n_ev50_bip.lt(config.min_ev50_bip) | refs.ref_ev50.isna()
    for method in ("batter", "ev50"):
        refs.loc[refs[f"fallback_{method}"], f"ref_{method}"] = global_ref
    refs["ref_global"] = global_ref
    refs["season"] = int(swings.season.iloc[0])
    if "relative_intercept_x" in swings:
        implied = (swings[X] - swings.relative_intercept_x).replace([np.inf, -np.inf], np.nan)
        check = implied.groupby(swings.batter).agg(["min", "max", "mean"])
        if ((check["max"] - check["min"]) > 1e-7).any():
            raise ValueError("Stored relative X is not a constant batter reference")
        refs["ref_existing"] = check["mean"]
    return refs

def predict_contact(swings, refs, methods=METHODS):
    """B1 selects each original curve; B2 holds abs(x_global) exactly fixed."""
    out = swings.loc[swings.description.eq("hit_into_play")].copy()
    if out.empty:
        return out
    for col in refs.columns:
        if col != "season":
            out[col] = out.batter.map(refs[col])
    params = json.loads((ROOT / "external/statcast-contact-point/reports/discovery_2025.json").read_text())
    with np.errstate(invalid="ignore", divide="ignore"):
        feats = geometry.candidate_features(out, int(out.season.iloc[0]),
            y_com_ft=float(params["y_com_ft"]), tilt_sign=float(params["tilt_sign"]))
    out["ball_speed_mph"] = feats.ball_speed_mph
    out["ev_max"] = reconstruct.ev_ceiling(out.bat_speed, out.ball_speed_mph)
    out["ev_deficit"] = out.ev_max - out.launch_speed
    valid = np.isfinite(out[[X, "bat_speed", "ball_speed_mph", "launch_speed", "ev_deficit"]]).all(axis=1)
    for method in methods:
        ok = valid & np.isfinite(out[f"ref_{method}"])
        handle = out[X].lt(out[f"ref_{method}"])
        out[f"toward_handle_{method}"] = handle.where(ok).astype("boolean")
        values = np.full(len(out), np.nan)
        values[ok.to_numpy(dtype=bool, na_value=False)] = reconstruct.x_from_ev_deficit(out.loc[ok, "ev_deficit"], handle.loc[ok])
        out[f"x_{method}"] = values
    for method in methods:
        handle = out[X].lt(out[f"ref_{method}"]).fillna(False).to_numpy(dtype=bool)
        out[f"x_{method}_sign_only"] = np.where(handle, -1, 1) * out.x_global.abs()
        out.loc[out[f"x_{method}"].isna(), f"x_{method}_sign_only"] = np.nan
    out["valid_outcome"] = valid
    out["primary_eligible"] = (valid & ~out.fallback_batter.fillna(True).astype(bool)
        & ~out.fallback_ev50.fillna(True).astype(bool)
        & np.isfinite(out[[f"x_{m}" for m in methods]]).all(axis=1))
    common = out.loc[out.primary_eligible]
    for method in methods:
        np.testing.assert_array_equal(common.x_global.abs(), common[f"x_{method}_sign_only"].abs())
    return out

def category_rates(pred, methods=METHODS):
    """One shared, complete set of swing rows for every method and experiment."""
    if pred.empty:
        return pd.DataFrame()
    if not np.isfinite(pred[[f"x_{m}" for m in methods]]).all().all():
        raise ValueError("Missing predictions cannot enter category denominators")
    records = []
    for experiment, suffix in [("B1_faithful", ""), ("B2_sign_only", "_sign_only")]:
        for method in methods:
            v = pred[f"x_{method}{suffix}"]
            d = pd.DataFrame({"batter": pred.batter, "tied_up": v.lt(-4),
                              "centered": v.between(-4, 4), "flail": v.gt(4),
                              "handle_changed": pred[X].lt(pred[f"ref_{method}"]) != pred[X].lt(pred.ref_global)})
            rate = d.groupby("batter").agg(n=("centered", "size"),
                tied_up=("tied_up", "mean"), centered=("centered", "mean"),
                flail=("flail", "mean"), handle_changed=("handle_changed", "mean"))
            rate["season"] = int(pred.season.iloc[0])
            rate["method"] = method
            rate["experiment"] = experiment
            records.append(rate.reset_index())
    return pd.concat(records, ignore_index=True)

def normalize_official(frame, season):
    required = {"id", "year", "bat_contact_code", "n_swings", "tied_up_percent", "centered_percent", "flailed_percent"}
    if not required.issubset(frame):
        raise ValueError("Official response lacks required fields including year")
    if not pd.to_numeric(frame.year, errors="raise").eq(season).all():
        raise ValueError("Endpoint returned the wrong season")
    d = frame.loc[pd.to_numeric(frame.bat_contact_code).eq(2)].copy()
    if d.empty or d.id.duplicated().any():
        raise ValueError("Missing BIP records or duplicate official batter IDs")
    cols = ["tied_up_percent", "centered_percent", "flailed_percent"]
    rates = d[cols].apply(pd.to_numeric, errors="raise")
    totals = rates.sum(axis=1)
    if np.isclose(totals, 1, atol=0.005).all():
        scale = 1
    elif np.isclose(totals, 100, atol=0.5).all():
        scale = 100
    else:
        raise ValueError("Official rate units or sums are invalid/ambiguous")
    rates /= scale
    if not np.isfinite(rates).all().all() or ((rates < 0) | (rates > 1)).any().any():
        raise ValueError("Invalid official rates")
    counts = pd.to_numeric(d.n_swings, errors="raise")
    if not (np.isfinite(counts) & counts.gt(0) & counts.eq(np.floor(counts))).all():
        raise ValueError("Invalid official denominators")
    result = pd.DataFrame({"batter": pd.to_numeric(d.id).astype(int), "official_n": counts.astype(int)})
    for col, cat in zip(cols, CATEGORIES):
        result[f"official_{cat}"] = rates[col]
    if "name" in d:
        result["official_name"] = d.name
    result.attrs["input_scale"] = scale
    return result

def verify_schedule(season, local_games, out, fetch=False):
    """Refuse full-season validation unless all official final R games are present."""
    path = out / "official" / f"schedule_{season}.json"
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R"
    result = dict(season=season, url=url, full_season_verified=False)
    try:
        cached = json.loads(path.read_text()) if path.exists() else {}
        if fetch or not path.exists() or cached.get("schema_version") != 2:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            payload = response.json()
            # Store only provenance and schedule metadata needed for coverage.
            games = [dict(game_pk=g["gamePk"], game_type=g["gameType"],
                          status=g["status"]["abstractGameState"],
                          status_code=g["status"]["codedGameState"],
                          detailed_state=g["status"]["detailedState"], date=day["date"])
                     for day in payload["dates"] for g in day["games"]]
            dump_json(path, dict(schema_version=2, url=url, fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(), games=games))
        cached = json.loads(path.read_text())
        games = cached["games"]
        # MLB also labels canceled games abstractGameState='Final'. They have
        # no pitches and must not be treated as missing played games.
        finished = {g["game_pk"] for g in games if g["game_type"] == "R" and g["status"] == "Final" and g["status_code"] != "C"}
        scheduled = [g for g in games if g["game_type"] == "R" and g["status"] in ("Preview", "Live")]
        local = set(int(g) for g in local_games)
        result.update(official_final_games=len(finished), local_games=len(local),
            missing_games=sorted(finished-local), unexpected_games=sorted(local-finished),
            cancelled_games=[g["game_pk"] for g in games if g["status_code"] == "C"],
            unfinished_games=len(scheduled), fetched_utc=cached["fetched_utc"],
            full_season_verified=bool(finished and not (finished-local) and not (local-finished) and not scheduled))
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        result["reason"] = str(exc)[:500]
    return result

def denominator_audit(swings, pred, official, config, complete):
    counts = swings.loc[swings.description.eq("hit_into_play")].groupby("batter").size().rename("local_tracked_bip")
    evaluated = pred.loc[pred.primary_eligible].groupby("batter").size().rename("evaluated_n")
    # Outer merge keeps official-only and local-only players visible.
    d = official.merge(counts, on="batter", how="outer").merge(evaluated, on="batter", how="outer")
    d[["local_tracked_bip", "evaluated_n"]] = d[["local_tracked_bip", "evaluated_n"]].fillna(0).astype(int)
    d["tracked_mismatch"] = (d.local_tracked_bip-d.official_n).abs()/d.official_n
    d["evaluation_mismatch"] = (d.evaluated_n-d.official_n).abs()/d.official_n
    d["denominators_match"] = (d.tracked_mismatch.le(config.denominator_tolerance)
        & d.evaluation_mismatch.le(config.denominator_tolerance))
    d["eligible"] = bool(complete) & d.denominators_match & d.evaluated_n.gt(0)
    return d

def bootstrap_difference(delta, config):
    a = np.asarray(delta, float)
    if not len(a):
        return np.nan, np.nan
    rng = np.random.default_rng(config.seed)
    boot = np.empty(config.bootstrap_reps)
    for start in range(0, len(boot), 250):
        end = min(start + 250, len(boot))
        boot[start:end] = a[rng.integers(len(a), size=(end-start, len(a)))].mean(axis=1)
    return tuple(np.quantile(boot, [.025, .975]))

def validation_metrics(rates, denominators, config):
    if rates.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = denominators.loc[denominators.eligible]
    joined = rates.merge(eligible, on="batter", how="inner", validate="many_to_one")
    if joined.empty:
        return pd.DataFrame(), joined
    for cat in CATEGORIES:
        joined[f"error_{cat}"] = (joined[cat]-joined[f"official_{cat}"]).abs()*100
    joined["error_combined"] = joined[[f"error_{c}" for c in CATEGORIES]].mean(axis=1)
    base = joined.loc[joined.method.eq("global"), ["experiment", "batter", "error_combined"]].rename(columns={"error_combined": "global_error"})
    joined = joined.merge(base, on=["experiment", "batter"], validate="many_to_one")
    joined["delta_mae_pp"] = joined.error_combined-joined.global_error
    rows = []
    for (experiment, method), d in joined.groupby(["experiment", "method"]):
        row = dict(season=int(d.season.iloc[0]), experiment=experiment, method=method,
            n_batters=len(d), n_swings=int(d.n.sum()), combined_mae=float(d.error_combined.mean()),
            handle_changed_percent=float(np.average(d.handle_changed, weights=d.n)*100),
            delta_mae_pp=float(d.delta_mae_pp.mean()))
        for cat in CATEGORIES:
            row[f"{cat}_mae"] = float(d[f"error_{cat}"].mean())
            row[f"{cat}_pearson"] = (d[cat].corr(d[f"official_{cat}"])
                if d[cat].nunique()>1 and d[f"official_{cat}"].nunique()>1 and len(d)>1 else np.nan)
        row["delta_ci_low"], row["delta_ci_high"] = bootstrap_difference(d.delta_mae_pp, config)
        rows.append(row)
    return pd.DataFrame(rows), joined
def original_swing_descriptions():
    # Parse the literal without importing pull.py (which loads database settings).
    tree = ast.parse((UPSTREAM / "cp_lib/pull.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SWING_DESCRIPTIONS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Upstream pull swing definitions missing")
