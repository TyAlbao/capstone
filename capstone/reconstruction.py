from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ROOT, UPSTREAM, AXES, RATES, PITCH_MAP, reconstruct, geometry, calibrate, discover, join
from .references import KEEP, X, Y, calculate_references, prepare_season, original_swing_descriptions
METHODS = {'x': ['author', 'global', 'batter', 'ev50', 'ev50_mean', 'existing'],
           'y': ['author', 'global', 'batter', 'ev50', 'tango_time', 'existing'], 'z': ['author']}

STAGES = ('raw', 'calibrated')

KEYS = ['season', 'batter', 'contact_type', 'group']

ROLE = {2024: 'backward_transfer', 2025: 'development', 2026: 'forward_transfer'}

SPECS, YCOM, TILT = reconstruct.load_verdicts(UPSTREAM/'reports/discovery_2025.json')

def assert_dev(d):
    if d.empty or not d.season.eq(2025).all(): raise ValueError('Development-only fitting required')

def load_swings():
    columns = pq.ParquetFile(ROOT/'data/statcast_2024_2026.parquet').schema.names
    keep = [c for c in KEEP if c in columns]
    raw = pd.read_parquet(ROOT/'data/statcast_2024_2026.parquet', columns=keep)
    raw['source_index'] = raw.index
    parts = []
    for season in (2024, 2025, 2026):
        s = prepare_season(raw, season)
        s = s[s.description.isin(original_swing_descriptions())].copy()
        s['group'] = s.pitch_type.map(PITCH_MAP).fillna('UNKNOWN')
        parts.append(s)
    d = pd.concat(parts, ignore_index=True)
    if d.duplicated(['game_pk','at_bat_number','pitch_number']).any(): raise ValueError('Duplicate pitch keys')
    d['batter'] = d.batter.astype(int); d['pitcher'] = d.pitcher.astype(int)
    return d

def original_predictions(d):
    out, feats = [], []
    for season, s in d.groupby('season'):
        p = reconstruct.reconstruct(s, int(season), SPECS, YCOM, TILT)
        f = geometry.candidate_features(s, int(season), y_com_ft=YCOM, tilt_sign=TILT)
        f['stand'] = s.stand; f['miss_distance'] = s.miss_distance
        out.append(p); feats.append(f)
    return pd.concat(out).sort_index(), pd.concat(feats).sort_index()

def fit_references(dev, yraw):
    assert_dev(dev)
    r = calculate_references(dev)
    valid_y = np.isfinite(dev[Y])
    gb = dev.loc[valid_y].groupby('batter')
    r['y_batter'] = gb[Y].median(); r['n_y_batter'] = gb.size()
    r['y_global'] = dev.loc[valid_y, Y].median()
    bip = dev[dev.contact_type.eq('in_play') & np.isfinite(dev[X]) & np.isfinite(dev.launch_speed)].copy()
    cut = bip.groupby('batter').launch_speed.transform('median')
    upper = bip[bip.launch_speed.ge(cut)].copy()
    upper['timing_raw'] = yraw.reindex(upper.index)
    r['ref_ev50_mean'] = upper.groupby('batter')[X].mean()
    r['y_ev50'] = upper.groupby('batter')[Y].median()
    r['n_y_ev50'] = bip.loc[np.isfinite(bip[Y])].groupby('batter').size()
    r['time_ev50'] = upper.groupby('batter').timing_raw.median()
    r['time_global'] = upper.timing_raw.median()
    r['fallback_y_batter'] = r.n_y_batter.fillna(0).lt(50) | r.y_batter.isna()
    r['fallback_y_ev50'] = r.n_y_ev50.fillna(0).lt(50) | r.y_ev50.isna() | r.time_ev50.isna()
    r.loc[r.fallback_ev50, 'ref_ev50_mean'] = r.ref_global
    r.loc[r.fallback_y_batter, 'y_batter'] = r.y_global
    r.loc[r.fallback_y_ev50, 'y_ev50'] = r.y_global
    r.loc[r.fallback_y_ev50, 'time_ev50'] = r.time_global
    return r

def raw_predictions(d, author, f, refs):
    p = d[['source_index','game_pk','at_bat_number','pitch_number','game_date','season','batter','pitcher',
           'contact_type','pitch_type','group','stand','p_throws','description']].copy()
    for name in ['bat_speed','launch_speed','launch_angle',X,Y,'relative_intercept_x','relative_intercept_y']:
        p[name] = d[name]
    for name in ('ball_speed_mph','ball_in_per_ms'): p[name] = f[name]
    for axis in AXES:
        p[f'source_{axis}_author'] = author[f'source_{axis}']
        p[f'{axis}_author_raw'] = author[f'{axis}_hat']
    p['has_launch_speed'] = np.isfinite(d.launch_speed)
    p['has_launch_angle'] = np.isfinite(d.launch_angle)
    p['has_intercept_xy'] = np.isfinite(d[[X,Y]]).all(axis=1)
    p['has_trajectory'] = np.isfinite(d[['vx0','vy0','vz0','ax','ay','az','plate_x','plate_z']]).all(axis=1)
    p['unknown_pitch_group'] = p.group.eq('UNKNOWN')
    p['unseen_development_batter'] = ~d.batter.isin(refs.index)
    p['reference_fit_season'] = 2025
    p['author_x_reference_season'] = d.season
    p['legacy_reference_temporal_leakage'] = True
    p['x_batter_reference_source'] = np.where(d.batter.map(refs.fallback_batter).fillna(True),'2025_global_fallback','2025_batter_median')
    p['x_ev50_reference_source'] = np.where(d.batter.map(refs.fallback_ev50).fillna(True),'2025_global_fallback','2025_batter_ev50')
    p['y_ev50_reference_source'] = np.where(d.batter.map(refs.fallback_y_ev50).fillna(True),'2025_global_fallback','2025_batter_ev50')
    global_x = float(refs.ref_global.iloc[0]); global_y = float(refs.y_global.iloc[0])
    for m in ('global','batter','ev50','ev50_mean'):
        p[f'x_reference_{m}'] = d.batter.map(refs[f'ref_{m}']).fillna(global_x)
    p['x_reference_author'] = d.groupby('season')[X].transform('median')
    p['x_reference_existing'] = d[X]-d.relative_intercept_x
    for m in ('batter','ev50'):
        p[f'fallback_x_{m}'] = d.batter.map(refs[f'fallback_{m}']).fillna(True).astype(bool)
    for m in ('global','batter','ev50'):
        p[f'y_reference_{m}'] = d.batter.map(refs[f'y_{m}']).fillna(global_y)
    for m in ('batter','ev50'):
        p[f'fallback_y_{m}'] = d.batter.map(refs[f'fallback_y_{m}']).fillna(True).astype(bool)
    p['y_reference_tango_time_ms'] = d.batter.map(refs.time_ev50).fillna(float(refs.time_global.iloc[0]))
    p['y_reference_existing'] = d[Y]-d.relative_intercept_y
    contact = ~d.contact_type.eq('whiff')
    inversion = contact & author.source_x.eq('inversion')
    deficit = reconstruct.ev_ceiling(d.bat_speed, f.ball_speed_mph)-d.launch_speed
    p['ev_deficit_mph'] = deficit
    p['author_nan_deficit_zero_flag'] = inversion & ~np.isfinite(deficit)
    p['author_missing_x_sign_flag'] = inversion & ~np.isfinite(d[X])
    for m in METHODS['x'][1:]:
        vals = author.x_hat.to_numpy(float).copy()
        ok = inversion & np.isfinite(deficit) & np.isfinite(d[X]) & np.isfinite(p[f'x_reference_{m}'])
        vals[inversion] = np.nan
        vals[ok.to_numpy(dtype=bool, na_value=False)] = reconstruct.x_from_ev_deficit(deficit[ok], d.loc[ok,X].lt(p.loc[ok,f'x_reference_{m}']))
        p[f'x_{m}_raw'] = vals
    slope = d.stand.map({s: v['coef']['icpt_y_over_ballspeed'] for s,v in SPECS['y'].stand_models.items()})
    for m in ('global','batter','ev50'):
        p[f'y_{m}_raw'] = slope*(d[Y]-p[f'y_reference_{m}'])/f.ball_in_per_ms.replace(0,np.nan)
    p['y_existing_raw'] = slope*d.relative_intercept_y/f.ball_in_per_ms.replace(0,np.nan)
    p['y_tango_time_raw'] = author.y_hat-p.y_reference_tango_time_ms
    whiff = d.contact_type.eq('whiff')
    for ax in ('x','z'):
        p[f'{ax}_whiff_candidate_raw'] = np.where(whiff, reconstruct._apply_formula(SPECS[ax], f), np.nan)
    # For diagnostic EV50 plots only: current-season local outcome selection, never fitting.
    hip = d.contact_type.eq('in_play') & np.isfinite(d.launch_speed)
    med = d.loc[hip].groupby(['season','batter']).launch_speed.transform('median')
    p['descriptive_ev50'] = False
    p.loc[hip,'descriptive_ev50'] = d.loc[hip].launch_speed.ge(med)
    for ax in AXES:
        p[f'common_{ax}'] = np.isfinite(p[[f'{ax}_{m}_raw' for m in METHODS[ax]]]).all(axis=1)
        if ax in ('x','z'): p.loc[whiff,f'common_{ax}'] = np.isfinite(p.loc[whiff,f'{ax}_whiff_candidate_raw'])
    return p

def target_rates():
    t = pd.read_csv(ROOT/'data/savant_calibration/pitcher_split_2025.csv')
    if not t.year.eq(2025).all(): raise ValueError('Holdout targets forbidden')
    out = {}
    for ct, code in [('in_play',2),('foul',4),('whiff',9)]:
        g = t[t.bat_contact_code.eq(code)]
        for ax in AXES:
            v = np.average(g[list(RATES[ax])], weights=g.n_swings, axis=0)
            if np.isclose(v.sum(),100): v /= 100
            if not np.isclose(v.sum(),1): raise ValueError('Invalid calibration rates')
            out[(ax,ct)] = dict(zip(('lo','mid','hi'), v))
    return out

def fit_maps(dev, hist, rates):
    assert_dev(dev); assert_dev(hist)
    maps, diag = {}, []
    for ax in AXES:
        h = calibrate._hist_for(hist, ax)
        for m in METHODS[ax] + (['whiff_candidate'] if ax in ('x','z') else []):
            for ct in ('in_play','foul','whiff'):
                if (ax in ('x','z')) and ((m == 'whiff_candidate') != (ct == 'whiff')): continue
                sel = dev.contact_type.eq(ct) & dev[f'common_{ax}']
                v = dev.loc[sel,f'{ax}_{m}_raw'].to_numpy(float)
                if not np.isfinite(v).any(): continue
                mp = calibrate.QuantileMap().fit(v, calibrate.per_type_target(h, ax, rates[(ax,ct)]), axis=ax)
                key = f'{ax}|{m}|{ct}'
                maps[key] = dict(src=mp.src.tolist(), dst=mp.dst.tolist(), fit_season=2025)
                diag.append(dict(axis=ax, method=m, contact_type=ct, n_fit=len(v),
                                 unique_knots=len(np.unique(mp.src)), total_knots=len(mp.src),
                                 min_target=mp.dst[0], max_target=mp.dst[-1],
                                 target_source='2025_pitcher_histogram_plus_contact_rates'))
        if ax == 'y':
            sel = dev.common_y
            mp = calibrate.QuantileMap().fit(dev.loc[sel,'y_author_raw'],h,axis=ax)
            maps['y|author_pooled|all'] = dict(src=mp.src.tolist(),dst=mp.dst.tolist(),fit_season=2025)
            diag.append(dict(axis=ax,method='author_pooled',contact_type='all',n_fit=int(sel.sum()),
                unique_knots=len(np.unique(mp.src)),total_knots=len(mp.src),min_target=mp.dst[0],max_target=mp.dst[-1],
                target_source='2025_pitcher_all_swing_histogram'))
    return maps, pd.DataFrame(diag)

def transform(values, spec):
    if spec['fit_season'] != 2025: raise ValueError('Invalid map provenance')
    m = calibrate.QuantileMap(); m.src = np.asarray(spec['src']); m.dst = np.asarray(spec['dst'])
    return m.transform(values)

def apply_maps(p, maps):
    saturation = []
    for ax in AXES:
        for m in METHODS[ax] + (['whiff_candidate'] if ax in ('x','z') else []):
            out = np.full(len(p),np.nan)
            for ct in ('in_play','foul','whiff'):
                spec = maps.get(f'{ax}|{m}|{ct}')
                if spec is None: continue
                sel = p.contact_type.eq(ct)
                v = p.loc[sel,f'{ax}_{m}_raw'].to_numpy(float)
                out[sel] = transform(v,spec)
                for season in (2024,2025,2026):
                    vv = v[p.loc[sel,'season'].eq(season).to_numpy()]
                    ok = np.isfinite(vv)
                    saturation.append(dict(axis=ax,method=m,contact_type=ct,season=season,n_finite=int(ok.sum()),
                        n_below_fit=int(np.sum(vv<spec['src'][0])),n_above_fit=int(np.sum(vv>spec['src'][-1])),
                        n_at_lower_knot=int(np.sum(vv==spec['src'][0])),n_at_upper_knot=int(np.sum(vv==spec['src'][-1]))))
            p[f'{ax}_{m}_calibrated'] = out
    p['y_author_pooled_calibrated'] = transform(p.y_author_raw,maps['y|author_pooled|all'])
    return p, pd.DataFrame(saturation)
