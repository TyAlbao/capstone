from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'
PARAMETERS = ROOT/'parameters'
UPSTREAM = ROOT/'external/statcast-contact-point'
sys.dont_write_bytecode = True
sys.path.insert(0, str(UPSTREAM))
from cp_lib import calibrate, discover, savant_truth, geometry, reconstruct, join, validate
AXES = ('x', 'y', 'z')

RATES = calibrate.BAND_RATES

TAILS = {'x': ('avg_x_tied_up', 'avg_x_flail'), 'y': ('avg_y_late', 'avg_y_early'),
         'z': ('avg_z_over', 'avg_z_under')}

STAT_COLS = [c for a in AXES for c in (*RATES[a], *TAILS[a])]

CTS = ('in_play', 'foul', 'whiff', 'all')

GROUP_MAP = {'FF': 'FB', 'CU': 'BB', 'CH': 'CH'}

PITCH_MAP = {p: g for g, ps in {'FB': ['FF', 'SI', 'FC', 'FA'],
             'BB': ['SL', 'ST', 'CU', 'KC', 'SV', 'CS', 'SC'],
             'CH': ['CH', 'FS', 'FO', 'EP']}.items() for p in ps}

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def normalize_leaderboard(d, ct, split):
    d = d.copy()
    d['season'] = pd.to_numeric(d['year'], errors='raise').astype(int)
    d['contact_type'] = ct
    d['group'] = d.api_pitch_group.map(GROUP_MAP).fillna('UNKNOWN') if split else 'ALL'
    d['is_league'] = d['name'].eq('League')
    if d.loc[~d.is_league, 'id'].isna().any(): raise ValueError('Unnamed missing player ID')
    d['batter'] = pd.to_numeric(d.id, errors='raise').astype('Int64')
    if d.duplicated(['season', 'batter', 'group']).any(): raise ValueError('Duplicate leaderboard keys')
    if not d.season.isin([2024, 2025, 2026]).all(): raise ValueError('Unexpected season')
    if not (d.n_swings.gt(0) & d.n_swings.eq(np.floor(d.n_swings))).all(): raise ValueError('Invalid count')
    for ax in AXES:
        c = list(RATES[ax]); v = d[c].astype(float)
        if np.allclose(v.sum(axis=1), 100, atol=.005): v /= 100
        if not np.isfinite(v).all().all() or not np.allclose(v.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f'Invalid {ax} proportions')
        if (v.lt(0) | v.gt(1)).any().any(): raise ValueError('Rate outside [0,1]')
        d[c] = v
        for tail, rate in zip(TAILS[ax], (c[0], c[2])):
            d[tail] = pd.to_numeric(d[tail], errors='raise')
            # Zero-event means have no meaning, even if an API supplies zero.
            d.loc[d[rate].eq(0), tail] = np.nan
    return d

def weighted_stats(d):
    out = {'n_swings': d.n_swings.sum()}
    for ax in AXES:
        for col in RATES[ax]: out[col] = np.average(d[col], weights=d.n_swings)
        for col, rate in zip(TAILS[ax], (RATES[ax][0], RATES[ax][2])):
            w = d.n_swings * d[rate]
            ok = d[col].notna() & w.gt(0)
            out[col] = np.average(d.loc[ok, col], weights=w[ok]) if ok.any() else np.nan
    return out

def ingest(write=True):
    rows, inventory = [], []
    for split in (False, True):
        for ct in CTS:
            suffix = 'hip' if ct == 'in_play' else ct
            name = f"savant_timing_leaderboard_2024_2026_{'pitch_group' if split else 'no_split'}_{suffix}.csv"
            matches = [ROOT/'data'/name] if (ROOT/'data'/name).is_file() else []
            if len(matches) != 1: raise ValueError(f'Expected one source: {name}')
            p = matches[0]; raw = pd.read_csv(p)
            d = normalize_leaderboard(raw, ct, split); rows.append(d)
            inventory.append(dict(file=str(p.relative_to(ROOT)), sha256=sha(p), rows=len(d),
                                  players=int((~d.is_league).sum()), league_rows=int(d.is_league.sum()),
                                  unknown_group_rows=int(d.group.eq('UNKNOWN').sum()),
                                  missing_tail_cells=int(d[[c for a in AXES for c in TAILS[a]]].isna().sum().sum())))
    all_rows = pd.concat(rows, ignore_index=True)
    player = all_rows.loc[~all_rows.is_league].copy()
    league_checks = []
    for key, g in player.groupby(['season', 'contact_type', 'group']):
        expected = weighted_stats(g)
        provided = all_rows.loc[all_rows.is_league & all_rows.season.eq(key[0]) & all_rows.contact_type.eq(key[1]) & all_rows.group.eq(key[2])]
        for col, val in expected.items():
            got = provided.iloc[0][col] if len(provided) == 1 else np.nan
            league_checks.append(dict(season=key[0], contact_type=key[1], group=key[2], statistic=col,
                                      reconstructed=val, supplied=got, difference=val-got))
    rec = []
    # Reconcile only complete observed partitions; missing components are never filled with zero.
    for partition in ('contact_type', 'group'):
        by = ['season', 'batter', 'group'] if partition == 'contact_type' else ['season', 'batter', 'contact_type']
        total_value = 'all' if partition == 'contact_type' else 'ALL'
        pieces = set(('in_play', 'foul', 'whiff') if partition == 'contact_type' else ('FB', 'BB', 'CH'))
        for key, g in player.groupby(by):
            total = g[g[partition].eq(total_value)]
            sub = g[g[partition].isin(pieces)]
            complete = len(total) == 1 and set(sub[partition]) == pieces
            tags = dict(zip(by, key))
            r = dict(partition=partition, **tags, complete=complete)
            if complete:
                w = weighted_stats(sub); t = total.iloc[0]
                r.update(count_difference=w['n_swings']-t.n_swings,
                         max_rate_gap=max(abs(w[c]-t[c]) for a in AXES for c in RATES[a]),
                         max_tail_gap=max([abs(w[c]-t[c]) for a in AXES for c in TAILS[a] if np.isfinite(w[c]) and np.isfinite(t[c])] or [np.nan]))
            rec.append(r)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(inventory).to_csv(OUT/'source_inventory.csv', index=False)
        pd.DataFrame(league_checks).to_csv(OUT/'derived_league_checks.csv', index=False)
        pd.DataFrame(rec).to_csv(OUT/'partition_reconciliation.csv', index=False)
        player.to_parquet(OUT/'official_players.parquet', index=False)
    return player, pd.DataFrame(rec)

def rebuild_development_histograms():
    cache = ROOT/'data/savant_calibration'
    manifest = json.loads((cache/'manifest.json').read_text())
    pitchers = pd.read_csv(cache/'pitcher_plain_2025.csv')
    ids = set(pitchers.id.dropna().astype(int))
    rows, seen = [], set()
    from urllib.parse import urlparse, parse_qs
    for name, meta in manifest.items():
        url = meta.get('url', '')
        if '/swing-timing-distribution/player/' not in url: continue
        q = parse_qs(urlparse(url).query)
        if q.get('season[]') != ['2025'] or q.get('type') != ['pitcher']: continue
        pid = int(urlparse(url).path.rsplit('/', 1)[1]); p = cache/name
        if sha(p) != meta['sha256']: raise ValueError('Histogram cache checksum changed')
        payload = json.loads(p.read_text())
        if not set(AXES).issubset(payload): raise ValueError('Missing histogram axis')
        seen.add(pid)
        for ax in AXES:
            for r in payload[ax]:
                if int(r['year']) != 2025: raise ValueError('Holdout histogram forbidden')
                rows.append(dict(id=pid, axis=ax, bin=float(r['bin']), n=float(r['n']), season=2025))
    if seen != ids: raise ValueError('Incomplete development pitcher histogram population')
    h = pd.DataFrame(rows)
    h['bin'] = savant_truth._lower_edge(h.axis, h.bin)
    if h.n.lt(0).any() or not np.isfinite(h[['bin', 'n']]).all().all(): raise ValueError('Invalid histogram')
    return h


def development_histograms():
    """Verified 2025 target artifact; raw-cache reconstruction remains available."""
    path = PARAMETERS/'development_histograms.parquet'
    expected = json.loads((PARAMETERS/'provenance.json').read_text())['artifacts'][path.name]
    if sha(path) != expected: raise ValueError('Development histogram checksum changed')
    h = pd.read_parquet(path)
    if h.empty or not h.season.eq(2025).all(): raise ValueError('Holdout histogram forbidden')
    return h
