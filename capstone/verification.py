"""Regression evidence independent of the old experiment directories."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import ROOT, PARAMETERS, UPSTREAM, sha, dump, discover



def protected_state():
    paths=[ROOT/'notebooks/scratch.ipynb', ROOT/'data/statcast_2024_2026.parquet']
    paths+=list((ROOT/'data').glob('savant_timing_leaderboard*.csv'))
    paths+=list((ROOT/'data/savant_whiff_tails').glob('*.parquet'))
    paths+=[p for p in UPSTREAM.rglob('*') if p.is_file() and not {'.git','__pycache__'}.intersection(p.relative_to(UPSTREAM).parts)]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}


def assert_frame(a,b,label):
    if len(a)!=len(b):raise AssertionError(f'{label}: rows {len(a)} != {len(b)}')
    missing=set(b)-set(a)
    if missing:raise AssertionError(f'{label}: missing columns {missing}')
    a=a[list(b)].reset_index(drop=True);b=b.reset_index(drop=True)
    # Tables may have different construction order; sort on descriptive keys.
    keys=[c for c in b if c in ('season','contact_type','group','axis','method','stage','protocol','route','category','tail','cohort','contrast','statistic','metric','perspective','experiment','population','rate')]
    if keys:
        a=a.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
        b=b.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
    for c in b:
        av,bv=a[c],b[c]
        if pd.api.types.is_numeric_dtype(bv) and not pd.api.types.is_bool_dtype(bv):
            np.testing.assert_allclose(pd.to_numeric(av).to_numpy(float),bv.to_numpy(float),rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=f'{label}/{c}')
        else:
            pd.testing.assert_series_equal(av.astype('string'),bv.astype('string'),check_names=False,obj=f'{label}/{c}')


def assert_parquet(new,old):
    oldfile=pq.ParquetFile(old)
    if pq.ParquetFile(new).metadata.num_rows!=oldfile.metadata.num_rows:raise AssertionError(f'Population mismatch: {new}')
    tested=[]
    for c in oldfile.schema_arrow.names:
        a=pd.read_parquet(new,columns=[c]);b=pd.read_parquet(old,columns=[c])
        assert_frame(a,b,f'{new.name}/{c}')
        if c[:2] in ('x_','y_','z_') and c.endswith(('_raw','_calibrated')):
            np.testing.assert_array_equal(discover.label_of(c[0],a[c]),discover.label_of(c[0],b[c]))
        tested.append(c)
    return dict(rows=oldfile.metadata.num_rows,columns=len(tested),all_values_and_missingness_equal=True,classifications_exact=True)


def digest_frame(path):
    """Exact per-column digests after numerical parity is independently established."""
    result={}
    for c in pq.ParquetFile(path).schema_arrow.names:
        d=pd.read_parquet(path,columns=[c])
        result[c]=hashlib.sha256(pd.util.hash_pandas_object(d,index=False).values.tobytes()).hexdigest()
    return dict(rows=pq.ParquetFile(path).metadata.num_rows,columns=result)


def frame_digest(frame,columns):
    result={}
    for c in columns:
        column=frame[[c]].copy()
        # Parquet canonicalizes NaN payload/sign bits; match the frozen on-disk
        # fingerprints without changing any finite value, mask or classification.
        if pd.api.types.is_float_dtype(column[c]):column.loc[column[c].isna(),c]=np.nan
        result[c]=hashlib.sha256(pd.util.hash_pandas_object(column,index=False).values.tobytes()).hexdigest()
    return result


def check_frames(frames,full_references=True):
    manifest=json.loads((PARAMETERS/'regression/outputs.json').read_text())
    for filename,expected in manifest.items():
        name=filename.removesuffix('.parquet')
        if name not in frames or (name.startswith('reference_') and not full_references):continue
        frame=frames[name]
        if len(frame)!=expected['rows']:raise AssertionError(f'{name}: changed population')
        got=frame_digest(frame,expected['columns'])
        bad=[c for c,h in expected['columns'].items() if got[c]!=h]
        if bad:raise AssertionError(f'{name}: changed values/classifications in {bad}')


def configured_table(name,frame,config):
    if name.startswith('whiff_'):
        if 'whiffs' not in config['experiments']:return frame.iloc[:0]
        if {'axis','method'}.issubset(frame):
            return frame.loc[[m in config['whiff_methods'][a] for a,m in zip(frame.axis,frame.method)]]
    elif name.startswith('reference_'):
        if 'references' not in config['experiments']:return frame.iloc[:0]
        names={'reference_raw':'same_season_raw','reference_prior_season':'prior_season_raw','reference_author_reproduction':'author_reproduction'}
        if name in names and names[name] not in config['reference_protocols']:return frame.iloc[:0]
        if name=='reference_calibrated':return frame[frame.protocol.isin(config['reference_protocols'])]
    elif {'axis','method'}.issubset(frame):
        return frame.loc[[m in config['methods'][a] or m in ('dev_null','whiff_candidate') for a,m in zip(frame.axis,frame.method)]]
    return frame


def verify(out,population_frames=None,require_report=True):
    from . import outputs
    out=Path(out);expected=PARAMETERS/'regression'
    required=outputs.FILES if require_report else outputs.FILES-{'report.md'}
    if {p.name for p in out.iterdir()}!=required:raise AssertionError('Output directory violates the four-file contract')
    config=outputs.metadata(out/'players.parquet')['config']
    full_refs=set(config['reference_protocols'])=={'same_season_raw','prior_season_raw','frozen_2025_references','same_season_descriptive','author_reproduction'}
    tables=outputs.read_tables(out);checks={}
    for path in expected.glob('*.parquet'):
        name=path.stem
        wanted=configured_table(name,pd.read_parquet(path),config)
        if wanted.empty and name not in tables:continue
        if name not in tables:raise AssertionError(f'Missing historical metric table: {name}')
        assert_frame(tables[name],wanted,name);checks[name]=len(tables[name])
    for name,comparison in tables.items():
        if not name.startswith('paired_') or 'improvement' not in comparison:continue
        finite=comparison.ci_low.notna()&comparison.ci_high.notna()
        if not comparison.loc[finite,'ci_low'].le(comparison.loc[finite,'ci_high']).all():raise AssertionError(f'{name}: reversed confidence interval')
        excludes=(comparison.loc[finite,'ci_low']>0)|(comparison.loc[finite,'ci_high']<0)
        if not np.array_equal(comparison.loc[finite,'excludes_zero'].astype(bool),excludes):raise AssertionError(f'{name}: incorrect excludes_zero')
        if not comparison.bootstrap_reps.eq(1000).all() or not comparison.bootstrap_seed.eq(180).all():raise AssertionError('Bootstrap settings changed')
    manifest=json.loads((expected/'outputs.json').read_text())
    for name in ('predictions','players','whiffs','reference_predictions'):
        available=outputs.metadata(out/'swings.parquet')['frames']
        if name!='players' and name not in available:continue
        frame=outputs.legacy_frame(out,name)
        check_frames({name:frame},full_references=full_refs)
        del frame
    if population_frames is None:population_frames=outputs.population_views(out)
    check_frames(population_frames,full_references=full_refs)
    players=pd.read_parquet(out/'players.parquet')
    if players.duplicated(['season','batter','contact_type','pitch_group','method_id']).any():raise AssertionError('Duplicate player cells')
    swings=pd.read_parquet(out/'swings.parquet',columns=outputs.PITCH_KEYS)
    if len(swings)!=974886 or swings.duplicated(outputs.PITCH_KEYS).any():raise AssertionError('Duplicate or missing swings')
    if protected_state()!=json.loads((expected/'protected.json').read_text()):raise AssertionError('An original protected source changed')
    evidence=dict(passed=True,metrics=checks,outputs=sorted(outputs.FILES),protected_inputs_unchanged=True)
    print(json.dumps(evidence,indent=2),flush=True)
    return evidence
