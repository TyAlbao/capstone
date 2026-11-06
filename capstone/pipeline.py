"""Unified, offline experiment orchestration. No automatic method selection."""
from __future__ import annotations
import gc,json
from pathlib import Path
import numpy as np
import pandas as pd
from . import data, reconstruction as r, evaluation as e, references as ref
from . import reference_calibration as rc, whiffs as w
from . import comparisons, outputs


def native_maps(p,maps):
    for ax in data.AXES:
        values=np.full(len(p),np.nan)
        for ct in ('in_play','foul','whiff'):
            spec=maps.get(f'{ax}|{ct}')
            if spec is not None:
                sel=p.contact_type.eq(ct)
                values[sel]=r.transform(p.loc[sel,f'{ax}_author_raw'],spec)
        p[f'{ax}_author_native_calibrated']=values
    return p


def classes(p):
    cats={c+'_category':pd.Categorical(data.discover.label_of(c[0],p[c])) for c in p
          if c[:2] in ('x_','y_','z_') and c.endswith(('_raw','_calibrated'))}
    return pd.concat([p,pd.DataFrame(cats,index=p.index)],axis=1)


def tendencies(player,cells):
    def direction(v):
        v=np.asarray(v,float)
        return np.where(np.isclose(v,0.,rtol=0.,atol=1e-12),0.,np.sign(v))
    rows=[];keys=['season','contact_type','axis','method','stage']
    for key,g in cells.groupby(keys+['contrast','statistic']):
        tags=dict(zip(keys+['contrast','statistic'],key))
        if tags['statistic'] not in data.RATES[tags['axis']]:continue
        a,b=g.predicted,g.official
        rows.append(dict(**tags,population='within_batter_pitch_group_contrast',cells=len(g),
            direction_accuracy=float((direction(a)==direction(b)).mean()),spearman_r=e.correlation(a.rank(),b.rank())))
    for key,g in player[player.eligible].groupby(keys+['group']):
        tags=dict(zip(keys+['group'],key))
        for stat in data.RATES[tags['axis']]:
            a,b=g[stat+'_pred'],g[stat+'_official']
            pa=a-np.average(a,weights=g.local_n);pb=b-np.average(b,weights=g.n_swings)
            rows.append(dict(**tags,statistic=stat,population='batter_relative_to_eligible_league',cells=len(g),
                direction_accuracy=float((direction(pa)==direction(pb)).mean()),spearman_r=e.correlation(a.rank(),b.rank())))
    return pd.DataFrame(rows)


def xyz(config,out,refit=False):
    print('XYZ: reconstructing all tracked swings',flush=True)
    d=r.load_swings();author,features=r.original_predictions(d);params=data.PARAMETERS
    refs=r.fit_references(d[d.season.eq(2025)],author.y_hat) if refit else pd.read_parquet(params/'references.parquet').set_index('batter')
    p=r.raw_predictions(d,author,features,refs)
    if refit:
        hist=data.development_histograms()
        maps,diag=r.fit_maps(p[p.season.eq(2025)],hist,r.target_rates())
        dev=p[p.season.eq(2025)]
        hats=pd.DataFrame({a+'_hat':dev[a+'_author_raw'] for a in data.AXES},index=dev.index)
        fitted=data.calibrate.fit_calibration(hats,dev,2025,hist=hist,rates=r.target_rates())
        native={f'{a}|{ct}':dict(src=m.src.tolist(),dst=m.dst.tolist(),fit_season=2025) for (a,ct),m in fitted.items()}
    else:
        maps=json.loads((params/'calibration_maps.json').read_text())
        native=json.loads((params/'author_native_maps.json').read_text())
    p,saturation=r.apply_maps(p,maps);p=native_maps(p,native)
    for col in ('miss_distance','plate_x','plate_z','sz_top','sz_bot','attack_angle','attack_direction','swing_path_tilt'):
        p[col]=d[col]
    del d,author,features
    gc.collect()
    official,reconciliation=data.ingest(write=False)
    den=e.denominator_audit(p,official)
    print('XYZ: shared populations and player/group rates',flush=True)
    player,league,coverage=e.aggregate_predictions(p,official,den)
    player=e.add_null(player,official);player,overall,cats,tails=e.summarize(player)
    print('XYZ: paired intervals and pitch-group tendencies',flush=True)
    pairs=e.paired_comparisons(player);contrast,cells=e.contrasts(player)
    tables=dict(overall_metrics=overall,category_metrics=cats,tail_metrics=tails,paired_comparisons=pairs,
        pitch_group_contrasts=contrast,league_metrics=league,coverage=coverage,
        relative_tendencies=tendencies(player,cells),author_reproduction_metrics=e.native_author_metrics(p,official,den),
        partition_reconciliation=reconciliation,calibration_saturation=saturation)
    # Selection changes reporting only: the benchmark population stays fixed.
    for name,frame in tables.items():
        if {'axis','method'}.issubset(frame):
            keep=[m in config['methods'][a] or m in ('dev_null','whiff_candidate') for a,m in zip(frame.axis,frame.method)]
            tables[name]=frame.loc[keep].reset_index(drop=True)
    tables['paired_relative_accuracy']=comparisons.aggregate_comparisons(player,cells)
    table=tables['paired_relative_accuracy']
    tables['paired_relative_accuracy']=table.loc[[m in config['methods'][a] or m=='dev_null' for a,m in zip(table.axis,table.method)]]
    return tables,dict(predictions=classes(p),players=player,populations=den)


def whiffs(config,out):
    print('Whiffs: published formulas, proxies and development baselines',flush=True)
    d,params=w.prepare()
    tables=w.diagnostics(d,params)
    tables['whiff_season']=tables.pop('whiff_season_metrics')
    tables['whiff_pitch_group']=tables.pop('whiff_pitch_group_metrics')
    tables['whiff_stratified_metrics']['stratum']=tables['whiff_stratified_metrics']['stratum'].astype(str)
    frozen=json.loads((data.PARAMETERS/'whiff_baselines.json').read_text())
    if params!=frozen:raise ValueError('Development whiff baseline differs from the preserved artifact')
    for name,frame in tables.items():
        if {'axis','method'}.issubset(frame):
            tables[name]=frame.loc[[m in config['whiff_methods'][a] for a,m in zip(frame.axis,frame.method)]].reset_index(drop=True)
    tables['paired_whiff_accuracy']=comparisons.whiff_comparisons(d,params)
    table=tables['paired_whiff_accuracy']
    tables['paired_whiff_accuracy']=table.loc[[m in config['whiff_methods'][a] for a,m in zip(table.axis,table.method)]]
    return tables,dict(whiffs=d)


def references(config,out):
    print('References: original X protocols and their distinct cohorts',flush=True)
    cache=data.ROOT/'data/savant_calibration';source=data.ROOT/'data/statcast_2024_2026.parquet'
    schema=data.pq.ParquetFile(source).schema.names
    metrics=[];prospective=[];populations=[];predictions=[];prior=None
    for season in (2024,2025,2026):
        raw=pd.read_parquet(source,columns=[c for c in ref.KEEP if c in schema],filters=[('game_year','==',season)])
        s=ref.prepare_season(raw,season);refs=ref.calculate_references(s)
        truth=ref.normalize_official(pd.read_csv(cache/f'reference_leaderboard_{season}.csv'),season)
        sched=json.loads((cache/f'reference_schedule_{season}.json').read_text())
        played={g['game_pk'] for g in sched['games'] if g['game_type']=='R' and g['status']=='Final' and g['status_code']!='C'}
        complete=bool(played) and played==set(raw.loc[raw.game_type.eq('R'),'game_pk'].astype(int)) and not any(g['status'] in ('Preview','Live') for g in sched['games'])
        if not complete:raise ValueError(f'Incomplete schedule coverage: {season}')
        variants=[('same_season_raw',refs,ref.METHODS)]
        if prior is not None:variants.append(('prior_season_raw',prior.drop(columns=['ref_existing'],errors='ignore'),ref.METHODS[:3]))
        for protocol,rr,methods in variants:
            if protocol not in config['reference_protocols']:continue
            pred=ref.predict_contact(s,rr,methods)
            den=ref.denominator_audit(s,pred,truth,ref.Config(),complete)
            rates=ref.category_rates(pred[pred.primary_eligible],methods)
            met,_=ref.validation_metrics(rates,den,ref.Config())
            if protocol=='prior_season_raw':met['reference_season']=season-1
            (metrics if protocol=='same_season_raw' else prospective).append(met)
            populations.append(den.assign(season=season,protocol=protocol))
            cols=['game_pk','at_bat_number','pitch_number','batter','season','primary_eligible']+[f'x_{m}{suf}' for m in methods for suf in ('','_sign_only')]
            predictions.append(pred[cols].assign(protocol=protocol))
        prior=refs
    tables={}
    if metrics:tables['reference_raw']=pd.concat(metrics,ignore_index=True)
    if prospective:tables['reference_prior_season']=pd.concat(prospective,ignore_index=True)
    protocols=set(config['reference_protocols'])&{'frozen_2025_references','same_season_descriptive'}
    if protocols:
        hist=data.development_histograms();hist=hist[hist.axis.eq('x')]
        dev=rc.load_season(2025);dev_refs=ref.calculate_references(dev)
        truth25=pd.concat([rc.official(2025,'pitcher',ct) for ct in ('in_play','foul')],ignore_index=True)
        maps,_=rc.fit_comparison(dev,dev_refs,hist,truth25)
        null=np.average(truth25.loc[truth25.bat_contact_code.eq(2),list(rc.XCOLS)],weights=truth25.loc[truth25.bat_contact_code.eq(2),'n_swings'],axis=0)
        parts=[]
        for season in (2025,2024,2026):
            s=dev if season==2025 else rc.load_season(season)
            for protocol,rr in [('frozen_2025_references',rc.references_for_target(dev_refs,s)),('same_season_descriptive',ref.calculate_references(s))]:
                if protocol not in protocols:continue
                pred=rc.common_predictions(s,rr);pred=pred[pred.primary_eligible].reset_index(drop=True)
                for stage in ('raw','calibrated'):
                    values={m:pred[f'x_{m}'].to_numpy() if stage=='raw' else rc.apply_maps(pred[f'x_{m}'],pred,maps[m]) for m in rc.METHODS}
                    for kind in ('batter','pitcher'):
                        met,_,_,den=rc.compare(values,pred,s[s.contact_type.eq('in_play')],rc.official(season,kind,'in_play'),kind,dict(season=season,protocol=protocol,stage=stage),True,null)
                        parts.append(met)
                        if stage=='raw':populations.append(den.assign(season=season,protocol=protocol,perspective=kind))
                    predictions.append(pred[['game_pk','at_bat_number','pitch_number','batter','season','primary_eligible']].assign(protocol=protocol,stage=stage,**{f'x_{m}':v for m,v in values.items()}))
        tables['reference_calibrated']=pd.concat(parts,ignore_index=True)
    if 'author_reproduction' in config['reference_protocols']:
        hist=data.development_histograms();hist=hist[hist.axis.eq('x')]
        dev=rc.load_season(2025)
        truth25=pd.concat([rc.official(2025,'pitcher',ct) for ct in ('in_play','foul')],ignore_index=True)
        rawdev=data.reconstruct.reconstruct(dev,2025,r.SPECS,r.YCOM,tilt_sign=r.TILT)
        author_maps=rc.fit_maps(rawdev,dev,hist,truth25)
        parts=[]
        for season in (2025,2024,2026):
            swings=dev if season==2025 else rc.load_season(season)
            raw=rawdev if season==2025 else data.reconstruct.reconstruct(swings,season,r.SPECS,r.YCOM,tilt_sign=r.TILT)
            calibrated=rc.apply_maps(raw.x_hat,swings,author_maps)
            for ct in ('in_play','foul'):
                sel=swings.contact_type.eq(ct).to_numpy();sub=swings.loc[sel].reset_index(drop=True)
                for kind in ('batter','pitcher'):
                    truth=rc.official(season,kind,ct)
                    for population,matched in [('author_rules',False),('matched_10pct',True)]:
                        for stage,values in [('raw',raw.x_hat.to_numpy()[sel]),('calibrated',calibrated[sel])]:
                            met,_,_,den=rc.compare({'author_global':values},sub,sub,truth,kind,dict(season=season,contact_type=ct,population=population,stage=stage),matched)
                            parts.append(met)
                            if stage=='raw':populations.append(den.assign(season=season,contact_type=ct,protocol='author_reproduction',population=population,perspective=kind))
        tables['reference_author_reproduction']=pd.concat(parts,ignore_index=True)
    frames={}
    if populations:frames['reference_populations']=pd.concat(populations,ignore_index=True)
    if predictions:frames['reference_predictions']=pd.concat(predictions,ignore_index=True)
    return tables,frames


def validate_config(c):
    if c.get('development_season')!=2025:raise ValueError('Preserved artifacts/protocols require 2025 development data')
    if not set(c['experiments']).issubset({'xyz','whiffs','references'}):raise ValueError('Unknown experiment')
    for a in data.AXES:
        if not set(c['methods'][a]).issubset(r.METHODS[a]+(['author_pooled'] if a=='y' else [])):raise ValueError(f'Unknown {a} method')
        if not set(c['whiff_methods'][a]).issubset(w.methods(a)):raise ValueError(f'Unknown {a} whiff method')
    if not set(c['reference_protocols']).issubset({'same_season_raw','prior_season_raw','frozen_2025_references','same_season_descriptive','author_reproduction'}):raise ValueError('Unknown reference protocol')


def run(config,out,refit=False):
    import tempfile,os
    from .verification import verify,check_frames
    validate_config(config)
    if 'xyz' not in config['experiments']:
        raise ValueError('The four-file output requires xyz to preserve the fixed swing population')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    old_names={'config.json','metrics.parquet','predictions.parquet','populations.parquet','whiffs.parquet',
               'reference_predictions.parquet','reference_populations.parquet','verification.json'}
    unexpected={p.name for p in out.iterdir()}-outputs.FILES-old_names
    if unexpected:raise ValueError(f'Unexpected output files; refusing to overwrite or delete them: {sorted(unexpected)}')
    tables={};frames={}
    for name in config['experiments']:
        ts,fs=globals()[name](config,out,refit) if name=='xyz' else globals()[name](config,out)
        tables.update(ts);frames.update(fs);gc.collect()
    if 'reference_predictions' in frames:
        tables['paired_reference_accuracy']=comparisons.reference_comparisons(frames['reference_predictions'],frames['reference_populations'],frames['predictions'])
    # Frozen regression checks precede any publication or removal of previous results.
    full_refs=set(config['reference_protocols'])=={'same_season_raw','prior_season_raw','frozen_2025_references','same_season_descriptive','author_reproduction'}
    check_frames(frames,full_references=full_refs)
    print('Packing four outputs and checking reversible regression views',flush=True)
    long,schemas=outputs.encode_metrics(tables)
    players,player_info=outputs.pack_players(frames['players'])
    swings,swing_info=outputs.pack_swings(frames)
    with tempfile.TemporaryDirectory(prefix='.results-',dir=out.parent) as tmp:
        staging=Path(tmp).resolve()
        if staging.parent!=out.parent:raise ValueError('Invalid staging directory')
        long.to_csv(staging/'metrics.csv',index=False)
        outputs.write_parquet(players,staging/'players.parquet',dict(players=player_info,metric_schemas=schemas,config=config,
            population_schemas={n:outputs.schema(f) for n,f in frames.items() if n.endswith('populations')}))
        outputs.write_parquet(swings,staging/'swings.parquet',dict(frames=swing_info,config=config))
        # Release large tables before reading the verification views back from disk.
        del long,players,swings
        evidence=verify(staging,population_frames={k:v for k,v in frames.items() if k.endswith('populations')},require_report=False)
        outputs.write_report(staging/'report.md',tables,config,evidence)
        if {p.name for p in staging.iterdir()}!=outputs.FILES:raise AssertionError('Unexpected staging output')
        for name in sorted(outputs.FILES):
            target=(out/name).resolve()
            if target.parent!=out:raise ValueError('Invalid publication target')
            os.replace(staging/name,target)
        for name in old_names:
            target=(out/name).resolve()
            if target.parent!=out:raise ValueError('Invalid cleanup target')
            if target.is_file():target.unlink()
    print('Published exactly metrics.csv, players.parquet, swings.parquet and report.md',flush=True)
    return tables
