"""Paired comparisons; positive differences always favor the candidate method."""
import numpy as np
import pandas as pd
from . import evaluation as e, data


def paired_summary(pair, value='improvement', seed=180, reps=1000):
    """Resample whole batters, preserving all of each batter's paired cells."""
    pair=pair[np.isfinite(pair[value])].copy()
    lo,hi=e.clustered_ci(pair,value,seed=seed,reps=reps)
    return dict(improvement=pair[value].mean(),ci_low=lo,ci_high=hi,
                excludes_zero=bool(lo>0 or hi<0) if np.isfinite(lo+hi) else None,
                cells=len(pair),batters=pair.batter.nunique(),bootstrap_reps=reps,
                bootstrap_seed=seed,confidence_level=.95)


def aggregate_comparisons(player,cells):
    """Extend existing author intervals to groups, nulls and group contrasts.

    ALL/GROUPS classification-vs-author intervals already exist in
    evaluation.paired_comparisons and are retained there (with output sign reversed).
    """
    rows=[]
    for (season,ct,ax),g in player[player.eligible].groupby(['season','contact_type','axis']):
        for group in ('ALL','FB','BB','CH','GROUPS'):
            sub=g[g.group.ne('ALL') if group=='GROUPS' else g.group.eq(group)]
            for (method,stage),a in sub.groupby(['method','stage']):
                base_stages=('raw','calibrated') if stage=='baseline' else (stage,)
                for baseline in ('author','dev_null'):
                    if baseline=='dev_null' and method=='dev_null':continue
                    for base_stage in base_stages:
                        if baseline=='author' and group in ('ALL','GROUPS'):continue
                        b=sub[sub.method.eq(baseline)&sub.stage.eq('baseline' if baseline=='dev_null' else base_stage)]
                        pair=a[['batter','group','rate_error_pp','local_n','n_swings']].merge(
                            b[['batter','group','rate_error_pp']],on=['batter','group'],suffixes=('_method','_baseline'),validate='one_to_one')
                        if pair.empty:continue
                        pair['improvement']=pair.rate_error_pp_baseline-pair.rate_error_pp_method
                        rows.append(dict(season=season,contact_type=ct,axis=ax,group=group,method=method,stage=stage,
                            baseline=baseline,baseline_stage='baseline' if baseline=='dev_null' else base_stage,
                            target_metric='classification_rate_mae_pp',swings=int(pair.local_n.sum()),official_swings=int(pair.n_swings.sum()),
                            **paired_summary(pair)))
    # Same per-batter, per-category, per-contrast cells as the original contrast MAE.
    rates=cells[[s in data.RATES[a] for a,s in zip(cells.axis,cells.statistic)]]
    keys=['batter','contrast','statistic']
    for (season,ct,ax),g in rates.groupby(['season','contact_type','axis']):
        for contrast in ('FB-BB','FB-CH','BB-CH','ALL_CONTRASTS'):
            sub=g if contrast=='ALL_CONTRASTS' else g[g.contrast.eq(contrast)]
            for (method,stage),a in sub.groupby(['method','stage']):
                for baseline in ('author','dev_null'):
                    if baseline=='dev_null' and method=='dev_null':continue
                    for base_stage in (('raw','calibrated') if stage=='baseline' else (stage,)):
                        b=sub[sub.method.eq(baseline)&sub.stage.eq('baseline' if baseline=='dev_null' else base_stage)]
                        pair=a[keys+['error']].merge(b[keys+['error']],on=keys,suffixes=('_method','_baseline'),validate='one_to_one')
                        if pair.empty:continue
                        pair['improvement']=pair.error_baseline-pair.error_method
                        rows.append(dict(season=season,contact_type=ct,axis=ax,group='GROUPS',contrast=contrast,statistic='all_classification_categories',method=method,stage=stage,
                            baseline=baseline,baseline_stage='baseline' if baseline=='dev_null' else base_stage,
                            target_metric='within_batter_pitch_group_contrast_mae_pp',**paired_summary(pair)))
    return pd.DataFrame(rows)


def whiff_comparisons(d,params):
    """Existing selected-tail category accuracy, paired against published regressions.

    These are pitch-level classification errors, not aggregate contact rate MAE.
    """
    from . import whiffs as w
    rows=[]
    for season,g in d.groupby('season'):
        for group in ['ALL',*sorted(g.pitch_group.unique())]:
            sub=g if group=='ALL' else g[g.pitch_group.eq(group)]
            for ax in data.AXES:
                use=sub[sub[f'common_{ax}']]
                for method in w.methods(ax):
                    for baseline in (w.PRIMARY[ax],'dev_majority'):
                        if method==baseline and baseline=='dev_majority':continue
                        pair=use[['batter']].copy()
                        truth=use[w.CAT[ax]]
                        pair['improvement']=(use[f'category_{ax}_{baseline}'].ne(truth).astype(float)-use[f'category_{ax}_{method}'].ne(truth).astype(float))*100
                        rows.append(dict(season=season,contact_type='whiff',axis=ax,group=group,method=method,
                            stage='baseline' if method.startswith('dev_') else 'raw',protocol='selected_whiff_tails',
                            baseline=baseline,baseline_stage='baseline' if baseline=='dev_majority' else 'raw',
                            target_metric='selected_whiff_classification_error_pp',swings=len(pair),**paired_summary(pair)))
    return pd.DataFrame(rows)

def reference_comparisons(predictions,populations,swings):
    """Pair historical X variants with author predictions on their ORIGINAL cohorts."""
    import json
    from . import reference_calibration as rc
    rows=[];pitch_keys=['game_pk','at_bat_number','pitch_number']
    null=np.asarray(json.loads((data.PARAMETERS/'reference_calibration.json').read_text())['development_null_rates'])
    author=swings[pitch_keys+['x_author_raw','x_author_native_calibrated']]
    for (season,protocol,stage),p in predictions.groupby(['season','protocol','stage'],dropna=False,sort=False):
        stage='raw' if pd.isna(stage) else stage
        den=populations[populations.season.eq(season)&populations.protocol.eq(protocol)]
        old_raw=protocol in ('same_season_raw','prior_season_raw')
        if not old_raw:den=den[den.perspective.eq('batter')]
        eligible=den[den.eligible.fillna(False)] if old_raw else den[den.qualifies.fillna(False)]
        id_col='batter' if old_raw else 'id';ids=eligible[id_col]
        p=p[p.primary_eligible & p.batter.isin(ids)].merge(author,on=pitch_keys,how='left',validate='one_to_one')
        if p.empty:continue
        if old_raw:
            truth=eligible.set_index('batter')[['official_tied_up','official_centered','official_flail']]
        else:
            truth=rc.official(int(season),'batter','in_play').set_index('id')[list(rc.XCOLS)]
        def errors(values):
            a=e.category_stats(p,values,'x',['batter']).set_index('batter')
            y=truth.reindex(a.index).to_numpy(float)
            err=np.abs(a[list(data.RATES['x'])].to_numpy()-y).mean(axis=1)*100
            return pd.DataFrame({'batter':a.index,'error':err,'local_n':a.local_n.to_numpy()})
        base=errors(p['x_author_raw' if stage=='raw' else 'x_author_native_calibrated'])
        null_error=pd.DataFrame({'batter':base.batter,'error':np.abs(truth.reindex(base.batter).to_numpy()-null).mean(axis=1)*100})
        columns=[c for c in predictions if c.startswith('x_') and c not in ('x_author_raw','x_author_native_calibrated') and p[c].notna().any()]
        for col in columns+([] if old_raw else ['development_rate_null']):
            method=col.removeprefix('x_');a=errors(p[col]) if col!='development_rate_null' else null_error
            for baseline,b in [('author_native',base),('development_rate_null',null_error)]:
                if col==baseline:continue
                pair=a[['batter','error']].merge(b[['batter','error']],on='batter',suffixes=('_method','_baseline'),validate='one_to_one')
                pair['improvement']=pair.error_baseline-pair.error_method
                rows.append(dict(season=season,contact_type='in_play',axis='x',group='ALL',method=method,stage=stage,
                    protocol=protocol,baseline=baseline,baseline_stage=stage if baseline=='author_native' else 'baseline',
                    target_metric='classification_rate_mae_pp',swings=len(p),**paired_summary(pair)))
    return pd.DataFrame(rows)
