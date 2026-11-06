from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .data import AXES, CTS, RATES, TAILS, weighted_stats, discover
from .reconstruction import METHODS, STAGES, KEYS
def native_author_metrics(p,official,den):
    rows=[]
    for ax in AXES:
        for stage in STAGES:
            col=f'{ax}_author_raw' if stage=='raw' else f'{ax}_author_native_calibrated'
            for ct in ('in_play','foul'):
                d=p[p.contact_type.eq(ct)]
                a=category_stats(d,d[col],ax,['season','batter'])
                off=official[official.contact_type.eq(ct)&official.group.eq('ALL')]
                a=a.merge(off[['season','batter','n_swings',*RATES[ax],*TAILS[ax]]],on=['season','batter'],suffixes=('_pred','_official'))
                a=a.merge(den[den.contact_type.eq(ct)&den.group.eq('ALL')][['season','batter','eligible']],on=['season','batter'])
                for cohort in ('author_min50','matched_10pct'):
                    g=a[a.n_swings.ge(50)] if cohort=='author_min50' else a[a.eligible]
                    for season,s in g.groupby('season'):
                        for c in (*RATES[ax],*TAILS[ax]):
                            ok=s[[c+'_pred',c+'_official']].notna().all(axis=1);v=s[ok]
                            scale=100 if c in RATES[ax] else 1
                            rows.append(dict(season=season,contact_type=ct,axis=ax,stage=stage,cohort=cohort,statistic=c,
                                players=len(v),swings=v.local_n.sum(),coverage=np.average(s.coverage,weights=s.local_n),
                                mae=scale*(v[c+'_pred']-v[c+'_official']).abs().mean(),pearson_r=correlation(v[c+'_pred'],v[c+'_official'])))
    return pd.DataFrame(rows)

def distribution_diagnostics(p):
    rows=[];mask_rows=[]
    for ax in AXES:
        for m,stage in method_variants(ax):
            v=p[f'{ax}_{m}_{stage}'].copy()
            if ax!='y':
                wh=p.contact_type.eq('whiff');v.loc[wh]=p.loc[wh,f'{ax}_whiff_candidate_{stage}']
            for (season,ct),idx in p.groupby(['season','contact_type']).groups.items():
                values=v.loc[idx]; finite=values[np.isfinite(values)]
                mask_rows.append(dict(season=season,contact_type=ct,axis=ax,method=m,stage=stage,
                    native_finite=len(finite),shared_finite=int(p.loc[idx,f'common_{ax}'].sum()),
                    shared_mask_loss=int((np.isfinite(values)&~p.loc[idx,f'common_{ax}']).sum())))
                for cohort,sub in [('all_contact_type',finite),('descriptive_ev50',finite[p.loc[finite.index,'descriptive_ev50']])]:
                    if sub.empty:continue
                    q=sub.quantile([.01,.05,.25,.5,.75,.95,.99])
                    rows.append(dict(season=season,contact_type=ct,axis=ax,method=m,stage=stage,cohort=cohort,
                        n=len(sub),mean=sub.mean(),sd=sub.std(),skew=sub.skew(),**{f'p{int(k*100):02d}':val for k,val in q.items()}))
    return pd.DataFrame(rows),pd.DataFrame(mask_rows)

def category_stats(frame, values, axis, keys):
    """Missing predictions remain in denominator; empty tails remain NaN."""
    v = np.asarray(values,float); good = np.isfinite(v); t = discover.THRESH[axis]
    w = frame[keys].copy()
    w['local_n'] = 1; w['known_n'] = good.astype(int)
    masks = (good & (v < -t), good & (v >= -t) & (v <= t), good & (v > t))
    for col, mask in zip(RATES[axis], masks): w[col] = mask.astype(int)
    for tail, mask in zip(TAILS[axis], (masks[0],masks[2])):
        w[tail+'_n'] = mask.astype(int)
        w[tail] = np.where(mask,v,0.)
    out = w.groupby(keys, dropna=False, observed=True).sum().reset_index()
    for col in RATES[axis]: out[col] /= out.local_n
    for col in TAILS[axis]: out[col] /= out[col+'_n'].replace(0,np.nan)
    out['coverage'] = out.known_n/out.local_n
    return out

def counts_table(p):
    records = []
    for ct in CTS:
        d = p if ct == 'all' else p[p.contact_type.eq(ct)]
        for group in ('ALL','FB','BB','CH','UNKNOWN'):
            g = d if group == 'ALL' else d[d.group.eq(group)]
            a = g.groupby(['season','batter']).size().rename('local_n').reset_index()
            a['contact_type'] = ct; a['group'] = group
            for col in ['has_launch_speed','has_launch_angle','has_intercept_xy','has_trajectory','common_x','common_y','common_z',
                        'fallback_x_batter','fallback_x_ev50','fallback_y_batter','fallback_y_ev50','unseen_development_batter']:
                a = a.merge(g.groupby(['season','batter'])[col].mean().rename(col),on=['season','batter'],how='left')
            records.append(a)
    return pd.concat(records,ignore_index=True)

def denominator_audit(p,official):
    a = counts_table(p)
    den = official[KEYS+['n_swings']].rename(columns={'n_swings':'official_n'}).merge(a,on=KEYS,how='outer',indicator=True)
    den['local_n'] = den.local_n.fillna(0).astype(int)
    den['count_difference'] = den.local_n-den.official_n
    den['relative_count_difference'] = den.count_difference/den.official_n
    den['absolute_count_gap'] = den.relative_count_difference.abs()
    den['minimum_required'] = np.where(den.contact_type.eq('all'),100,50)
    den['eligible'] = den.official_n.ge(den.minimum_required) & den.local_n.ge(den.minimum_required) & den.absolute_count_gap.le(.1)
    return den

def method_variants(ax):
    return [(m,s) for m in METHODS[ax] for s in STAGES] + ([('author_pooled','calibrated')] if ax=='y' else [])

def aggregate_predictions(p, official, den):
    outputs, leagues, covers = [], [], []
    for ax in AXES:
        for m,stage in method_variants(ax):
            vals = p[f'{ax}_{m}_{stage}'].to_numpy(float).copy()
            if ax in ('x','z'):
                wh = p.contact_type.eq('whiff').to_numpy()
                vals[wh] = p.loc[wh,f'{ax}_whiff_candidate_{stage}']
            vals[~p[f'common_{ax}'].to_numpy()] = np.nan
            for ct in CTS:
                ct_mask = np.ones(len(p),bool) if ct=='all' else p.contact_type.eq(ct).to_numpy()
                if ct=='whiff' and ax in ('x','z') and m!='author': continue  # same candidate, no duplicate leaderboard entries
                for group in ('ALL','FB','BB','CH'):
                    sel = ct_mask if group=='ALL' else ct_mask & p.group.eq(group).to_numpy()
                    d = p.loc[sel]; v = vals[sel]
                    a = category_stats(d,v,ax,['season','batter'])
                    a['contact_type'] = ct; a['group'] = group
                    off = official[official.contact_type.eq(ct) & official.group.eq(group)]
                    merged = a.merge(off[KEYS+['n_swings',*RATES[ax],*TAILS[ax]]],on=KEYS,how='inner',suffixes=('_pred','_official'))
                    merged = merged.merge(den[KEYS+['eligible','absolute_count_gap','minimum_required', 'fallback_x_ev50','fallback_y_ev50']],on=KEYS,validate='one_to_one')
                    merged['axis'] = ax; merged['method'] = m; merged['stage'] = stage
                    merged['route'] = 'linear_whiff_composite' if ct=='all' and ax in ('x','z') else ('linear_whiff_candidate' if ct=='whiff' and ax in ('x','z') else 'contact_or_timing_formula')
                    merged['protocol'] = 'historical_leaky' if m=='existing' else ('author_season_reference' if m=='author' and ax=='x' else 'frozen_2025')
                    errors = np.column_stack([100*(merged[c+'_pred']-merged[c+'_official']).abs() for c in RATES[ax]])
                    merged['rate_error_pp'] = errors.mean(axis=1)
                    for tail in TAILS[ax]: merged[tail+'_error'] = (merged[tail+'_pred']-merged[tail+'_official']).abs()
                    outputs.append(merged)
                    for season, g in a.groupby('season'):
                        covers.append(dict(axis=ax,method=m,stage=stage,contact_type=ct,group=group,season=season,
                            local_swings=int(g.local_n.sum()), known_swings=int(g.known_n.sum()),coverage=g.known_n.sum()/g.local_n.sum()))
                        # Pair complete official player roster, no minimum-count exclusion for league summaries.
                        o = off[off.season.eq(season)]
                        mine = g[g.batter.isin(o.batter)]
                        if o.empty or mine.empty: continue
                        rr = dict(axis=ax,method=m,stage=stage,contact_type=ct,group=group,season=season,
                                  official_n=o.n_swings.sum(),local_n=mine.local_n.sum(),
                                  official_players=len(o),matched_players=len(mine),
                                  coverage=mine.known_n.sum()/mine.local_n.sum())
                        ostats = weighted_stats(o)
                        for c in RATES[ax]:
                            rr[c+'_pred'] = np.average(mine[c],weights=mine.local_n)
                            rr[c+'_official'] = ostats[c]
                        for c in TAILS[ax]:
                            ok = mine[c].notna() & mine[c+'_n'].gt(0)
                            rr[c+'_pred'] = np.average(mine.loc[ok,c],weights=mine.loc[ok,c+'_n']) if ok.any() else np.nan
                            rr[c+'_official'] = ostats[c]
                        leagues.append(rr)
    return pd.concat(outputs,ignore_index=True), pd.DataFrame(leagues), pd.DataFrame(covers)

def add_null(player, official):
    dev = official[official.season.eq(2025) & official.group.eq('ALL')]
    targets = {ct:weighted_stats(g) for ct,g in dev.groupby('contact_type')}
    records=[]
    for ax in AXES:
        base = player[player.axis.eq(ax) & player.method.eq('author') & player.stage.eq('raw')].copy()
        base['method']='dev_null'; base['stage']='baseline'; base['protocol']='frozen_2025'; base['route']='aggregate_null'
        for c in (*RATES[ax],*TAILS[ax]): base[c+'_pred'] = base.contact_type.map({ct:v[c] for ct,v in targets.items()})
        base['rate_error_pp'] = np.mean([100*(base[c+'_pred']-base[c+'_official']).abs() for c in RATES[ax]],axis=0)
        for c,rate in zip(TAILS[ax],(RATES[ax][0],RATES[ax][2])):
            base[c+'_error']=(base[c+'_pred']-base[c+'_official']).abs()
            base[c+'_n']=base.local_n*base[rate+'_pred']
        base['coverage']=1.; base['known_n']=base.local_n
        records.append(base)
    return pd.concat([player,*records],ignore_index=True)

def correlation(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<3 or np.std(a[good])<1e-12 or np.std(b[good])<1e-12: return np.nan
    return np.corrcoef(a[good],b[good])[0,1]

def summarize(player):
    # Common finite-tail cells across every method/stage; preserve individual coverage too.
    cellkeys=KEYS+['axis']
    for ax in AXES:
        sel=player.axis.eq(ax)
        for c in TAILS[ax]:
            valid = player.loc[sel,[c+'_pred',c+'_official']].notna().all(axis=1)
            tmp=player.loc[sel,cellkeys].assign(valid=valid)
            common=tmp.groupby(cellkeys).valid.transform('all')
            player.loc[sel,c+'_common']=common.to_numpy()
    overall=[]; cats=[]; tails=[]
    keys=['season','contact_type','group','axis','method','stage','protocol','route']
    for key,g in player[player.eligible].groupby(keys,dropna=False):
        tags=dict(zip(keys,key)); ax=tags['axis']
        overall.append(dict(**tags,players=len(g),swings=int(g.local_n.sum()),official_swings=int(g.n_swings.sum()),
            mae_pp=g.rate_error_pp.mean(),median_error_pp=g.rate_error_pp.median(),
            coverage=np.average(g.coverage,weights=g.local_n),
            fallback_ev50=np.average(g['fallback_x_ev50' if ax=='x' else 'fallback_y_ev50'],weights=g.local_n) if ax!='z' else np.nan))
        for c in RATES[ax]:
            a,b=g[c+'_pred'],g[c+'_official']
            cats.append(dict(**tags,category=c,players=len(g),mae_pp=100*(a-b).abs().mean(),
                pearson_r=correlation(a,b),spearman_r=correlation(a.rank(),b.rank())))
        for c in TAILS[ax]:
            good=g[[c+'_pred',c+'_official']].notna().all(axis=1)
            for cohort,mask in [('available',good),('common',g[c+'_common'].fillna(False).astype(bool))]:
                t=g[mask]; a,b=t[c+'_pred'],t[c+'_official']
                tails.append(dict(**tags,tail=c,cohort=cohort,units='ms' if ax=='y' else 'inches',
                    eligible_players=len(g),players=len(t),excluded=len(g)-len(t),coverage=len(t)/len(g),
                    predicted_tail_swings=t[c+'_n'].sum(),mae=(a-b).abs().mean(),median_error=(a-b).abs().median(),
                    pearson_r=correlation(a,b)))
    return player,pd.DataFrame(overall),pd.DataFrame(cats),pd.DataFrame(tails)

def clustered_ci(d, value, seed=180, reps=1000):
    # Sample whole batter clusters, preserving group-cell weighting within each draw.
    g=d.groupby('batter')[value].agg(['sum','count'])
    if not len(g): return (np.nan,np.nan)
    rng=np.random.default_rng(seed); boot=[]
    for i in range(0,reps,100):
        ix=rng.integers(len(g),size=(min(100,reps-i),len(g)))
        boot.extend(g['sum'].to_numpy()[ix].sum(axis=1)/g['count'].to_numpy()[ix].sum(axis=1))
    return tuple(np.quantile(boot,[.025,.975]))

def paired_comparisons(player):
    rows=[]
    for group_selection in ('ALL','GROUPS'):
        use=player[player.eligible & (player.group.eq('ALL') if group_selection=='ALL' else player.group.ne('ALL'))]
        for (season,ct,ax),g in use.groupby(['season','contact_type','axis']):
            key=['batter','group']
            for stage in STAGES:
                base=g[g.method.eq('author') & g.stage.eq(stage)]
                for (method,st),a in g.groupby(['method','stage']):
                    if st!=stage and method!='dev_null': continue
                    metrics=['rate_error_pp',*TAILS[ax]]
                    for metric in metrics:
                        col=metric if metric=='rate_error_pp' else metric+'_error'
                        b=base; aa=a
                        if metric!='rate_error_pp':
                            b=b[b[metric+'_common'].fillna(False).astype(bool)]
                            aa=aa[aa[metric+'_common'].fillna(False).astype(bool)]
                        pair=aa[key+[col]].merge(b[key+[col]],on=key,suffixes=('_alt','_base')).dropna()
                        pair['delta']=pair[col+'_alt']-pair[col+'_base']
                        lo,hi=clustered_ci(pair,'delta')
                        rows.append(dict(season=season,contact_type=ct,axis=ax,group=group_selection,stage=stage,
                            method=method,baseline='author',metric=metric,cells=len(pair),batters=pair.batter.nunique(),
                            difference=pair.delta.mean(),ci_low=lo,ci_high=hi))
            # Calibration gain, paired on the same eligible batter and tail cells.
            for method,a in g.groupby('method'):
                pair=a[a.stage.eq('calibrated')][key+['rate_error_pp']].merge(a[a.stage.eq('raw')][key+['rate_error_pp']],on=key,suffixes=('_alt','_base'))
                if pair.empty: continue
                pair['delta']=pair.rate_error_pp_alt-pair.rate_error_pp_base
                lo,hi=clustered_ci(pair,'delta')
                rows.append(dict(season=season,contact_type=ct,axis=ax,group=group_selection,stage='calibrated_minus_raw',
                    method=method,baseline=method,metric='rate_error_pp',cells=len(pair),batters=pair.batter.nunique(),
                    difference=pair.delta.mean(),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)

def contrasts(player):
    cells=[]; summary=[]
    use=player[player.eligible & player.group.ne('ALL')]
    keys=['season','contact_type','axis','method','stage']
    for key,g in use.groupby(keys):
        tags=dict(zip(keys,key)); ax=tags['axis']
        for left,right in [('FB','BB'),('FB','CH'),('BB','CH')]:
            a=g[g.group.eq(left)].merge(g[g.group.eq(right)],on='batter',suffixes=('_a','_b'))
            for stat in (*RATES[ax],*TAILS[ax]):
                is_rate=stat in RATES[ax]; scale=100 if is_rate else 1
                ok=np.isfinite(a[[stat+'_pred_a',stat+'_pred_b',stat+'_official_a',stat+'_official_b']]).all(axis=1)
                if not is_rate: ok &= a[stat+'_common_a'].fillna(False).astype(bool)&a[stat+'_common_b'].fillna(False).astype(bool)
                b=a[ok]
                pred=scale*(b[stat+'_pred_a']-b[stat+'_pred_b']); truth=scale*(b[stat+'_official_a']-b[stat+'_official_b'])
                err=(pred-truth).abs()
                summary.append(dict(**tags,contrast=left+'-'+right,statistic=stat,units='pp' if is_rate else ('ms' if ax=='y' else 'inches'),
                    players=len(b),eligible_pairs=len(a),mae=err.mean(),median_error=err.median(),pearson_r=correlation(pred,truth)))
                for pid,pr,tr,er in zip(b.batter,pred,truth,err): cells.append(dict(**tags,contrast=left+'-'+right,statistic=stat,batter=pid,predicted=pr,official=tr,error=er))
    return pd.DataFrame(summary),pd.DataFrame(cells)

def sensitivities(player):
    rows=[]
    for tol in (0.,.02,.05,.1):
        for minimum in (25,50,100):
            use=player[player.absolute_count_gap.le(tol+1e-12) & player.n_swings.ge(minimum) & player.local_n.ge(minimum)]
            for supported in (False,True):
                s=use[use.fallback_x_ev50.eq(0)] if supported else use
                z=s.groupby(['season','contact_type','group','axis','method','stage']).agg(players=('batter','size'),swings=('local_n','sum'),mae_pp=('rate_error_pp','mean')).reset_index()
                z['tolerance']=tol; z['minimum']=minimum; z['supported_x_ev50_only']=supported
                rows.append(z)
    return pd.concat(rows,ignore_index=True)
