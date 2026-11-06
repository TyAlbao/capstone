"""Four-file output contract and reversible, compact views for regression checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from . import data

FILES={'metrics.csv','players.parquet','swings.parquet','report.md'}
PITCH_KEYS=['game_pk','at_bat_number','pitch_number']


def schema(frame):
    return {c:str(frame[c].dtype) for c in frame}


def cast(frame,types):
    result=frame.copy()
    for c,t in types.items():
        if c not in result:result[c]=np.nan
        if t=='bool':result[c]=result[c].map({'True':True,'False':False,True:True,False:False})
        result[c]=result[c].astype(t)
    return result[list(types)]


def write_parquet(frame,path,metadata):
    table=pa.Table.from_pandas(frame,preserve_index=False)
    meta=dict(table.schema.metadata or {})
    meta[b'capstone']=json.dumps(metadata,default=str).encode()
    table=table.replace_schema_metadata(meta)
    pq.write_table(table,path,compression='zstd',compression_level=9)


def metadata(path):
    return json.loads(pq.read_schema(path).metadata[b'capstone'])


def protocol(axis,method,contact_type='in_play'):
    if method=='existing':return 'historical_leaky'
    if axis=='x' and method=='author' and contact_type!='whiff':return 'author_season_reference'
    return 'frozen_2025'


def method_id(axis,method,stage,reference=None,contact_type='in_play'):
    axis=axis if isinstance(axis,str) else 'all_axes'
    method=method if isinstance(method,str) else 'author'
    stage=stage if isinstance(stage,str) else 'raw'
    reference=reference if isinstance(reference,str) else protocol(axis,method,contact_type)
    if method=='author_global':method='author_native' if stage=='calibrated' else 'author'
    if method=='author_native':
        if stage=='raw':method='author';reference=protocol(axis,method,contact_type)
        else:reference='author_native_2025'
    if method in ('dev_null','dev_mean','dev_median','dev_majority','development_rate_null'):stage='baseline'
    if method=='whiff_candidate' or (method=='author' and axis in ('x','z') and contact_type=='whiff'):
        method='published_whiff_regression';reference='frozen_2025'
    return '_'.join((axis,method,stage,reference))


def estimate_id(column):
    ax,rest=column.split('_',1)
    stage='calibrated' if rest.endswith('_calibrated') else 'raw'
    method=rest.removesuffix('_'+stage)
    return method_id(ax,method,stage,'author_native_2025' if method=='author_native' else None)


def pack_swings(frames):
    original=frames['predictions'];out=original.copy();rename={'group':'pitch_group'}
    for c in original:
        base=c.removesuffix('_category')
        if base[:2] in ('x_','y_','z_') and base.endswith(('_raw','_calibrated')):
            rename[c]=estimate_id(base)+('__classification' if c.endswith('_category') else '__value')
    out=out.rename(columns=rename)
    info={'predictions':{'types':schema(original),'columns':{c:rename.get(c,c) for c in original}}}
    index=pd.MultiIndex.from_frame(original[PITCH_KEYS])
    if len(out)!=974886 or not index.is_unique:raise ValueError('The fixed 974,886-swing population changed')
    def positions(frame):
        ix=index.get_indexer(pd.MultiIndex.from_frame(frame[PITCH_KEYS]))
        if (ix<0).any() or len(np.unique(ix))!=len(ix):raise ValueError('Export would lose or duplicate swings')
        return ix
    def scatter(values,ix,name):
        # Nullable storage avoids converting integer IDs to imprecise floats.
        v=values.reset_index(drop=True)
        if pd.api.types.is_integer_dtype(v.dtype):v=v.astype('Int64')
        elif pd.api.types.is_bool_dtype(v.dtype):v=v.astype('boolean')
        if isinstance(v.dtype,pd.CategoricalDtype):v=v.astype('string')
        sparse=pd.Series(v.array,index=ix).reindex(out.index)
        out[name]=sparse
    if 'whiffs' in frames:
        w=frames['whiffs'];ix=positions(w);mapping={}
        # Raw selection JSON is already retained in the immutable label dataset.
        omit={'selected_by_json','lists'}|{c for c in w if c.startswith('error_')}
        for c in w:
            if c in omit:continue
            candidate=rename.get(c,c)
            if not c.startswith('common_') and candidate in out and out.loc[ix,candidate].reset_index(drop=True).equals(w[c].reset_index(drop=True)):
                mapping[c]=candidate;continue
            if c.startswith(('pred_','category_')):
                prefix,ax,m=c.split('_',2)
                stage='baseline' if m.startswith('dev_') else 'raw'
                name=method_id(ax,m,stage,'selected_whiff_tails')+('__value' if prefix=='pred' else '__classification')
            else:name={'sav_x':'observed_x_inches','sav_y':'observed_y_ms','sav_z':'observed_z_inches',
                       'play_id':'savant_play_id','timing_x_tiedupflail':'observed_x_classification',
                       'timing_y_earlylate':'observed_y_classification','timing_z_overunder':'observed_z_classification'}.get(c,'tail_'+c)
            scatter(w[c],ix,name);mapping[c]=name
        info['whiffs']={'types':schema(w),'columns':mapping}
    if 'reference_predictions' in frames:
        ref=frames['reference_predictions'];blocks=[]
        for (year,reference,stage),g in ref.groupby(['season','protocol','stage'],sort=False,dropna=False):
            ix=positions(g);stage=None if pd.isna(stage) else stage
            flag=f'reference_{reference}_eligible'
            # Raw/calibrated views share the same membership and eligibility.
            if flag not in out:out[flag]=pd.Series(pd.array([pd.NA]*len(out),dtype='boolean'))
            out.loc[ix,flag]=g.primary_eligible.to_numpy()
            mapping={}
            for c in ref:
                if not c.startswith('x_') or g[c].isna().all():continue
                name=method_id('x',c[2:],stage or 'raw',reference)+'__value'
                if name not in out:out[name]=np.nan
                out.loc[ix,name]=g[c].to_numpy();mapping[c]=name
                label=name.removesuffix('__value')+'__classification'
                if label not in out:out[label]=pd.Series(pd.NA,index=out.index,dtype='string')
                out.loc[ix,label]=data.discover.label_of('x',g[c])
            blocks.append(dict(season=int(year),protocol=reference,stage=stage,flag=flag,columns=mapping))
        info['reference_predictions']={'types':schema(ref),'blocks':blocks}
    return out,info


def unpack_swings(out,info,name):
    spec=info[name]
    if name=='predictions':
        return cast(pd.DataFrame({c:out[new].array for c,new in spec['columns'].items()}),spec['types'])
    if name=='whiffs':
        source=pd.read_parquet(data.ROOT/'data/savant_whiff_tails/whiff_tail_matched.parquet',columns=['play_id','selected_by_json','lists']).set_index('play_id')
        pid=spec['columns']['play_id']
        selected=out[out[pid].notna()].set_index(pid,drop=False).loc[source.index].reset_index(drop=True)
        result=pd.DataFrame({c:selected[new].array for c,new in spec['columns'].items()})
        # Source-only provenance and row order come from the immutable label dataset.
        for c in ('selected_by_json','lists'):result[c]=result.play_id.map(source[c])
        for c in spec['types']:
            if c.startswith('error_'):
                _,ax,m=c.split('_',2);result[c]=result[f'pred_{ax}_{m}']-result['sav_'+ax]
        return cast(result,spec['types'])
    parts=[]
    for b in spec['blocks']:
        select=out.season.eq(b['season'])&out[b['flag']].notna()
        base=out.loc[select]
        p=base[PITCH_KEYS+['batter','season']].copy()
        p['primary_eligible']=base[b['flag']].to_numpy()
        p['protocol']=b['protocol'];p['stage']=b['stage']
        for c,col in b['columns'].items():p[c]=base[col].to_numpy()
        parts.append(p)
    return cast(pd.concat(parts,ignore_index=True),spec['types'])


def pack_players(player):
    generic={};mapped=set()
    for ax in data.AXES:
        for position,rate in zip(('low','middle','high'),data.RATES[ax]):
            for side in ('pred','official'):
                old=rate+'_'+side;new=('predicted' if side=='pred' else 'official')+'_'+position+'_rate'
                generic.setdefault(new,{})[ax]=old;mapped.add(old)
        for position,tail in zip(('low','high'),data.TAILS[ax]):
            for suffix in ('pred','official','n','error','common'):
                old=tail+'_'+suffix;new=f'tail_{position}_{suffix}'
                generic.setdefault(new,{})[ax]=old;mapped.add(old)
    rename={'group':'pitch_group','local_n':'n_swings','known_n':'n_known','n_swings':'n_official_swings','rate_error_pp':'classification_mae_pp'}
    out=player[[c for c in player if c not in mapped]].rename(columns=rename).copy()
    for new,by_axis in generic.items():
        result=pd.Series(np.nan,index=player.index,dtype=object if new.endswith('_common') else float)
        for ax,col in by_axis.items():result.loc[player.axis.eq(ax)]=player.loc[player.axis.eq(ax),col]
        out[new]=result.astype('boolean') if new.endswith('_common') else result
    out.loc[out.axis.eq('x')&out.method.eq('author')&out.contact_type.eq('whiff'),'protocol']='frozen_2025'
    out['method_id']=[method_id(a,m,s,p,ct) for a,m,s,p,ct in zip(out.axis,out.method,out.stage,out.protocol,out.contact_type)]
    keys=['season','batter','contact_type','pitch_group','method_id']
    if out.duplicated(keys).any():raise ValueError('Duplicate player/method cells')
    groups=['season','contact_type','pitch_group','axis','method','stage']
    eligible=out[out.eligible]
    for pos in ('low','middle','high'):
        for side,weight in [('predicted','n_swings'),('official','n_official_swings')]:
            col=f'{side}_{pos}_rate'
            temp=eligible[groups].copy();temp['weighted']=eligible[col]*eligible[weight];temp['n']=eligible[weight]
            sums=temp.groupby(groups,dropna=False)[['weighted','n']].transform('sum')
            out[f'{side}_{pos}_league_delta_pp']=np.nan
            out.loc[eligible.index,f'{side}_{pos}_league_delta_pp']=100*(eligible[col]-sums.weighted/sums.n)
    return out,dict(types=schema(player),rename=rename,generic=generic)


def unpack_players(frame,info):
    out=frame.rename(columns={v:k for k,v in info['rename'].items()}).copy()
    for new,by_axis in info['generic'].items():
        for ax,old in by_axis.items():
            out[old]=np.nan if not old.endswith('_common') else pd.Series(np.nan,index=out.index,dtype=object)
            sel=out.axis.eq(ax);out.loc[sel,old]=frame.loc[sel,new].to_numpy()
    out.loc[out.axis.eq('x')&out.method.eq('author')&out.contact_type.eq('whiff'),'protocol']='author_season_reference'
    return cast(out,info['types'])


COUNTS={'players':'n_batters','batters':'n_batters','n_batters':'n_batters','swings':'n_swings','n_swings':'n_swings',
        'official_swings':'n_official_swings','cells':'n_cells','n':'n_observations','n_total':'n_available',
        'n_excluded':'n_excluded','pitchers':'n_pitchers','eligible_players':'n_eligible_batters','excluded':'n_excluded_by_cohort',
        'predicted_tail_swings':'n_tail_swings','eligible_pairs':'n_eligible_pairs','local_swings':'n_swings',
        'known_swings':'n_known','official_n':'n_official_swings','local_n':'n_swings','official_players':'n_official_batters','matched_players':'n_matched_batters'}
DIMENSIONS={'season','batter','player_id','reference_season','bootstrap_seed','bootstrap_reps','confidence_level'}
ALIASES={'group':'pitch_group','metric':'target_metric','units':'axis_units'}
CI={'ci_low','ci_high','delta_ci_low','delta_ci_high','excludes_zero'}


def metric_units(metric,axis,axis_units=None):
    if metric in ('improvement','difference','delta_mae_pp') or metric.endswith(('_pp','_percent')):return 'percentage_points'
    if metric in ('combined_mae','tied_up_mae','centered_mae','flail_mae','handle_changed_percent','unknown_percent'):return 'percentage_points'
    if 'accuracy' in metric or metric in ('coverage','fallback_ev50'):return 'proportion'
    if metric in ('mae','rmse','bias','median_error','macro_mae','median_player_mae','p90_player_mae'):
        return axis_units if isinstance(axis_units,str) else ('ms' if axis=='y' else 'inches')
    if metric.endswith(('_r','_pearson')) or metric in ('r2','actual_on_pred_slope'):return 'unitless'
    if metric.startswith('n_') or metric in ('n','count','count_difference','boundary_total'):return 'count'
    if metric.endswith(('_pred','_official')) and 'percent' in metric:return 'proportion'
    if metric=='max_rate_gap':return 'proportion'
    if metric=='max_tail_gap':return 'mixed_axis_units'
    if metric.startswith('avg_') or metric=='actual_on_pred_intercept':return 'ms' if axis=='y' else 'inches'
    raise ValueError(f'Undefined units for metric: {metric}')


def encode_metrics(tables):
    parts=[];schemas={}
    for table,source in tables.items():
        source=source.reset_index(drop=True).copy()
        if source.empty:continue
        dims=[c for c in source if c in DIMENSIONS or c in COUNTS or (not pd.api.types.is_numeric_dtype(source[c]) or pd.api.types.is_bool_dtype(source[c])) and c not in CI]
        dims=[c for c in dims if c not in CI]
        values=[c for c in source if c not in dims and c not in CI]
        if not values:
            # A confusion matrix's count is the metric, not merely a sample-size field.
            values=[c for c in ('n',) if c in source];dims=[c for c in dims if c not in values]
        if not values:raise ValueError(f'No numeric metrics in {table}')
        rename={c:COUNTS.get(c,ALIASES.get(c,c)) for c in dims}
        if len(set(rename.values()))!=len(rename):raise ValueError(f'Ambiguous sample columns in {table}')
        if source.duplicated(dims).any():raise ValueError(f'Non-unique metric context in {table}')
        schemas[table]=dict(types=schema(source),dimensions=dims,values=values,rename=rename)
        src=source.copy();src['_record']=np.arange(len(src))
        long=src.rename(columns=rename).melt(id_vars=[rename[c] for c in dims]+['_record'],value_vars=values,var_name='metric',value_name='value')
        long['evaluation']=table
        low='ci_low' if 'ci_low' in source else 'delta_ci_low'
        high='ci_high' if 'ci_high' in source else 'delta_ci_high'
        ci_metric='difference' if 'difference' in source else ('improvement' if 'improvement' in source else 'delta_mae_pp')
        long['ci_low']=long._record.map(source[low]) if low in source else np.nan
        long['ci_high']=long._record.map(source[high]) if high in source else np.nan
        has_ci=long.metric.eq(ci_metric)
        long.loc[~has_ci,['ci_low','ci_high']]=np.nan
        # Preserve old calculations but express paired error differences as improvements.
        if ci_metric in ('difference','delta_mae_pp'):
            select=long.metric.eq(ci_metric)
            long.loc[select,'value']=-long.loc[select,'value']
            lo=long.loc[select,'ci_low'].copy();long.loc[select,'ci_low']=-long.loc[select,'ci_high'];long.loc[select,'ci_high']=-lo
            long.loc[select,'metric']='improvement'
        long['excludes_zero']=pd.Series(pd.NA,index=long.index,dtype='boolean')
        known=long.ci_low.notna()&long.ci_high.notna()
        long.loc[known,'excludes_zero']=(long.loc[known,'ci_low']>0)|(long.loc[known,'ci_high']<0)
        if table=='whiff_player_summary':
            from .whiffs import PRIMARY
            long['method']=long.axis.map(PRIMARY)
        for c,default in [('axis','x' if table.startswith('reference_') else None),('method','author_native' if table=='author_reproduction_metrics' else 'population_audit'),
                          ('stage','raw'),('contact_type','whiff' if table.startswith('whiff_') or table=='paired_whiff_accuracy' else ('in_play' if table.startswith('reference_') else 'all')),
                          ('pitch_group','ALL')]:
            if c not in long:long[c]=default
        if 'protocol' not in long:
            if table.startswith('whiff_') or table=='paired_whiff_accuracy':long['protocol']='selected_whiff_tails'
            elif table=='reference_raw':long['protocol']='same_season_raw'
            elif table=='reference_prior_season':long['protocol']='prior_season_raw'
            elif table=='reference_author_reproduction':long['protocol']='author_reproduction'
            else:long['protocol']=[protocol(a,m,ct) for a,m,ct in zip(long.axis,long.method,long.contact_type)]
        if table=='calibration_saturation':long['stage']='calibrated'
        actual_methods=long.method.copy()
        if 'experiment' in long:
            actual_methods=actual_methods.where(~long.experiment.eq('B2_sign_only'),actual_methods+'_sign_only')
        long['method_id']=[method_id(a,m,s,p,ct) for a,m,s,p,ct in zip(long.axis,actual_methods,long.stage,long.protocol,long.contact_type)]
        long['baseline_method_id']=pd.NA
        if 'baseline' in long:
            for idx,row in long[long.baseline.notna()].iterrows():
                bs=row.get('baseline_stage',row.stage)
                bp=row.protocol if table=='paired_reference_accuracy' or row.protocol=='selected_whiff_tails' else protocol(row.axis,row.baseline,row.contact_type)
                long.at[idx,'baseline_method_id']=method_id(row.axis,row.baseline,bs,bp,row.contact_type)
        elif ci_metric=='delta_mae_pp':
            long.loc[long.metric.eq('improvement'),'baseline_method_id']=[method_id(a,'global',s,p,ct) for a,s,p,ct in zip(long.loc[long.metric.eq('improvement'),'axis'],long.loc[long.metric.eq('improvement'),'stage'],long.loc[long.metric.eq('improvement'),'protocol'],long.loc[long.metric.eq('improvement'),'contact_type'])]
        if 'target_metric' not in long:long['target_metric']=pd.NA
        long.loc[long.metric.eq('improvement')&long.target_metric.isna(),'target_metric']='classification_rate_mae_pp'
        long['baseline_kind']=''
        paired=long.baseline_method_id.notna()
        for idx,row in long[paired].iterrows():
            ident=row.baseline_method_id
            kind='development_null' if any(v in ident for v in ('dev_null','dev_majority','development_rate_null')) else ('published_whiff_regression' if row.contact_type=='whiff' else ('author_contact_plus_published_whiff_regression' if row.contact_type=='all' and row.axis in ('x','z') else 'author_formula'))
            long.at[idx,'baseline_kind']=kind
        long['units']=[metric_units(m,a,u) for m,a,u in zip(long.metric,long.axis,long.get('axis_units',pd.Series(None,index=long.index)))]
        improvement=long.metric.eq('improvement')
        if 'bootstrap_reps' not in long:long['bootstrap_reps']=np.nan
        if 'bootstrap_seed' not in long:long['bootstrap_seed']=np.nan
        if 'confidence_level' not in long:long['confidence_level']=np.nan
        long.loc[known & long.bootstrap_reps.isna(),'bootstrap_reps']=5000 if table.startswith('reference_') else 1000
        long.loc[known & long.bootstrap_seed.isna(),'bootstrap_seed']=180
        long.loc[known & long.confidence_level.isna(),'confidence_level']=.95
        tail_improvement=improvement & long.target_metric.fillna('').str.startswith('avg_')
        long.loc[tail_improvement,'units']=np.where(long.loc[tail_improvement,'axis'].eq('y'),'ms','inches')
        if table=='author_reproduction_metrics':
            long.loc[long.metric.eq('mae')&long.statistic.str.endswith('_percent'),'units']='percentage_points'
        if table=='paired_comparisons':
            # Null targets retain baseline-stage context, but their own method ID is baseline.
            gain=long.stage.eq('calibrated_minus_raw') & long.baseline.notna()
            long.loc[gain,'baseline_method_id']=[method_id(a,m,'raw',p,ct) for a,m,p,ct in zip(long.loc[gain,'axis'],long.loc[gain,'method'],long.loc[gain,'protocol'],long.loc[gain,'contact_type'])]
        if table in ('reference_raw','reference_prior_season'):
            sign_only=long.metric.eq('improvement')&long.experiment.eq('B2_sign_only')
            long.loc[sign_only,'baseline_method_id']=[method_id(a,'global_sign_only',st,p,ct) for a,st,p,ct in zip(long.loc[sign_only,'axis'],long.loc[sign_only,'stage'],long.loc[sign_only,'protocol'],long.loc[sign_only,'contact_type'])]
        if table=='reference_author_reproduction':
            long.loc[improvement,'baseline_method_id']=long.loc[improvement,'method_id']
        if table in ('reference_raw','reference_prior_season','reference_calibrated'):
            long.loc[improvement,'baseline_kind']='global_reference'
        if table=='paired_comparisons':
            long.loc[long.stage.eq('calibrated_minus_raw'),'baseline_kind']='same_method_raw'
        # Stored metric contexts uniquely identify records; no synthetic CSV row IDs.
        parts.append(long.drop(columns='_record'))
    result=pd.concat(parts,ignore_index=True)
    for col in set(COUNTS.values())|{'season','reference_season','bootstrap_seed','bootstrap_reps'}:
        if col in result and pd.api.types.is_numeric_dtype(result[col]):
            known=result[col].dropna()
            if known.eq(np.floor(known)).all():result[col]=result[col].astype('Int64')
    priority=['evaluation','metric','method_id','axis','method','stage','protocol','season','contact_type','pitch_group',
              'value','units','n_batters','n_swings','n_official_swings','n_cells','n_observations',
              'baseline_method_id','baseline_kind','target_metric','ci_low','ci_high','excludes_zero',
              'confidence_level','bootstrap_reps','bootstrap_seed']
    return result[[c for c in priority if c in result]+[c for c in result if c not in priority]],schemas


def decode_metrics(long,schemas):
    result={}
    for table,s in schemas.items():
        d=long[long.evaluation.eq(table)].copy();keys=[s['rename'][c] for c in s['dimensions']]
        context=d[keys].drop_duplicates().reset_index(drop=True)
        context['_row']=np.arange(len(context))
        joined=d.merge(context,on=keys,how='left',validate='many_to_one')
        # Reverse the display sign convention before checking the frozen benchmarks.
        original_ci='difference' if 'difference' in s['types'] else ('improvement' if 'improvement' in s['types'] else 'delta_mae_pp')
        if original_ci in ('difference','delta_mae_pp'):
            use=joined.metric.eq('improvement');joined.loc[use,'metric']=original_ci;joined.loc[use,'value']=-joined.loc[use,'value']
            lo=joined.loc[use,'ci_low'].copy();joined.loc[use,'ci_low']=-joined.loc[use,'ci_high'];joined.loc[use,'ci_high']=-lo
        vals=joined.pivot(index='_row',columns='metric',values='value').reindex(context._row)
        frame=context.rename(columns={v:k for k,v in s['rename'].items()}).drop(columns='_row')
        for c in s['values']:frame[c]=vals[c].to_numpy()
        ci=joined[joined.metric.eq(original_ci)].set_index('_row').reindex(context._row)
        for col in ('ci_low','delta_ci_low','ci_high','delta_ci_high'):
            if col in s['types']:frame[col]=ci['ci_low' if col.endswith('low') else 'ci_high'].to_numpy()
        if 'excludes_zero' in s['types']:frame['excludes_zero']=ci.excludes_zero.to_numpy()
        result[table]=cast(frame,s['types'])
    return result


def read_tables(out):
    info=metadata(Path(out)/'players.parquet')
    long=pd.read_csv(Path(out)/'metrics.csv',float_precision='round_trip',keep_default_na=True,low_memory=False)
    return decode_metrics(long,info['metric_schemas'])


def legacy_frame(out,name):
    out=Path(out)
    if name=='players':return unpack_players(pd.read_parquet(out/'players.parquet'),metadata(out/'players.parquet')['players'])
    frame=pd.read_parquet(out/'swings.parquet');info=metadata(out/'swings.parquet')['frames']
    return unpack_swings(frame,info,name)

def population_views(out):
    """Recompute the old denominator views from exported membership and source counts."""
    from . import reconstruction as r, evaluation as e, references as ref, reference_calibration as rc
    out=Path(out);info=metadata(out/'players.parquet')['population_schemas']
    swings=legacy_frame(out,'predictions');official,_=data.ingest(write=False)
    den=e.denominator_audit(swings,official)
    result={'populations':cast(den,info['populations'])}
    if 'reference_populations' not in info:return result
    pred=legacy_frame(out,'reference_predictions');rows=[]
    cache=data.ROOT/'data/savant_calibration';source=data.ROOT/'data/statcast_2024_2026.parquet'
    columns=pq.ParquetFile(source).schema_arrow.names
    seasons={}
    for year in (2024,2025,2026):
        raw=pd.read_parquet(source,columns=[c for c in ref.KEEP if c in columns],filters=[('game_year','==',year)])
        s=ref.prepare_season(raw,year);seasons[year]=s
        truth=ref.normalize_official(pd.read_csv(cache/f'reference_leaderboard_{year}.csv'),year)
        for protocol in ('same_season_raw','prior_season_raw'):
            p=pred[pred.season.eq(year)&pred.protocol.eq(protocol)]
            if p.empty:continue
            a=ref.denominator_audit(s,p,truth,ref.Config(),True)
            rows.append(a.assign(season=year,protocol=protocol))
    for year in (2025,2024,2026):
        s=seasons[year];s=s[s.description.isin(ref.original_swing_descriptions())]
        for protocol in ('frozen_2025_references','same_season_descriptive'):
            p=pred[pred.season.eq(year)&pred.protocol.eq(protocol)&pred.stage.eq('raw')]
            if p.empty:continue
            p=p.merge(s[PITCH_KEYS+['pitcher']],on=PITCH_KEYS,validate='one_to_one')
            for kind in ('batter','pitcher'):
                a=rc.eligible_ids(p,s[s.contact_type.eq('in_play')],rc.official(year,kind,'in_play'),kind,True)
                rows.append(a.assign(season=year,protocol=protocol,perspective=kind))
    # Author population rules do not depend on any fitted value.
    config=metadata(out/'players.parquet')['config']
    if 'author_reproduction' in config['reference_protocols']:
        for year in (2025,2024,2026):
            s=seasons[year];s=s[s.description.isin(ref.original_swing_descriptions())]
            for ct in ('in_play','foul'):
                sub=s[s.contact_type.eq(ct)].reset_index(drop=True)
                for kind in ('batter','pitcher'):
                    for population,matched in [('author_rules',False),('matched_10pct',True)]:
                        a=rc.eligible_ids(sub,sub,rc.official(year,kind,ct),kind,matched)
                        rows.append(a.assign(season=year,contact_type=ct,protocol='author_reproduction',population=population,perspective=kind))
    result['reference_populations']=cast(pd.concat(rows,ignore_index=True),info['reference_populations'])
    return result


def write_report(path,tables,config,verification):
    lines=['# XYZ method comparison','','## Outputs and interpretation','',
        '- `metrics.csv`: one numeric metric per row, with population/context, sample sizes and applicable paired confidence intervals.',
        '- `players.parquet`: unique batter/season/contact type/pitch group/method cells. Classification rates are proportions; league-relative deviations are percentage points within the same eligible cohort.',
        '- `swings.parquet`: exactly 974,886 unique tracked swings. Columns ending `__value` are continuous estimates; `__classification` holds their labels. Missing inputs and unevaluated historical/sample routes remain missing.',
        '- This report contains configuration, verification and decision guidance. No separate manifests, figures or intermediate datasets are produced.','',
        'X and Z are inches; Y is timing in milliseconds. Category bands are X ±4, Y ±7 and Z ±2, inclusive in the middle. Low/middle/high player-rate columns mean tied-up/centered/flail for X, late/on-time/early for Y, and over/lined-up/under for Z.','',
        'Development is 2025; 2024 is backward transfer and 2026 forward transfer. Both transfer seasons were previously examined. Raw and calibrated methods remain separate. All comparisons retain the original eligibility rules and shared input masks.','',
        '## Paired uncertainty','',
        'Improvement = baseline error minus method error, in percentage points: positive favors the method. Intervals use 1,000 paired bootstrap resamples of whole batters (seed 180), retaining all their group/category/contrast cells. Classification MAE averages paired batter/group errors. Contrast comparisons average the three category-error cells per batter/contrast; ALL_CONTRASTS pools those cells across the three pitch-group pairs. Each method is paired with the same-stage author comparison route (original formulas with the existing common-population calibration) and with development nulls where applicable. Legacy calibration-gain and global-reference comparisons retain their original intervals and are labeled separately. Missing intervals mean they were not computed. Intervals are descriptive, without adjustment for multiple comparisons.','',
        'The X/Z whiff author rows in the aggregate experiment use published whiff regression formulas. These are available coefficient baselines, not recovered original trained advanced models. The latter remain unavailable. Selected-tail whiff classification comparisons are separately labeled and cannot establish accuracy for all whiffs. Contact validation is aggregate, not pitch-level contact ground truth.','',
        'Historical X reference protocols retain their original populations and references. `existing` is the historical pooled/leaky sensitivity; same-season reference experiments are descriptive. Reference confidence intervals compare the author on the identical historical subset, not on a different full-season cohort.','',
        '## 2026 aggregate classification-rate comparisons','',
        '| Contact | Axis | Method | Stage | Batters | MAE pp | Improvement vs author pp [95% CI] |','|---|---|---|---|---:|---:|---|']
    main=tables.get('overall_metrics',pd.DataFrame())
    if not main.empty:
        score=main[main.season.eq(2026)&main.group.eq('ALL')]
        old_pairs=tables['paired_comparisons']
        for q in score.itertuples():
            pair=old_pairs[old_pairs.season.eq(2026)&old_pairs.contact_type.eq(q.contact_type)&old_pairs.axis.eq(q.axis)&old_pairs.method.eq(q.method)&old_pairs.stage.eq(q.stage)&old_pairs.group.eq('ALL')&old_pairs.metric.eq('rate_error_pp')]
            interval='not applicable'
            if len(pair):
                b=pair.iloc[0];interval=f'{-b.difference:.4f} [{-b.ci_high:.4f}, {-b.ci_low:.4f}]'
            lines.append(f'| {q.contact_type} | {q.axis} | {q.method} | {q.stage} | {q.players} | {q.mae_pp:.4f} | {interval} |')
    lines+=['','## Within-batter pitch-group comparisons','',
        '| Contact | Axis | Method | Stage | Improvement vs author pp | 95% CI | Excludes zero |','|---|---|---|---|---:|---|---|']
    pairs=tables.get('paired_relative_accuracy',pd.DataFrame())
    if not pairs.empty:
        use=pairs[pairs.season.eq(2026)&pairs.baseline.eq('author')&pairs.target_metric.eq('within_batter_pitch_group_contrast_mae_pp')&pairs.contrast.eq('ALL_CONTRASTS')]
        lines += [f'| {q.contact_type} | {q.axis} | {q.method} | {q.stage} | {q.improvement:.4f} | [{q.ci_low:.4f}, {q.ci_high:.4f}] | {q.excludes_zero} |' for q in use.itertuples()]
    lines+=['','## Decision guidance','',
        'Choose methods within an axis, contact type and calibration stage. Prioritize reproducible pitch-group contrast improvements, then category-rate errors and batter-rank correlations. A lower point estimate with an interval spanning zero is not a clear improvement. Always check the null baseline, coverage and population. Physical deviation MAE is secondary.','']
    if not pairs.empty:
        use=pairs[pairs.season.eq(2026)&pairs.baseline.eq('author')&pairs.target_metric.eq('within_batter_pitch_group_contrast_mae_pp')&pairs.contrast.eq('ALL_CONTRASTS')&pairs.method.ne('dev_null')&pairs.method.ne('existing')]
        for (ct,axis,stage),g in use.groupby(['contact_type','axis','stage']):
            best=g.sort_values('improvement',ascending=False).iloc[0]
            label='Supported improvement' if best.ci_low>0 else 'No clear improvement over the author'
            lines.append(f'- **{ct}, {axis.upper()}, {stage}: {label}.** Best observed contrast improvement: `{best.method}`, {best.improvement:.3f} pp (95% CI {best.ci_low:.3f} to {best.ci_high:.3f}). Check its classification-rate and rank rows before selecting it; this is not automatic model selection.')
    lines+=['','## Verification','',f"Status: {'PASS' if verification['passed'] else 'FAIL'}. Historical metric tables: {len(verification.get('metrics',{}))}. Unique swings: 974,886. Preserved regression fixtures and protected inputs were checked; no source datasets or frozen parameters were rewritten.",'',
        '## Reproduce','', '```powershell','uv run --no-sync python main.py run','uv run --no-sync python main.py verify','```','',
        'Configuration for this run (also embedded in Parquet metadata):','```json',json.dumps(config,indent=2),'```','']
    path.write_text('\n'.join(lines),encoding='utf-8')
