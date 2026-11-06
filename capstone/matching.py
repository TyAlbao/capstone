"""Deduplicate, conservatively match, and audit cached Savant whiff tails.

python main.py match
No model training, calibration, or model evaluation occurs in this script.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True

sys.path.insert(0,str(ROOT/"external/statcast-contact-point"))
from .collection import CACHE,REPORT,LISTS,atomic_json,validate_tail,utcnow
from cp_lib import join, savant_truth as st

DATA=ROOT/"data/savant_whiff_tails"
SOURCE=ROOT/"data/statcast_2024_2026.parquet"
NUMERIC={"sav_x":"delta_batball_tiedup_neg_x2","sav_y":"delta_batball_late_neg_y2_msec",
         "sav_z":"ball_pos_above_plane","miss_distance_inches":"miss_distance_inches"}
CRITICAL=["game_pk","pitcher","savant_batter_id","game_date","pitch_type",*NUMERIC.values(),
          "timing_x_tiedupflail","timing_y_earlylate","timing_z_overunder"]


def number(value):
    try:return float(value) if value is not None else np.nan
    except (TypeError,ValueError):return np.nan


def equivalent(a,b):
    na,nb=number(a),number(b)
    if np.isfinite(na) and np.isfinite(nb):return bool(np.isclose(na,nb,rtol=0,atol=1e-9))
    return a==b


def deduplicate(occurrences):
    """Stable first-nonmissing canonical fields; never hide conflicting versions."""
    groups={};orphans=[]
    for o in occurrences:
        row=json.loads(o["raw_row_json"])
        pid=row.get("play_id")
        if not pid:
            orphans.append(o);continue
        groups.setdefault(str(pid),[]).append((o,row))
    labels=[];conflicts=[]
    for play_id,items in sorted(groups.items()):
        canonical={};values={}
        for _,r in items:
            for field,value in r.items():
                if value is None:continue
                if field not in canonical:canonical[field]=value
                distinct=values.setdefault(field,[])
                if not any(equivalent(value,v) for v in distinct):distinct.append(value)
        conflict_fields=[k for k,v in values.items() if len(v)>1]
        for field in conflict_fields:
            conflicts.append(dict(play_id=play_id,field=field,critical=field in CRITICAL,values_json=json.dumps(values[field])))
        c={k:v for k,v in canonical.items() if not isinstance(v,(list,dict))}
        c["play_id"]=play_id
        c["canonical_row_json"]=json.dumps(canonical,sort_keys=True)
        sources=[dict(perspective=o["perspective"],player_id=o["player_id"],list=o["list"],rank=o["rank"],
                      requested_season=o["requested_season"],response_sha256=o["response_sha256"]) for o,_ in items]
        requested=sorted(set(o["requested_season"] for o,_ in items))
        date=pd.to_datetime(c.get("game_date"),utc=True,errors="coerce")
        c["season"]=int(date.year) if pd.notna(date) else requested[0]
        c["game_date"]=date.strftime("%Y-%m-%d") if pd.notna(date) else None
        c["requested_seasons_json"]=json.dumps(requested)
        c["requested_season_mismatch"]=any(y!=c["season"] for y in requested)
        c["cross_season_play_id"]=len(requested)>1
        c["selected_by_json"]=json.dumps(sources,sort_keys=True)
        c["lists"]=",".join(sorted({f"{o['perspective']}:{o['list']}" for o,_ in items}))
        c["perspectives"]=",".join(sorted({o["perspective"] for o,_ in items}))
        c["n_selections"]=len(items)
        c["n_selection_lists"]=len({o["list"] for o,_ in items})
        c["n_selecting_players"]=len({(o["perspective"],o["player_id"]) for o,_ in items})
        c["conflict_fields"]=",".join(conflict_fields)
        c["critical_conflict"]=bool(set(conflict_fields)&set(CRITICAL))
        c["batter"]=c.get("savant_batter_id")
        c["selector_mismatch"]=any(number(o["player_id"])!=number(r.get("savant_batter_id" if o["perspective"]=="batter" else "pitcher")) for o,r in items)
        for alias,field in NUMERIC.items():
            c[alias]=number(canonical.get(field))
            c[f"invalid_numeric_{alias}"]=canonical.get(field) is not None and not np.isfinite(c[alias])
        for axis,limit in (("x",36.),("z",18.)):
            v=c[f"sav_{axis}"]
            c[f"{axis}_at_positive_cap"]=bool(np.isfinite(v) and np.isclose(v,limit,rtol=0,atol=1e-8))
            c[f"{axis}_at_negative_cap"]=bool(np.isfinite(v) and np.isclose(v,-limit,rtol=0,atol=1e-8))
            c[f"{axis}_beyond_candidate_cap"]=bool(np.isfinite(v) and abs(v)>limit+1e-8)
        c["candidate_cap_flag"]=any(c[k] for k in ("x_at_positive_cap","x_at_negative_cap","z_at_positive_cap","z_at_negative_cap"))
        # Additional endpoints observed in the partial cache, retained as candidate
        # reporting boundaries rather than asserted physical limits.
        c["x_at_minus24"]=bool(np.isfinite(c["sav_x"]) and np.isclose(c["sav_x"],-24,rtol=0,atol=1e-8))
        c["z_at_plus12"]=bool(np.isfinite(c["sav_z"]) and np.isclose(c["sav_z"],12,rtol=0,atol=1e-8))
        c["candidate_cap_flag"] |= c["x_at_minus24"] or c["z_at_plus12"]
        c["miss_at_point1"]=bool(np.isfinite(c["miss_distance_inches"]) and np.isclose(c["miss_distance_inches"],.1,rtol=0,atol=1e-8))
        c["finite_xyz"]=all(np.isfinite(c[f"sav_{a}"]) for a in "xyz")
        # A metadata consistency check, not evaluation of any reconstructed model.
        expected={"x":("Tiedup","Centered","Flail",4),"y":("Late","OnTime","Early",7),"z":("Over","Linedup","Under",2)}
        cat_cols={"x":"timing_x_tiedupflail","y":"timing_y_earlylate","z":"timing_z_overunder"}
        for axis,(low,mid,high,threshold) in expected.items():
            v=c[f"sav_{axis}"];label=c.get(cat_cols[axis])
            c[f"{axis}_category_mismatch"]=bool(np.isfinite(v) and label is not None and label!=(low if v < -threshold else high if v > threshold else mid))
        labels.append(c)
    frame=pd.DataFrame(labels)
    if len(frame):
        for field in set(NUMERIC.values())|{"abs_delta_batball_tiedup_neg_x2","abs_delta_batball_late_neg_y2_msec","abs_ball_pos_above_plane"}:
            if field in frame:frame[field]=pd.to_numeric(frame[field],errors="coerce")
        for field in ("batter","savant_batter_id","pitcher","game_pk","season"):
            frame[field]=pd.to_numeric(frame[field],errors="coerce").astype("Int64")
    return frame,pd.DataFrame(conflicts,columns=["play_id","field","critical","values_json"]),pd.DataFrame(orphans)


def load_occurrences(seasons):
    occurrences=[];responses=[]
    for season in seasons:
        registry=pd.read_csv(REPORT/f"player_registry_{season}.csv")
        for r in registry.sort_values(["perspective","player_id"]).itertuples():
            path=CACHE/f"{season}/{r.perspective}/{r.player_id}.json"
            meta_path=path.with_suffix(".meta.json")
            rec=dict(season=season,perspective=r.perspective,player_id=int(r.player_id),n_swings=r.n_swings)
            if not path.exists() or not meta_path.exists():
                responses.append(dict(**rec,status="missing"));continue
            body=path.read_bytes();meta=json.loads(meta_path.read_text())
            if hashlib.sha256(body).hexdigest()!=meta["sha256"]:raise ValueError(f"Invalid checksum: {path}")
            payload=json.loads(body);schema=validate_tail(payload)
            counts={f"n_{key}":len(payload.get(key,[])) for key in LISTS}
            responses.append(dict(**rec,status="ok",schema=schema,**counts,total_entries=sum(len(v) for v in payload.values()),
                                  fetched_utc=meta["started_utc"],response_sha256=meta["sha256"]))
            for key,rows in payload.items():
                for rank,row in enumerate(rows,1):
                    occurrences.append(dict(requested_season=season,perspective=r.perspective,player_id=int(r.player_id),
                        list=key,rank=rank,play_id=row.get("play_id"),response_sha256=meta["sha256"],
                        response_url=meta["url"],raw_row_json=json.dumps(row,sort_keys=True)))
    return occurrences,pd.DataFrame(responses)


def load_local(season):
    cols=["game_year","game_type","game_date","game_pk","at_bat_number","pitch_number","batter","pitcher",
          "pitch_type","description","bat_speed","miss_distance"]
    d=pd.read_parquet(SOURCE,columns=cols,filters=[("game_year","==",season)]).reset_index(drop=True)
    d["local_row"]=np.arange(len(d))
    d=join.derive_columns(d,season)
    tree=ast.parse((ROOT/"external/statcast-contact-point/cp_lib/pull.py").read_text())
    descriptions=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SWING_DESCRIPTIONS" for t in n.targets))
    kept=d.loc[d.game_type.eq("R")&d.description.isin(descriptions)&join.swing_filter(d,"bat_speed_nobunt")].copy()
    return d,kept.reset_index(drop=True)


def audit_match(labels,local):
    """Same upstream keys/tolerance; refuse null joins, fan-outs and reuse of a pitch."""
    right=local.copy().reset_index(drop=True)
    right["miss_key"]=pd.to_numeric(right.miss_distance,errors="coerce").round(6)
    exact_keys=list(join.JOIN_KEYS)
    near_keys=["game_pk","batter","pitcher","pitch_type"]
    live=right.loc[np.isfinite(right.miss_key)]
    exact={k:list(v) for k,v in live.groupby(exact_keys).groups.items()}
    near={k:list(v) for k,v in live.groupby(near_keys).groups.items()}
    audits=[];candidates=[]
    for r in labels.itertuples(index=False):
        record=dict(play_id=r.play_id,season=int(r.season),match_status="unmatched",candidate_count=0,
                    author_join_method="unmatched",author_local_row=np.nan,selected_local_row=np.nan,
                    source_game_pk=np.nan,source_at_bat_number=np.nan,source_pitch_number=np.nan,
                    miss_distance_difference=np.nan,metadata_mismatch=False,matched_whiff=False,
                    date_mismatch=False,pitch_type_mismatch=False,
                    reason="no_candidate")
        ids=[getattr(r,k) for k in ("game_pk","batter","pitcher")]
        distance=number(r.miss_distance_inches)
        if any(pd.isna(v) for v in ids) or not np.isfinite(distance):
            record["reason"]="missing_join_key";audits.append(record);continue
        positions=exact.get((*ids,float(np.round(distance,6))),[])
        stage="exact"
        if not positions:
            stage="approximate"
            possible=near.get((*ids,r.pitch_type),[]) if pd.notna(r.pitch_type) else []
            positions=[i for i in possible if abs(number(right.loc[i,"miss_distance"])-distance)<=1e-3]
        record["candidate_count"]=len(positions)
        if positions:
            chosen=min(positions,key=lambda i:abs(number(right.loc[i,"miss_distance"])-distance)) if stage=="approximate" else positions[0]
            record["author_join_method"]="nearest" if stage=="approximate" else "exact"
            record["author_local_row"]=int(right.loc[chosen,"local_row"])
            for i in positions:
                s=right.loc[i]
                candidates.append(dict(play_id=r.play_id,season=int(r.season),stage=stage,local_row=int(s.local_row),
                    game_pk=int(s.game_pk),at_bat_number=int(s.at_bat_number),pitch_number=int(s.pitch_number),
                    description=s.description,pitch_type=s.pitch_type,game_date=str(s.game_date)[:10],miss_distance=float(s.miss_distance),
                    difference=abs(number(s.miss_distance)-distance),author_choice=i==chosen))
            if len(positions)>1:
                record.update(match_status="ambiguous",reason=f"multiple_{stage}_candidates")
            else:
                s=right.loc[chosen]
                same_date=str(s.game_date)[:10]==r.game_date
                same_type=pd.isna(r.pitch_type) or r.pitch_type==s.pitch_type
                record.update(match_status=stage,reason="unique_candidate",selected_local_row=int(s.local_row),
                    source_game_pk=int(s.game_pk),source_at_bat_number=int(s.at_bat_number),source_pitch_number=int(s.pitch_number),
                    miss_distance_difference=abs(number(s.miss_distance)-distance),
                    metadata_mismatch=not(same_date and same_type),date_mismatch=not same_date,
                    pitch_type_mismatch=not same_type,matched_whiff=s.description in join.WHIFF)
        audits.append(record)
    a=pd.DataFrame(audits)
    if a.empty:return a,pd.DataFrame(candidates)
    assigned=a.selected_local_row.notna()
    reused=a.loc[assigned,"selected_local_row"].duplicated(keep=False)
    indices=reused.index[reused]
    a.loc[indices,"match_status"]="ambiguous"
    a.loc[indices,"reason"]="multiple_play_ids_for_source_pitch"
    # Keep author/proposed IDs for investigation, but never export ambiguous features as valid.
    a=a.merge(labels[["play_id","critical_conflict","requested_season_mismatch","selector_mismatch","finite_xyz"]],on="play_id",validate="one_to_one")
    a["valid_match"]=a.match_status.isin(["exact","approximate"])&~a.date_mismatch&~a.critical_conflict&~a.requested_season_mismatch&~a.selector_mismatch&a.matched_whiff
    a["valid_xyz_match"]=a.valid_match&a.finite_xyz
    return a,pd.DataFrame(candidates)


def upstream_check(labels,local,audited):
    """Run the original function on non-null join keys as a convention/parity audit."""
    required=["game_pk","batter","pitcher","miss_distance_inches"]
    left=labels.dropna(subset=required)
    minimal=left[["play_id","game_pk","batter","pitcher","pitch_type","game_date","miss_distance_inches"]].copy()
    minimal["miss_distance_inches"]=minimal.miss_distance_inches.astype(float)
    got=join.join_tails(local,minimal)
    proposed=got[["play_id","join_method","local_row"]].merge(
        audited[["play_id","author_join_method","author_local_row","candidate_count"]],on="play_id",how="left")
    proposed["method_agrees"]=proposed.join_method.eq(proposed.author_join_method)
    proposed["chosen_row_agrees"]=proposed.local_row.eq(proposed.author_local_row)|(proposed.local_row.isna()&proposed.author_local_row.isna())
    # First-row choices in an ambiguous set are not reliable identities either way.
    unique=proposed.candidate_count.le(1)
    if not proposed.loc[unique,"method_agrees"].all() or not proposed.loc[unique,"chosen_row_agrees"].all():
        raise AssertionError("Unique-candidate matching differs from upstream")
    return got.attrs["join_stats"],proposed


def export_matched(labels,audits,path):
    """Stream source batches and retain only accepted pitch rows, preserving all source columns."""
    a=audits.loc[audits.valid_match].copy()
    keep=a[["play_id","source_game_pk","source_at_bat_number","source_pitch_number"]]
    parts=[]
    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=50000):
        d=batch.to_pandas().reset_index(drop=True)
        keys=d[["game_pk","at_bat_number","pitch_number"]].copy()
        keys["batch_position"]=np.arange(len(d))
        matched=keep.merge(keys,left_on=["source_game_pk","source_at_bat_number","source_pitch_number"],right_on=["game_pk","at_bat_number","pitch_number"],how="inner",validate="one_to_one")
        if matched.empty:continue
        subset=d.iloc[matched.batch_position.to_numpy()].reset_index(drop=True).add_prefix("statcast_")
        subset.insert(0,"play_id",matched.play_id.to_numpy())
        parts.append(subset)
    local=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["play_id"])
    if len(local)!=len(a) or local.play_id.duplicated().any():raise AssertionError("Source export lost or duplicated accepted matches")
    out=labels.merge(a.drop(columns=[c for c in a if c in labels and c!="play_id"]),on="play_id",validate="one_to_one").merge(local,on="play_id",validate="one_to_one")
    out.to_parquet(path,index=False)
    return len(out)


def markdown(frame,columns):
    return "| "+" | ".join(columns)+" |\n| "+" | ".join(["---"]*len(columns))+" |\n"+"\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in frame[columns].itertuples(index=False,name=None))


def build(seasons=(2024,2025,2026)):
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    occurrences,responses=load_occurrences(seasons)
    labels,conflicts,orphans=deduplicate(occurrences)
    if labels.empty:raise ValueError("No valid play IDs in cache")
    pd.DataFrame(occurrences).to_parquet(DATA/"tail_selections.parquet",index=False)
    labels.to_parquet(DATA/"whiff_tail_labels.parquet",index=False)
    conflicts.to_csv(REPORT/"field_conflicts.csv",index=False)
    if len(orphans):orphans.to_parquet(DATA/"orphan_selections.parquet",index=False)
    responses.to_csv(REPORT/"response_audit.csv",index=False)
    audits=[];candidates=[];summary=[];upstream={};bias=[];selection_numeric=[];unmatched_rows=[]
    for season in sorted(labels.season.dropna().unique()):
        t=labels.loc[labels.season.eq(season)].copy()
        all_local,local=load_local(int(season))
        a,c=audit_match(t,local)
        stats,parity=upstream_check(t,local,a)
        upstream[str(season)]=stats
        parity.to_csv(REPORT/f"upstream_matching_parity_{season}.csv",index=False)
        # Diagnose unmatched records against the broader file without treating them as accepted.
        unmatched=a.loc[a.match_status.eq("unmatched"),"play_id"]
        broad=all_local.groupby(["game_pk","batter","pitcher"],dropna=True).size()
        for r in t.loc[t.play_id.isin(unmatched)].itertuples():
            key=(r.game_pk,r.batter,r.pitcher)
            reason="missing_join_key" if not np.isfinite(number(r.miss_distance_inches)) else "matchup_present_but_distance_or_filter_failed" if key in broad.index else "matchup_not_in_local_file"
            a.loc[a.play_id.eq(r.play_id),"unmatched_diagnostic"]=reason
            nearby=all_local.loc[all_local.game_pk.eq(r.game_pk)&all_local.batter.eq(r.batter)&all_local.pitcher.eq(r.pitcher)&all_local.description.isin(join.WHIFF)].copy()
            a.loc[a.play_id.eq(r.play_id),"matchup_whiffs"]=len(nearby)
            a.loc[a.play_id.eq(r.play_id),"matchup_whiffs_missing_distance"]=int(nearby.miss_distance.isna().sum())
            a.loc[a.play_id.eq(r.play_id),"matchup_whiffs_missing_bat_speed"]=int(nearby.bat_speed.isna().sum())
            if len(nearby):
                nearby.insert(0,"unmatched_play_id",r.play_id)
                nearby["savant_pitch_type"]=r.pitch_type
                nearby["savant_miss_distance_inches"]=r.miss_distance_inches
                nearby["absolute_distance_difference"]=(nearby.miss_distance-r.miss_distance_inches).abs()
                nearby["eligible_for_author_join"]=nearby.local_row.isin(local.local_row)
                unmatched_rows.append(nearby)
        audits.append(a);candidates.append(c)
        request=responses.loc[responses.season.eq(season)]
        summary.append(dict(season=int(season),expected_requests=len(request),successful_requests=int(request.status.eq("ok").sum()),
            unique_labeled_pitches=len(t),selections=int(t.n_selections.sum()),exact=int(a.match_status.eq("exact").sum()),
            approximate=int(a.match_status.eq("approximate").sum()),ambiguous=int(a.match_status.eq("ambiguous").sum()),
            unmatched=int(a.match_status.eq("unmatched").sum()),valid_matches=int(a.valid_match.sum()),valid_xyz_matches=int(a.valid_xyz_match.sum()),
            cap_flagged=int(t.candidate_cap_flag.sum()),critical_conflicts=int(t.critical_conflict.sum()),
            selector_mismatches=int(t.selector_mismatch.sum()),metadata_mismatches=int(a.metadata_mismatch.sum()),
            tracked_whiffs=int(local.description.isin(join.WHIFF).sum()),
            local_total_pitches=len(all_local),local_regular_tracked_swings=len(local),
            first_label_date=t.game_date.min(),last_label_date=t.game_date.max(),
            local_first_game_date=str(all_local.game_date.min())[:10],local_last_game_date=str(all_local.game_date.max())[:10]))
        local_whiffs=local.loc[local.description.isin(join.WHIFF)]
        chosen=local.loc[local.local_row.isin(a.loc[a.valid_match,"selected_local_row"])]
        for population,g in (("all_tracked_whiffs",local_whiffs),("accepted_tail_matches",chosen)):
            v=pd.to_numeric(g.miss_distance,errors="coerce").dropna()
            selection_numeric.append(dict(season=int(season),population=population,pitches=len(g),
                finite_miss_distance=len(v),median_miss_inches=float(v.median()),p90_miss_inches=float(v.quantile(.9)),
                p99_miss_inches=float(v.quantile(.99))))
        for field in ("pitch_type","description"):
            total=local_whiffs[field].value_counts(dropna=False).rename("local_whiffs")
            selected=chosen[field].value_counts(dropna=False).rename("selected_whiffs")
            tab=pd.concat([total,selected],axis=1).fillna(0)
            tab["selected_fraction"]=tab.selected_whiffs/tab.local_whiffs.replace(0,np.nan)
            bias.append(tab.reset_index().rename(columns={field:"value"}).assign(season=season,field=field))
        print(summary[-1],flush=True)
    audit=pd.concat(audits,ignore_index=True);cand=pd.concat(candidates,ignore_index=True)
    audit.to_parquet(DATA/"match_audit.parquet",index=False)
    cand.to_parquet(DATA/"match_candidates.parquet",index=False)
    audit.groupby(["season","match_status","reason"]).size().rename("pitches").to_csv(REPORT/"match_status_details.csv")
    exceptions=audit.loc[~audit.valid_match].merge(labels[["play_id","game_pk","game_date","batter","pitcher","pitch_type","miss_distance_inches"]],on="play_id",validate="one_to_one")
    exceptions.to_csv(REPORT/"matching_exceptions.csv",index=False)
    if unmatched_rows:pd.concat(unmatched_rows,ignore_index=True).to_csv(REPORT/"unmatched_matchup_whiffs.csv",index=False)
    mismatches=audit.loc[audit.pitch_type_mismatch,["play_id","season"]].merge(labels[["play_id","pitch_type"]],on="play_id").merge(cand.loc[cand.author_choice,["play_id","pitch_type"]],on="play_id",suffixes=("_savant","_statcast"))
    mismatches.groupby(["season","pitch_type_savant","pitch_type_statcast"]).size().rename("pitches").to_csv(REPORT/"pitch_type_disagreements.csv")
    export_matched(labels,audit,DATA/"whiff_tail_matched.parquet")
    summary=pd.DataFrame(summary);summary.to_csv(REPORT/"summary.csv",index=False)
    pd.concat(bias,ignore_index=True).to_csv(REPORT/"selection_by_pitch_type_and_description.csv",index=False)
    pd.DataFrame(selection_numeric).to_csv(REPORT/"selection_numeric_summary.csv",index=False)
    atomic_json(REPORT/"upstream_join_stats.json",upstream)
    missing=[];caps=[]
    for season,g in labels.groupby("season"):
        for field in ("sav_x","sav_y","sav_z","miss_distance_inches","timing_x_tiedupflail","timing_y_earlylate","timing_z_overunder","game_date","pitch_type","game_pk","batter","pitcher","play_id"):
            missing.append(dict(season=int(season),field=field,missing=int(g[field].isna().sum()),total=len(g)))
        for axis,limit in (("x",36.),("z",18.)):
            v=g[f"sav_{axis}"]
            caps.append(dict(season=int(season),axis=axis,finite=int(np.isfinite(v).sum()),minimum=float(v.min()),maximum=float(v.max()),
                at_negative_cap=int(g[f"{axis}_at_negative_cap"].sum()),at_positive_cap=int(g[f"{axis}_at_positive_cap"].sum()),
                beyond_candidate_cap=int(g[f"{axis}_beyond_candidate_cap"].sum()),category_mismatches=int(g[f"{axis}_category_mismatch"].sum())))
    pd.DataFrame(missing).to_csv(REPORT/"missing_values.csv",index=False)
    quality_fields=["critical_conflict","selector_mismatch","requested_season_mismatch","cross_season_play_id",
                    "invalid_numeric_sav_x","invalid_numeric_sav_y","invalid_numeric_sav_z",
                    "invalid_numeric_miss_distance_inches","x_category_mismatch","y_category_mismatch","z_category_mismatch"]
    labels.groupby("season")[quality_fields].sum().to_csv(REPORT/"label_quality_flags.csv")
    cap=pd.DataFrame(caps);cap.to_csv(REPORT/"cap_diagnostics.csv",index=False)
    numeric=[]
    for year,g in labels.groupby("season"):
        for field in ("sav_x","sav_y","sav_z","miss_distance_inches"):
            v=g[field].dropna()
            numeric.append(dict(season=int(year),field=field,finite=int(np.isfinite(v).sum()),minimum=float(v.min()),
                maximum=float(v.max()),at_minimum=int(v.eq(v.min()).sum()),at_maximum=int(v.eq(v.max()).sum()),
                median=float(v.median()),unique_values=int(v.nunique())))
    pd.DataFrame(numeric).to_csv(REPORT/"numeric_label_summary.csv",index=False)
    extra=labels.groupby("season")[["x_at_minus24","z_at_plus12","miss_at_point1"]].sum().reset_index()
    extra.to_csv(REPORT/"additional_boundary_flags.csv",index=False)
    for axis in "xyz":labels.groupby("season")[f"sav_{axis}"].value_counts().rename("count").reset_index().to_csv(REPORT/f"label_frequencies_{axis}.csv",index=False)
    labels.groupby(["season","perspectives","n_selection_lists"]).size().rename("pitches").to_csv(REPORT/"selection_provenance_counts.csv")
    lengths=[]
    for (year,perspective),g in responses.loc[responses.status.eq("ok")].groupby(["season","perspective"]):
        for key in LISTS:
            n=g[f"n_{key}"]
            lengths.append(dict(season=int(year),perspective=perspective,list=key,players=len(g),empty=int(n.eq(0).sum()),
                short=int(n.between(1,9).sum()),ten=int(n.eq(10).sum()),over_ten=int(n.gt(10).sum()),maximum=int(n.max())))
    pd.DataFrame(lengths).to_csv(REPORT/"list_lengths.csv",index=False)
    complete=responses.status.eq("ok").all()
    text="# Savant whiff-tail collection and matching\n\n"
    text+=f"Collection {'complete' if complete else 'INCOMPLETE'} for requested seasons {list(seasons)}: {len(responses):,} player-perspective requests; {int(responses.status.eq('ok').sum()):,} valid cached responses. {len(occurrences):,} list selections yield {len(labels):,} unique play IDs. {int(audit.valid_match.sum()):,} accepted one-to-one matches are exported. No models were trained, calibrated, or evaluated.\n\n"
    text+=markdown(summary,["season","expected_requests","successful_requests","unique_labeled_pitches","exact","approximate","ambiguous","unmatched","valid_matches","valid_xyz_matches"])
    text+=f"\n\nMissing required label/category/date/type/identifier cells: {sum(r['missing'] for r in missing):,}; list occurrences without a play ID: {len(orphans):,}. Per-field counts are in `missing_values.csv`; numeric conversion and category-consistency flags are in `label_quality_flags.csv`. X, Z and miss distance are in inches; Y is timing in milliseconds."
    text+="\n\nWhiffs follow the author's definition, including foul tips. The `bat_speed_nobunt` filter excludes bunts and swings without bat-speed tracking; this is also the filter used by the author's `traits.py` tail-matching path."
    text+="\n\n## Retrieval and provenance\n\nIDs come from public unsplit season leaderboards for both batters and pitchers, min=1. URLs and User-Agent reuse `cp_lib.savant_truth`; seasons use `season[]=YYYY`. Four worker threads share a limiter of at most two request starts per second, with three attempts, exponential backoff and numeric Retry-After support. Atomic response and metadata writes, SHA-256 checks, and per-attempt request logs support resumption. Empty lists are valid and audited; missing responses never count as successful. Original JSON bodies and all original row fields remain in the cache and selection Parquet.\n\nEach canonical row is deduplicated by play_id across lists, requested players, perspectives, and seasons. `selected_by_json` preserves list, rank, requested season, selecting player, perspective and response checksum; `lists` also preserves the author's perspective:list notation. Conflicting values are retained in the occurrence/raw JSON data and `field_conflicts.csv`. Canonical fields use the first nonmissing value in stable order; critical conflicts are excluded from accepted matches. Numeric X/Y/Z aliases are `sav_x`, `sav_y`, `sav_z`; original field names, category names, miss distance and metadata are retained.\n\n## Matching rules and safeguards\n\nRegular-season local rows use the original pull swing descriptions and `bat_speed_nobunt` filter. Exact matching uses game_pk, batter, pitcher and miss distance rounded to six decimals. Unmatched exact keys fall back to game_pk, batter, pitcher and pitch_type within 0.001 inch of miss distance. No time-window or player-name guess is substituted. The original `join.join_tails` is also executed on nonmissing keys, and unique-candidate assignments are asserted equal.\n\nThe author chooses the first exact candidate or closest approximate candidate. This audit conservatively marks **any multiple-candidate set** ambiguous, even if one candidate is numerically closest; it also quarantines multiple play IDs assigned to one source pitch. Candidates and the author's proposed choice remain available in `match_candidates.parquet`. Missing join keys never match nulls. Date/pitch-type disagreements, selector/season inconsistencies, critical label/identifier conflicts, and non-whiff matches fail the valid-match gate. Missing XYZ labels do not erase pitch identity: `valid_xyz_match` separately identifies accepted matches with all three finite labels.\n\n## Candidate reporting caps\n\n"
    text+=markdown(cap,["season","axis","finite","minimum","maximum","at_negative_cap","at_positive_cap","beyond_candidate_cap","category_mismatches"])
    text+="\n\nFlags use an absolute tolerance of 1e-8 at X +/-36 and Z +/-18; beyond-bound observations are counted separately. Point masses at these values are evidence compatible with clipping, not proof of the undocumented endpoint's exact censoring mechanism. Preserve these values and flags; do not treat a potentially censored endpoint value as an exact physical measurement or silently discard it. Full frequency tables are provided.\n\n## Selection and missingness\n\nThese are extreme-tail selections, not random whiffs or representative swings. The same pitch may be selected by several lists and by both perspectives; deduplication removes repeated rows but does not remove selection bias. List rank is the returned ordering, not an inferred sampling probability. Coverage differs with player swing volume; short lists and empty responses are retained. Pitch-type/outcome selection rates, list/perspective overlap, missing label counts, numeric conversion flags and category-consistency checks are supplied. No inverse-selection weights are invented, and no model accuracy is inferred.\n\n## Files and rerunning\n\n- `data/savant_whiff_tails/whiff_tail_labels.parquet`: every unique labeled pitch, including unmatched/conflicted records and provenance.\n- `data/savant_whiff_tails/whiff_tail_matched.parquet`: accepted matches only; source Statcast columns have a `statcast_` prefix to avoid overwriting Savant fields.\n- `tail_selections.parquet`: every list occurrence and its complete raw row JSON.\n- `match_audit.parquet`: one status/provenance row per unique label.\n- `match_candidates.parquet`: candidate source identifiers and proposed author choices, including ambiguities.\n- `cache/`: raw responses, per-response checksum metadata and append-only request attempts.\n\n```text\npython main.py collect\npython main.py match\n```\n\nSuccessful cached responses are reused. Rerun collection to retry failures, then rerun auditing. All paths resolve from the project root. The current experiment branch, original Statcast Parquet, scratch notebook and submodule are unchanged; no commit or merge was performed.\n"
    text=text.replace("Four worker threads","Twelve worker threads (after an initial higher-concurrency trial)")
    text=text.replace("Date/pitch-type disagreements, selector/season inconsistencies", "Date disagreements, selector/season inconsistencies")
    text+="\nAdditional observed boundary flags (included in `candidate_cap_flag`) are retained separately from the requested symmetric checks:\n\n"+markdown(extra,["season","x_at_minus24","z_at_plus12"])+"\n\nThese X=-24 and Z=+12 boundaries emerged in the partial response audit. Numeric min/max frequencies across all four labels are recorded in `numeric_label_summary.csv`. Neither a plotted/reporting boundary nor an extreme-tail selection establishes an exact physical limit.\n"
    text+="\nAccepted labeled coverage of the local tracked-whiff population:\n\n"
    for r in summary.itertuples():
        text+=f"- {r.season}: {r.valid_matches:,} / {r.tracked_whiffs:,} ({100*r.valid_matches/r.tracked_whiffs:.2f}%).\n"
    text+="\n`selection_numeric_summary.csv` contrasts miss-distance medians and upper quantiles for accepted tail matches versus all tracked whiffs; this is a descriptive selection audit, not model evaluation.\n"
    text+="\nThe repeated miss-distance value 0.1 inch is separately flagged as `miss_at_point1`; `numeric_label_summary.csv` records its minimum-value frequency. A repeated rounded distance can create ambiguous join keys, so the audit never treats it as a unique pitch identifier. `matching_exceptions.csv` lists every unaccepted label with its reason and pitch metadata.\n"
    text+="\nFor unmatched labels, `unmatched_matchup_whiffs.csv` lists local whiffs from the same game/batter/pitcher matchup, including missing bat speed or miss distance. These rows are diagnostic possibilities, not accepted identities; a missing join key is never filled by guessing from the remaining metadata.\n"
    text+="\nPitch-type disagreements are preserved in `pitch_type_disagreements.csv` and flagged on matches. They do not independently invalidate a unique exact match because pitch type is not in the author's exact key. The partial audit identified repeated Savant CU versus local KC/CS values; neither field is overwritten. Approximate matching still requires identical pitch type, as in the author code. List-length diagnostics report empty, short, ten-entry and longer lists.\n"
    (REPORT/"report.md").write_text(text,encoding="utf-8")
    atomic_json(REPORT/"audit_manifest.json",dict(seasons=list(seasons),collection_complete=bool(complete),audited_utc=utcnow(),
        unique_labels=len(labels),valid_matches=int(audit.valid_match.sum()),orphan_selections=len(orphans),
        source_bytes=SOURCE.stat().st_size,source_mtime_ns=SOURCE.stat().st_mtime_ns,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"capstone/collection.py"]},
        upstream_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"external/statcast-contact-point/cp_lib"/name for name in ("savant_truth.py","join.py","pull.py")]}))
    return summary


