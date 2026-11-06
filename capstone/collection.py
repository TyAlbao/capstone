"""Resumable Savant tail acquisition; no model fitting and no upstream writes.

python main.py collect
The shared limiter allows at most 2 request starts/second across all workers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/savant_whiff_tails/audit"
CACHE = ROOT / "data/savant_whiff_tails/cache"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "external/statcast-contact-point"))
from cp_lib import savant_truth as st

LISTS = ("miss_distance", "x", "y", "z")


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def validate_tail(payload):
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("Unexpected tail response or error object")
    if payload == {}:
        return "empty_object"
    if not set(LISTS).issubset(payload):
        raise ValueError(f"Missing expected tail lists: {set(LISTS)-set(payload)}")
    if any(not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows) for rows in payload.values()):
        raise ValueError("Tail entries must be lists of objects")
    return "four_lists"


class Collector:
    def __init__(self, cache=CACHE, interval=.5, attempts=3):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.interval=max(float(interval),float(st.THROTTLE_S))
        self.attempts=attempts
        self.rate_lock=threading.Lock();self.log_lock=threading.Lock()
        self.last_start=0.;self.local=threading.local()

    def session(self):
        if not hasattr(self.local,"session"):
            self.local.session=requests.Session()
            self.local.session.headers["User-Agent"]=st.UA
        return self.local.session

    def limit(self):
        with self.rate_lock:
            time.sleep(max(0,self.interval-(time.monotonic()-self.last_start)))
            self.last_start=time.monotonic()

    def log(self,record):
        with self.log_lock:
            with (self.cache/"requests.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,default=str)+"\n")

    def fetch(self,url,path,kind,context):
        meta_path=path.with_suffix(".meta.json")
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            body=path.read_bytes()
            if meta.get("url")==url and hashlib.sha256(body).hexdigest()==meta.get("sha256"):
                if kind=="json":validate_tail(json.loads(body))
                return dict(**context,status="cached",file=str(path.relative_to(self.cache)),sha256=meta["sha256"])
            raise ValueError(f"Cache integrity failure: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        last=None
        for attempt in range(1,self.attempts+1):
            self.limit();start=time.monotonic()
            record=dict(**context,url=url,attempt=attempt,started_utc=utcnow())
            retry_after=None
            try:
                response=self.session().get(url,timeout=(10,60))
                record.update(http_status=response.status_code,content_type=response.headers.get("Content-Type"))
                retry_after=response.headers.get("Retry-After")
                response.raise_for_status()
                body=response.content
                invalid=st._valid(body,kind)
                if invalid:raise ValueError(invalid)
                if kind=="json":
                    payload=json.loads(body);record["schema"]=validate_tail(payload)
                else:
                    frame=pd.read_csv(io.BytesIO(body),encoding="utf-8-sig")
                    if not {"id","year","n_swings"}.issubset(frame) or not frame.year.eq(context["season"]).all():
                        raise ValueError("Wrong leaderboard schema or season")
                digest=hashlib.sha256(body).hexdigest()
                temp=path.with_suffix(path.suffix+".tmp");temp.write_bytes(body);temp.replace(path)
                record.update(status="ok",sha256=digest,bytes=len(body),elapsed_s=round(time.monotonic()-start,3))
                atomic_json(meta_path,record);self.log(record)
                return dict(**context,status="downloaded",file=str(path.relative_to(self.cache)),sha256=digest)
            except (requests.RequestException,ValueError) as exc:
                last=str(exc)[:1000]
                record.update(status="failed_attempt",error=last,elapsed_s=round(time.monotonic()-start,3))
                self.log(record)
                if attempt<self.attempts:
                    delay=min(30,2**attempt)
                    if retry_after:
                        try:delay=max(delay,min(60,float(retry_after)))
                        except ValueError:pass
                    time.sleep(delay)
        return dict(**context,status="failed",error=last,url=url)

    def leaderboard(self,season,perspective):
        path=self.cache/f"{season}/leaderboard_{perspective}.csv"
        r=self.fetch(st.leaderboard_url(season,perspective),path,"csv",dict(season=season,perspective=perspective,kind="leaderboard"))
        if r["status"]=="failed":raise RuntimeError(r)
        d=pd.read_csv(path,encoding="utf-8-sig").dropna(subset=["id"])
        return d

    def player(self,season,perspective,player_id):
        path=self.cache/f"{season}/{perspective}/{int(player_id)}.json"
        context=dict(season=season,perspective=perspective,player_id=int(player_id),kind="tails")
        return self.fetch(st.tails_url(int(player_id),season,perspective),path,"json",context)


def collect(seasons=(2025,2024,2026),workers=12):
    REPORT.mkdir(parents=True,exist_ok=True)
    collector=Collector()
    status_path=REPORT/"collection_status.json"
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    for season in seasons:
        registry=[]
        for perspective in ("batter","pitcher"):
            board=collector.leaderboard(season,perspective)
            for _,r in board.sort_values("id").drop_duplicates("id").iterrows():
                registry.append(dict(season=season,perspective=perspective,player_id=int(r.id),
                    name=r.get("name"),n_swings=int(r.n_swings)))
        reg=pd.DataFrame(registry)
        reg.to_csv(REPORT/f"player_registry_{season}.csv",index=False)
        results=[];started=utcnow()
        print(f"Season {season}: {len(reg)} player-perspective requests",flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(collector.player,season,r["perspective"],r["player_id"]):r for r in registry}
            for future in as_completed(futures):
                results.append(future.result())
                n=len(results)
                if n%50==0 or n==len(reg):
                    failures=sum(r["status"]=="failed" for r in results)
                    status[str(season)]=dict(expected=len(reg),processed=n,succeeded=n-failures,failed=failures,
                        complete=n==len(reg) and failures==0,started_utc=started,updated_utc=utcnow())
                    atomic_json(status_path,status)
                    pd.DataFrame(results).to_csv(REPORT/f"requests_{season}.csv",index=False)
                    print(f"{season}: {n}/{len(reg)} processed; {failures} failures",flush=True)
        atomic_json(REPORT/f"collection_{season}.json",status[str(season)])
    return status


