import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capstone.matching import audit_match,deduplicate,upstream_check
from capstone.collection import Collector,validate_tail


def occurrence(play="abc",perspective="batter",list_name="x",distance=5.,**overrides):
    row=dict(play_id=play,game_pk=100,game_date="2025-06-01T00:00:00.000Z",savant_batter_id=1,
        pitcher=2,pitch_type="FF",miss_distance_inches=str(distance),
        delta_batball_tiedup_neg_x2=36.,delta_batball_late_neg_y2_msec=-8.,ball_pos_above_plane=-18.,
        timing_x_tiedupflail="Flail",timing_y_earlylate="Late",timing_z_overunder="Over")
    row.update(overrides)
    return dict(requested_season=2025,perspective=perspective,player_id=1 if perspective=="batter" else 2,
        list=list_name,rank=1,response_sha256="test",raw_row_json=json.dumps(row))


def local(distances=(5.,),**changes):
    d=pd.DataFrame([dict(local_row=i,game_pk=100,batter=1,pitcher=2,pitch_type="FF",game_date="2025-06-01",
        miss_distance=distance,description="swinging_strike",at_bat_number=1,pitch_number=i+1) for i,distance in enumerate(distances)])
    for k,v in changes.items():d[k]=v
    return d


class TailAudit(unittest.TestCase):
    def test_dedup_preserves_selectors_and_caps(self):
        labels,conflicts,orphans=deduplicate([occurrence(),occurrence(perspective="pitcher",list_name="z")])
        self.assertEqual(len(labels),1)
        r=labels.iloc[0]
        self.assertEqual(r.n_selections,2)
        self.assertEqual(r.n_selecting_players,2)
        self.assertEqual(len(json.loads(r.selected_by_json)),2)
        self.assertTrue(r.x_at_positive_cap and r.z_at_negative_cap)
        self.assertFalse(r.x_beyond_candidate_cap)
        self.assertTrue(conflicts.empty and orphans.empty)

    def test_conflict_not_silently_discarded(self):
        labels,conflicts,_=deduplicate([occurrence(),occurrence(delta_batball_tiedup_neg_x2=35.)])
        self.assertTrue(labels.critical_conflict.iloc[0])
        self.assertEqual(len(conflicts),1)
        a,_=audit_match(labels,local())
        self.assertFalse(a.valid_match.iloc[0])

    def test_exact_and_approximate_match_upstream(self):
        for distance,expected in [(5.,"exact"),(5.0004,"approximate")]:
            labels,_,_=deduplicate([occurrence()])
            d=local([distance]);a,c=audit_match(labels,d)
            self.assertEqual(a.match_status.iloc[0],expected)
            self.assertTrue(a.valid_match.iloc[0])
            _,parity=upstream_check(labels,d,a)
            self.assertTrue(parity.method_agrees.all())

    def test_exact_fanout_and_near_candidates_are_ambiguous(self):
        labels,_,_=deduplicate([occurrence()])
        for d in [local([5.,5.]),local([5.0004,5.0006])]:
            a,c=audit_match(labels,d)
            self.assertEqual(a.match_status.iloc[0],"ambiguous")
            self.assertEqual(len(c),2)
            self.assertFalse(a.valid_match.iloc[0])

    def test_no_null_join_or_tolerance_expansion(self):
        labels,_,_=deduplicate([occurrence(miss_distance_inches=None)])
        a,_=audit_match(labels,local([np.nan]))
        self.assertEqual(a.reason.iloc[0],"missing_join_key")
        labels,_,_=deduplicate([occurrence()])
        a,_=audit_match(labels,local([5.0011]))
        self.assertEqual(a.match_status.iloc[0],"unmatched")

    def test_metadata_and_nonwhiff_quarantine(self):
        labels,_,_=deduplicate([occurrence()])
        for d in [local(game_date="2025-06-02"),local(description="hit_into_play")]:
            a,_=audit_match(labels,d)
            self.assertEqual(a.match_status.iloc[0],"exact")
            self.assertFalse(a.valid_match.iloc[0])
        a,_=audit_match(labels,local(pitch_type="KC"))
        self.assertTrue(a.pitch_type_mismatch.iloc[0])
        self.assertTrue(a.valid_match.iloc[0])

    def test_many_play_ids_one_pitch_quarantined(self):
        labels,_,_=deduplicate([occurrence("one"),occurrence("two")])
        a,_=audit_match(labels,local())
        self.assertTrue(a.match_status.eq("ambiguous").all())
        self.assertFalse(a.valid_match.any())

    def test_missing_labels_do_not_erase_pitch_identity(self):
        labels,_,_=deduplicate([occurrence(delta_batball_tiedup_neg_x2=None)])
        a,_=audit_match(labels,local())
        self.assertTrue(a.valid_match.iloc[0])
        self.assertFalse(a.valid_xyz_match.iloc[0])

    def test_cross_season_and_selector_flags(self):
        labels,_,_=deduplicate([occurrence(game_date="2024-06-01",savant_batter_id=7)])
        self.assertTrue(labels.requested_season_mismatch.iloc[0])
        self.assertTrue(labels.selector_mismatch.iloc[0])


class TailCollection(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(validate_tail({}),"empty_object")
        self.assertEqual(validate_tail({k:[] for k in ("miss_distance","x","y","z")}),"four_lists")
        with self.assertRaises(ValueError):validate_tail({"error":"failed"})
        with self.assertRaises(ValueError):validate_tail({"x":[]})

    def test_cache_resumption_retry_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            c=Collector(Path(folder),attempts=2)
            response=Mock();response.status_code=200;response.headers={};response.content=json.dumps({k:[] for k in ("miss_distance","x","y","z")}).encode()
            session=Mock();session.get.return_value=response
            with patch.object(c,"session",return_value=session),patch.object(c,"limit"):
                a=c.player(2025,"batter",1);b=c.player(2025,"batter",1)
                self.assertEqual(a["status"],"downloaded");self.assertEqual(b["status"],"cached")
                self.assertEqual(session.get.call_count,1)
                path=Path(folder)/"2025/batter/1.json";path.write_text('{}')
                with self.assertRaises(ValueError):c.player(2025,"batter",1)

    def test_transient_failure_retries_and_failed_request_resumes(self):
        import requests
        with tempfile.TemporaryDirectory() as folder:
            c=Collector(Path(folder),attempts=2)
            response=Mock();response.status_code=200;response.headers={};response.content=json.dumps({k:[] for k in ("miss_distance","x","y","z")}).encode()
            session=Mock();session.get.side_effect=[requests.Timeout("test"),response]
            with patch.object(c,"session",return_value=session),patch.object(c,"limit"),patch("capstone.collection.time.sleep"):
                result=c.player(2025,"pitcher",1)
                self.assertEqual(result["status"],"downloaded")
                self.assertEqual(session.get.call_count,2)
                session.get.side_effect=requests.Timeout("test")
                self.assertEqual(c.player(2025,"pitcher",2)["status"],"failed")
                session.get.side_effect=None;session.get.return_value=response
                self.assertEqual(c.player(2025,"pitcher",2)["status"],"downloaded")


if __name__=="__main__":unittest.main()
