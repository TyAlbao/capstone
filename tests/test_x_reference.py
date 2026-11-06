"""Behavioral tests; run with uv run python -m unittest discover -s tests -v."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capstone.references import (
    X, Y, Config, METHODS, bootstrap_difference, calculate_references,
    category_rates, denominator_audit, normalize_official, predict_contact,
    prepare_season, reconstruct, validation_metrics, verify_schedule,
)


def fixture():
    n = 8
    d = pd.DataFrame({"batter": [1]*4+[2]*4, "game_year": [2025]*n,
        "game_type": ["R"]*n, "description": ["hit_into_play"]*n,
        X: [20., 30., 40., 50., 40., 45., 50., 65.],
        Y: [24.]*n, "launch_speed": [80.,90.,90.,100.]*2,
        "bat_speed": [75.]*n, "stand": ["R"]*n,
        "relative_intercept_x": [-10.,0.,10.,20.,-10.,-5.,0.,15.]})
    constants = dict(vx0=3., vy0=-130., vz0=-4., ax=-5., ay=25., az=-20.,
        plate_x=.2, plate_z=2.5, attack_angle=10., attack_direction=5.,
        swing_path_tilt=25., launch_angle=15., miss_distance=np.nan, sz_top=3.5, sz_bot=1.5)
    for col, val in constants.items(): d[col] = val
    return d


class References(unittest.TestCase):
    def setUp(self):
        self.config = Config(min_batter_swings=4, min_ev50_bip=4, bootstrap_reps=200)
        self.swings = prepare_season(fixture(), 2025)

    def test_calculation_and_ties(self):
        r = calculate_references(self.swings, self.config)
        self.assertEqual(r.loc[1, "ref_batter"], 35.)
        self.assertEqual(r.loc[1, "ref_ev50"], 40.)
        self.assertEqual(r.loc[1, "n_ev50_selected"], 3)
        self.assertEqual(r.loc[1, "ref_global"], 42.5)
        self.assertEqual(r.loc[1, "ref_existing"], 30.)
        self.assertFalse(r.fallback_ev50.any())

    def test_threshold_before_upper_half_and_fallback(self):
        r = calculate_references(self.swings, Config(5, 5))
        self.assertTrue(r.fallback_batter.all() and r.fallback_ev50.all())
        np.testing.assert_array_equal(r.ref_global, r.ref_ev50)
        pred = predict_contact(self.swings, r)
        self.assertFalse(pred.primary_eligible.any())
        self.assertTrue(pred.x_ev50.notna().all())

    def test_full_reference_population_includes_fouls(self):
        extra = self.swings.iloc[[0]].copy()
        extra[X] = 1000.
        extra["description"] = "foul"
        s = pd.concat([self.swings, extra], ignore_index=True)
        s = s.drop(columns="relative_intercept_x")
        r = calculate_references(s, self.config)
        self.assertEqual(r.loc[1, "ref_batter"], 40.)
        self.assertEqual(r.loc[1, "n_ev50_bip"], 4)
        self.assertEqual(r.loc[1, "ref_global"], 45.)

    def test_exact_swing_filter_and_season_separation(self):
        d = fixture()
        d.loc[0, "description"] = "missed_bunt"
        d.loc[1, "bat_speed"] = np.nan
        d.loc[2, "game_type"] = "S"
        d.loc[3, "bat_speed"] = 20.  # no unintended typical-speed restriction
        other = fixture().assign(game_year=2026)
        s = prepare_season(pd.concat([d, other]), 2025)
        self.assertEqual(len(s), 5)
        self.assertIn(20., s.bat_speed.to_list())
        with self.assertRaises(ValueError):
            calculate_references(pd.concat([s, prepare_season(other, 2026)]))

    def test_missing_ev_x_and_infinite_ev_not_in_ev50(self):
        s = self.swings.copy()
        s.loc[0, X] = np.nan
        s.loc[1, "launch_speed"] = np.nan
        s.loc[2, "launch_speed"] = np.inf
        r = calculate_references(s, Config(1, 1))
        self.assertEqual(r.loc[1, "n_batter"], 3)
        self.assertEqual(r.loc[1, "n_ev50_bip"], 1)
        self.assertEqual(r.loc[1, "ref_ev50"], 50.)


class Predictions(unittest.TestCase):
    def setUp(self):
        self.s = prepare_season(fixture(), 2025)
        self.r = calculate_references(self.s, Config(1, 1))

    def test_author_inversion_and_magnitude(self):
        p = predict_contact(self.s, self.r)
        for method in METHODS:
            expected = reconstruct.x_from_ev_deficit(p.ev_deficit, p[X].lt(p[f"ref_{method}"]))
            np.testing.assert_allclose(p[f"x_{method}"], expected)
            np.testing.assert_array_equal(p[f"x_{method}_sign_only"].abs(), p.x_global.abs())
        change = p.toward_handle_ev50.ne(p.toward_handle_global)
        self.assertTrue(change.any())
        self.assertTrue((p.loc[change, "x_ev50"].abs() != p.loc[change, "x_global"].abs()).any())

    def test_missing_prediction_never_zero_and_shared_rows(self):
        self.s.loc[0, "launch_speed"] = np.nan
        self.s.loc[1, "vy0"] = np.nan
        p = predict_contact(self.s, self.r)
        for method in METHODS:
            self.assertTrue(p.loc[[0,1], f"x_{method}"].isna().all())
        rates = category_rates(p.loc[p.primary_eligible])
        self.assertTrue(rates.groupby("batter").n.nunique().eq(1).all())
        self.assertEqual(int(rates.groupby(["experiment","method"]).n.sum().iloc[0]), 6)
        with self.assertRaises(ValueError): category_rates(p)

    def test_threshold_endpoints(self):
        p = predict_contact(self.s, self.r).iloc[:4].copy()
        for method in METHODS:
            p[f"x_{method}"] = [-4.,4.,-4.1,4.1]
            p[f"x_{method}_sign_only"] = p[f"x_{method}"]
        r = category_rates(p)
        self.assertTrue(r.centered.eq(.5).all())
        self.assertTrue(r.tied_up.eq(.25).all())
        self.assertTrue(r.flail.eq(.25).all())

    def test_season_passed_to_geometry(self):
        from capstone.references import geometry
        self.s["season"] = 2026
        with patch.object(geometry, "candidate_features", wraps=geometry.candidate_features) as call:
            predict_contact(self.s, self.r)
            self.assertEqual(call.call_args.args[1], 2026)

    def test_input_columns_not_mutated(self):
        original = self.s.copy(deep=True)
        predict_contact(self.s, self.r)
        pd.testing.assert_frame_equal(self.s, original)

    def test_nullable_parquet_dtypes(self):
        s = self.s.convert_dtypes()
        s.loc[0, "relative_intercept_x"] = pd.NA
        s.loc[s.batter.eq(2), "relative_intercept_x"] = pd.NA
        r = calculate_references(s, Config(1,1))
        p = predict_contact(s, r)
        self.assertEqual(len(p), 8)
        self.assertTrue(p.x_ev50.notna().all())
        self.assertTrue(p.loc[p.batter.eq(2), "x_existing"].isna().all())
        self.assertTrue(p.loc[p.batter.eq(2), "x_existing_sign_only"].isna().all())


class Validation(unittest.TestCase):
    def official(self, scale=1):
        return pd.DataFrame({"id":[1,2], "year":[2025]*2, "bat_contact_code":[2]*2,
            "n_swings":[4,4], "tied_up_percent":[.2*scale]*2,
            "centered_percent":[.5*scale]*2, "flailed_percent":[.3*scale]*2})

    def test_fraction_percent_and_unexpected_responses(self):
        a = normalize_official(self.official(), 2025)
        b = normalize_official(self.official(100), 2025)
        pd.testing.assert_frame_equal(a, b)
        with self.assertRaises(ValueError): normalize_official(self.official(), 2026)
        with self.assertRaises(ValueError): normalize_official(self.official(10), 2025)
        with self.assertRaises(ValueError): normalize_official(pd.concat([self.official()]*2), 2025)
        bad = self.official(); bad.loc[0, "centered_percent"] = np.nan
        with self.assertRaises(ValueError): normalize_official(bad, 2025)

    def test_mismatch_partial_season_and_paired_populations(self):
        config = Config(1,1, bootstrap_reps=100)
        s = prepare_season(fixture(), 2025)
        p = predict_contact(s, calculate_references(s, config))
        off = normalize_official(self.official(), 2025)
        off.loc[off.batter.eq(2), "official_n"] = 10
        den = denominator_audit(s,p,off,config,True)
        self.assertEqual(den.loc[den.eligible,"batter"].to_list(), [1])
        self.assertFalse(denominator_audit(s,p,off,config,False).eligible.any())
        metrics, paired = validation_metrics(category_rates(p), den, config)
        self.assertTrue(metrics.n_swings.eq(4).all())
        self.assertTrue(metrics.n_batters.eq(1).all())
        self.assertEqual(paired.batter.unique().tolist(), [1])

    def test_bootstrap_reproducible_and_direction(self):
        c = Config(bootstrap_reps=500)
        self.assertEqual(bootstrap_difference([-1.,-1.,-1.], c), (-1.,-1.))
        self.assertEqual(bootstrap_difference([-1.,2.,3.],c), bootstrap_difference([-1.,2.,3.],c))

    def test_schedule_refuses_partial_game_coverage(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); (out/"official").mkdir()
            games = [{"game_pk":i,"game_type":"R","status":"Final","status_code":"F","date":"2025-09-28"} for i in [1,2]]
            games.append({"game_pk":3,"game_type":"R","status":"Final","status_code":"C","date":"2025-09-28"})
            (out/"official/schedule_2025.json").write_text(json.dumps({"schema_version":2,"games":games,"fetched_utc":"test"}))
            self.assertFalse(verify_schedule(2025,[1],out)["full_season_verified"])
            self.assertTrue(verify_schedule(2025,[1,2],out)["full_season_verified"])


if __name__ == "__main__":
    unittest.main()
