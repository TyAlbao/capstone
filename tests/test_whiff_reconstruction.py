import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capstone.whiffs as e


def fixture():
    r=dict(game_year=2025,stand="R",vx0=5.,vy0=-130.,vz0=-5.,ax=-4.,ay=28.,az=-20.,
           plate_x=.25,plate_z=2.5,attack_angle=10.,attack_direction=-5.,swing_path_tilt=30.,
           bat_speed=70.,sz_top=3.5,sz_bot=1.5,intercept_ball_minus_batter_pos_x_inches=30.,
           intercept_ball_minus_batter_pos_y_inches=36.,miss_distance=8.)
    return pd.DataFrame([r,dict(r,stand="L")]).add_prefix("statcast_")


class FormulaTests(unittest.TestCase):
    def test_extract_only_prefixed_allowlist(self):
        d=fixture().assign(sav_y=123.,relative_intercept_y=-500.,statcast_relative_intercept_y=999.)
        raw=e.extract_statcast(d)
        self.assertEqual(list(raw),e.INPUTS)
        self.assertTrue(raw.intercept_ball_minus_batter_pos_y_inches.eq(36).all())
        with self.assertRaises(ValueError):
            e.extract_statcast(d.drop(columns="statcast_vy0"))

    def test_exact_coefficients_pinned(self):
        specs,ycom,tilt=e.formulas()
        self.assertEqual((ycom,tilt),(0.,1.))
        self.assertEqual(specs["y"].stand_models["R"]["coef"]["icpt_y_over_ballspeed"],.9235093592055167)
        self.assertEqual(specs["y"].stand_models["L"]["intercept"],-18.343601678682262)
        self.assertEqual(len(specs["x"].stand_models["R"]["coef"]),9)
        self.assertEqual(len(specs["z"].stand_models["R"]["coef"]),6)
        self.assertEqual(specs["x"].status,"fallback")

    def test_speed_units_and_y_formula(self):
        raw=e.extract_statcast(fixture())
        state=e.geometry.ball_state_at_y(raw,2025,np.array([3.,3.]))
        speed_fps=np.sqrt(state.bvx**2+state.bvy**2+state.bvz**2)
        expected_in_ms=speed_fps*12/1000
        feats=e.geometry.candidate_features(raw,2025)
        np.testing.assert_allclose(feats.ball_in_per_ms,expected_in_ms,rtol=1e-13)
        got=e.predict(fixture()).pred_y_icpt_y_over_ballspeed
        expected=np.array([.9235093592055167,.9106727948752966])*36/expected_in_ms-np.array([17.918413523829468,18.343601678682262])
        np.testing.assert_allclose(got,expected,rtol=1e-13)

    def test_2026_geometry_anchors_and_pass_through(self):
        raw=e.extract_statcast(fixture())
        self.assertEqual(e.geometry.plate_y_ft(2025),17/12)
        self.assertEqual(e.geometry.plate_y_ft(2026),17/24)
        for year in (2025,2026):
            at_plate=e.geometry.ball_state_at_y(raw,year,e.geometry.plate_y_ft(year))
            np.testing.assert_allclose(at_plate.bx,raw.plate_x,atol=1e-12)
            np.testing.assert_allclose(at_plate.bz,raw.plate_z,atol=1e-12)
        a=e.geometry.ball_state_at_y(raw,2025,0.)
        b=e.geometry.ball_state_at_y(raw,2026,0.)
        self.assertGreater(np.max(np.abs(a.bz-b.bz)),.01)
        d=fixture();d.statcast_game_year=2026
        p=e.predict(d)
        f=e.geometry.candidate_features(e.extract_statcast(d),2026)
        f["stand"]=e.extract_statcast(d).stand;f["miss_distance"]=8.
        specs,_,_=e.formulas()
        np.testing.assert_allclose(p.pred_x_x_full_geometry,e.reconstruct._apply_formula(specs["x"],f))

    def test_all_published_coefficients_used_unchanged(self):
        raw=e.extract_statcast(fixture())
        f=e.geometry.candidate_features(raw,2025);f["stand"]=raw.stand;f["miss_distance"]=raw.miss_distance
        specs,_,_=e.formulas();p=e.predict(fixture())
        for ax in e.AXES:
            for i,stand in enumerate(("R","L")):
                spec=specs[ax].stand_models[stand]
                manual=spec["intercept"]+sum(v*f.loc[i,k] for k,v in spec["coef"].items())
                self.assertAlmostEqual(p.loc[i,f"pred_{ax}_{e.PRIMARY[ax]}"],manual,places=11)

    def test_no_truth_or_selection_leakage(self):
        d=fixture();first=e.predict(d)
        for col in ["sav_x","sav_y","sav_z",*e.CAT.values(),"selected_x","selected_y","lists",
                    "miss_distance_inches","relative_intercept_y","statcast_launch_speed","statcast_relative_intercept_y"]:
            d[col]=[99999.,-99999.]
        pd.testing.assert_frame_equal(first,e.predict(d))

    def test_missing_values_remain_unknown(self):
        d=fixture();d.loc[0,"statcast_intercept_ball_minus_batter_pos_y_inches"]=np.nan
        p=e.predict(d)
        self.assertTrue(np.isnan(p.loc[0,"pred_y_icpt_y_over_ballspeed"]))
        self.assertEqual(e.discover.label_of("y",[p.loc[0,"pred_y_icpt_y_over_ballspeed"]])[0],"Unknown")
        self.assertTrue(np.isfinite(p.loc[1,"pred_y_icpt_y_over_ballspeed"]))

    def test_categories_signs_and_inclusive_boundaries(self):
        for ax,t in (("x",4.),("y",7.),("z",2.)):
            labels=e.discover.LABELS[ax]
            got=e.discover.label_of(ax,[-t-.001,-t,0,t,t+.001,np.nan,np.inf])
            self.assertEqual(list(got),[labels[0],labels[1],labels[1],labels[1],labels[2],"Unknown","Unknown"])
        self.assertEqual(e.discover.label_of("z",[-3,3]).tolist(),["Over","Under"])
        self.assertEqual(e.discover.label_of("x",[-5,5]).tolist(),["Tiedup","Flail"])

    def test_missing_handedness_is_not_treated_as_right_handed(self):
        d=fixture();d.loc[0,"statcast_stand"]=None
        p=e.predict(d)
        self.assertTrue(p.loc[0].isna().all())
        self.assertTrue(np.isfinite(p.loc[1].to_numpy(float)).all())

    def test_axis_specific_boundary_flags_do_not_mutate(self):
        v=np.array([-24.,36.,-18.,12.,0.,np.nan]);before=v.copy()
        self.assertEqual(e.boundary("x",v).tolist(),[True,True,False,False,False,False])
        self.assertEqual(e.boundary("z",v).tolist(),[False,False,True,True,False,False])
        np.testing.assert_array_equal(v,before)
        self.assertTrue(e.boundary("x",[36-1e-9])[0])
        self.assertFalse(e.boundary("x",[36-1e-5])[0])

    def test_reproducible_metrics_and_regression_direction(self):
        r=e.numerical_metrics([-10,0,10],[-8,0,8],["Late","OnTime","Early"],"y")
        self.assertAlmostEqual(r["mae"],4/3)
        self.assertAlmostEqual(r["rmse"],np.sqrt(8/3))
        self.assertAlmostEqual(r["r2"],.96)
        self.assertAlmostEqual(r["actual_on_pred_slope"],1.25)
        self.assertAlmostEqual(r["actual_on_pred_intercept"],0)
        self.assertEqual(r["category_accuracy"],1)
        self.assertEqual(r["sign_accuracy"],1)
        self.assertEqual(r,e.numerical_metrics([-10,0,10],[-8,0,8],["Late","OnTime","Early"],"y"))

    def test_constant_and_missing_predictions_metrics(self):
        r=e.numerical_metrics([-10,0,10],[0,0,0],["Late","OnTime","Early"],"y")
        self.assertTrue(np.isnan(r["pearson_r"]))
        self.assertTrue(np.isnan(r["actual_on_pred_slope"]))
        self.assertEqual(r["r2"],0)
        self.assertAlmostEqual(r["category_accuracy"],1/3)
        r=e.numerical_metrics([-10,0,10],[-10,np.nan,10],["Late","OnTime","Early"],"y")
        self.assertEqual(r["n"],2);self.assertEqual(r["mae"],0)

    def test_baseline_separation(self):
        d=pd.DataFrame({"season":[2025]*3})
        for ax in e.AXES:
            d[f"sav_{ax}"]=[1.,2.,12.]
            d[f"common_{ax}"]=True
            d[e.CAT[ax]]=e.discover.label_of(ax,d[f"sav_{ax}"])
        params=e.fit_baselines(d)
        self.assertEqual(params["axes"]["y"]["mean"],5.)
        self.assertEqual(params["axes"]["y"]["median"],2.)
        frozen=copy.deepcopy(params)
        holdout=d.copy();holdout.season=2026
        with self.assertRaises(ValueError):e.fit_baselines(holdout)
        with self.assertRaises(ValueError):e.fit_baselines(pd.concat([d,holdout]))
        self.assertEqual(params,frozen)
        d["common_y"]=False
        with self.assertRaises(ValueError):e.fit_baselines(d)

    def test_common_population_scoring(self):
        d=pd.DataFrame({"common_y":[True,False,True],"batter":[1,2,3],"pitcher":[5,5,5],
                        "sav_y":[-10,100,10],e.CAT["y"]:["Late","Early","Early"],
                        "pred_y_dev_mean":[0,0,0]})
        r=e.score(d,"y","dev_mean",{})
        self.assertEqual((r["n"],r["n_total"],r["n_excluded"]),(2,3,1))
        self.assertEqual(r["mae"],10)

    def test_real_source_unique_and_group_mapping(self):
        cols=["play_id","season","valid_match","batter","pitcher","game_pk","statcast_game_year",
              "statcast_batter","statcast_pitcher","statcast_game_pk","statcast_pitch_type","statcast_pitch_group","pitch_type"]
        d=pd.read_parquet(e.SOURCE,columns=cols)
        e.validate_source(d)
        dup=d.copy();dup.loc[1,"play_id"]=dup.loc[0,"play_id"]
        with self.assertRaises(ValueError):e.validate_source(dup)
        mapped=d.statcast_pitch_type.map(e.PITCH_MAP).fillna("Unknown")
        self.assertTrue(mapped.eq(d.statcast_pitch_group.fillna("Unknown")).all())
        self.assertEqual(int(mapped.eq("Unknown").sum()),44)
        self.assertEqual(int(mapped.ne(d.pitch_type.map(e.PITCH_MAP).fillna("Unknown")).sum()),3)


if __name__=="__main__":unittest.main()
