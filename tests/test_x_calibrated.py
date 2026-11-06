"""Development/holdout isolation and upstream-equivalence regression checks."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capstone.reference_calibration import (METHODS, XCOLS, apply_maps, calibrate, compare,
    fit_maps, map_fingerprint, references_for_target, x_player_rates)
from capstone.references import Config, calculate_references, prepare_season
from test_x_reference import fixture


class CalibratedX(unittest.TestCase):
    def setUp(self):
        self.s=prepare_season(fixture().drop(columns="relative_intercept_x"),2025)
        self.s["pitcher"]=[10]*len(self.s)
        self.p=pd.DataFrame({"x_hat":[-10.,-7.,-4.,-1.,2.,5.,8.,11.]})
        self.hist=pd.DataFrame({"axis":["x"]*10,"bin":np.arange(-9,11,2),
            "n":[1,3,8,20,35,35,20,8,3,1],"season":[2025]*10})
        self.truth=pd.DataFrame({"year":[2025,2025],"bat_contact_code":[2,2],
            "n_swings":[100,200],"tied_up_percent":[.1,.2],
            "centered_percent":[.8,.6],"flailed_percent":[.1,.2]})

    def test_exact_original_quantile_fit(self):
        maps=fit_maps(self.p,self.s,self.hist,self.truth)
        rates=dict(zip(("lo","mid","hi"),np.average(self.truth[list(XCOLS)],weights=self.truth.n_swings,axis=0)))
        target=calibrate.per_type_target(self.hist.groupby("bin").n.sum(),"x",rates)
        expected=calibrate.QuantileMap().fit(self.p.x_hat,target,axis="x")
        np.testing.assert_array_equal(maps[("x","in_play")].src,expected.src)
        np.testing.assert_array_equal(maps[("x","in_play")].dst,expected.dst)
        self.assertEqual(len(expected.src),201)

    def test_reject_holdout_predictions_histograms_and_rates(self):
        with self.assertRaises(ValueError): fit_maps(self.p,self.s.assign(season=2026),self.hist,self.truth)
        with self.assertRaises(ValueError): fit_maps(self.p,self.s,self.hist.assign(season=2024),self.truth)
        with self.assertRaises(ValueError): fit_maps(self.p,self.s,self.hist,self.truth.assign(year=2026))
        with self.assertRaises(ValueError): fit_maps(self.p.set_axis(range(1,9)),self.s,self.hist,self.truth)

    def test_transform_frozen_and_missing_preserved(self):
        maps=fit_maps(self.p,self.s,self.hist,self.truth)
        before=map_fingerprint(maps)
        holdout=self.s.assign(season=2026)
        v=self.p.x_hat.copy();v.iloc[0]=np.nan
        got=apply_maps(v,holdout,maps)
        self.assertTrue(np.isnan(got[0]))
        self.assertEqual(map_fingerprint(maps),before)
        np.testing.assert_allclose(got,maps[("x","in_play")].transform(v),equal_nan=True)
        # Extreme holdout values clamp, and never re-estimate source quantiles.
        apply_maps(np.full(len(holdout),9999.),holdout,maps)
        self.assertEqual(map_fingerprint(maps),before)

    def test_original_unknown_denominators(self):
        r=x_player_rates(self.s,[np.nan,-4.,4.,5.,-5.,0.,0.,0.],"batter").set_index("id")
        self.assertEqual(r.loc[1,"n_swings"],4)
        self.assertEqual(r.loc[1,"unknown_x_percent"],.25)
        self.assertEqual(r.loc[1,"centered_percent"],.5)
        self.assertEqual(r.loc[1,"flailed_percent"],.25)

    def test_references_stay_development_only(self):
        refs=calculate_references(self.s,Config(1,1))
        original=refs.copy(deep=True)
        target=self.s.assign(batter=3)
        mapped=references_for_target(refs,target)
        self.assertTrue(mapped.loc[3,"fallback_ev50"])
        self.assertEqual(mapped.loc[3,"ref_ev50"],refs.ref_global.iloc[0])
        pd.testing.assert_frame_equal(refs,original)

    def test_identical_scoring_populations(self):
        s=pd.concat([self.s]*25,ignore_index=True)
        truth=pd.DataFrame({"id":[1,2],"name":["A","B"],"n_swings":[100,100],
            "tied_up_percent":[.2,.2],"centered_percent":[.6,.6],"flailed_percent":[.2,.2]})
        values={m:np.tile(self.p.x_hat.to_numpy(),25) for m in METHODS}
        metrics,details,pairs,den=compare(values,s,s,truth,"batter",dict(season=2026,stage="calibrated",protocol="test"))
        self.assertTrue(metrics.players.eq(2).all())
        self.assertTrue(metrics.swings.eq(200).all())
        self.assertTrue(metrics.delta_mae_pp.eq(0).all())
        self.assertTrue(pairs.groupby("method").id.apply(tuple).map(lambda x:x==(1,2)).all())


if __name__=="__main__":unittest.main()
