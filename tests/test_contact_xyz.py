"""Scientific invariants for the frozen contact XYZ experiment."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import capstone.data as audit
import capstone.reconstruction as exp
import capstone.evaluation as scoring
from capstone import outputs

def predictions(columns=None,filters=None):
    d=outputs.legacy_frame(audit.OUT,"predictions")
    for column,op,value in filters or []:
        if op!="==":raise ValueError(op)
        d=d[d[column].eq(value)]
    return (d if columns is None else d[columns]).reset_index(drop=True)

class ContactXYZTests(unittest.TestCase):
    def test_ingest_excludes_league_and_unique_keys(self):
        d=audit.ingest(write=False)[0]
        self.assertFalse(d.is_league.any())
        self.assertFalse(d.duplicated(exp.KEYS).any())
        self.assertTrue(d.batter.notna().all())
        for ax in audit.AXES: np.testing.assert_allclose(d[list(audit.RATES[ax])].sum(axis=1),1,atol=1e-8)

    def test_partition_reconciliation(self):
        d=audit.ingest(write=False)[1];d=d[d.complete]
        np.testing.assert_allclose(d.count_difference,0)
        self.assertLess(d.max_rate_gap.max(),1e-10)
        self.assertLess(d.max_tail_gap.max(),1e-9)

    def test_league_weighting_and_undefined_tail(self):
        d=pd.DataFrame({'n_swings':[10,90]})
        for ax in audit.AXES:
            for c,v in zip(audit.RATES[ax],([.5,0],[.5,.9],[0,.1])):d[c]=v
            d[audit.TAILS[ax][0]]=[-8,np.nan];d[audit.TAILS[ax][1]]=[np.nan,11]
        s=audit.weighted_stats(d)
        self.assertAlmostEqual(s['tied_up_percent'],.05)
        self.assertEqual(s['avg_x_tied_up'],-8)
        self.assertEqual(s['avg_x_flail'],11)

    def test_author_contact_types_and_bunts(self):
        d=pd.DataFrame({'description':['hit_into_play','foul','foul_tip','swinging_strike','missed_bunt','foul_bunt'], 'bat_speed':[70]*6})
        out=exp.join.derive_columns(d,2025)
        self.assertEqual(out.contact_type.tolist(),['in_play','foul','whiff','whiff','whiff','foul'])
        self.assertEqual(exp.join.swing_filter(out,'bat_speed_nobunt').tolist(),[True]*4+[False]*2)

    def test_pitch_mapping_unknown_is_retained(self):
        self.assertEqual(audit.PITCH_MAP['FC'],'FB');self.assertEqual(audit.PITCH_MAP['ST'],'BB');self.assertEqual(audit.PITCH_MAP['FS'],'CH')
        self.assertNotIn('KN',audit.PITCH_MAP);self.assertNotIn('UN',audit.PITCH_MAP)

    def test_thresholds_missing_denominator_and_conditional_mean(self):
        for ax,t in exp.discover.THRESH.items():
            d=pd.DataFrame({'batter':[1]*7})
            v=np.array([-t-1,-t,0,t,t+1,np.nan,np.inf])
            a=scoring.category_stats(d,v,ax,['batter']).iloc[0]
            self.assertEqual(a.local_n,7);self.assertEqual(a.known_n,5)
            np.testing.assert_allclose([a[c] for c in audit.RATES[ax]],[1/7,3/7,1/7])
            self.assertEqual(a[audit.TAILS[ax][0]],-t-1)
            self.assertEqual(a[audit.TAILS[ax][1]],t+1)
        a=scoring.category_stats(pd.DataFrame({'batter':[1,1]}),[0,np.nan],'x',['batter']).iloc[0]
        self.assertTrue(np.isnan(a.avg_x_tied_up));self.assertTrue(np.isnan(a.avg_x_flail))

    def test_all_swing_aggregation_uses_actual_counts(self):
        d=pd.DataFrame({'batter':[1]*10})
        a=scoring.category_stats(d,[-8]*2+[0]*8,'x',['batter']).iloc[0]
        self.assertAlmostEqual(a.tied_up_percent,.2)
        self.assertNotAlmostEqual(a.tied_up_percent,.5)

    def test_original_coefficients_and_units(self):
        r=exp.SPECS['y'].stand_models['R'];l=exp.SPECS['y'].stand_models['L']
        self.assertAlmostEqual(r['coef']['icpt_y_over_ballspeed'],.9235093592055167)
        self.assertAlmostEqual(r['intercept'],-17.918413523829468)
        self.assertAlmostEqual(l['coef']['icpt_y_over_ballspeed'],.9106727948752966)
        self.assertAlmostEqual(l['intercept'],-18.343601678682262)
        self.assertAlmostEqual(90*5280*12/3600/1000,1.584)
        self.assertAlmostEqual(float(exp.reconstruct.ev_ceiling(70,90)),106.8)

    def test_2026_geometry_and_original_z_branch(self):
        self.assertAlmostEqual(exp.geometry.plate_y_ft(2025),17/12)
        self.assertAlmostEqual(exp.geometry.plate_y_ft(2026),17/24)
        z=exp.reconstruct.z_from_launch([100,300,np.nan],[0,0,10],[True,False,False])
        np.testing.assert_allclose(z[:2],[2.75,6]);self.assertTrue(np.isnan(z[2]))
        self.assertFalse(exp.SPECS['z'].all_swing)

    def test_original_x_continuation_and_asymmetry(self):
        v=exp.reconstruct.x_from_ev_deficit([9,9,80],[True,False,False])
        self.assertEqual(v[0],-4);self.assertGreater(v[1],4);self.assertGreater(v[2],14)

    def test_bin_centers_and_calibration_no_leakage(self):
        np.testing.assert_allclose(audit.savant_truth._lower_edge(['x','y','z'],[0,0,0]),[-1,-1,-.5])
        for year in [2024,2026]:
            with self.assertRaises(ValueError):exp.assert_dev(pd.DataFrame({'season':[year]}))
        with self.assertRaises(ValueError):exp.assert_dev(pd.DataFrame({'season':[]}))
        with self.assertRaises(ValueError):exp.transform([0],dict(fit_season=2026,src=[0,1],dst=[0,1]))

    def test_map_missing_and_endpoint_behavior(self):
        s=dict(fit_season=2025,src=[-1,0,1],dst=[-4,0,4])
        np.testing.assert_allclose(exp.transform([-10,0,10,np.nan],s),[-4,0,4,np.nan],equal_nan=True)

    def test_reference_training_refuses_mixed_seasons(self):
        d=pd.DataFrame({'season':[2025,2026]})
        with self.assertRaises(ValueError):exp.fit_references(d,pd.Series([0,0]))

    @unittest.skipUnless((audit.PARAMETERS/'references.parquet').exists(),'Run experiment first')
    def test_saved_reference_fallbacks(self):
        r=pd.read_parquet(audit.PARAMETERS/'references.parquet')
        self.assertTrue(r.season.eq(2025).all())
        np.testing.assert_allclose(r.loc[r.fallback_ev50,'ref_ev50'],r.loc[r.fallback_ev50,'ref_global'])
        np.testing.assert_allclose(r.loc[r.fallback_y_ev50,'time_ev50'],r.loc[r.fallback_y_ev50,'time_global'])

    @unittest.skipUnless((audit.PARAMETERS/'calibration_diagnostics.csv').exists(),'Run experiment first')
    def test_calibration_methods_share_fit_populations(self):
        d=pd.read_csv(audit.PARAMETERS/'calibration_diagnostics.csv')
        for _,g in d.groupby(['axis','contact_type']):self.assertEqual(g.n_fit.nunique(),1)

    @unittest.skipUnless((audit.PARAMETERS/'author_native_maps.json').exists(),'Run experiment first')
    def test_native_maps_exactly_reproduce_upstream_fitting(self):
        cols=['season','contact_type',*[a+'_author_raw' for a in audit.AXES]]
        d=predictions(columns=cols,filters=[('season','==',2025)])
        hats=pd.DataFrame({a+'_hat':d[a+'_author_raw'] for a in audit.AXES})
        hist=audit.development_histograms()
        fitted=exp.calibrate.fit_calibration(hats,d,2025,hist=hist,rates=exp.target_rates())
        saved=json.loads((audit.PARAMETERS/'author_native_maps.json').read_text())
        self.assertEqual(set(saved),{f'{a}|{ct}' for a,ct in fitted})
        for (a,ct),m in fitted.items():
            np.testing.assert_array_equal(m.src,saved[f'{a}|{ct}']['src'])
            np.testing.assert_array_equal(m.dst,saved[f'{a}|{ct}']['dst'])

    @unittest.skipUnless((audit.OUT/'swings.parquet').exists(),'Run experiment first')
    def test_saved_y_formula_and_reference_provenance(self):
        cols=['season','stand',exp.Y,'ball_in_per_ms','y_author_raw','author_x_reference_season',
              'x_reference_global','x_reference_batter','fallback_x_batter','y_tango_time_raw','y_reference_tango_time_ms']
        d=predictions(columns=cols)
        expected=np.full(len(d),np.nan)
        for hand,m in exp.SPECS['y'].stand_models.items():
            take=d.stand.eq(hand)
            expected[take]=m['coef']['icpt_y_over_ballspeed']*d.loc[take,exp.Y]/d.loc[take,'ball_in_per_ms']+m['intercept']
        np.testing.assert_allclose(d.y_author_raw,expected,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(d.y_tango_time_raw,d.y_author_raw-d.y_reference_tango_time_ms,atol=1e-12,equal_nan=True)
        self.assertTrue(d.author_x_reference_season.eq(d.season).all())
        np.testing.assert_allclose(d.loc[d.fallback_x_batter,'x_reference_batter'],d.loc[d.fallback_x_batter,'x_reference_global'])

    def test_no_truth_predictor_boundary(self):
        # The raw prediction boundary takes only source Statcast, unchanged author
        # features and development references. No official pitch labels are loaded.
        src=(ROOT/'capstone/reconstruction.py').read_text()
        self.assertNotIn('whiff_tail_matched',src)
        self.assertNotIn('sav_x',src)
        self.assertNotIn('sav_y',src)
        self.assertNotIn('sav_z',src)

    def test_cluster_bootstrap_determinism(self):
        d=pd.DataFrame({'batter':[1,1,2,3,3], 'delta':[1,2,-1,0,1]})
        self.assertEqual(scoring.clustered_ci(d,'delta',reps=100),scoring.clustered_ci(d,'delta',reps=100))
        c=pd.DataFrame({'batter':[1,1,2,3], 'delta':[2,2,2,2]})
        self.assertEqual(scoring.clustered_ci(c,'delta',reps=100),(2,2))

    @unittest.skipUnless((audit.OUT/'swings.parquet').exists(),'Run experiment first')
    def test_artifact_counts_missingness_and_frozen_maps(self):
        d=predictions(columns=['season','contact_type','game_pk','at_bat_number','pitch_number','has_launch_angle','z_author_raw','reference_fit_season'])
        self.assertEqual(len(d),974886)
        self.assertFalse(d.duplicated(['game_pk','at_bat_number','pitch_number']).any())
        self.assertTrue(d.reference_fit_season.eq(2025).all())
        self.assertTrue(d.loc[~d.has_launch_angle,'z_author_raw'].isna().all())
        maps=json.loads((audit.PARAMETERS/'calibration_maps.json').read_text())
        self.assertTrue(all(v['fit_season']==2025 for v in maps.values()))
        self.assertTrue(all(len(v['src'])==201 for v in maps.values()))

    @unittest.skipUnless((audit.OUT/'players.parquet').exists(),'Run experiment first')
    def test_shared_cohorts_and_metric_recomputation(self):
        d=outputs.legacy_frame(audit.OUT,'players');d=d[d.eligible]
        for _,g in d.groupby(['season','contact_type','group','axis']):
            counts=g.groupby(['method','stage']).batter.nunique()
            self.assertEqual(counts.nunique(),1)
        overall=outputs.read_tables(audit.OUT)['overall_metrics']
        for ax in audit.AXES:
            g=d[d.season.eq(2026)&d.group.eq('ALL')&d.contact_type.eq('in_play')&d.axis.eq(ax)&d.method.eq('author')&d.stage.eq('calibrated')]
            error=np.mean([100*(g[c+'_pred']-g[c+'_official']).abs() for c in audit.RATES[ax]],axis=0).mean()
            row=overall[overall.season.eq(2026)&overall.group.eq('ALL')&overall.contact_type.eq('in_play')&overall.axis.eq(ax)&overall.method.eq('author')&overall.stage.eq('calibrated')].iloc[0]
            self.assertAlmostEqual(error,row.mae_pp,places=10)

    def test_relative_direction_does_not_invent_signal_for_constant_null(self):
        from capstone.pipeline import tendencies
        player=pd.DataFrame(dict(season=[2026]*3,contact_type=['in_play']*3,axis=['x']*3,
            method=['dev_null']*3,stage=['baseline']*3,group=['ALL']*3,eligible=[True]*3,
            local_n=[3]*3,n_swings=[3]*3))
        for category,predicted,truth in zip(audit.RATES['x'],[.2,.6,.2],[[.1,.2,.5],[.7,.6,.3],[.2,.2,.2]]):
            player[category+'_pred']=predicted;player[category+'_official']=truth
        cells=pd.DataFrame(columns=['season','contact_type','axis','method','stage','contrast','statistic'])
        result=tendencies(player,cells).set_index('statistic')
        self.assertEqual(result.loc['tied_up_percent','direction_accuracy'],0.)
        self.assertEqual(result.loc['flailed_percent','direction_accuracy'],1.)
        self.assertTrue(result.spearman_r.isna().all())

if __name__=='__main__': unittest.main()
