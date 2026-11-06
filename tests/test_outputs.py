"""Four-file serialization and paired, batter-clustered inference invariants."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from capstone import comparisons as c,outputs as o,verification as v

class PairedBootstrapTests(unittest.TestCase):
    def test_clusters_retain_unequal_numbers_of_observations(self):
        d=pd.DataFrame({'batter':[1,1,1,2,3,3],'improvement':[1,2,6,-8,4,5]})
        got=c.paired_summary(d,reps=1000)
        clusters=d.groupby('batter').improvement.agg(['sum','count'])
        ix=np.random.default_rng(180).integers(3,size=(1000,3))
        boot=clusters['sum'].to_numpy()[ix].sum(axis=1)/clusters['count'].to_numpy()[ix].sum(axis=1)
        np.testing.assert_allclose([got['ci_low'],got['ci_high']],np.quantile(boot,[.025,.975]))
        self.assertEqual(got['improvement'],d.improvement.mean())
        self.assertEqual(got['batters'],3);self.assertEqual(got['cells'],6)

    def test_positive_negative_and_self_improvements(self):
        for difference in (2.,-2.,0.):
            d=pd.DataFrame({'batter':[1,1,2,3],'improvement':[difference]*4})
            got=c.paired_summary(d)
            self.assertEqual((got['ci_low'],got['ci_high']),(difference,difference))
            self.assertEqual(got['excludes_zero'],difference!=0)

    def test_contrast_pairs_join_identities_and_keep_nulls(self):
        cells=[]
        for method,stage,error in [('author','raw',5.),('ev50','raw',3.),('dev_null','baseline',7.)]:
            for batter in (1,2,3):
                for contrast in ('FB-BB','FB-CH','BB-CH'):
                    for stat in ('tied_up_percent','centered_percent','flailed_percent'):
                        cells.append(dict(season=2026,contact_type='in_play',axis='x',method=method,stage=stage,
                                          batter=batter,contrast=contrast,statistic=stat,error=error))
        # No eligible player rows needed to test the independent contrast comparison.
        player=pd.DataFrame(columns=['season','contact_type','axis','eligible'])
        player['eligible']=player.eligible.astype(bool)
        result=c.aggregate_comparisons(player,pd.DataFrame(cells).sample(frac=1,random_state=7))
        row=result[result.method.eq('ev50')&result.baseline.eq('author')&result.contrast.eq('ALL_CONTRASTS')].iloc[0]
        self.assertEqual(row.improvement,2.);self.assertEqual(row.cells,27);self.assertEqual(row.batters,3)
        self.assertEqual(row.ci_low,2.);self.assertEqual(row.ci_high,2.)
        self.assertTrue(result.baseline.eq('dev_null').any())

class OutputSchemaTests(unittest.TestCase):
    def test_method_identity_keeps_stage_protocol_and_whiff_baseline_distinct(self):
        self.assertNotEqual(o.method_id('x','ev50','raw'),o.method_id('x','ev50','calibrated'))
        self.assertNotEqual(o.method_id('x','ev50','raw','same_season_raw'),o.method_id('x','ev50','raw','frozen_2025'))
        self.assertIn('published_whiff_regression',o.method_id('z','author','raw',contact_type='whiff'))

    def test_old_error_difference_is_displayed_as_positive_improvement(self):
        old=pd.DataFrame([dict(season=2026,contact_type='in_play',axis='x',group='ALL',stage='raw',
            method='ev50',baseline='author',metric='rate_error_pp',cells=20,batters=20,
            difference=-2.,ci_low=-3.,ci_high=-1.)])
        encoded,s=o.encode_metrics({'paired_comparisons':old})
        self.assertEqual(len(encoded),1)
        row=encoded.iloc[0]
        self.assertEqual(row.metric,'improvement');self.assertEqual(row.value,2.)
        self.assertEqual(row.ci_low,1.);self.assertEqual(row.ci_high,3.);self.assertTrue(row.excludes_zero)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metrics.csv';encoded.to_csv(path,index=False)
            decoded=o.decode_metrics(pd.read_csv(path,float_precision='round_trip'),s)
        v.assert_frame(decoded['paired_comparisons'],old,'interval round trip')

    def test_undefined_intervals_stay_missing(self):
        old=pd.DataFrame([dict(season=2026,axis='y',method='author',stage='raw',mae=2.,players=50)])
        encoded,_=o.encode_metrics({'test':old})
        self.assertTrue(encoded.ci_low.isna().all());self.assertTrue(encoded.excludes_zero.isna().all())

    def test_nan_canonicalization_never_rounds_finite_values(self):
        a=pd.DataFrame({'value':[1.,np.nan]})
        b=pd.DataFrame({'value':[1.,np.copysign(np.nan,-1.)]})
        self.assertEqual(v.frame_digest(a,['value']),v.frame_digest(b,['value']))
        b.loc[0,'value']=np.nextafter(1.,2.)
        self.assertNotEqual(v.frame_digest(a,['value']),v.frame_digest(b,['value']))

    def test_output_contract_has_no_manifest_or_intermediate(self):
        self.assertEqual(o.FILES,{'metrics.csv','players.parquet','swings.parquet','report.md'})

if __name__=='__main__':unittest.main()
