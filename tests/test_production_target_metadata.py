"""Reject mixed-shower coordinates without fixing the runtime to one target size."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from production_model.model import ProductionModel

class TargetMetadataTests(unittest.TestCase):
    def fixture(self, root):
        source={'shower_sha256':'test-shower','incident_protons':20000,'samples_per_pool':131072}
        target={'length_m':1.46,'diameter_m':.5,'tungsten_length_m':1.33,'z_origin':'front face','beam_ring_radius_m':.05,'beam_sigma_xy_m':.016}
        docs={'manifest.json':{'target':target,'shower_sha256':'test-shower'},
              'results/idm_production/sources/provenance.json':source,
              'results/propagation/origins/provenance.json':{'source_provenance':source}}
        for name,data in docs.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
        return docs
    def test_segmented_target_loads_with_exact_source_origin_pair(self):
        with tempfile.TemporaryDirectory() as t,patch('production_model.model.Inputs') as inputs:
            r=Path(t);self.fixture(r);p=ProductionModel(r)
            self.assertEqual(p.metadata['target']['tungsten_length_m'],1.33)
            inputs.assert_called_once_with(with_origins=True,data_root=r)
    def test_wrong_coordinate_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as t,patch('production_model.model.Inputs') as inputs:
            r=Path(t);d=self.fixture(r);d['manifest.json']['target']['z_origin']='centre'
            (r/'manifest.json').write_text(json.dumps(d['manifest.json']))
            with self.assertRaisesRegex(ValueError,'front face'):ProductionModel(r)
            inputs.assert_not_called()
    def test_old_origin_pool_cannot_be_combined_with_new_production(self):
        with tempfile.TemporaryDirectory() as t,patch('production_model.model.Inputs') as inputs:
            r=Path(t);self.fixture(r);p=r/'results/propagation/origins/provenance.json'
            d=json.loads(p.read_text());d['source_provenance']['shower_sha256']='previous-shower';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'different showers'):ProductionModel(r)
            inputs.assert_not_called()
    def test_manifest_must_identify_the_actual_shower(self):
        with tempfile.TemporaryDirectory() as t,patch('production_model.model.Inputs'):
            r=Path(t);d=self.fixture(r);d['manifest.json']['shower_sha256']='previous-shower'
            (r/'manifest.json').write_text(json.dumps(d['manifest.json']))
            with self.assertRaisesRegex(ValueError,'different showers'):ProductionModel(r)
    def test_invalid_target_dimensions_are_rejected(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as t,patch('production_model.model.Inputs'):
                r=Path(t);d=self.fixture(r);d['manifest.json']['target']['length_m']=value
                (r/'manifest.json').write_text(json.dumps(d['manifest.json']))
                with self.assertRaisesRegex(ValueError,'positive target dimensions'):ProductionModel(r)
if __name__=='__main__':unittest.main()
