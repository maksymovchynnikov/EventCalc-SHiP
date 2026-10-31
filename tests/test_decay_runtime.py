from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from funcs.simulation_config import config_from_mapping
from eventcalc.decay_runtime import RuntimeDecayAdapter
from eventcalc.production.external import ParentBatch


class DecayRuntimeTests(unittest.TestCase):
    def config(self, model='Dark-photons', **kwargs):
        return config_from_mapping(dict(model=model,events=8,masses=[.3],c_taus=[1.],
            decay_channels=['e-pair'],exhad='off',**kwargs),check_files=False)

    def test_external_parents_do_not_load_production(self):
        with patch('eventcalc.production.dark_photon.compact.table',side_effect=AssertionError('Production loaded')), \
             patch('funcs.initLLP.read_distribution',side_effect=AssertionError('Spectrum loaded')):
            adapter=RuntimeDecayAdapter(self.config(),mass=.3,seed=52)
            p=np.array([[.1,0,100,np.sqrt(10000+.01+.09)]]*8)
            parent=ParentBatch(p,np.zeros((8,3)),np.array([1,-1]*4)*.1,.3)
            b=adapter.decay(parent)
            b.arrays()
            np.testing.assert_array_equal(b.pdg_ids,np.tile([-11,11],(8,1)))
            np.testing.assert_allclose(b.weights,parent.weights*adapter.generator.visible_branching_ratio)
            self.assertEqual(b.weights.sum(),0)
            with self.assertRaises(ValueError):
                adapter.decay(ParentBatch(p,np.zeros((8,3)),parent.weights,.4))

    def test_decay_tables_preserve_existing_model_values(self):
        from funcs.initLLP import LLP
        root=Path(__file__).resolve().parents[1]/'Distributions'
        models=[('Dark-photons',dict(dp_production_mode='combined')),
                ('Scalar-mixing',{}),('ALP-photon',dict(alp_production_mode='primary')),
                ('ALP-fermion',{}),('ALP-SU2L',{}),
                ('ALP-mixed',dict(xi=.35,interference='constructive')),
                ('HNL',dict(mixing_pattern=np.array([.3,.4,.3])))]
        for model,kwargs in models:
            with self.subTest(model=model):
                args=dict(mass=None,particle_selection={'LLP_name':model,'particle_path':str(root/model)},**kwargs)
                try:
                    original=LLP(**args,defer_production_tables=True)
                except Exception as exc:
                    from funcs.hnl_source_validation import HNLSourceValidationError
                    if model!='HNL' or not isinstance(exc,HNLSourceValidationError): raise
                    # CRLF-converted checkouts can invalidate the existing signed pair.
                    # Both paths must fail closed; never rewrite source authority here.
                    with self.assertRaises(HNLSourceValidationError): LLP(**args,decay_only=True)
                    continue
                extracted=LLP(**args,decay_only=True)
                low=max(original.m_min_tabulated,extracted.m_min_tabulated,.01)
                high=min(original.m_max_tabulated,extracted.m_max_tabulated,2.)
                for mass in np.geomspace(low*1.001,high*.999,3):
                    np.testing.assert_allclose(extracted.get_Br(mass),original.get_Br(mass),rtol=1e-12,atol=1e-15)
                    np.testing.assert_allclose(extracted.get_ctau(mass),original.get_ctau(mass),rtol=1e-12)


if __name__=='__main__': unittest.main()
