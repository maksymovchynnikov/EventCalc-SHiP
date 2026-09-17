"""Weight normalization, bounded export and native random-stream regression."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weighted_runtime import WeightedEventBatch,HadronizationNormalization,export_weighted
from funcs import decayProducts


def batch(start,weights,groups,key=('alp',3.,'central',.1,'release')):
    n=len(weights);records=np.zeros((n,22));records[:,3]=3.;records[:,4]=3.;records[:,6]=.25
    records[:,10:16]=[0.,0.,1.5,1.5,0.,22.]
    records[:,16:]=[0.,0.,-1.5,1.5,0.,22.]
    label=f'{decayProducts.MATCHED_PROCESS_LABEL}:Jets-cc'
    return WeightedEventBatch(0,start,start+n,records,np.full(n,label),
        np.ones(n,dtype=bool),np.full(n,.25),np.asarray(weights,dtype=float),np.asarray(groups),key)


class WeightedTests(unittest.TestCase):
    def test_global_normalization_not_each_batch(self):
        a=batch(0,[2.,1.],['fragmentation','external'])
        b=batch(2,[8.,1.],['fragmentation','external'])
        counter=HadronizationNormalization();counter.add(a);counter.add(b)
        result=counter.as_dict()
        self.assertEqual(result['fragmentation_factor'],.2)
        self.assertAlmostEqual(result['hadronization_effective_sample_size'],16./4.72)
        with self.assertRaises(TypeError):a.event(0)

    def test_no_cross_mass_or_release_normalization(self):
        a=batch(0,[1.],['external']);counter=HadronizationNormalization();counter.add(a)
        for key in (('alp',2.,'central',.1,'release'),('alp',3.,'central',.1,'other')):
            with self.assertRaises(ValueError):counter.add(replace(a,normalization_key=key))

    def test_bad_metadata_fails_before_modifying_totals(self):
        for weights,groups in (([np.nan],['fragmentation']),([-1.],['fragmentation']),
                               ([2.],['external']),([1.],['unknown'])):
            c=HadronizationNormalization()
            with self.assertRaises(ValueError):c.add(batch(0,weights,groups))
            self.assertEqual(c.events,0)

    def test_two_pass_export_has_normalized_analysis_weights(self):
        a=batch(0,[2.,1.],['fragmentation','external']);b=batch(2,[8.,1.],['fragmentation','external'])
        generator=SimpleNamespace(init=lambda:True,generate=lambda *a,**k:iter([a_batch,b_batch]),
            mass=3.,c_tau=10.,mode='fiducial',resolved_seed=123,weight_floor_fraction=.1,
            llp=SimpleNamespace(),experiment=SimpleNamespace(as_dict=lambda:{}),
            yields=lambda:SimpleNamespace(as_dict=lambda:{}))
        a_batch,b_batch=a,b
        with tempfile.TemporaryDirectory() as temp,patch('funcs.exhad_integration.metadata',return_value={}):
            directory=Path(temp)/'result';report=export_weighted(generator,4,directory,batch_size=2)
            files=report['files'];self.assertEqual(len(files),2)
            with np.load(directory/files[0]) as data:
                np.testing.assert_array_equal(data['hadronization_weights'],[.4,1.])
                np.testing.assert_array_equal(data['analysis_weights'],[.1,.25])
                np.testing.assert_array_equal(data['decay_weights'],[.25,.25])
            with np.load(directory/files[1]) as data:
                np.testing.assert_array_equal(data['hadronization_weights'],[1.6,1.])
            self.assertEqual(json.loads((directory/'metadata.json').read_text())['normalization']['events'],4)
            self.assertEqual(len(list(directory.glob('.weighted-raw-*'))),0)
            with self.assertRaises(FileExistsError):export_weighted(generator,4,directory)

    def test_pythia_stream_is_the_seed_the_block_was_given(self):
        primary=[[0.,0.,1.,1.,0.,22.,0.,1., 0.,0.,-1.,1.,0.,22.,0.,1.]]
        def settings(seed):
            fake=Mock();fake.init.return_value=True;fake.forceHadronLevel.return_value=True
            fake.event=MagicMock();particle=Mock()
            for name,value in dict(isFinal=True,px=0.,py=0.,pz=1.,e=1.,m=0.,id=22).items():
                getattr(particle,name).return_value=value
            fake.event.__iter__.side_effect=lambda:iter([particle])
            fake.event.size.return_value=1
            fake.event.__getitem__.return_value=particle
            module=SimpleNamespace(Pythia=lambda:fake)
            with patch.object(decayProducts,'_PYTHIA_INSTANCE',None),\
                 patch.object(decayProducts,'load_pythia8',return_value=module):
                decayProducts.process_events_with_pythia(primary,3.,seed=seed)
            return [call.args[0] for call in fake.readString.call_args_list]
        first,second,repeated=settings(11),settings(12),settings(11)
        # Pythia's own energy-momentum closure test, at the tolerance the
        # blocks have always run with rather than ErrorChecks.xml's 1e-4.
        self.assertIn('Check:event = on',first)
        self.assertIn('Check:epTolErr = 2e-6',first)
        self.assertIn('Random:seed = 11',first)
        self.assertIn('Random:seed = 12',second)
        self.assertEqual(first,repeated)
        self.assertNotEqual(first,second)
        # a block with no seed runs on Pythia's own default stream
        self.assertFalse([line for line in settings(None) if line.startswith('Random:seed')])


if __name__=='__main__':unittest.main()
