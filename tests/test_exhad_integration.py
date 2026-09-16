"""Fast routing/ownership tests; tests/smoke_exhad.py exercises real Pythia."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from funcs import decayProducts
from funcs import exhad_integration as integration
from funcs.simulation_config import ConfigurationError, config_from_mapping


class IntegrationTests(unittest.TestCase):
    def test_parallel_generator_configuration_and_metadata(self):
        binding = dict(root='unused', python='unused', model='alp-fermion', version='test')
        module = SimpleNamespace(ParallelGenerator=Mock())
        generator = SimpleNamespace(workers=3, chunk_size=17, close=Mock())
        module.ParallelGenerator.return_value = generator
        with patch.object(integration, 'release', return_value=module), patch.dict(
                integration.os.environ, EXHAD_WORKERS='3', EXHAD_CHUNK_SIZE='17'), \
                patch.object(integration, '_GENERATORS', {}):
            self.assertIs(integration._generator(binding), generator)
            self.assertIs(integration._generator(binding), generator)
            module.ParallelGenerator.assert_called_once_with('alp-fermion', workers=3,
                chunk_size=17, root='unused', python='unused')
            self.assertEqual(binding['execution'], dict(workers=3, chunk_size=17,
                seed_stream='chunked-v1'))
            integration.close_generators()
            generator.close.assert_called_once()

    def test_invalid_parallel_settings_fail_before_loading(self):
        for workers, size in [('0','512'), ('65','512'), ('2','0')]:
            with patch.dict(integration.os.environ, EXHAD_WORKERS=workers,
                            EXHAD_CHUNK_SIZE=size), self.assertRaises(ValueError):
                integration._generator({})

    def test_release_without_host_interface_is_rejected(self):
        llp = SimpleNamespace(LLP_name='ALP-fermion')
        config = SimpleNamespace(hadronization='exhad', exhad_root='unused', exhad_python=None)
        with patch.object(integration, 'release', return_value=SimpleNamespace(Generator=Mock())), \
                self.assertRaisesRegex(ValueError, 'model_info'):
            integration.configure_llp(llp, config)
        self.assertIsNone(llp._exhad_binding)

    def test_exclusive_alp_threshold_steps_are_numeric(self):
        from funcs.initLLP import LLP
        llp = LLP.__new__(LLP)
        llp.Matrix_elements_expr = []
        func, = llp.compile_matrix_elements(['UnitStep(E1 - 0.3) + UnitStep(E3 - 0.5)'])
        np.testing.assert_array_equal(func(1.8, np.array([.2, .4]), np.array([.6, .4])), [1., 1.])

    def setUp(self):
        self.llp = SimpleNamespace(_exhad_binding={
            'model': 'dark-photon', 'start': 1.7, 'end': 5.0, 'root': 'unused', 'python': 'unused'})
        self.pdgs = [np.array([-1, 1]), np.array([-2, 2]), np.array([22, 22])]
        self.br = [.4, .4, .2]
        self.event = [[0., 0., 1., 1., 0., 22.], [0., 0., -1., 1., 0., 22.]]

    def run_pool(self, selected=(0, 1, 2), n=10):
        def direct(mass, pdgs, branching, count, *args):
            return np.array([[v for p in self.event for v in p]]*count), np.array([count])
        fake = Mock()
        fake.generate.side_effect = lambda mass, count, **kw: [self.event]*count
        with patch.object(integration, '_generator', return_value=fake), patch.object(
                decayProducts, 'simulateDecays_rest_frame', side_effect=direct):
            result = integration.simulate_decays(self.llp, decayProducts, 2., self.pdgs,
                self.br, n, None, list(selected), sum(self.br[i] for i in selected), seed=8)
        return result, fake

    def test_pool_sampled_once_and_total_count_preserved(self):
        (events, sizes), fake = self.run_pool()
        fake.generate.assert_called_once_with(2., 8, seed=8)
        self.assertEqual(events.shape, (10, 12))
        self.assertEqual(list(sizes), [4, 4, 2])
        self.assertEqual(self.llp._exhad_pooled_indices, (0, 1))

    def test_partial_positive_hadronic_pool_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete hadronic'):
            self.run_pool(selected=(0, 2))

    def test_zero_weight_omission_is_safe(self):
        self.br = [.8, 0., .2]
        (events, _), fake = self.run_pool(selected=(0, 2))
        self.assertEqual(len(events), 10)
        fake.generate.assert_called_once()

    def test_generator_failure_has_no_raw_fallback(self):
        with patch.object(integration, '_generator', side_effect=RuntimeError('broken card')):
            with self.assertRaisesRegex(RuntimeError, 'broken card'):
                integration.simulate_decays(self.llp, decayProducts, 2., self.pdgs,
                    self.br, 10, None, [0, 1, 2], 1.)

    def test_below_support_keeps_native_exclusive_path(self):
        expected = (np.ones((2, 12)), np.array([2]))
        with patch.object(decayProducts, 'simulateDecays_rest_frame', return_value=expected) as raw:
            observed = integration.simulate_decays(self.llp, decayProducts, 1.5,
                self.pdgs, self.br, 2, None, [0, 1, 2], 1.)
        self.assertIs(observed, expected)
        raw.assert_called_once()

    def test_above_support_requires_explicit_raw_choice(self):
        with self.assertRaisesRegex(ValueError, 'through 5'):
            integration.simulate_decays(self.llp, decayProducts, 5.1, self.pdgs,
                self.br, 10, None, [0, 1, 2], 1.)

    def test_alp_support_comes_from_the_selected_deployment(self):
        self.llp._exhad_binding.update(model='alp-fermion', start=1.911, end=3.5)
        with self.assertRaisesRegex(ValueError, 'through 3.5'):
            integration.simulate_decays(self.llp, decayProducts, 3.5001,
                self.pdgs, self.br, 10, None, [0, 1, 2], 1.)
        metadata = integration.metadata(SimpleNamespace(_exhad_binding={
            **self.llp._exhad_binding, 'version': 'test'}))
        self.assertEqual(metadata['matched_support_gev'], [1.911, 3.5])

    def test_photon_alp_is_not_fermion_benchmark(self):
        values = dict(model='ALP-SU2L', hadronization='exhad', events=10,
                      masses=[1.], c_taus=[10.], decay_channels=['all'])
        with self.assertRaisesRegex(ConfigurationError, 'No exHad'):
            config_from_mapping(values, check_files=False)

    def test_explicit_raw_needs_no_exhad_installation(self):
        llp = SimpleNamespace()
        integration.configure_llp(llp, SimpleNamespace(hadronization='raw'))
        self.assertIsNone(llp._exhad_binding)

    def test_text_export_groups_pool_and_preserves_parent_pairing(self):
        self.llp._exhad_pooled_indices = (0, 2)
        self.llp.decayChannels = ['quarks-a', 'leptons', 'quarks-b']
        mothers = np.arange(5).reshape(-1, 1)
        daughters = mothers + 10
        m, d, labels, counts, indices = integration.legacy_output(
            self.llp, mothers, daughters, [2, 1, 2], [0, 1, 2])
        self.assertEqual(labels, ['leptons', 'Hadronic-exHad'])
        self.assertEqual(list(counts), [1, 4])
        self.assertEqual(indices, [0, 1])
        self.assertEqual(list(m[:, 0]), [2, 0, 1, 3, 4])
        np.testing.assert_array_equal(d-m, np.full((5, 1), 10))


class HNLIntegrationTests(unittest.TestCase):
    def test_hnl_config_is_supported(self):
        config = config_from_mapping(dict(model='HNL', events=10, masses=[2.],
            c_taus=[10.], mixing_pattern=[1.,0.,0.], decay_channels=['all']), check_files=False)
        self.assertEqual(config.hadronization, 'exhad')

    def test_signed_rows_are_not_pooled_and_native_poles_are_untouched(self):
        llp = SimpleNamespace(LLP_name='HNL', _exhad_binding={
            'model':'hnl', 'start':.02, 'end':5.27}, _exhad_pooled_indices=())
        pdgs = [np.array([11,2,-1]), np.array([-11,-2,1]), np.array([11,211,-999])]
        fake = Mock()
        fake.hadronize_hnl.return_value = [[[0.,0.,0.,2.,2.,22.]]]*3
        native = Mock(side_effect=lambda *args, **kw:
                      ([[0.]*(24 if kw['primary_only'] else 6)]*args[3], np.array([args[3]])))
        with patch.object(integration, '_generator', return_value=fake), \
                patch.object(decayProducts, 'simulateDecays_rest_frame', native):
            rows, counts = integration.simulate_decays(llp, decayProducts, 2., pdgs,
                [.3,.3,.4], 10, None, [0,1,2], 1., seed=19)
        self.assertEqual(fake.hadronize_hnl.call_count, 2)
        self.assertEqual(fake.hadronize_hnl.call_args_list[0].args[1], [11,2,-1])
        self.assertEqual(fake.hadronize_hnl.call_args_list[1].args[1], [-11,-2,1])
        self.assertFalse(native.call_args_list[-1].kwargs['primary_only'])
        self.assertEqual(llp._exhad_pooled_indices, ())
        self.assertEqual(len(rows), 10)
        np.testing.assert_array_equal(counts, [3,3,4])

    def test_hnl_outside_support_fails_before_sampling(self):
        with self.assertRaisesRegex(ValueError, '5.27'):
            integration._simulate_hnl({'start':.02,'end':5.27}, decayProducts,
                5.28, [], [], 0, None, [], 1., 1)


class NativePythiaRecordTests(unittest.TestCase):
    def fake(self, outcomes):
        generator = Mock()
        generator.init.return_value = True
        generator.forceHadronLevel.side_effect = outcomes
        generator.event = MagicMock()
        particle = Mock()
        for name, value in dict(isFinal=True, px=0., py=0., pz=1., e=1., m=0., id=22).items():
            getattr(particle,name).return_value = value
        generator.event.__iter__.side_effect = lambda: iter([particle])
        generator.event.size.return_value = 1
        generator.event.__getitem__.return_value = particle
        return generator

    def test_system_record_is_initialized_and_failed_attempt_is_rebuilt(self):
        generator = self.fake([False, True])
        primary = [[0.,0.,1.,1.,0.,22.,0.,1., 0.,0.,-1.,1.,0.,22.,0.,1.]]
        with patch.object(decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
            result = decayProducts.process_events_with_pythia(primary,2.)
        self.assertEqual(generator.event.reset.call_count,2)
        self.assertEqual(generator.event.__getitem__.return_value.p.call_count,2)
        generator.event.__getitem__.return_value.p.assert_called_with(0.,0.,0.,2.)
        generator.event.__getitem__.return_value.m.assert_any_call(2.)
        self.assertEqual(generator.event.append.call_args_list[:3],
                         generator.event.append.call_args_list[3:])
        generator.event.__iter__.assert_not_called()
        generator.event.size.assert_called_once_with()
        self.assertEqual(len(result),1)

    def test_persistent_failure_returns_no_event(self):
        generator = self.fake([False]*10)
        with patch.object(decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
            with self.assertRaisesRegex(RuntimeError,'after 10 attempts'):
                decayProducts.process_events_with_pythia([[0.,0.,0.,2.,2.,221.,0.,0.]],2.)
        self.assertEqual(generator.forceHadronLevel.call_count,10)
        generator.event.__iter__.assert_not_called()

    def test_failed_initialization_is_not_ignored(self):
        generator=self.fake([])
        generator.init.return_value=False
        with patch.object(decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
            with self.assertRaisesRegex(RuntimeError,'initialization failed'):
                decayProducts.process_events_with_pythia([],2.)


class BackendSwitchTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(model='ALP-fermion', events=10,
                           masses=[2.], c_taus=[10.], decay_channels=['all'])

    def test_exhad_is_the_default_for_supported_models(self):
        for model in ('ALP-fermion', 'Dark-photons', 'Scalar-mixing', 'Scalar-quartic'):
            with self.subTest(model=model):
                values = {**self.values, 'model': model}
                if model == 'Dark-photons':
                    values['uncertainty'] = 'central'
                config = config_from_mapping(values, check_files=False)
                self.assertEqual(config.hadronization, 'exhad')

    def test_explicit_raw_card_is_respected(self):
        config = config_from_mapping({**self.values, 'hadronization': 'raw'}, check_files=False)
        self.assertEqual(config.hadronization, 'raw')

    def test_unsupported_model_requires_explicit_raw_switch(self):
        values = {**self.values, 'model': 'ALP-SU2L'}
        with self.assertRaisesRegex(ConfigurationError, '--rawPythia'):
            config_from_mapping(values, check_files=False)
        config = config_from_mapping({**values, 'hadronization': 'raw'}, check_files=False)
        self.assertEqual(config.hadronization, 'raw')

    def test_original_launcher_flag_overrides_card(self):
        from funcs import simulation_config as module
        for flag, card_backend, expected in (
                ('--rawPythia', 'exhad', 'raw'), ('--raw-pythia', 'exhad', 'raw'),
                ('--exhad', 'raw', 'exhad')):
            with self.subTest(flag=flag), patch.object(module, 'load_card',
                    return_value={**self.values, 'hadronization': card_backend}), patch.object(
                        module, 'config_from_mapping', side_effect=lambda values, **kw:
                        config_from_mapping(values, check_files=False)):
                config, _ = module.config_from_command_line(['--card', 'unused.json', flag])
                self.assertEqual(config.hadronization, expected)

    def test_runtime_helpers_override_card(self):
        import runtime_generator as runtime
        for helper in ('generator_from_card', 'scan_from_card'):
            constructor = 'RuntimeEventGenerator' if helper == 'generator_from_card' else 'RuntimeModelScan'
            for override, card_backend, expected in (
                    ('raw', 'exhad', 'raw'), ('exhad', 'raw', 'exhad'), (None, 'raw', 'raw')):
                with self.subTest(helper=helper, override=override), patch.object(runtime, 'load_card',
                        return_value={**self.values, 'hadronization': card_backend}), patch.object(
                            runtime, 'config_from_mapping', side_effect=lambda values, **kw:
                            config_from_mapping(values, check_files=False)), patch.object(
                                runtime, constructor) as factory:
                    kwargs = dict(mass=2., c_tau=10.) if helper == 'generator_from_card' else {}
                    getattr(runtime, helper)(Path('unused.json'), hadronization=override, **kwargs)
                    self.assertEqual(factory.call_args.args[0].hadronization, expected)

    def test_runtime_cli_forwards_raw_switch(self):
        import runtime_generator as runtime
        with patch.object(runtime, 'generator_from_card', side_effect=RuntimeError('test stop')) as factory:
            with self.assertRaisesRegex(RuntimeError, 'test stop'):
                runtime.main(['--card', 'unused.json', '--mass', '2', '--ctau', '10',
                              '--events', '10', '--output-dir', 'unused', '--rawPythia'])
        self.assertEqual(factory.call_args.kwargs['hadronization'], 'raw')

    def test_interactive_launcher_supports_only_the_raw_switch(self):
        import simulate
        with patch.object(simulate, '_interactive_main') as interactive:
            self.assertEqual(simulate.main(['--rawPythia']), 0)
            interactive.assert_called_once_with(hadronization='raw')

    def test_interactive_launcher_defaults_to_exhad(self):
        import inspect
        import simulate
        self.assertEqual(inspect.signature(simulate._interactive_main).parameters['hadronization'].default,
                         'exhad')
        with patch.object(simulate, '_interactive_main') as interactive:
            self.assertEqual(simulate.main([]), 0)
            interactive.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
