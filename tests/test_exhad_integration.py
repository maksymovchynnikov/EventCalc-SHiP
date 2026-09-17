"""Contracts of the exHad compatibility shim and of the decay entry point it
calls; tests/smoke_exhad.py exercises real Pythia."""
import contextlib
from contextlib import ExitStack
import inspect
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = str(Path(__file__).resolve().parents[1])

from funcs import decayProducts
from funcs import exhad_release
from funcs import exhadDecays
from funcs import exhad_integration as integration
from funcs.simulation_config import ConfigurationError, config_from_mapping


SHIM_NAMES = ('MODELS', 'configure_llp', 'metadata', 'matched_process_labels',
              'simulate_decays_weighted', '_generator')

REST_FRAME_ORDER = ('mass', 'PDGdecay', 'BrRatio', 'size', 'Msquared3BodyLLP',
                    'selected_decay_indices', 'br_visible_val')


def call_as_dict(call):
    """Positional and keyword arguments of one rest-frame call, by name."""
    values = dict(zip(REST_FRAME_ORDER, call.args))
    values.update(call.kwargs)
    return values


class ShimTests(unittest.TestCase):
    """funcs/exhad_integration.py holds no logic: every name it offers is the
    one funcs/exhadDecays.py defines, and the two calls it still makes go to
    the merged decay entry point."""

    def test_shim_reexports_the_adapter(self):
        for name in SHIM_NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(integration, name),
                              getattr(exhadDecays, name))

    def test_fermion_alp_import_is_the_llp_method(self):
        llp = Mock()
        integration.import_fermion_alp(llp)
        llp.import_ALP_fermion.assert_called_once_with()

    def test_unweighted_decays_go_to_the_merged_entry_point(self):
        llp = SimpleNamespace(LLP_name='Dark-photons',
                              particle_path='Distributions/Dark-photons',
                              scalar_lifetime=None)
        module = Mock()
        module.simulateDecays_rest_frame.return_value = ('events', 'sizes', 'labels')
        pdgs = [np.array([-1, 1]), np.array([22, 22])]
        result = integration.simulate_decays(llp, module, 2., pdgs, [.8, .2], 10,
                                             'matrix', [0, 1], 1., seed=17)
        self.assertEqual(result, ('events', 'sizes', 'labels'))
        module.simulateDecays_rest_frame.assert_called_once()
        values = call_as_dict(module.simulateDecays_rest_frame.call_args)
        self.assertEqual(values['mass'], 2.)
        self.assertIs(values['PDGdecay'], pdgs)
        self.assertEqual(values['BrRatio'], [.8, .2])
        self.assertEqual(values['size'], 10)
        self.assertEqual(values['Msquared3BodyLLP'], 'matrix')
        self.assertEqual(list(values['selected_decay_indices']), [0, 1])
        self.assertEqual(values['br_visible_val'], 1.)
        self.assertEqual(values['llp_name'], 'Dark-photons')
        self.assertEqual(values['particle_path'], 'Distributions/Dark-photons')
        self.assertIsNone(values['exhad_variant'])
        self.assertEqual(values['seed'], 17)
        self.assertIs(values['return_process_labels'], True)

    def test_weighted_decays_go_to_the_importance_sampling_path(self):
        llp = SimpleNamespace(LLP_name='ALP-fermion', particle_path='unused',
                              scalar_lifetime=None)
        module = Mock()
        with patch.object(integration, 'simulate_decays_weighted',
                          return_value='weighted') as weighted:
            result = integration.simulate_decays(llp, module, 2., [], [], 10, None,
                                                 [0], 1., seed=5, weighted=True,
                                                 weight_floor_fraction=.1)
        self.assertEqual(result, 'weighted')
        weighted.assert_called_once()
        self.assertEqual(weighted.call_args.kwargs['seed'], 5)
        self.assertEqual(weighted.call_args.kwargs['weight_floor_fraction'], .1)
        module.simulateDecays_rest_frame.assert_not_called()

    def test_text_export_preserves_row_order_and_labels_every_row(self):
        """R7: the rows are written in the order they were sampled, and each
        pooled row carries its own label."""
        llp = SimpleNamespace(decayChannels=['quarks-a', 'leptons', 'quarks-b'],
                              _exhad_pooled_indices=(0, 2))
        mothers = np.arange(5).reshape(-1, 1)
        daughters = mothers + 10
        labels = ['Jets-matched:quarks-a', None, 'Jets-matched:quarks-b']
        with patch.object(integration, 'matched_process_labels',
                          return_value=labels) as naming:
            m, d, produced, counts, indices = integration.legacy_output(
                llp, mothers, daughters, [2, 1, 2], [0, 1, 2])
        np.testing.assert_array_equal(m, mothers)
        np.testing.assert_array_equal(d, daughters)
        self.assertEqual(produced, labels)
        self.assertEqual(list(counts), [2, 1, 2])
        self.assertEqual(list(indices), [0, 1, 2])
        self.assertEqual(naming.call_args.args[0], llp.decayChannels)
        self.assertEqual(list(naming.call_args.args[1]), [0, 1, 2])


class GeneratorTests(unittest.TestCase):
    def test_generator_configuration_and_metadata(self):
        """The release exposes one Generator, exercised against the real
        release rather than a mock."""
        root = os.environ.get('EXHAD_ROOT')
        if not root:
            self.skipTest('EXHAD_ROOT is not configured')
        module = exhad_release.release(root)
        self.assertTrue(hasattr(module, 'Generator'))
        self.assertFalse(hasattr(module, 'ParallelGenerator'))
        binding = exhad_release.bind(root, 'alp-fermion')
        with patch.dict(os.environ, EXHAD_WORKERS='2', EXHAD_CHUNK_SIZE='17'), \
                patch.object(exhad_release, '_GENERATORS', {}):
            first = integration._generator(binding)
            self.assertIs(integration._generator(binding), first)
            self.assertEqual(binding['execution'],
                             dict(workers=2, chunk_size=17, seed_stream='chunked-v1'))
            exhad_release.close_generators()

    def test_invalid_parallel_settings_fail_before_loading(self):
        for workers, size in [('0', '512'), ('65', '512'), ('2', '0')]:
            with patch.dict(os.environ, EXHAD_WORKERS=workers,
                            EXHAD_CHUNK_SIZE=size), self.assertRaises(ValueError):
                integration._generator({})

    def test_release_without_host_interface_is_rejected(self):
        """A release with no model_info is not bound and not run around.

        configure_llp reaches the release through exhad_release.model_info,
        which refuses a module that does not offer the host interface.  The
        refusal has to leave the LLP unbound rather than half-bound.
        """
        root = os.environ.get('EXHAD_ROOT')
        if not root:
            self.skipTest('EXHAD_ROOT is not configured')
        exhadDecays.set_selection('ALP-fermion', 'Distributions/ALP-fermion', None)
        llp = SimpleNamespace(LLP_name='ALP-fermion',
                              particle_path='Distributions/ALP-fermion')
        config = SimpleNamespace(exhad_mode='on', exhad_root=root, exhad_python=None)
        with patch.object(exhad_release, 'release',
                          return_value=SimpleNamespace(Generator=Mock())), \
                self.assertRaisesRegex(ValueError, 'model_info'):
            integration.configure_llp(llp, config)
        self.assertIsNone(llp._exhad_binding)

    def test_exhad_on_without_a_usable_release_stops_the_run(self):
        """R9: `on` keeps the hard failure for anyone who wants it."""
        exhadDecays.set_selection(None, None, None)
        llp = SimpleNamespace(LLP_name='ALP-fermion')
        config = SimpleNamespace(exhad_mode='on', exhad_root=None, exhad_python=None)
        with self.assertRaises((RuntimeError, ValueError)):
            integration.configure_llp(llp, config)
        self.assertIsNone(llp._exhad_binding)

    def test_exhad_auto_without_a_usable_release_runs_on_the_baseline(self):
        """R9: with `auto` and no usable card or bridge the run proceeds."""
        exhadDecays.set_selection(None, None, None)
        llp = SimpleNamespace(LLP_name='ALP-fermion')
        integration.configure_llp(llp, SimpleNamespace(
            exhad_mode='auto', exhad_root=None, exhad_python=None))
        self.assertIsNone(llp._exhad_binding)

    def test_switching_exhad_off_needs_no_exhad_installation(self):
        llp = SimpleNamespace(LLP_name='ALP-fermion')
        integration.configure_llp(llp, SimpleNamespace(
            exhad_mode='off', exhad_root=None, exhad_python=None))
        self.assertIsNone(llp._exhad_binding)

    def test_below_support_uses_the_release_exclusive_rows(self):
        """Below the matched support exHad supplies exact exclusive rows.

        Flat phase space there left the 1.699-1.700 GeV dark-photon gap and the
        sub-threshold neutral-kaon NaNs.  Checked against the real release:
        generate_exclusive returns exactly the requested counts for rows that
        are open at the mass.
        """
        root = os.environ.get('EXHAD_ROOT')
        if not root:
            self.skipTest('EXHAD_ROOT is not configured')
        binding = exhad_release.bind(root, 'dark-photon')
        self.assertEqual(exhad_release.check_mass(binding, 1.6995), 'exclusive')
        self.assertEqual(exhad_release.check_mass(binding, 1.7), 'matched')
        label = next(iter(binding['exclusive_rows']))
        try:
            produced = exhad_release.generate_exclusive(binding, 1.6995, {label: 8}, seed=3)
            self.assertEqual(len(produced[label]), 8)
        finally:
            exhad_release.close_generators()


class PooledRowTests(unittest.TestCase):
    """The fixed-mass pool, now that the routing lives in the decay entry
    point rather than in a driver layer above it."""

    def setUp(self):
        self.pdgs = [np.array([-1, 1]), np.array([-2, 2]), np.array([22, 22])]
        self.branching = [.4, .4, .2]

    def routing(self, **overrides):
        """Pin the predicates the entry point consults for a fixed-mass pool."""
        pinned = dict(is_hnl_bench=False, exclusive_enabled=False,
                      uses_portable_model1_backend=False,
                      fixed_partonic_row_supported=True, enabled=True)
        pinned.update(overrides)
        stack = ExitStack()
        stack.enter_context(patch.object(exhadDecays, 'set_selection'))
        for name, value in pinned.items():
            stack.enter_context(
                patch.object(exhadDecays, name, return_value=value))
        return stack

    def generate(self, selected, size=10, mass=2., **overrides):
        def two_body(_mass, count, *args):
            return np.zeros((count, 16), dtype=float)

        def exhad_events(count, _mass, seed=1):
            rows = np.zeros((count, 6), dtype=float)
            rows[:, 5] = 22.
            return rows

        pool = Mock(side_effect=exhad_events)
        with self.routing(**overrides), \
                patch.object(decayProducts.TwoBodyDecay, 'decay_products',
                             side_effect=two_body), \
                patch.object(exhadDecays, 'process_events_with_exhad', pool), \
                contextlib.redirect_stdout(io.StringIO()):
            np.random.seed(20260916)
            produced = decayProducts.simulateDecays_rest_frame(
                mass, self.pdgs, self.branching, size, None, list(selected),
                sum(self.branching[index] for index in selected), seed=8)
        return produced, pool

    def test_pool_is_drawn_once_and_the_row_counts_are_preserved(self):
        (events, sizes), pool = self.generate((0, 1, 2))
        pool.assert_called_once()
        self.assertEqual(pool.call_args.args[0], int(sizes[0]) + int(sizes[1]))
        self.assertEqual(len(events), 10)
        self.assertEqual(int(np.sum(sizes)), 10)

    def test_partial_hadronic_pool_is_refused(self):
        """A complete-pool backend cannot represent part of its own pool."""
        config = dict(eventcalc_rows=[
            dict(authority_row_id='Jets-a', pdg_signature=[-1, 1]),
            dict(authority_row_id='Jets-b', pdg_signature=[-2, 2])])
        labels = {(-1, 1): 'Jets-a', (-2, 2): 'Jets-b'}
        with self.routing(uses_portable_model1_backend=True), \
                patch.object(exhadDecays, 'portable_model1_configuration',
                             return_value=config), \
                patch.object(exhadDecays, 'generation_window',
                             return_value=(0., 5.)), \
                patch.object(exhadDecays, 'portable_model1_eventcalc_row_label',
                             side_effect=lambda ids: labels.get(
                                 tuple(int(value) for value in ids))):
            with self.assertRaisesRegex(ValueError, 'complete hadronic'):
                decayProducts._fixed_exhad_context(
                    2., self.pdgs, [0, 2], llp_name='Scalar-mixing')

    def test_a_zero_rate_row_may_be_omitted(self):
        self.branching = [.8, 0., .2]
        (events, sizes), pool = self.generate((0, 2))
        self.assertEqual(len(events), 10)
        pool.assert_called_once()

    def test_a_generator_failure_is_not_a_silent_baseline_run(self):
        def two_body(_mass, count, *args):
            return np.zeros((count, 16), dtype=float)

        with self.routing(), \
                patch.object(decayProducts.TwoBodyDecay, 'decay_products',
                             side_effect=two_body), \
                patch.object(exhadDecays, 'process_events_with_exhad',
                             side_effect=RuntimeError('broken card')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'broken card'):
                decayProducts.simulateDecays_rest_frame(
                    2., self.pdgs, self.branching, 10, None, [0, 1, 2], 1.,
                    seed=8)


class MassWindowTests(unittest.TestCase):
    def tearDown(self):
        exhadDecays.set_selection(None, None, None)
        exhadDecays.set_force(None)

    def test_below_the_matched_window_the_exclusive_rows_are_used(self):
        """R4: the exclusive branch owns the rows the release names."""
        pdgs = [np.array([211, -211]), np.array([22, 22])]
        with patch.object(exhadDecays, 'set_selection'), \
                patch.object(exhadDecays, 'is_hnl_bench', return_value=False), \
                patch.object(exhadDecays, 'exclusive_enabled',
                             return_value=True), \
                patch.object(exhadDecays, 'exclusive_row_label',
                             side_effect=lambda ids, **kw:
                             'Pip_Pim' if len(ids) == 2 and 211 in ids
                             else None):
            module, positions = decayProducts._fixed_exhad_context(
                1.5, pdgs, [0, 1], llp_name='Dark-photons')
        self.assertIs(module, exhadDecays)
        self.assertEqual(positions, [0])

    def test_above_the_generation_window_the_run_stops(self):
        with patch.object(exhadDecays, 'set_selection'), \
                patch.object(exhadDecays, 'is_hnl_bench', return_value=False), \
                patch.object(exhadDecays, 'exclusive_enabled',
                             return_value=False), \
                patch.object(exhadDecays, 'uses_portable_model1_backend',
                             return_value=True), \
                patch.object(exhadDecays, 'portable_model1_configuration',
                             return_value=dict(eventcalc_rows=[])), \
                patch.object(exhadDecays, 'generation_window',
                             return_value=(1.911, 5.)), \
                patch.object(exhadDecays, 'get_bench', return_value='alp'):
            with self.assertRaisesRegex(ValueError, 'up to 5'):
                decayProducts._fixed_exhad_context(
                    5.1, [np.array([-1, 1])], [0], llp_name='ALP-fermion')

    def test_alp_fermion_boundaries_come_from_the_installed_card(self):
        """R11: the two boundaries are the card's, not a constant in code."""
        from funcs import alp_fermion
        path = os.path.join(ROOT, 'Distributions', 'ALP-fermion')
        self.assertEqual(alp_fermion.load_exhad_boundaries(path),
                         (1.911, 3.741))

    def test_photon_alp_does_not_borrow_the_fermion_benchmark(self):
        """The photon-coupled ALP is a different benchmark and has no card."""
        self.assertNotIn('ALP-photon', exhadDecays.BENCH_BY_LLP)
        exhadDecays.set_selection(
            'ALP-photon', os.path.join(ROOT, 'Distributions', 'ALP-photon'))
        self.assertIsNone(exhadDecays.get_bench())
        self.assertFalse(exhadDecays.can_use_exhad())


class MatrixElementTests(unittest.TestCase):
    def test_exclusive_alp_threshold_steps_are_numeric(self):
        from funcs.initLLP import LLP
        llp = LLP.__new__(LLP)
        llp.Matrix_elements_expr = []
        func, = llp.compile_matrix_elements(['UnitStep(E1 - 0.3) + UnitStep(E3 - 0.5)'])
        # The decay tables write daughter masses symbolically, so every
        # compiled matrix element takes (mLLP, E1, E3, m1, m2, m3).
        np.testing.assert_array_equal(
            func(1.8, np.array([.2, .4]), np.array([.6, .4]), .1396, .1396, .1349),
            [1., 1.])


class HNLIntegrationTests(unittest.TestCase):
    def test_hnl_config_is_supported(self):
        config = config_from_mapping(dict(model='HNL', events=10, masses=[2.],
            c_taus=[10.], mixing_pattern=[1., 0., 0.], decay_channels=['all']),
            check_files=False)
        self.assertEqual(config.exhad_mode, 'auto')

    def test_hnl_mass_ceiling_is_the_end_of_generation(self):
        """The HNL mass runs to the end of generation.

        The upper edge of the release's matched support bounds W, the invariant
        mass of the hadronic system, which an HNL shares with a charged lepton
        or a neutrino; an HNL mass above that edge is generated.
        """
        binding = {'model': 'hnl', 'matched_start': .02, 'matched_w_end': 5.27,
                   'generation_start': .02, 'generation_end': 40., 'exclusive_gev': None}
        for mass in (5.28, 6., 40.):
            self.assertEqual(exhad_release.check_mass(binding, mass, allow_exclusive=False),
                             'matched')


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
        with patch.object(decayProducts,'_PYTHIA_INSTANCE',None), patch.object(
                decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
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
        with patch.object(decayProducts,'_PYTHIA_INSTANCE',None), patch.object(
                decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
            with self.assertRaisesRegex(RuntimeError,'after 10 attempts'):
                decayProducts.process_events_with_pythia([[0.,0.,0.,2.,2.,221.,0.,0.]],2.)
        self.assertEqual(generator.forceHadronLevel.call_count,10)
        generator.event.__iter__.assert_not_called()

    def test_failed_initialization_is_not_ignored(self):
        generator=self.fake([])
        generator.init.return_value=False
        with patch.object(decayProducts,'_PYTHIA_INSTANCE',None), patch.object(
                decayProducts,'load_pythia8',return_value=SimpleNamespace(Pythia=lambda:generator)):
            with self.assertRaisesRegex(RuntimeError,'initialization failed'):
                decayProducts.process_events_with_pythia([[0.,0.,0.,2.,2.,221.,0.,0.]],2.)


class HadronizationModeTests(unittest.TestCase):
    """--exhad {auto,on,off}, its aliases, and the card key that maps onto it."""

    def setUp(self):
        self.values = dict(model='ALP-fermion', events=10,
                           masses=[2.], c_taus=[10.], decay_channels=['all'])

    def test_default_is_auto(self):
        for model in ('ALP-fermion', 'Dark-photons', 'Scalar-mixing', 'Scalar-quartic',
                      'ALP-SU2L'):
            with self.subTest(model=model):
                values = {**self.values, 'model': model}
                if model == 'Dark-photons':
                    values['uncertainty'] = 'central'
                config = config_from_mapping(values, check_files=False)
                self.assertEqual(config.exhad_mode, 'auto')

    def test_card_hadronization_key_maps_onto_the_mode(self):
        for card_value, expected in (('exhad', 'on'), ('rawPythia', 'off')):
            with self.subTest(hadronization=card_value):
                config = config_from_mapping(
                    {**self.values, 'hadronization': card_value}, check_files=False)
                self.assertEqual(config.exhad_mode, expected)

    def test_command_line_flags_and_their_aliases(self):
        from funcs import simulation_config as module
        for argv, expected in (
                (['--rawPythia'], 'off'),
                (['--raw-pythia'], 'off'),
                (['--exhad'], 'on'),
                (['--exhad', 'on'], 'on'),
                (['--exhad', 'off'], 'off'),
                (['--exhad', 'auto'], 'auto'),
                ([], 'auto')):
            with self.subTest(argv=argv), patch.object(
                    module, 'load_card', return_value=dict(self.values)), patch.object(
                        module, 'config_from_mapping',
                        side_effect=lambda values, **kw: config_from_mapping(
                            values, check_files=False)):
                config, _ = module.config_from_command_line(['--card', 'unused.json', *argv])
                self.assertEqual(config.exhad_mode, expected)

    def test_interactive_launcher_defaults_to_auto(self):
        import simulate
        self.assertEqual(
            inspect.signature(simulate._interactive_main).parameters['exhad_mode'].default,
            'auto')

    def test_interactive_launcher_forwards_the_raw_switch(self):
        import simulate
        with patch.object(simulate, '_interactive_main') as interactive:
            self.assertEqual(simulate.main(['--rawPythia']), 0)
            interactive.assert_called_once_with(exhad_mode='off')

    def test_runtime_helpers_override_card(self):
        import runtime_generator as runtime
        for helper in ('generator_from_card', 'scan_from_card'):
            constructor = ('RuntimeEventGenerator'
                           if helper == 'generator_from_card'
                           else 'RuntimeModelScan')
            for override, card_mode, expected in (
                    ('off', 'on', 'off'), ('on', 'off', 'on'),
                    (None, 'off', 'off')):
                with self.subTest(helper=helper, override=override), \
                        patch.object(runtime, 'load_card',
                                     return_value={**self.values,
                                                   'exhad_mode': card_mode}), \
                        patch.object(runtime, 'config_from_mapping',
                                     side_effect=lambda values, **kw:
                                     config_from_mapping(values,
                                                         check_files=False)), \
                        patch.object(runtime, constructor) as factory:
                    kwargs = (dict(mass=2., c_tau=10.)
                              if helper == 'generator_from_card' else {})
                    getattr(runtime, helper)(Path('unused.json'),
                                             exhad_mode=override, **kwargs)
                    self.assertEqual(factory.call_args.args[0].exhad_mode,
                                     expected)

    def test_runtime_cli_forwards_the_mode(self):
        import runtime_generator as runtime
        with patch.object(runtime, 'generator_from_card',
                          side_effect=RuntimeError('test stop')) as factory:
            with self.assertRaisesRegex(RuntimeError, 'test stop'):
                runtime.main(['--card', 'unused.json', '--mass', '2',
                              '--ctau', '10', '--events', '10',
                              '--output-dir', 'unused', '--rawPythia'])
        self.assertEqual(factory.call_args.kwargs['exhad_mode'], 'off')

    def test_a_model_with_no_release_model_refuses_on(self):
        """A model exHad could hadronize but has no release model for is
        refused; one with nothing to hadronize reports the request instead."""
        values = {**self.values, 'model': 'ALP-SU2L'}
        with self.assertRaisesRegex(ConfigurationError, 'No exHad'):
            config_from_mapping({**values, 'hadronization': 'exhad'},
                                check_files=False)
        self.assertEqual(
            config_from_mapping({**values, 'hadronization': 'rawPythia'},
                                check_files=False).exhad_mode, 'off')
        photon = config_from_mapping(
            {**self.values, 'model': 'ALP-photon', 'hadronization': 'exhad',
             'alp_production_mode': 'primary'},
            check_files=False)
        self.assertEqual(photon.exhad_mode, 'on')


if __name__ == '__main__':
    unittest.main()
